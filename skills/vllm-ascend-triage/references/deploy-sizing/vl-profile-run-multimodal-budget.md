# A VL model's engine start OOMs inside the vision tower, and the reporter blames TP

Scope: a **VL/omni** serve or offline `LLM(...)` whose start dies with
`RuntimeError: NPU out of memory. Tried to allocate X GiB (NPU n; T total capacity; A
already allocated; …)` and whose **innermost frame is in the vision tower**, typically

```text
worker/worker.py:… determine_num_available_blocks
  → worker/model_runner.py:… profile_run → execute_model
  → models/qwen2_5_vl.py:… get_input_embeddings_v0 → _process_video_input → self.visual(...)
  → <downstream>_vl.py:… split_qkv → all_gather_interleave → torch.cat
```

The report usually sets **only** `tensor_parallel_size` and then argues from two runs
that "more TP OOMed while less TP worked, so it must be a bug in TP". It is not:
the failing forward is the **dummy profiling batch**, whose size the launch derives
from `max_num_batched_tokens` and from the per-modality limits — and the ViT peak per
card does not fall with TP (step 2: it grows). Deliverable: which flag sizes the dummy batch, the one
ablation that holds TP fixed, and the budget the pin's own recipe uses.

Not this topic: a VL start that **stalls** after
`Encoder cache will be initialized with a budget of N tokens` with no traceback
(`../startup-failures/mm-encoder-profile-stall.md`), a `TP4`-up/`TP8`-OOM pair on a
**text** model (`tp-scaling-oom-at-startup.md`), or a VL serve that starts and then
answers garbage (`../accuracy-benchmarks/vl-vision-tower-degeneracy.md`).

Verified at vllm **v0.8.5.post1** (`3015d563`) + vllm-ascend **v0.8.5rc1**
(`ec27af34`) for recorded **#1617** (Qwen2.5-VL-7B, 4×910B, 60.96 GiB/card,
`LLM(model=…, trust_remote_code=True, tensor_parallel_size=4)` and nothing else).
Re-grep the file names at another ref: V1's runner moved the same decision to
`vllm/v1/worker/gpu_model_runner.py` and renamed the numbers
(`../startup-failures/mm-encoder-profile-stall.md` has the v0.13 mapping).

## Step 1 — the three numbers are the reporter's defaults, not their traffic

In the V0 runner (`vllm/worker/model_runner.py:1233-1340`):

```python
def profile_run(self):
    max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
    max_num_seqs = self.scheduler_config.max_num_seqs
    self._dummy_run(max_num_batched_tokens, max_num_seqs)
...
max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(self.model_config)   # sum over modalities
if max_mm_tokens > 0:
    max_num_seqs = min(max_num_seqs, max_num_batched_tokens // max_mm_tokens)
for group_id in range(max_num_seqs):
    seq_len = max_num_batched_tokens // max_num_seqs + (…)
    dummy_data = self.input_registry.dummy_data_for_profiling(…, seq_len, self.mm_registry)
```

So the dummy batch is **`max_num_batched_tokens` tokens of max-feature-size visual
data**, split into `min(max_num_seqs, max_num_batched_tokens // max_mm_tokens)`
sequences. On a VL checkpoint the small `max_mm_tokens` collapses that to one or two
sequences, each of which carries a **maximum-resolution image and a maximum-resolution
video frame** in one forward. None of it comes from the reporter's prompts.

With `LLM(model=…)` and no scheduler flags, the three inputs are:

| input | value on #1617 | source |
|---|---|---|
| `max_model_len` | the checkpoint's own value (32768 for Qwen2.5-VL-7B) — **only because this construction passes no `--max-model-len`**; a serve that carries the flag logs the flag instead | `ModelConfig` resolves it from the HF config (`vllm/config/model.py:1898-2026`: the user's value is returned verbatim when given, else the derived length) |
| `max_num_batched_tokens` | `max(max_model_len, _DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048)` = **32768**, then raised to the multimodal floor `_MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120` | `vllm/config.py:68,1979-1983`, floor at `:70,1992-1997` |
| `max_num_seqs` | **256** (then clamped by `max_mm_tokens`) | `vllm/engine/arg_utils.py:1561-1563` — `_set_default_args_v0`, "Set max_num_seqs to 256 for VLLM_V0", which `create_engine_config` runs before building `SchedulerConfig` (`:1118-1122`). The dataclass's own `128` (`vllm/config.py:1961-1965`) is dead on any CLI / `LLM(...)` path |
| `limit_mm_per_prompt` | empty ⇒ **1 item for every supported modality** | `vllm/config.py:2800-2808` (`999 if envs.VLLM_USE_V1 else 1`), `vllm/multimodal/processing.py:1038-1050` |

The last row is the one reporters never expect: Qwen2.5-VL supports **image and
video**, so the V0 dummy prompt carries one of each — which is why the #1617 paste's
deepest frames are `_process_video_input` on a benchmark that only ever sends images.

The 5120 floor bounds the knobs, so say where it bites: it applies only while
`max_num_batched_tokens` is **unset** (`:1979`), and it means a VL model's dummy batch
never drops below **5120 tokens** however small `max_model_len` gets. The #1617 working
config's `max_model_len=4096` therefore still profiles a 5120-token dummy batch (then
`min(256, 5120 // max_mm_tokens) = 0` → the `Setting it to the minimum value of 1`
warning, `vllm/worker/model_runner.py:1296-1310`): that change buys 6.4×, not 8×, and
going below 4096 buys nothing. With `--enable-chunked-prefill` unset, V0 also decides
chunked prefill **off** for a multimodal model (`vllm/engine/arg_utils.py:1506-1509`),
which is why the `max(max_model_len, 2048)` branch is the live one rather than the flat
2048 of the chunked branch above it.

Say the arithmetic in the answer, because it answers the literal question: the model's
weights are `Loading model weights took 4.7006 GB` per card at TP4, so on a card whose
paste reads `52.87 GiB already allocated; 2.78 GiB free` the ~48 GiB residual is
**vision-tower activations of the dummy batch**, not weights, KV cache (not yet
allocated at `profile_run`), or the 7B checkpoint.

## Step 2 — TP is not the lever, and one code read shows it

`all_gather_interleave` (`vllm/model_executor/models/qwen2_5_vl.py:200-217`) allocates
`tp_size` buffers of the **full** qkv size, `dist.all_gather`s the rank-local qkv into
them, splits and `torch.cat`s — every rank ends up holding the **unsharded** qkv, and
the paste's failing request (`Tried to allocate 4.50 GiB` at `torch.cat`) is exactly
that cat. `split_qkv` (`:258-264`) enters it only when `tp_size > 1`, so:

- TP=1 skips the gather and the cat (that is the whole of the "TP1 works" half);
- TP>1 adds `tp_size` full-size copies **plus** the cat on top of the same per-rank
  ViT activations, i.e. the peak **grows** with TP while only the LLM weights divide.

**Later pins close the TP half of the shape.** From vllm v0.15.0 (`f1764434`)
`--mm-encoder-tp-mode` exists (default `weights`, `vllm/config/multimodal.py:117`):
`data` runs the ViT data-parallel — full weights on every rank, the **batch** sliced
across the TP group and gathered after the tower
(`vllm/model_executor/models/vision.py:384-421` `run_dp_sharded_mrope_vision_model`,
entered from `is_vit_use_data_parallel()` `:127-143`) — so the rank-local ViT activation
is *divided* instead of multiplied. It is accepted only for models declaring
`supports_encoder_tp_data` (Kimi-K2.5: `kimi_k25.py:294`, read by MoonViT at
`moonvit.py:314,350`); every other model gets `This model does not support
--mm-encoder-tp-mode data. Falling back to --mm-encoder-tp-mode weights`
(`config/model.py:580-587`, a `warning_once`, so a log tail can miss it). No such flag
exists at v0.8.5.post1 — on a ≥0.15 VL MoE check the serve line for it **before**
deriving a peak from `all_gather_interleave`, because the reporter-confirmed Kimi-K2.5
line already carries it (`moe-load-oom-expert-shard.md`).

Therefore "TP=4 OOMs and TP=1 doesn't" is not evidence of a TP defect, and the
reporter's own control is confounded anyway: on #1617 the working config also changed
`gpu_memory_utilization` 0.9→0.8, `max_model_len` 32768→4096, `max_num_seqs` 256→4 and
added `limit_mm_per_prompt={"image": 1}` — five variables at once.

## Step 3 — the discriminator and the ladder

1. **Hold TP fixed**: rerun the *working* config with `tensor_parallel_size=4`. It
   passes ⇒ TP is out, the budget page is in (this is the one run that settles the
   reporter's question).
2. `--limit-mm-per-prompt '{"image": 1, "video": 0}'` on the failing config: a modality
   limited to 0 contributes no dummy items, so the vision forward loses the video tail —
   the paste's `_process_video_input` frame predicts a visible peak drop (the same
   lever `../startup-failures/mm-encoder-profile-stall.md` uses one version later).
3. `--max-num-batched-tokens 2048` (or a `max_model_len` the KV cache can hold):
   the dummy batch shrinks linearly, which is the cheapest budget lever and needs no
   code change.
4. `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` — the pin's tutorials export it for
   fragmentation (the OOM line itself suggests it); it is a *secondary* lever that
   cannot absorb a 4.5 GiB request, so never ship it as the answer to this shape.

The pin's own recipe for this exact checkpoint is the answer to "what budget is
validated": `docs/source/tutorials/single_npu_multimodal.md:54-59` runs
`LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", max_model_len=16384,
limit_mm_per_prompt={"image": 10})`, and its serve form is
`--max_model_len 16384 --max-num-batched-tokens 16384` (`:147`). Quote it: the project
never leaves `max_model_len` at the checkpoint default and never leaves the mm limits
empty.

## Recorded case

**#1617** — vllm `0.8.5.post1+empty`, vllm-ascend `0.8.5rc1`, `CANN 8.1.RC1`,
4×910B1 64 GB, MathVista harness copied from a CUDA setup: `LLM(…, tensor_parallel_size=4)`
OOMs at `torch.cat` in `all_gather_interleave`, `Tried to allocate 4.50 GiB … 52.88 GiB
already allocated … 2.78 GiB free`, on **every** rank at the same second, after
`Loading model weights took 4.7006 GB`. `tensor_parallel_size=1` with
`gpu_memory_utilization=0.8, max_num_seqs=4, max_model_len=4096,
limit_mm_per_prompt={"image":1}` serves. The recorded session verified the mechanism
above (defaults, `max_mm_tokens` clamp, the gather+cat) and **shipped no answer**:
18 assistant turns / 27 tool calls, 43.4 K characters of reasoning against 2.2 K of
message text, cut at the 16 384-token cap on an empty final turn. The verdict it needed was one sentence — "the
dummy profile batch, not TP: cap `max_num_batched_tokens`/mm limits, then rerun your
working config at TP=4" — available from step 1's three defaults.
