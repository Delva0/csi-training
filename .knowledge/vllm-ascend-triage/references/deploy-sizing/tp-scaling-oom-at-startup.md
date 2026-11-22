# "Same flags, TP4 comes up but TP8 OOMs" at engine start

Scope: two `vllm serve` commands that differ **only** by `--tensor-parallel-size`
(usually 4 vs 8) on one node; the smaller one serves, the larger dies during
engine start with `NPU out of memory` / `torch.OutOfMemoryError` /
`Worker failed with error '... out of memory ...'`, or with the parent-only
`WorkerProc initialization failed` lane. Often a long `--max-model-len`
(128K-256K) and a hybrid linear-attention model (Qwen3-Next / Coder-Next,
Qwen3.5/3.6 GDN, DeepSeek-V4, GLM). Sometimes nothing is pasted but the two
commands and the model path.

Three sub-shapes, and they do **not** get the same answer:

- **A — a post-sizing phase outgrew the reserve** (the #6572 shape): the OOM frame
  is graph capture, a workspace or a collective, the KV pool was sized from the
  profile run and the reserve `(1-gmu) × T` is what ran out.
- **B — the pool itself does not fit while it is being allocated**: the `torch.zeros`
  in the traceback sits in `initialize_kv_cache_tensors` (worker
  `initialize_kv_cache_config` → `model_runner_v1.py:initialize_kv_cache_tensors`),
  right after the `GPU KV cache size:` / `Maximum concurrency` lines, and the
  context is *short* (`--max-model-len 4096`). #4474 is this sub-shape on
  Qwen3-Next-80B-A3B, v0.11.0rc2, TP4 up / TP8 OOM. Here `gmu × T` being exceeded
  is **expected**, not evidence of a post-sizing phase (step 2).
- **C — the model is a VL/omni checkpoint and the innermost frame is in the vision
  tower** (`split_qkv` → `all_gather_interleave` → `torch.cat`, reached from
  `get_input_embeddings_v0` → `_process_video_input`): the failing forward is the
  dummy *profiling* batch, its size is `max_num_batched_tokens`, and TP makes the
  ViT peak larger rather than smaller — `vl-profile-run-multimodal-budget.md`
  (recorded #1617), not this page's A/B arithmetic. Note that this sub-shape is the
  only one where "the larger TP OOMed" is a red herring rather than the report's own
  frame.

This is a **budget** question (`index.md` rules 1 and 4: arithmetic first), not a
TP8 defect and not something to resolve by reading the allocator. Worked case for
A: vllm-ascend #6572, Qwen3-Coder-Next on 910B3-64G, `--max-model-len 262144
--gpu-memory-utilization 0.80 --max-num-batched-tokens 4096
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, TP4 up / TP8 OOM.

Not this page: the failure is after a healthy serving period
(`../runtime-memory/index.md`), the OOM accompanies a collective/device error
(`../hccl-and-hardware/index.md`), the report is about wrong output on the
card that does start (`../accuracy-benchmarks/index.md`), or the OOM is raised
**inside model construction** on a 384-512-expert MoE under
`--enable-expert-parallel` (`w4a16.py get_weight` → `torch.empty`), which is an
experts-per-card question — `moe-load-oom-expert-shard.md`.

## Step 1 — diff the command against the project's own envelope (one read)

The tree ships the validated serve line for each tutorial model. Read it at the
pinned commit before any arithmetic on the reporter's numbers:

```bash
rg --files docs/source/tutorials | rg -i qwen3_next     # the file name moves per release
rg -n 'tensor-parallel-size|max-model-len|gpu-memory-utilization|enforce-eager' \
  docs/source/tutorials/<that file>          # and tests/e2e/**/test_<model>*.py
```

The tutorial file name **moves between releases** — `docs/source/tutorials/Qwen3-Next.md`
at v0.14.0rc1, `docs/source/tutorials/multi_npu_qwen3_next.md` at v0.11.0 — so
locate it (`rg --files docs/source/tutorials | rg -i qwen3_next`) instead of
assuming the path:

| Ref | Validated Qwen3-Next-80B-A3B line | Card rule on the same page |
|---|---|---|
| v0.14.0rc1 (`52d4acf`) | `Qwen3-Next.md:64`: TP4, `--max-model-len 32768`, gmu 0.8, `--max-num-batched-tokens 4096`, `FULL_DECODE_ONLY` | — |
| v0.13.0 (`6281c12`) | same file and line as v0.14.0rc1 — `Qwen3-Next.md:64`: TP4, `--max-model-len 32768`, gmu 0.8, `--max-num-batched-tokens 4096`, `FULL_DECODE_ONLY` | — (the v0.11 card-count note is **gone** at this ref) |
| v0.11.0rc2 (`a2e4c3f`) | `multi_npu_qwen3_next.md:89`: TP4, `--max-model-len 4096`, gmu 0.7, `--enforce-eager` — byte-for-byte the #4474 reporter's *working* command | `:86` "for an Atlas A2 with 64 GB of NPU card memory, tensor-parallel-size should be at least 4, and for 32 GB of memory, at least 8" |

So "TP4 at 256K" (#6572) is already outside the documented envelope and "TP8 at
256K" was never validated at any context; on #4474 TP4 at 4K *is* the shipped
recipe and TP8 at 4K is the **32 GB-card** recipe run on 64 GB cards. Say that
first: it is the answer to the literal question ("is TP8 broken?" — no, it is
unvalidated), and it costs one file read. The CI file
(`tests/e2e/multicard/4-cards/test_qwen3_next.py` at v0.14.0rc1, and unchanged
at v0.13.0) is TP4 as well.

## Step 2 — place the allocation by its frame, then read it as a budget

The traceback's **innermost frame decides the sub-shape**, before any arithmetic.
If it is `... initialize_kv_cache_config(kv_cache_config)` →
`model_runner_v1.py:<n> in initialize_kv_cache_tensors` → `torch.zeros(kv_cache_tensor.size)`,
the engine is filling the pool it sized in the previous phase, so `already
allocated` already contains the weights **plus the layers allocated so far**, and
`A > gmu × T` is the normal shape of a pool that does not fit. Do not read that
line as "a later phase outgrew the reserve" — that reading only applies when the
frame is capture, a workspace or a collective.

If the paste does carry
`NPU out of memory. Tried to allocate X GiB (NPU 0; T total capacity; A already
allocated; R reserved; F free)`:

```bash
bash <skill_root>/scripts/oom-budget-split.sh --line '<the OOM line>' --gmu 0.8 \
     --params 80 --tp 8            # add --dtype to price W8A8/W4A8 checkpoints
```

- frame **not** in the KV-pool loop, `A > gmu × T` → sub-shape A: the failing
  request is **outside** the profiled budget, the reserve `(1-gmu) × T` is the
  thing that is too small.
- `A ≤ gmu × T` → the request failed **inside** the budget the engine granted
  itself, so the accounting is wrong, not the model.
- `A` within a block of `gmu × T` → a coarse-granularity tail: on a hybrid model
  the pool allocates `page_size = 12 × attn_page + padded mamba page` per block, so
  the last block is tens to hundreds of MiB and the free remainder can be too small
  for the *next* one.

In #6572 the fellow reporter's line was `Tried to allocate 2.36 GiB (60.94 GiB
total; 58.82 GiB already allocated; 823.69 MiB free)` at gmu 0.8, i.e.
`A = 58.82 > 0.8 × 60.94 = 48.75` by ~10 GiB with no KV frame in the trace — a
sub-shape A failure, which is why the touch is the reserve/phase and not "more
cards".

### Sub-shape B: pool sized ≠ pool allocated

The sizing identity, at vllm v0.11.0 (`vllm/v1/core/kv_cache_utils.py`): the
hybrid/general path takes `group_size = max(len(group.layer_names))` (`:1054`,
36 for a 12-full-attention/36-linear model), `num_blocks = available_memory //
page_size // group_size` (`get_num_blocks`, `:813-827`, where `available_memory`
is vllm-ascend's `total × gmu − peak_memory`, `worker_v1.py:219-260`), and builds
`group_size` `KVCacheTensor`s of `size = page_size × num_blocks` each, with
`shared_by` holding **one layer per group** (`:1066-1070`). So

```
measured pool bytes  ==  (#tensors the runner allocates) × page_size × num_blocks
sized pool bytes     ==  group_size                            × page_size × num_blocks
```

and the two are equal **only if the runner shares one tensor per row** — which is
what upstream does (`vllm/v1/worker/gpu_model_runner.py:3770-3776`: one
`torch.zeros` per `kv_cache_tensor`, then every `shared_by` name maps to it).
vllm-ascend v0.11.0rc2 does not: `model_runner_v1.py:2916-2976` allocates a fresh
`torch.zeros(kv_cache_tensor.size)` per layer name in the `shared_by` list (mamba
branch) and k/v halves per attention layer, under the comment `# TODO: REFACTOR ME
to sharing hybrid cache` (`:2933`), and the file itself carries
`# TODO: remove this after the OOM issue is located and fixed` at `:2995-2996` (it is
that function's own reshape loop).
On a two-group hybrid a row carries one full-attention and one mamba layer, so the
rows that carry both are paid for twice: 48 tensors for 36 rows = **×4/3** of the
budget.

At v0.13.0 (`6281c12`) it reads the same way under new line numbers —
`_allocate_kv_cache_tensors` at `model_runner_v1.py:2469`, the
`# TODO: REFACTOR ME to sharing hybrid cache` at `:2492`, the `linear_attn` branch's
full-size `torch.zeros(kv_cache_tensor.size)` at `:2499`, and the function's own
`# TODO: remove this after the OOM issue is located and fixed` at `:2627` — while
upstream vllm v0.13.0 (`72506c9`) still allocates **one** `torch.zeros` per
`kv_cache_tensor` and maps every `shared_by` name onto it
(`v1/worker/gpu_model_runner.py:5056-5075`). So on a v0.13.0 report the divergence is
current behaviour, not a stale v0.11 reading. Recorded #6611 (910B3-64G, `v0.13.0`
reported on both sides, `Tried to allocate 2.57 GiB … 57.49 GiB already allocated …
1.96 GiB free, 57.62 GiB reserved`) died on `:2499` — the mamba branch, i.e. **inside**
the divergent loop — and its four frame line numbers (`worker.py:455`, `:2426`,
`:2450`, `:2499`) match the pinned tree exactly, so the installed tree is the tag and
the frame needs no re-derivation. That body carried no serve command at all, so the
envelope row above is the first thing to quote and the TP/gmu/context are the ask.

That is a verified upstream↔downstream divergence and the first thing to check when
the frame is the pool loop — but it is not by itself the #4474 answer: on that
report `A = 57.94 GiB` against `0.7 × 60.96 = 42.67 GiB` is ~15 GiB over, more
than ×4/3 of a ~22 GiB pool covers. **State the identity, quote the two log lines,
and leave the residual open** rather than deriving head counts from the screenshot:

- `Available memory: N, total memory: M` (`worker_v1.py:257`) — the number the
  engine granted the pool;
- `Setting attention block size to N tokens` / `Padding mamba page size by X%`
  (`patch_mamba_config.py`) — the page geometry it then used;
- `GPU KV cache size: N tokens` and `Maximum concurrency for M tokens per request:
  X` — **two different formulas**: `num_blocks // len(kv_cache_groups) ×
  min_block_size` (`kv_cache_utils.py:1080`) versus `num_blocks / num_block_per_request`
  with `num_block_per_request = cdiv(num_layer_per_group × max_memory_usage_bytes,
  page_size × num_layer_per_group)` (`:780-795`, `FullAttentionSpec.max_memory_usage_bytes
  = cdiv(max_model_len, block_size) × page_size`, `kv_cache_interface.py:83-91`).
  On a hybrid they differ by the group count and by the per-request block count, so
  `tokens = concurrency × max_model_len` is **not** an identity — #4474's paste
  reads `488,192 tokens` beside `401.58x` for 4,096-token requests, which those two
  formulas can produce without contradiction (either digit may still be
  OCR-garbled). Never reverse-engineer block size from screenshot digits; ask for
  the text log.

The lever that actually caps this class: `--num-gpu-blocks-override N` exists
upstream (`vllm/config/cache.py:64`, `may_override_num_blocks` at
`kv_cache_utils.py:798-810`) and vllm-ascend never touches it, so it caps
`num_blocks` under either sub-shape. On a hybrid remember the measured pool is
`#tensors × page × num_blocks`, so size it against the larger factor.

## Step 3 — which per-rank terms actually shrink with TP

Compute the three terms per card; only the first one halves:

| Term | Scaling | Source at vllm v0.14.1 / vllm-ascend v0.14.0rc1 |
|---|---|---|
| weights | `params × bytes ÷ TP` | checkpoint size; `scripts/card-budget.sh` |
| attention KV per token | `max(1, num_kv_heads // TP)` — **saturates** | `vllm/config/model.py:1142-1153` |
| mamba/GDN state per block | `conv_dim ÷ TP` and `num_v_heads ÷ TP` | `vllm/model_executor/layers/mamba/mamba_utils.py:170-193` |

Qwen3-Next-80B-A3B declares `num_hidden_layers=48`,
`num_key_value_heads=2`, `linear_num_{key,value}_heads` 16/32
(`vllm/transformers_utils/configs/qwen3_next.py:191-193,204-213`), so TP4 and TP8 both
resolve to **one** KV head per rank: the attention half of the KV cache is
identical at both sizes, and only the (small) mamba state halves.

The hybrid page geometry then moves too, because vllm-ascend *derives* the block
size from the mamba page and only ever raises it
(`vllm_ascend/patch/platform/patch_mamba_config.py:37-88`:
`attn_block_size = 128 × cdiv(mamba_page, 128 × attn_page_1_token)`, then
`:86-93` pads the mamba page up to the attention page; `vllm_ascend/utils.py:1139`
exempts `model_type == "qwen3_next"` from the usual "block size is 128" clamp).
Arithmetic for the 80B-A3B defaults (every 4th layer full attention → 36 linear /
12 full, head dims 128, conv kernel 4, bf16 state):

| TP | KV heads/rank | mamba page | derived block size | per block (12 attn + mamba) | blocks for 262144 | KV minimum for one 256K request |
|---|---|---|---|---|---|---|
| 4 | 1 | 9.42 MiB | 19,328 tokens | 122.7 MiB | 14 | 1.68 GiB |
| 8 | 1 | 4.71 MiB | 9,728 tokens | 61.8 MiB | 27 | 1.63 GiB |

Two consequences to state: the KV a 256K context needs is **TP-invariant**
(≈1.7 GiB per rank at either size), and the block is *thousands* of tokens, so
the pool's granularity is coarse and the effective block size is a number the
reporter's own log prints (`Setting attention block size to N tokens`,
`Padding mamba page size by X%`). Quote those lines back: they either match this
table (the geometry is as computed) or they do not (the checkpoint's head counts
differ from the defaults — ask for `config.json`).

Per-rank totals for this model: TP4 ≈ 40 GB weights + ~1.8 GB KV; TP8 ≈ 20 GB +
~1.8 GB. TP8 is *smaller* on every term above. Read that against the sub-shape:

- sub-shape A (post-sizing frame): the difference is an **unprofiled per-rank cost**
  (capture/workspace/collective buffers) or a reserve that is too small at that gmu
  — not the KV arithmetic; the fix is a budget/phase change.
- sub-shape B (pool-loop frame): the terms above are *per token / per block*, so the
  pool the engine can afford is **bigger** at TP8 (weights halve, `available_memory`
  grows), and a fixed multiplicative over-allocation in the allocation loop therefore
  costs many more GiB at TP8 than at TP4 — which is why the smaller TP starts. Here
  the number to compute is `#tensors × page_size × num_blocks` (step 2), not the KV
  minimum for one request.

Do not blend the two: say which frame the paste has before offering either reading.

## Step 4 — do not lead with the full-graph warning

Every `FULL_DECODE_ONLY` launch prints the canned block "You have enabled the
*full graph* feature … A known problem is that capturing too many batch sizes can
lead to OOM … consider reducing `gpu_memory_utilization`" (`vllm_ascend/platform.py:313-327` at
v0.14.0rc1). It is printed unconditionally; it is not evidence that capture owns
this OOM. On #6572 the reporter's later public follow-up (2026-02-10) states they
run full graph on many other models without this effect and only Qwen3-Next
behaves this way, and that they suspect the model-family-specific optimisations
(reporter judgement, not verified). So rank capture as one candidate with one
ablation — `--enforce-eager`, or `cudagraph_capture_sizes` reduced to `[1]`
(`scripts/aclgraph-size-budget.sh` prices the size set) — instead of deriving its
footprint from the warning text.

## Answer shape

1. Direct answer: TP8 was never a validated configuration for this model at that
   context; the shipped recipe at the ref is TP4 (32K window at v0.14.0rc1, 4K at
   v0.11.0rc2), and the reporter's working TP4 command is that recipe verbatim
   (#4474). TP8 is the 32 GB-card recipe on 64 GB cards (#4474's own page).
2. The two per-card numbers from step 3 for their model and both TP sizes, plus
   the block size / per-block page from step 2-3 and the log lines that confirm
   them.
3. The sub-shape and its verdict: which frame holds the failing `torch.zeros`, and
   — for A — whether `A` is inside or outside `gmu × T`; for B, the
   `#tensors × page × num_blocks` identity against the quoted `Available memory:`
   line, with the residual left open.
4. Ranked actions, cheapest first: keep TP4 and the context the recipe validates;
   cap the pool with `--num-gpu-blocks-override` (sub-shape B) or drop
   `--gpu-memory-utilization` in ~0.05 steps and quote the resulting
   `GPU KV cache size` line; two TP4 instances instead of one TP8 (the
   framework-agnostic answer the #6572 reporter reached themselves — the two measured
   points below); then the capture ablation; then a quantised checkpoint.
5. Gaps: the version/image pairing, `config.json` head counts, and the full
   startup log (block size, KV size, frame of the failing allocation).

### Reported `--gpu-memory-utilization` ceilings on this class

Both points are the #6572 reporter's own tests of Qwen3-Coder-Next at
`--max-model-len 262144`, and they are a curve in (dtype, card), not a constant:

| Checkpoint | Card | TP | What starts | The other bound |
|---|---|---|---|---|
| BF16 | 910B3-64G | 8 | ≈0.60 is already the most that starts | TP4 runs at 0.80; the reporter's verdict was that **two TP4 instances beat one TP8** |
| W8A8 | 32 GB A2 | 4 | 0.85 | 0.85 is a *floor* as well: any less and the 256K window is not filled ("多了拉不起来，少了开不满256K") |

Read the second row whenever the report is quantized: a W8A8 checkpoint on 32 GB cards
does **not** buy the 256K window cheaply — its required gmu sits *above* the BF16/64 GB
figure — so "it is only W8A8, gmu can be low" is not the answer. Quote both numbers as
*reported*, per dtype/card/context, never as a default for another combination.
**Both rows arrived as later comments on #6572 itself**, so when the report you are
answering *is* #6572 they are that reporter's own future measurements: give them as
「你自己后续回复里测到的」 or leave them out, and never as an unattributed 「已记录的实测」
— the recorded replay led its first answer with ~0.60 on that wording and had to spend
its second message withdrawing the attribution (`../evidence-workspace/case-layout.md`,
own-issue rule). Attributed, they belong to the gmu rung of the ladder (*Answer shape*
item 4), not to the classification and not to another reporter's box.

## Stop rule — this class does not need an allocator mechanism

#6572 used 20 assistant turns, never emitted an answer, and died inside a
28,405-character reasoning block that was still oscillating between
"TP-invariant reserve", "capture footprint" and "the loader materialises full
tensors" when the output cap hit. Everything needed was already in hand by turn
12: the tutorial envelope (TP4/32768), the reporter's gmu ceilings, and the OOM
line. For this class the answer is steps 1-3 plus the action ladder; the
mechanism of the last 2 GiB is **not** required, and the per-rank table above is
the deliverable that replaces it. A replay against this page did exactly that:
two messages (the budget/envelope verdict, then the delta once the reporter's OOM
line and ceilings arrived), 10 assistant turns and 3.6 K output tokens, with the
sub-shape left open until the frame existed — the only defect was the ceiling
attribution above, which needed no source read to avoid.

#4474 is the same class one sub-shape over (TP4 up / TP8 OOM at
`--max-model-len 4096`, v0.11.0rc2) and repeated it: 16 K output tokens, twelve
tool calls, and no answer — the reasoning was still reconciling two OCR'd
screenshot numbers (`GPU KV cache size: 488,192 tokens` against
`Maximum concurrency … 401.58x`) when the cap hit, after it had already found the
tutorial line, `determine_available_memory`, `--num-gpu-blocks-override` and the
`# TODO: REFACTOR ME to sharing hybrid cache` comment. For B the deliverable is
step 1 + the frame + the identity + the two log lines to quote + the override, in
one message; the mamba padding algebra and the last 15 GiB are not required, and
the pair of printed numbers above is *not* a contradiction to resolve.

## Scope and recheck

Sub-shape A verified at vllm-ascend v0.14.0rc1 (`52d4acf`) + vllm v0.14.1
(`d7de043d`); sub-shape B at vllm-ascend v0.11.0rc2 (`a2e4c3f`) + vllm v0.11.0
(`b8b302c`) and re-verified at vllm-ascend v0.13.0 (`6281c12`) + vllm v0.13.0
(`72506c9`), where `patch_mamba_config.py` is `vllm_ascend/patch/platform/` (it
moves between releases). Re-read before reusing on another release: the tutorial's
serve line **and file name**, the `patch_mamba_config` page arithmetic (it has
changed shape across releases — see `../accuracy-benchmarks/hybrid-block-size-corruption.md`
for a later ref), `get_num_kv_heads`, whether the block-size clamp exemption for
`qwen3_next` still exists (`utils.py:1139`), and — for B — whether
`initialize_kv_cache_tensors` still allocates per layer instead of sharing per
`kv_cache_tensor` (upstream `gpu_model_runner.py:_allocate_kv_cache_tensors`). The
reporter's gmu ceilings are one host's measurements on one checkpoint (two points:
BF16/64 GB and W8A8/32 GB), not a constant.
