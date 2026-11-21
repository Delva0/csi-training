# A gather's device assert: the bound names the source table

## Scope

The paste carries a CANN AscendC device assert of this shape (recorded #2349,
verbatim apart from the masked home directory):

```text
[ASSERT] [CANN_VERSION : 8.2.RC1][TimeStamp : 0] /[HOME_DIRECTORY_1]/Ascend/8.2.RC1/opp/built-in/
  op_impl/ai_core/tbe/impl/dynamic/../ascendc/gather_v3/gather_v3_base.h:137:
  Assertion `(0 <= val && val < this->gxSize_)' Index 41513 out of range[0 40960)!
```

usually beside `rtKernelLaunchWithHandleV2 failed: 507035` /
`reason=[vector core exception]`, `OpExecCache run fail`, `ERR01005 OPS internal
error`, and on another rank `call aclnnAddRmsNorm failed, detail:EZ9903 … Assert
((rtKernelLaunchWithHandleV2(…)) == 0) failed`. The asserting kernel is a
**built-in CANN TBE** op, so no `csrc/<op>/` line and no vllm-ascend kernel owns
it: one `rg -ln 'GatherV3' csrc` to confirm the absence, then stop looking there.
The same assert also fires for other tables and other models (#866 Baichuan2-7B,
#1682, #1731 Qwen2.5-VL), so what identifies this page's shape is the bound being
a plausible **table row count** (next section), not the file name.

## Rule: two numbers, one table

`gxSize_` is the row count of the gather's source and `Index` the element asked
for. Both were computed on the host and handed to the kernel, which asserts on
the pair — so the finding is an **index-space overflow upstream**, not a card:
skip `npu-smi`, `asys health` and the driver/CANN pairing for this shape
(contrast the aivector/MTE dump family in `aclnn-error-dump-reading.md`, where
hardware is a real candidate). Read the bound as a table identity, then ask which
code can produce an index ≥ M:

| Table | rows = | row count comes from | index is |
|---|---|---|---|
| RoPE `cos_sin_cache` | the checkpoint's `max_position_embeddings` | `qwen3.py:178` / `qwen2.py:219` → `rotary_embedding.py:132` | a token position |
| KV block table | `num_gpu_blocks` | the `Memory profiling … GPU blocks: N` line | a block id |
| token / sequence buffers | `max_num_batched_tokens`, `max_num_seqs`, **or one request's own token count** | serve command / scheduler config, else the fault-time `dump_input.py` block (`prompt_token_ids_len=N`, `SchedulerStats(… queries=N, hits=M)`) | a slot in the padded batch, or an absolute position |
| vocab / embedding table | `vocab_size` (padded) | checkpoint `config.json` | a token id |

`Index` need not be a position. #866 hit the same assert at `Index 5330 out of
range[0 4096)` on Baichuan2-7B with `--input-len 512 --output-len 1`; no position
in that run could reach 5330, so the table there was not the RoPE cache. Two
independent facts must line up before a table is named: **M equals a row count
computed from `config.json` or the serve line**, *and* **N is plausible in that
table's index space**. Bound-and-index alone is a hypothesis.

### When M matches no config value, read the engine's own fault dump first

`M` is not always a table you can compute from the checkpoint or the serve line,
and a mismatch there does **not** refute the table reading. #1603 (vllm-ascend
`v0.9.0rc2` `8dd686d` + vllm `v0.9.0`, Qwen3-32B bf16 TP4, `--max-model-len 32768`,
`--enable-prefix-caching`, chunked prefill at `max_num_batched_tokens=2048`,
PIECEWISE ACL graph `cudagraph_capture_sizes [512 … 1]`) asserted at
`Index 775 … 829 out of range[0 391)`, and 391 is neither the checkpoint's
`max_position_embeddings` (a 40960-row Qwen3 cache) nor `num_gpu_blocks` (the same
log's `GPU KV cache size: 631,424 tokens`). The bound is printed by V1's own
disaster dump, in the block that precedes the assert:

```text
ERROR [dump_input.py:78] Dumping scheduler output for model execution:
ERROR [dump_input.py:79] SchedulerOutput(scheduled_new_reqs=[NewRequestData(
  req_id=chatcmpl-…, prompt_token_ids_len=391, … block_ids=[[47,48,116,117]],
  num_computed_tokens=256, …)], num_scheduled_tokens={chatcmpl-…: 135}, …
ERROR [dump_input.py:81] SchedulerStats(num_running_reqs=1, …,
  prefix_cache_stats=PrefixCacheStats(requests=1, queries=391, hits=256))
```

So `M = 391` is **that request's prompt token count** (`queries=391`,
`hits=256`) — a per-step, per-request buffer, not a config constant. The second
fact then decides it: the step scheduled `135` tokens for that request
(`391 - 256`), whose positions can only be `256 … 390` and whose block ids are
`47, 48, 116, 117`, so **no index of that request can produce 775-829** — the
offending index comes from the co-scheduled decoding request (or a physical slot),
i.e. a request-sized table indexed with batch-global positions. Same rule as
above, applied to a table only the dump reveals: read `scheduled_new_reqs`,
`num_scheduled_tokens`, `block_ids` and `SchedulerStats` before any `jq` or block
arithmetic, and treat "matches nothing in `config.json`" as a pointer to this
family rather than as a dead end.

Trigger shape: the assert lands on the step that **first mixes a new request with
a decoding one** (here `Running: 1 reqs` for an hour, then six requests inside 8 s),
on every rank in the same second, followed by `RuntimeError: ACL stream synchronize
failed, error code507035` (`ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`,
`Ascend/pytorch` `third_party/acl/inc/acl/error_codes/rt_error_codes.h:94`) and
`EngineDeadError` — one fault, not two. The op name is *not* in the paste: the
deepest frame is `eval_with_keys-1.1 line 49, in forward` →
`torch/ops.py:1116` in `__call__` inside a compiled piecewise subgraph, so nothing
here distinguishes RoPE from paged attention from an `index_select`, and hunting
the op is not what decides the reply.

Ladder, cheapest first — the step mix is the variable, not the card:

1. `--no-enable-prefix-caching` on the same command: the crashing step is a prefix
   hit (`queries=391, hits=256`), so this removes the partial-prefill shape in one
   launch.
2. `--enforce-eager`, which removes the ACL-graph/piecewise subgraph named by the
   deepest frame.
3. One client at a time against the same serve, the control that separates
   concurrency from model/length.

Then ask for the artefacts the paste does not carry: the plog
`fault kernel_name=GatherV3_<hash>…`, the `dfx/ops/*.json` that names input shapes
and dtypes (a bf16 `[M, head_dim]` input is a RoPE cache, an int block table is
KV indexing), and `jq '.max_position_embeddings'` run on **the model directory the
dump prints** rather than the one the prose names.

Resolution class, and why it matters for the reply: a **code patch**. The original
reporter later commented on this issue that the patch helped (`Thx, this patch is
indeed useful`, 2025-07-03 — later public observation, not a reply to these
investigators). Which line it touched is not in this snapshot (the pin predates
it), so quote the direction — upstream index space — and skip card, driver and
CANN-pairing advice entirely.

## The recurring one: a context length past the checkpoint's RoPE table

Chain verified at vllm-ascend `b5b7e0e` (v0.9.2rc1, the reported ref) + vllm
`a5dd03c` (v0.9.2):

1. `Qwen3Attention` builds its RoPE from the **checkpoint** value, not from
   `--max-model-len`: `max_position=config.max_position_embeddings`
   (`vllm/model_executor/models/qwen3.py:178`; same at `qwen2.py:219`).
2. `RotaryEmbedding._compute_cos_sin_cache` creates exactly
   `max_position_embeddings` rows —
   `t = torch.arange(self.max_position_embeddings, dtype=torch.float)`
   (`vllm/model_executor/layers/rotary_embedding.py:132`).
3. On Ascend that cache is indexed by `positions`:
   `RotaryEmbedding.forward_oot = rope_forward_oot`
   (`vllm_ascend/ops/rotary_embedding.py:286`), whose default branch calls
   `torch_npu._npu_rotary_embedding(positions, query, key, self.head_size,
   self.cos_sin_cache, neox_style)` (`:78-85`). The AscendC
   `torch.ops._C.rotary_embedding` branch above it (`:60-70`) needs fp16 + neox
   + `head_size % 32 == 0` (`:29-31`), so a bf16 serve takes the `torch_npu`
   call — check the dtype before claiming which kernel indexed the cache.
4. `--max-model-len` above the derived maximum is only allowed with
   `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, which downgrades the check to a warning:
   "User-specified max_model_len (N) is greater than the derived
   max_model_len … This may lead to incorrect model outputs or CUDA errors."
   (`vllm/config.py:3507-3523`; the `raise ValueError` variant at `:3521-3523`
   is the one that names the env var). Nothing re-sizes the RoPE cache, so
   positions run to `max_model_len - 1` while the table stops at the checkpoint
   value.

Two commands decide it, and they are the first thing to ask for:

```bash
jq '.max_position_embeddings, .rope_scaling' <model_dir>/config.json
#   beside the failing serve line's --max-model-len and VLLM_ALLOW_LONG_MAX_MODEL_LEN
```

M == `max_position_embeddings` and N < `max_model_len` → this is the cause;
`rope_scaling: null` is the confirming detail. (The recorded bound 40960 is what
Qwen3-era checkpoints carry; the reporter's own `jq` output is the proof, not
this page — never quote a checkpoint value from memory, `SKILL.md` rule 6.)

Fixes, cheapest first:

1. Set `--max-model-len` ≤ the checkpoint's `max_position_embeddings` and drop
   `VLLM_ALLOW_LONG_MAX_MODEL_LEN`; the cache then cannot be overrun, and no
   code change is needed.
2. If the longer context is required, enable a scaling policy in the
   checkpoint's `config.json` or via `--hf-overrides '{"rope_scaling": …}'`
   (`original_max_position_embeddings` = that value). The scaling subclasses
   build their cache at `max_position_embeddings * scaling_factor`
   (`rotary_embedding.py:434`, `:521`, `:664`), which is why the index then
   fits — but YaRN multiplies the cached cos/sin by `mscale` (`:668-669`), so
   accuracy must be re-baselined; present it as a correctness change, not a
   no-op.
3. Do not propose clamping positions inside `rope_forward_oot`: the model then
   attends with wrong rotations instead of failing, converting a crash into a
   silent accuracy defect.

## Artefacts that name the table outright

- plog `[AIC_INFO]` / `Aicore kernel execute failed … fault kernel_name=
  GatherV3_<hash>_high_performance_<n>` — the compiled instance.
- CANN's own `asys` / msaicerr parser resolves that hash into
  `dfx/ops/GatherV3_<hash>.json` carrying `tiling_key`, input shapes and dtypes
  (parser and fixtures in `cann/oam-tools`; the sentence it parses is
  `cann/oam-tools/test/st/msaicerr/testcase/conftest.py:47`, the resolved file
  `…/asys_output_*/dfx/ops/GatherV3_<hash>_high_performance.json`). A bf16
  `[M, head_dim]` input is the RoPE cache; an int
  `[num_blocks, block_size]`/`[max_reqs, max_blocks]` input is a block table.
  Ask for that file when the paste does not carry it.
- Also ask for the prompt + generated length of the request in flight (an
  agent/conversation client crossing the checkpoint's context is what
  "运行一段时间后" means in the recorded case) and the `Using max model len N`
  line.

Order in the paste is not order of causation
(`../hccl-and-hardware/collective-error-chain.md`): `507035`/`507011` is the
runtime wrapper, and the other rank's `aclnnAddRmsNorm … EZ9903 …
rtKernelLaunchWithHandleV2 … failed` is the *latched* device error, not a
layernorm defect.

## Delivery

Two commands produce this answer (`jq` + the bound comparison, or one read of the
paste's own dump block): send it then, with the ladder and the version scope above.
Recorded #2349 had exactly this diagnosis in hand from its fifth turn and shipped
**386 characters** of text across 16 turns, the complete ranked answer sitting in a
6.3 K-character reasoning block that was cut at the completion cap. #1603 is the
same failure one table over: the bound matched its own dump line by the eighth turn,
the rest of the budget went into naming the op behind `eval_with_keys-1.1 line 49`,
and **no message text shipped at all**. For this shape the op owner is a *supporting*
detail, never a precondition — the index-space verdict plus the three-rung ladder is
already a complete reply.
