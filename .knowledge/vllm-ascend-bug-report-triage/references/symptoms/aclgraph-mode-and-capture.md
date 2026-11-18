# ACL-graph mode configuration: which graph mode a model may run, the derived capture sizes, and the uniform-decode padding

## Scope

A run whose failure or limit is decided by `compilation_config` graph settings rather than by model
math: `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`, explicit
`cudagraph_capture_sizes`, or a model/config the capture-size helper cannot size. Eight verified
members, one per phase:

- **A.** The full-graph decode request kills the engine at request time on a model whose backend has no
  working full-graph decode path at that release (#4326 — reported on `0.11.0`, read at `v0.11.0rc1`;
  Qwen3-30B-A3B W8A8, DP8 + EP, `--max-num-seqs 128`).
- **B.** FULL decode + data parallelism: `_prepare_inputs` pads `query_start_loc` past the end of its
  buffer and raises `RuntimeError: The expanded size of the tensor (N) must match the existing size
  (M) ... Target sizes: [N]` (#5076, `v0.12.0rc1`).
- **C.** Config validation dies before any weight load with `AttributeError: '<Xxx>Config' object has
  no attribute 'num_hidden_layers'` inside `update_aclgraph_sizes` (#1183, `v0.9.0rc1`, Gemma-3).
- **D.** Decode returns garbled text **only** in graph mode on a DP rank with sparse (DSA-family)
  attention: a dummy batch writes through stale KV slots (#17593, `0.20.2rc1`).
- **E.** The legacy `full_cuda_graph: true` boolean of `≤ v0.9.x` (not the `cudagraph_mode` enum): a
  post-startup freeze whose py-spy frame is `graph_task_update_end`, i.e. the per-layer task-update loop
  the flag alone reaches (#4180, `v0.9.1`).
- **F.** `cudagraph_mode: "FULL"` on a **hybrid linear-attention** model (Qwen3.5/Qwen3-Next GDN, whose
  builder declares `UNIFORM_BATCH`): no crash and no wrong output — the mode is silently rewritten to
  `FULL_DECODE_ONLY`, so decode is captured and *prefill is not* ("the prefill phase is not captured by the
  graph, and there are still gaps between operators"), and the one warning line names the backend and the
  enum (#11947, `v0.21.0rc1`).
- **G.** A worker death inside `execute_dummy_batch` on a `FULL_DECODE_ONLY` deployment: the dummy run's
  padded counts disagree with their source lists, so the replay reads a slot that was never written
  (#7290, #7460, #7401; `v0.16.0rc1`–`v0.20.2rc1`).
- **H.** Capture runs out of **events** on 310P/300I DUO: `Insufficient_Event_Resources(EL0008)` /
  `rtEventRecord … 207007`, then `capture_end … 507903`, while TP1 on the same command starts (#12989,
  `v0.23.0rc1` + `-310p` image). The fork's own mitigation text does not fire for this signature (§H).

Not this page: a capture-window refusal (`107041` / `rtMemcpy … the current capture mode does not
support this operation`, the replay-branch `torch.npu.synchronize()`), a missing Meta kernel or an op
the SoC's store lacks — `graph-capture-and-custom-ops.md`; a capture that fails at `capture_begin`/
`capture_end` for **resource** reasons, where the resource named is a stream or an event and the fix is a
smaller capture-size list — §H below (not a device fault, not a padding bug); a device fault under capture —
`device-kernel-faults.md`; a producer/consumer metadata mismatch in spec decoding —
`pcp-dcp-and-mtp.md`; a *crash* at request time rather than wrong output (§A below) still routes here,
but wrong output with no crash and no DP dummy batch usually belongs to the parallel feature instead.
A 310P/300I DUO capture that fails on the **event-id pool** — `Insufficient_Event_Resources(EL0008)`,
`rtEventRecord … 207007`, `507903` at `capture_end`, TP>1 — is §H below; the surrounding 310P product
line (the tag's own 310P page, its per-TP capture-size lists, `--max-model-len`, the sharded W8A8SC
checkpoint gate) is `310p-capability.md`.

## Common rules

- **Read the effective mode from the log, not from the flag.** `vllm_ascend/platform.py`
  `check_and_update_config` rewrites the request: `FULL_AND_PIECEWISE` → `PIECEWISE` (`:202-203` at
  `v0.11.0rc1`, the default for a V1 dense model at these tags) and `NONE` → `level =
  NO_COMPILATION`, printing `PIECEWISE compilation enabled on NPU` / `FULL_DECODE_ONLY compilation
  enabled on NPU` immediately above the frames that follow (`:184-186` and `:219-222` at
  `v0.11.0rc1`). Quote that line: it is the mode that actually ran, and it is usually two lines
  before the reported traceback.
- **The hook accepts `FULL_DECODE_ONLY` with a warning and no capability check.** `platform.py:219-241`
  at `v0.11.0rc1` logs the red "full graph … early experimental stage … capturing too many batch sizes
  can lead to OOM or inference hangs" banner and sets `use_inductor = False`; only modes the branch
  does not know fall back to `NONE`. There is no per-model check anywhere in the hook.
- **`--enforce-eager` (or `-O0`) is the lever that removes every branch below**, at the price of ACL
  graph capture and every compile-time optimisation. When the reporter's own text says the run works
  without the graph config, that is the confirmed workaround and it belongs in the first line of the
  answer.
- **`aclgraph_support` is a classvar on the attention backend, and `UNIFORM_SINGLE_TOKEN_DECODE` does
  not block `FULL_DECODE_ONLY`.** At `v0.11.0rc1` `attention/attention_v1.py:174-175` declares
  `UNIFORM_SINGLE_TOKEN_DECODE` and the file has no `full_graph_attention`; the runner's check
  (`worker/model_runner_v1.py:3405-3430`) only downgrades the *mixed* `FULL` mode (`!= ALWAYS` →
  `FULL_AND_PIECEWISE`/`FULL_DECODE_ONLY` with a warning) and raises only for `NEVER`. Upstream vLLM's
  own gate agrees: `initialize_cudagraph_capture` (vLLM `v0.11.0`
  `vllm/v1/worker/gpu_model_runner.py:3611-3660`) takes the minimum `cudagraph_support` over the
  attention metadata builders and rewrites only a mixed `FULL` mode below `ALWAYS`; an answer built on
  "the enum should have stopped it" is wrong. A backend that nominally supports decode-only capture
  can therefore still be broken at runtime: the symptom is an engine death while serving, not a
  configuration error.
- **Capture sizes are derived, so their failures happen at config validation.** `update_aclgraph_sizes`
  in `vllm_ascend/utils.py` prunes the vLLM default list by `(num_hidden_layers + 1) ×
  parallel_factor` (`:134-147` at `v0.9.0rc1`; a `resources_per_graph` form on later tags) and runs
  from `platform.py:189`, inside `VllmConfig.__post_init__` — before any device work, which is why §C
  has no NPU frame at all.

## A. `FULL_DECODE_ONLY` kills the engine on this model at this release — a support gap, not a line

Signature (#4326): a Qwen3-MoE W8A8 service starts, captures and serves, then requests end with the
API server spamming `serving_chat.py:1145 … raise output` and finally
`vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above)
for the root cause.` — while the "stack trace (above)" is not in the paste. A CANN plog in the same
paste carrying `[CANNKB] cann_kb_finalize … AttributeError: 'ForkAwareLocal' object has no attribute
'connection'` / `ConnectionRefusedError` is TUNE/knowledge-bank teardown from a dying process, not the
cause.

Answer shape:

- **Workaround:** remove `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` (keep the
  default PIECEWISE, the tag's own default) or add `--enforce-eager`; cost is the full-graph decode
  speedup / the piecewise re-execution. In #4326 the last line of the report body already says it:
  "after removed `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` it will work" —
  reporter-confirmed. Their `VLLM_ASCEND_ENABLE_FLASHCOMM=1` + `--async-scheduling` + DP8/EP are not
  required for the crash. **Not** a lever: capture-size tuning — #4199/#4205 only change the *default*
  capture sizes for Qwen3-MoE under full DP (a throughput fix), so they neither fix nor explain an
  engine death.
- **Root cause:** at `v0.11.0rc1` the captured decode attention runs `torch_npu._npu_paged_attention`,
  refreshed every step through `graph_task_update_begin/end` in `compilation/acl_graph.py:216-240`,
  with an extra `_npu_paged_attention_get_workspace` whose own comment reads "there are some rare bugs
  for FULL_DECODE_ONLY mode with GQA … on some rare cases … a bigger workspace". That workaround
  reached the 0.11 branch as a **cherry-pick (#3987, `f9494d978`), not as #3986's sha** — read the
  tag's tree, not ancestry (`../workflow/fix-discovery.md`). The maintainer answer on the thread
  (`leo-pony`, 2025-12-15) is `reported`: "DeepSeek 3/3.1 and Qwen3 doesn't work with FULL_DECODE_ONLY
  graph mode on main branch currently. We'll fix it in next release." Together with the tag, that
  sentence settles the workaround and the status; the mechanism above is the version-scoped half a
  clone can show.
- **Fix range, verified:** `e38ef2c43` (#3970, "support FULL graph mode for GQA", first tag
  `v0.12.0rc1`, in **no** `v0.11.0*` tag — check with `git merge-base --is-ancestor`) replaces that
  refresh with `full_graph_attention()` on `npu_fused_infer_attention_score` with a per-token cached
  workspace, and sets `aclgraph_support = AttentionCGSupport.ALWAYS`
  (`attention/attention_v1.py:232-234`, `:388` at the tag); DeepSeek-V3.2 gains its own full-decode
  support in `5b179c53f` (#4706), same first tag. One command separates the releases:
  `git grep -n '_npu_paged_attention' v0.12.0rc1 -- vllm_ascend/compilation/acl_graph.py` is empty.
- **Label the gap:** the worker-side crash line was never pasted, so which of the "rare bugs" fired is
  `hypothesis`; the falsifier is the worker traceback in `error.log`, or the same load on
  `v0.12.0rc1`+.

## B. FULL decode + DP: the `query_start_loc` padding runs past its buffer (#5076)

Verbatim shape (`v0.12.0rc1`, Qwen3-Coder-480B-A35B, 2 nodes, DP4 (2 local per node) + TP8 + EP,
`--max-num-seqs 64`, `--max-num-batched-tokens 8192`, 7168-token prompts, `--enable-chunked-prefill`,
`--async-scheduling`, `{"cudagraph_mode":"FULL_DECODE_ONLY"}`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`,
crash only at high concurrency):

```
File ".../vllm_ascend/worker/model_runner_v1.py", line 1035, in _prepare_inputs
    self.query_start_loc.gpu[num_reqs + 1:num_reqs_padded +
RuntimeError: The expanded size of the tensor (63) must match the existing size (6151)
at non-singleton dimension 0.  Target sizes: [63].  Tensor sizes: [6151]
```

Read it as a buffer overflow and check it with the numbers the report already gives — one or two
commands, **never prose arithmetic**:

- `query_start_loc` is `_make_buffer(self.max_num_reqs + 1)` (`:392`), and `max_num_reqs =
  scheduler_config.max_num_seqs` (`:202`) → 64 + 1 = 65 slots here.
- The gate is only `cudagraph_mode.decode_mode() == FULL and uniform_decode` (`:1021`) with no upper
  bound on `num_input_tokens`; `num_reqs_padded = num_input_tokens // uniform_decode_query_len`
  (`:1023`) and `fill_values` carries `pad_size = num_reqs_padded - num_reqs` elements (`:1024-1034`),
  written straight into the GPU slice at `:1035-1036`.
- The destination slice is clipped at the buffer end, so with `num_reqs = 1` the target is
  `65 - 2 = 63` while the source keeps `6151 = 6152 - 1` elements: **`num_input_tokens` was 6152**.
  The two extents are exactly `a = max_num_reqs - num_reqs` and `b = pad_size`; if they are not, this
  is another class — do not force it.
- `num_input_tokens` is not this rank's batch: `_sync_metadata_across_dp` (`:450`, result assigned at
  `:655` `num_input_tokens = maybe_padded_num_tokens`) replaces it with `max_tokens_across_dp` from an
  all_reduce over the DP CPU group, so one rank's ~6.2 K-token chunked-prefill step sets the value on
  every rank — including a rank whose own batch is a single decoding request (`uniform_decode` is
  computed from that rank's tokens, at `:1016`). Without DP the value is
  `pad_for_cudagraph(total_num_scheduled_tokens)` (`:617-620`) and stays small.
- Hence the observed conditions: FULL decode graphs **+ DP>1 + an unbalanced step** (prefill on one
  rank, decode on another). With DP=1 the value comes from `pad_for_cudagraph(total)` and stays inside
  the buffer, so this case needs the all_reduce to be in play. Which DP rank held which count is
  `hypothesis` (nobody printed it); the sizes fit exactly.
- The same inflation is a correctness hazard whenever the slice does *not* clip: `seq_lens`,
  `block_table_tensor` and the attention metadata are all handed `num_reqs = num_reqs_padded`.

Fix and status: both repairs are in `v0.13.0rc1` — maintainer `wangxiyuan` "try 0.13.0rc1", reporter
"Great! It has been fixed." `0686b32d8` (#4963) writes the padding on the CPU mirror with one
`copy_to_gpu(num_reqs_padded + 1)` and adds `num_input_tokens <= cudagraph_batch_sizes[-1]`
("prevents padding when the number of input tokens exceeds the maximum ACL graph batch size to avoid
out-of-bounds errors"); `ff3914e31` (#5164) then replaces that ceiling with
`uniform_decode_query_len <= num_input_tokens <= max_num_seqs * uniform_decode_query_len`, because
`cudagraph_batch_sizes[-1]` "is actually not the max number of tokens to be padded in
`FULL_DECODE_ONLY` mode, much larger instead". The fixed block reads:

```python
max_decode_tokens = self.scheduler_config.max_num_seqs * self.uniform_decode_query_len
if self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
    uniform_decode and self.uniform_decode_query_len <= num_input_tokens <= max_decode_tokens:
    num_reqs_padded = num_input_tokens // self.uniform_decode_query_len
    ...
    self.query_start_loc.np[num_reqs + 1:num_reqs_padded + 1] = \
        self.arange_np[1:pad_size + 1] * self.uniform_decode_query_len + last_query_loc
    self.query_start_loc.copy_to_gpu(num_reqs_padded + 1)
```

Workaround on `v0.12.0rc1`: upgrade to `v0.13.0rc1`+ (reporter-confirmed, the crash gone), or drop
`FULL_DECODE_ONLY` (PIECEWISE or `--enforce-eager`), or remove the deployment's rank imbalance — the
last two trade the decode speedup or throughput and were not confirmed by the reporter. The
maintainer's own note in the thread ("error lines do not exist in the main branch") is why a `main`
read cannot answer this case.

## C. `AttributeError: '<Xxx>Config' object has no attribute 'num_hidden_layers'` at config validation

Signature (#1183, image `vllm-ascend:v0.9.0rc1-openeuler`, Gemma-3-27B-it, TP2, chunked prefill):
the log prints `platform.py:183 PIECEWISE compilation enabled on NPU`, then the process dies inside
`engine_args.create_engine_config → VllmConfig.__post_init__ → check_and_update_config →
update_aclgraph_sizes` with `AttributeError: 'Gemma3Config' object has no attribute
'num_hidden_layers'`. No weight load and no NPU op has run, so the device, TP size and `VLLM_USE_V1`
are all irrelevant — the flag is only what selected the ACL-graph branch.

Route here when the class in the message is a **wrapper** config (`Gemma3Config`, `Qwen2AudioConfig`,
Qwen3.5-VL, DeepSeek-V4 VL) and the field is a transformer field (`num_hidden_layers`,
`num_attention_heads`, `max_position_embeddings`, `hidden_size`); vLLM's own accessor for that case is
`ModelConfig.hf_text_config`. Not here: a missing attribute on a genuine flat text config (a
`transformers` version skew), a `KeyError` over a weight dict (`quantization-load.md`), or a crash
after the first forward.

Mechanism (verified at `v0.9.0rc1`): `utils.py:134` reads
`vllm_config.model_config.hf_config.num_hidden_layers` unconditionally and uses it at `:147` as
`MAX_CAPTURE_SIZE / (num_hidden_layers + 1) / parallel_factor`. A **multimodal** config keeps the
transformer fields under `text_config` (`Gemma3Config`, `Qwen2AudioConfig`), so the top-level object
has no such attribute. Consequences worth stating in a report: it is TP/DP-independent (any parallel
size reaches the same line), it has nothing to do with the V1 engine or the model's attention, and it
reproduces from the start command alone. Workaround: **`--enforce-eager`** — it takes the
`NO_COMPILATION` branch of `platform.py`, so `update_aclgraph_sizes` is never called (maintainer
`yiz-liu`: "Try add --enforce-eager, should be OK then"; cost: no ACL graph). Fix: `d1c640841` (#1803)
replaces the read with

```python
hf_config = vllm_config.model_config.hf_config
if hasattr(hf_config, 'num_hidden_layers'):
    num_hidden_layers = hf_config.num_hidden_layers
else:
    num_hidden_layers = get_max_hidden_layers(hf_config)   # recursive over hf_config.to_dict(), max()
```

`get_max_hidden_layers` raises `ValueError("Not found num_hidden_layers in model config.")` when no
sub-config carries the key. First tag `v0.10.0rc1`, **not** in `v0.9.0rc1`; its message lists the
earlier reports it closes (#359 for this very model, #1276, #1760, #1780).

**Version shape of the same line**

| ref | what the code does |
| --- | --- |
| `v0.9.0rc1` (and the `v0.9.x` line) | direct read at `utils.py:134` → the `AttributeError` above |
| `v0.10.0rc1` – `v0.20.2rc1` | `hasattr` + `get_max_hidden_layers` (`v0.20.2rc1` `utils.py:567` definition, `:596` call); a config with no such key anywhere now raises `ValueError: Not found num_hidden_layers in model config.` instead |
| `v0.23.0rc1`+ | `platform.py` no longer calls `update_aclgraph_sizes` (`git grep -n 'update_aclgraph_sizes(' <ref> -- vllm_ascend` returns no call site) — this traceback at such a ref means the version is misidentified, not that a fix vanished |
| `main` | `get_max_hidden_layers` survives (`utils.py:755`) with no in-tree caller |

Gate differs by ref, so the workaround does too: at `v0.9.0rc1` the gate is the compilation-level chain
above; at `v0.20.2rc1` it is `compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE`
(`platform.py:425`), so `--enforce-eager` or `cudagraph_mode: "FULL_DECODE_ONLY"` both skip it.

Limits worth relaying: that max-over-sub-configs is a heuristic, and `dec04ec8d` (#7064, first tag
`v0.16.0rc1`) had to fix the *draft* half — MTP stores its depth in `num_nextn_predict_layers`
(DeepSeek-V3) or `mtp_num_hidden_layers` (Qwen3.5), so `draft.hf_config.num_hidden_layers` counted the
main model and the `resources_per_graph` estimate (hence the batch-size ceiling) came out wrong
(`draft.get_total_num_hidden_layers()` now; the same `model_arch_config_convertor` path the KV
connectors use). So on `v0.16.0rc1`+ a layer-count complaint from this function is a **performance**
symptom (too few capture sizes), not the `AttributeError`. And a second maintainer's "I think gemma3
will not work in graph mode and eager mode" on the same thread predates `d1c640841` and contradicts
its `Closes #359`: report it as `reported` and unverified rather than resolving the conflict for the
reader.

**Two checks when a new report names this frame**

1. Which read, at the reported ref: `git show <ref>:vllm_ascend/utils.py | grep -n 'hf_config\.'`
   plus `git show <ref>:vllm_ascend/platform.py | sed -n '/check_and_update_config/,/worker_cls/p'`.
2. Whether the gate is reachable with the reporter's flags (see the table above), and whether the
   model is a wrapper: the exception's own class name says so; `hf_text_config` is where the field is.

**Pitfalls**

- **An empty grep for the model family is not "unsupported".** `git grep -i gemma <ref> --
  vllm_ascend` returns nothing at `v0.9.x` because the model class lives in the `vllm` clone; Ascend
  gemma-specific code appears only much later (`_310p/ops/layernorm.py`, the `gemma4` gates in
  `attention/utils.py` at HEAD). The failing line is model-agnostic — any wrapper config reaches it.
- **Do not answer `--hf-overrides '{"num_hidden_layers": …}'` as verified.** It is untested here and
  it writes a main-model field into a wrapper config; the documented levers are the graph-mode switch
  and the release containing `d1c640841`.
- The class is about *config plumbing*, so a report at a ref ≥ `v0.10.0rc1` that still shows the
  `AttributeError` needs the built commit fingerprinted before the line is cited
  (`../workflow/version-pinning.md`).

## D. Garbled decode only in graph mode, DP + sparse attention: a dummy batch writing into released KV blocks (#17593)

Signature: multi-node DP + TP, an attention backend of the DSA/sparse family, spec decode on, `FULL` or
`FULL_DECODE_ONLY`; the answers are 乱码 / fluent garbage while the same launch with the graph mode off
is correct — no traceback, no accuracy gate, and the reporter usually says so themselves. Worked report
#17593: `v0.20.2rc1` (vLLM `0.20.2`), CANN `9.0.0`, 2 nodes × 8×910B3, DP4 + TP4, a customer 30B bf16
"DSA" model, `--num-speculative-tokens 4`, `--max-num-batched-tokens 1024`, `--max-num-seqs 320`,
`--max-model-len 65536`, `--compilation-config FULL_DECODE_ONLY`; 「关闭图模式则正常」.

**The mechanism one fix names.** A DP rank with no real request still executes `_dummy_run` for DP
alignment. If the dummy batch keeps stale `slot_mapping` slots, the *captured* decode writes through them
into logical KV blocks that a finished request already released and that may have been reassigned to a
live request — so the corruption is in the KV cache and the output text, not in a tensor shape. One change
fixes it, on three lines of the repo (all merged 2026-09-02, same title "[BugFix][Worker] Invalidate dummy
slots before attention metadata build"):

| sha | PR | line | first tag containing it |
| --- | --- | --- | --- |
| `fe8f6b0d1` | #15417 | `main` | `v0.27.1rc1` |
| `04a4549f9` | #15362 | `releases/v0.23.0` (head `backport/record-stream-v0.23.0`) | `v0.23.0.post1` |
| `23dd2e411` | #15409 | `releases/v0.26.0rc` | `v0.26.0rc1` |

The hunk moves the per-KV-group `slot_mapping.gpu.fill_(-1)` (guarded by `not is_graph_capturing`) from
*after* `_build_attention_metadata()` to *before* it, so a backend that derives or copies backend-specific
slot metadata during that build sees `-1`; the PR body's probe on DP2 in FULL graph mode: stale source
slots `1498..1505`, and after the same dummy forward 0 of 4096 BF16 values were finite (4096 NaN) in the
persistent KV row. Read the hunk, not the title: `git show --format= fe8f6b0d1 --
vllm_ascend/worker/model_runner_v1.py`.

**Answer shape for a repeat report.**

- **Workaround:** the reporter-confirmed lever is the one they already have — drop the graph mode
  (`--enforce-eager`, or a mode that does not capture decode) — at the cost of the decode graph. The
  durable lever is a build whose tree contains the fix **on their release line** (table above); a
  hand-port into an older tag is not the same change (see below).
- **Root cause candidate:** the `_dummy_run` ordering, labelled `hypothesis` for any build that lacks the
  fix, with the falsifier "same launch at a tag containing it, original command".
- **Status for #17593:** maintainer `kunpengW-code` (2026-09-28): `0.20.2rc1` is a relatively old
  historical release, the customer's own model was never validated against it in graph mode, try the
  latest stable first; `bowgneo` (2026-10-08) named #15362 and #15409; the reporter (2026-10-08) applied
  #15362 to `0.20.2rc1` by hand and reported the identical 乱码 with the modified files attached. That
  result is consistent with the ref: at `v0.20.2rc1` `_dummy_run` (`:3055`) calls
  `_build_attention_metadata` at `:3210` and **no** `slot_mapping.gpu.fill_(-1)` exists anywhere in the
  file — `git show v0.20.2rc1:vllm_ascend/worker/model_runner_v1.py | grep -n 'fill_(-1)'` returns only
  the `query_start_loc` pad (`:873`) and the two in-metadata pad fills (`:2796`, `:2806`), so the clear
  the fix relocates had not been written yet. Do not read "the PR was applied and it still garbles" as
  "the class is wrong": the port changed the code's shape.
- **DSA facts to have at the ref** (`vllm_ascend/attention/dsa_v1.py` at `v0.20.2rc1`): with spec decode
  the builder allocates `num_speculative_tokens` slot-mapping buffers of width
  `scheduler_config.max_num_batched_tokens` (`:369-377`), sets `decode_threshold = 1 +
  num_speculative_tokens` with an assert `<= 16` (the TND `npu_fused_infer_attention_score` limit
  quoted in the message), reuses it as `reorder_batch_threshold`, and declares
  `get_cudagraph_support → AttentionCGSupport.UNIFORM_BATCH` (`:433-440`) with `graph_pad_size` state.
  Changing `--num-speculative-tokens` re-sizes every one of them, so a config diff between "works" and
  "garbles" runs is a real lead here.
- A customer-defined architecture means the model class is in *their* tree: a vllm-ascend-only mechanism
  is at most a class statement. Ask for the model class or the ported `model_runner` diff instead of
  deriving their forward from the tag.

## E. The legacy `full_cuda_graph: true` boolean (`≤ v0.9.x`): a freeze whose py-spy frame is `graph_task_update_end`

Worked report #4180 — `v0.9.1`, Qwen2.5-32B-Instruct, `--compilation-config '{"full_cuda_graph": true}'`:
inference hangs after the service is up, and a `py-spy dump` parks the thread in the NPU call
`graph_task_update_end`. The thread has no diagnosis (`watch-Ultra` got only 「为啥不用新版本镜像呢」,
linked #4135, then feedback-bot → stale close), so everything below is read from the tag.

**Version scope first — the flag is a `v0.9` spelling, not the `cudagraph_mode` one.**

| ref | `full_cuda_graph` in the fork | upstream field |
| --- | --- | --- |
| `v0.9.1` | 7 sites: `platform.py:239`, `utils.py:185`, `compilation/piecewise_backend.py:103/178/278`, `attention/attention_v1.py:298` | vLLM `v0.9.1` `vllm/config.py:3961` `full_cuda_graph: bool = False`; `set_splitting_ops_for_v1` empties `splitting_ops` (`:4164-4171`) |
| `v0.10.0rc1` | **none** — `git grep -c full_cuda_graph v0.10.0rc1 -- vllm_ascend` is empty | vLLM `v0.10.0` still has it; from `v0.10.1` it is `Optional[bool]` and logs "full_cuda_graph is deprecated, use …" (`vllm/config/compilation.py:292`, `:452-458`) → `cudagraph_mode` |
| `v0.11.0rc0`+ | the same per-layer refresh lives in `compilation/acl_graph.py` (`update_attn_params`), keyed by request shape — §A | — |

So at ≥ `v0.10` the fork no longer special-cases the boolean while upstream still honours it until
`v0.10.1` deprecates it: the supported spelling there is `cudagraph_mode`, and a `graph_task_update_end`
freeze is §A's per-step update class, not this one. Two consequences: a serve block copied from a 0.9-era
report sets **nothing** on a newer tag (Trap A of `../vllm-ascend/config-flag-lifetimes.md`), and
`bash scripts/flag_lifecycle.sh full_cuda_graph` is the one call that dates the spelling before you blame
it.

**Mechanism, verified at `v0.9.1`.** The flag means "nothing is split out", so the attention op stays
inside the captured graph: `platform.py:239` extends `splitting_ops` with
`vllm.unified_ascend_attention_with_output` **only when the flag is off**. The paged-attention kernel
then has to be refreshed inside the graph on every replay:

- capture half — `attention/attention_v1.py:399-443`: under `forward_context.capturing` each layer
  allocates a `torch.npu.ExternalEvent`, `wait`s and `reset`s it on the current stream, appends it plus
  the `(query, k_cache, v_cache, …, block_tables, seq_lens, output)` tuple to
  `graph_params.events/attn_params[num_tokens]`, and keeps the task-group handle (`:439`).
- the private stream itself is created and fed by the backend: `piecewise_backend.py:104-105` builds
  `self.update_stream` and calls `set_graph_params(...)`; nothing else configures it.
- replay half — `compilation/piecewise_backend.py:276-280` calls `entry.aclgraph.replay()` and then,
  **only under the flag**, `update_attn_params` (`:128-160`): for every attention layer it opens
  `torch.npu.stream(self.update_stream)`, runs `graph_task_update_begin` (`:145`) →
  `torch_npu._npu_paged_attention` → `graph_task_update_end` (`:157`), then records an event on that
  private stream (`:159`). Nothing in the tag waits on or queries those events: the only
  `wait_event`/`wait_stream` hit (`model_runner_v1.py:135`) joins the runner's own stream, and no
  `synchronize()` names `update_stream` — so the replay and update streams are unordered relative to each
  other.
- The fork warns about this mode in its own words: `utils.py:185-200` prints, for `full_cuda_graph`,
  "This is an early experimental stage and may involve various unknown issues. A known problem is that
  capturing too many batch sizes can lead to OOM (Out of Memory) errors or inference hangs. If you
  encounter such issues, consider reducing `gpu_memory_utilization` or manually specifying a smaller
  batch size for graph capture." The same red banner is quoted in §A for `FULL_DECODE_ONLY` at later tags.

**Why the frame is `graph_task_update_end`** — `hypothesis`, with its falsifier: that call is where the
ACL task-group update blocks, and it is reachable only from `update_attn_params`, so a blocking update
(one per attention layer per decode step, on a stream the replay never joins) parks py-spy exactly
there. Kill it if the same launch without the flag still freezes inside a `graph_task_update*` frame, or
if the dump's top frame is `aclgraph.replay()` / the capture loop (startup capture, not the per-step
update).

**Answer shape.**

- **Workaround (structural, verified):** drop `full_cuda_graph` and keep the tag's default PIECEWISE
  mode — attention is split out (`platform.py:239`) and `update_attn_params` is never entered
  (`piecewise_backend.py:278` guards it), so the reported frame is unreachable; `--enforce-eager` removes
  the capture as well. Cost: the full-graph decode speedup / all ACL-graph throughput. If the mode must
  stay, the fork's own banner is the lever — pass an explicit small `cudagraph_capture_sizes` list (this
  branch truncates to `max_num_seqs`, `utils.py:185-190`) and/or lower `gpu_memory_utilization`.
- **Root cause:** the per-layer `graph_task_update_begin` / `_npu_paged_attention` / `graph_task_update_end`
  loop on `update_stream` with its recorded events never awaited, in a mode the fork itself labels
  experimental. Present it as the tag's known limitation, not as a defect with a fix commit: no commit
  fixing this boolean was identified, and the mode is gone from the fork by `v0.10.0rc1`.
- **Status and limits:** the thread is a stale close with no maintainer diagnosis — cite it as no
  evidence of a fix. The report carries no worker log, plog or `collect_env`, and the image is not
  inspectable, so the mode attribution rests on the version plus the requested flag: that is why the first
  reply line carries the two workarounds rather than a `path:line`.

## F. `FULL` requested, decode captured, prefill not — a hybrid (GDN) model, and the downgrade is one warning line

Reported (#11947, image `quay.io/ascend/vllm-ascend:v0.21.0rc1`, Qwen3.5-0.8B, TP1, `--max-model-len 1024
--max-num-batched-tokens 1024 --max-num-seqs 16`, `--async-scheduling`, `--enable-prefix-caching`,
`--additional-config '{"enable_npugraph_ex": true, "fuse_muls_add": true}'`, torch profiler on, no exception
anywhere):

```text
--compilation-config '{"cudagraph_mode": "FULL", "cudagraph_capture_sizes": [1,2,3,…,1024]}'
```

The complaint is a *missing capture*, not a crash: "Although I have set `cudagraph_mode: FULL`, the prefill
phase is not captured by the graph, and there are still gaps between operators." The issue was closed by the
reporter: not supported, on GPU either, because Mamba/linear-attention-family backends cannot be full-graph
captured.

Chain, read at both refs (`v0.21.0rc1` ascend, `v0.21.0` vLLM):

1. **The Ascend hook does not rewrite `FULL`.** In `check_and_update_config` the mode branches are
   `NONE` → `mode = NO_COMPILATION`, then `elif cudagraph_mode.requires_piecewise_compilation()` — and
   `FULL.requires_piecewise_compilation()` is `has_mode(PIECEWISE)` on a non-tuple mode, i.e. **False** — so
   `FULL` falls into `elif cudagraph_mode.has_full_cudagraphs():`, which sets
   `compilation_config.splitting_ops = []` (`vllm_ascend/platform.py:576-582` at `v0.21.0rc1`, assignment at
   `:576-579`, the comment reads "so we set splitting_ops to empty manually") and leaves
   `ascend_config.ascend_compilation_config.enable_npugraph_ex` as the reporter set it. The hook's own
   rewrites at that tag are xlite mode (`:474-481`), a non-`VLLM_COMPILE` mode (`:496`), encoder-decoder
   (`:529-542`) and an unknown enum (`:582-584`) — none touches `FULL`, so reading `platform.py` alone
   suggests FULL is honoured. Do not answer "platform.py downgraded it": at config time the mode is still
   `FULL`.
2. **Upstream rewrites it at runner init.** `CompilationConfig.resolve_cudagraph_mode_and_sizes`
   (`vllm/config/compilation.py:1301-1345` at `v0.21.0`, comparison at `:1319-1323`): for
   `cudagraph_mode.mixed_mode() == FULL and min_cg_support != AttentionCGSupport.ALWAYS` it raises only for
   `NEVER`, otherwise logs `CUDAGraphMode.<requested> is not supported with <backend> backend (support:
   <enum>); setting cudagraph_mode=…` and picks `FULL_AND_PIECEWISE` when `splitting_ops_contain_attention()`,
   else **`FULL_DECODE_ONLY`**. `splitting_ops_contain_attention()` is `all(op in self.splitting_ops for op in
   self._attention_ops)` (`:1209-1212`), and step 1 emptied the list — so the fallback on this stack is always
   decode-only. That is the trap: the downgrade happens after config, in the runner's
   `initialize_cudagraph_capture`.
3. **Which backend capped it.** `min_cg_support` is the minimum over attention metadata builders
   (`vllm/v1/worker/gpu_model_runner.py:6468-6484`), and Qwen3.5's hybrid linear attention is served by
   `GDNAttentionMetadataBuilder`, whose class attribute is
   `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH` (`vllm/v1/attention/backends/gdn_attn.py:77`);
   `vllm/model_executor/models/qwen3_5.py:135` selects that path on `layer_type == "linear_attention"`, while
   the fork's dense backends declare `ALWAYS` (`vllm_ascend/attention/attention_v1.py:264-267` at
   `v0.21.0rc1`). `UNIFORM_BATCH` is enough for decode capture and not for a mixed batch, which is exactly
   the reported split: decode graphed, prefill eager. One grep of the log's `(support: …)` value settles
   which builder set the minimum.

Answer shape:

- **No workaround restores full-graph prefill for that model at that release** — state it plainly rather
  than selling a flag. [reported] the thread is closed with the reporter's own conclusion that it is
  unsupported, including on GPU, because Mamba/GDN-family backends cannot be fully captured (their source:
  vLLM's `design/cuda_graphs` page, "CUDA Graphs compatibility of attention backends" — the upstream table
  is the cross-check that this is a backend property, not an Ascend gap). [verified] the mode that actually
  runs is `FULL_DECODE_ONLY`, the tag's default `PIECEWISE` path is equivalent for decode, and
  `--enforce-eager` only removes what is left. Prefill capture needs a backend at `ALWAYS`; do not propose
  `cudagraph_capture_sizes` tuning or `enable_npugraph_ex` as a fix for it, and do not go looking for a
  fork-side patch — the resolver is upstream and the declaring class is the model's.
- **The deliverable is the confirmation line**, which the reporter can grep in their own log:
  `grep 'is not supported with .* backend (support:' server.log` names both the backend and the enum. That
  turns "not supported" into a code-level why without NPU access.
- Root cause: `vllm-ascend` `platform.py:576-582` (`splitting_ops = []` for any full-graph request) plus
  upstream `compilation.py:1301-1345` (min-support downgrade) at the paired tag; the owner of the cap is the
  model's attention builder (`gdn_attn.py:77`), not vllm-ascend.

Evidence classes: the resolver, the declaring class and the empty-`splitting_ops` branch are `verified` at
the two refs above; the reporter's closing comment ("not supported … including on GPU") is `reported`; that
the profiler gaps *are* the eager prefill is `hypothesis`, with two falsifiers — `min_cg_support == ALWAYS` in
the log (no downgrade warning, so the prefill gap has another cause) or the same run on a dense model with
`FULL` showing no prefill gaps.

## G. `FULL_DECODE_ONLY` + `execute_dummy_batch`: a worker death and plog noise from `num_reqs_padded` (#7401)

Signature: a 2-node PD mixed deployment (P node + D node), GLM5-class model, W4A8, `FULL_DECODE_ONLY`.
The D worker dies mid-run inside the **dummy batch**, not in a request: `execute_dummy_batch` →
`_dummy_run(uniform_decode=True)` → `_model_forward` → the acl-graph `__call__` → `rtStreamSynchronize*`,
with an aivec line `DDR address of the MTE instruction is out of range`, `EL0004`, and — earlier in the same
plog — `TransposeKvCacheByBlock` op errors carrying `input2.blockIDs`. The Mooncake/ADXL `transfer failed`
lines in the same report are a *second, non-fatal* fault of the P node's KV-pool child; they do not explain a
worker death.

- **#7448 is a duplicate of this report** (`yiz-liu`, 2026-03-19: "This seems like a duplicate of #7401,
  we'll trace the issue there, closing this one in one day"). It was filed by `ZYang6263` — the same second
  reporter who posts the single-node A3 DP2 TP8, GLM5-W4A8, 8-concurrent recurrence here — and its
  `rtMemcpy … the current capture mode does not support this operation` twin plus `--max-num-batched-tokens
  16` add nothing: a run on #7448 spends batch 1 on the duplicate marker and batch 2 on this page.
- What the padded metadata is: `_dummy_run` sizes the device-side inputs from the *descriptor*, not from
  the request list. Verified at `v0.16.0rc1` in `worker/model_runner_v1.py`: `:2289 num_reqs_padded =
  batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs`, then `:2293
  num_scheduled_tokens.repeat(num_reqs_padded)`, `:2320-2321 self.seq_lens.np[:num_reqs_padded] = seq_lens`
  / `.np[num_reqs_padded:] = 0`, `:2325 self.query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens`,
  and `:2328` re-derives the count again through `_pad_query_start_loc_for_fia`. Three counts
  (`num_reqs`, the descriptor's padded count, the FIA-padded count) and three source lists
  (`seq_lens_list`, `num_scheduled_tokens`, the block table) must agree; when one is short, the capture
  replays over a `seq_lens`/block index that was never written and the kernel faults on an address.
- Two verified instances of that disagreement: **#7290** (the eagle3 drafter's `attn_metadata`
  `seq_lens_list` is one entry shorter than `num_reqs_padded` under `FULL_DECODE_ONLY`, per the PR body) and
  **#7460** (`_prepare_inputs` took `num_reqs_padded = num_reqs` whenever the *runtime* mode was `FULL`,
  even when the *configured* mode was `FULL_DECODE_ONLY`; `cbf46fad3` adds `and
  self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL` at `:567-570`, first tag `v0.18.0rc1`).
  Which of the two the report hit is `hypothesis` unless the traceback names the dummy batch's own
  arrays.
- **The guard exists in *partial* form at `v0.17.0rc1`, so "the check is missing at my tag" is not the
  finding.** Verified: `_pad_query_start_loc_for_fia` already reads `if cudagraph_runtime_mode ==
  CUDAGraphMode.FULL: num_reqs_padded = num_reqs` (`worker/model_runner_v1.py:557-558` there) — it keys on
  the **runtime** mode of the step, while #7460 adds the *configured* `compilation_config.cudagraph_mode`
  to the same branch. A report on `v0.17.0rc1` therefore sits between the two: the branch is reachable for
  a decode step that is not dispatched as `FULL` even though the model was captured in `FULL_DECODE_ONLY`.
  Say which side of that line the tag is on instead of quoting a line number from a later ref. And read
  #7460's own body: it describes the defect as *accuracy degradation in Qwen3-Next*, so citing it as the
  **crash** fix is the maintainer's thread attribution (`reported`), not the commit's claim.
- **The same branch produces *quietly wrong* decode, not only a worker death — and that is what #7460's
  body describes.** vllm-ascend #9666: Qwen3.5-9B on 910B4 (A2, four devices, a `VLLMWorker_TP` per card while the
  pasted command says `--tensor-parallel-size 1 --data-parallel-size 1`), `--speculative-config` **absent**
  (MTP off), `cudagraph_mode: FULL_DECODE_ONLY`, `--async-scheduling`, `--max-num-seqs 8` with 30 concurrent
  clients; an answer degenerates into one high-probability token (`{"intentName` followed by ~450 `!`) until
  `max_tokens`, HTTP 200, no error and no traceback (the body carries neither), and the reporter confirmed on
  2026-06-05 that
  **`v0.18.0` no longer shows it** (「v0.18.0版本已经没有这个问题：关闭MTP(投机推理)高并发会偶发输出重复的"!"」). Only
  #7460's hunk differs in that branch between `v0.17.0rc1` and `v0.18.0rc1`, so read #7460 as the
  *accuracy/garbage* fix it says it is instead of dismissing it as crash-only. Levers on the reporter's
  build: drop the decode graph (`--enforce-eager`, or `PIECEWISE`) or move to `>= v0.18.0rc1`; cost = the
  decode-graph speedup. Falsifier: the loop persists with `--enforce-eager` ⇒ not this padding (then it is a
  sampler / weight-format question; MTP was already off, so that control is spent).
- The maintainer thread also names #7235 (`4e62a2ae1`, "fix TransposeKvCacheByBlock op error report in
  plog"): it only *silences* plog errors that do not by themselves stop the engine, so it is not the fix for
  a worker death — quote it as noise reduction. Verified wording, in the release notes added by #7436
  (`adc57c595`, contained in `v0.18.0rc1`+):
  "GLM5 has an issue in the 2-node PD mixed deployment scenario where inference may hang when concurrency
  exceeds 8 (fixed in PR #7235 #7290)".
- Workaround (`proposed`, not reporter-confirmed): leave `FULL_DECODE_ONLY` (`--cudagraph-mode PIECEWISE`,
  or `--enforce-eager`) so the padded dummy-run path is not replayed, or move to a tag carrying #7460
  (`≥ v0.18.0rc1`); cost = the decode-graph speedup. Reporters on `0.18.0rc1`/`0.20.2rc1` still see it, so
  the known-issue line is not the whole story — say so.
- Reading order: the release note first (it is one grep at the reporter's tag:
  `git show <tag>:docs/source/user_guide/release_notes.md | grep -n -i '2-node PD'`), then the
  `num_reqs_padded` branch, then the plog. Do not spend the search on the Mooncake half of the same body.

Falsifier: the death happens outside `execute_dummy_batch` (a real request's step), or the configured mode in
the log is `FULL` rather than `FULL_DECODE_ONLY` — then this padding branch is not the one that ran.

## H. 310P/300I DUO: capture runs out of **events** — `EL0008` / `207007`, then `capture_end … 507903` (#12989)

Signature, in this order (`v0.23.0rc1`, image `quay.io/ascend/vllm-ascend:v0.23.0rc1-310p`, Atlas 300I DUO,
CANN `9.1.0-beta.1`, `npu-smi 25.5.2`, Qwen3-14B-W8A8SC, TP2, `cudagraph_mode: FULL_DECODE_ONLY` with five
sizes):

```text
Capturing CUDA graphs (decode, FULL):  80%|…| 4/5          # dies on the last capture, not the first
[ERROR] ERR00100 PTA call acl api failed.
Insufficient_Event_Resources(EL0008): The event resources are insufficient.
        Create capture event failed, stream_id=[27->36], event_id=[-1->640], error=117571609  # api_impl_capture_event.cc:52
        rtEventRecord execution failed, reason=driver error:no event resource                  # 207007
During handling of the above exception, another exception occurred:
RuntimeError: capture_end:…NPUGraph.cpp:247 … AclmdlRICaptureEnd(capture_stream_, &model_ri), error code is 507903
        rtStreamEndCapture execution failed, reason=an error has invalidated the stream capture sequence
RuntimeError: NPUModelRunner failed, error is capture_end:…
```

- **The `507903` is the consequence; the event allocation is the fault.** Read the *first* error
  (`EL0008`, `207007`, `GetCaptureEvent`, `event_id=-1`), not the capture-end wrapper: `507903` only says
  the sequence was already invalidated. Keep it apart from the capture-window refusal in
  `graph-capture-and-custom-ops.md` §C (a host copy issued *inside* the window): here nothing illegal was
  issued — the runtime had no event left to hand out.
- **The fork's own mitigation text does not fire for this variant.** `vllm_ascend/compilation/acl_graph.py`
  at `v0.23.0rc1` carries `_STREAM_RESOURCE_ERROR_CODE = "207008"`, `_STREAM_RESOURCE_ERROR_MARKERS =
  ("insufficient_stream_resources", "stream resources are insufficient")` (`:28-32`) and re-raises with
  `_STREAM_RESOURCE_GUIDANCE` (`:33-40`: "upgrading to a newer HDK/CANN stack, reducing
  cudagraph_capture_sizes, lowering max_cudagraph_capture_size, preferring FULL or FULL_DECODE_ONLY …,
  or temporarily disabling graph mode"); `_is_stream_resource_capture_error` (`:42-47`) matches only
  those. The reporter's `EL0008` / `207007` / `Insufficient_Event_Resources` is the **event** sibling and
  reaches them bare — which is why the report looks unhandled. The tag's own FAQ documents only the stream
  variant too (`docs/source/faqs.md:209-228`, quoting `207008` + `Insufficient_Stream_Resources(EL0009)`),
  so "read the guide" (the maintainer's `请认真查看指导文档`) points at the right *class* with the wrong code.
  A one-line fix (add `207007` / `insufficient_event_resources` / `EL0008` to the markers) is a legitimate
  follow-up issue; do not present it as the reporter's workaround.
- **No code-side cap exists: the capture-size list is whatever the user passed.** `platform.py`'s
  `_get_default_max_cudagraph_capture_size` (`:207-229`) returns `None` once `cudagraph_capture_sizes` **or**
  `max_cudagraph_capture_size` is set, so the fork injects no default and prunes nothing; there is no
  310P branch anywhere near it (`git grep -n 'capture_sizes' <ref> -- vllm_ascend/_310p` is empty at that
  tag; the only prune helper is `prune_capture_sizes_for_950`, `:109`). The old
  `update_aclgraph_sizes()` shrink heuristic is **gone** — `docs/source/developer_guide/Design_Documents/
  ACL_Graph.md:57-63` says so and names `cudagraph_capture_sizes` / `max_cudagraph_capture_size` as "the
  primary tuning levers". So "fewer captured graphs" is the user's edit, not a version bump.
- **Workaround, in the order to state it.** (1) `--enforce-eager` — *reporter-confirmed* on exactly this
  build (their TP2 command plus `--enforce-eager`, `--max-model-len 20480`: "TP2正常启动"); cost = no decode
  graph, so decode throughput/latency regress (small on 310P). (2) `proposed`: cut
  `cudagraph_capture_sizes` to one or two entries (`[1]`, `[1,2]`) — the failure lands at the 5th capture
  (`4/5`), and `event_id=[-1->640]` reads as the next event id being 640, i.e. a ceiling reached by
  hundreds of events accumulated across the five captures, so the *count* is the lever and not the shape. (3) `proposed`: `cudagraph_mode: PIECEWISE`/`NO_COMPILATION`. All
  three are the same trade in different amounts.
- **Status: driver-bound, not fixable in the fork.** Maintainer `Tflowers-0129`: 「硬件资源限制，导致图捕获数量
  有限」 and 「这个问题需要依赖未来驱动版本的发布」 [reported]; #12989 is closed (`bug`/`310p`/`triaged`).
  Say that plainly instead of hunting a code line: the ceiling is CANN/runtime event accounting
  (`api_impl_capture_event.cc`, `rtEventRecord`), outside every clone here.
- **Why TP2 and not TP1** is `hypothesis`: the `npu-smi` table shows the 8 `310P3` dies in DUO pairs behind
  one bus-id each (`0000:01:00.0` for devices 0 and 1, …), and `ASCEND_RT_VISIBLE_DEVICES=0,1` puts both
  ranks of TP2 on one package, so the per-device event pool is shared and per-rank stream/event demand
  doubles while the pool does not. Falsifier: TP2 with the two ranks on dies of *different* packages (or a
  TP1 run at the same capture-size count) that fails identically kills the sharing story; a TP2 that starts
  after only cutting the size list is consistent with it.

Falsifier for the class: a `capture_end` failure whose first error is not a resource error (a `107041` sync,
an `AclOpKernelInit failed`, an `EL0004` inside `execute_dummy_batch`) belongs to the neighbour page or to
§G, and one that fails at `capture_begin` with the *stream* code (`207008`/`EL0009`) is the documented
variant whose guidance already printed.

## Checklist

```bash
# which mode actually ran?
grep -n 'compilation enabled on NPU' <log>
# A: is the decode full-graph path the paged-attention one at this ref?
git grep -n '_npu_paged_attention\|full_graph_attention' <ref> -- \
  vllm_ascend/compilation/acl_graph.py vllm_ascend/attention/attention_v1.py
# B: is the padding block guarded, and does DP feed the value?
git show <ref>:vllm_ascend/worker/model_runner_v1.py | sed -n '/uniform_decode = (max_num/,+26p'
# C: how does the helper read the layer count?
git show <ref>:vllm_ascend/utils.py | sed -n '/def update_aclgraph_sizes/,+30p'
# D: is the dummy-run slot clear present, and on which side of the metadata build?
git show <ref>:vllm_ascend/worker/model_runner_v1.py | grep -n 'fill_(-1)'
# F: was FULL accepted and then downgraded? the warning names backend + enum
grep -n 'is not supported with .* backend (support:' <log>
git show <ref>:vllm_ascend/platform.py | grep -n -B2 -A6 'has_full_cudagraphs()'
git grep -n '_cudagraph_support' <vllm-ref> -- vllm/v1/attention/backends/<family>_attn.py
# E: legacy boolean or the enum? and does the replay branch update tasks on a private stream?
git grep -n 'full_cuda_graph' <ref> -- vllm_ascend | head
git grep -n -A3 'aclgraph.replay()' <ref> -- vllm_ascend/compilation | head -20
git log --oneline -S'Invalidate dummy slots' --all | head
# H: resource exhaustion at capture? which resource, and does the handler know the code?
grep -n 'EL0008\|EL0009\|207007\|207008\|Insufficient_' <log>
git show <ref>:vllm_ascend/compilation/acl_graph.py | nl -ba | sed -n '26,52p'
git show <ref>:docs/source/faqs.md | nl -ba | grep -n -A20 'stream resource exhaustion'
git show <ref>:vllm_ascend/platform.py | nl -ba | sed -n '206,232p'   # no default, no prune
git grep -n 'capture_sizes' <ref> -- vllm_ascend/_310p
```

## Recheck conditions

- `aclgraph_support` values and the downgrade block move between releases: re-read
  `attention/attention_v1.py` and the `mixed_mode()` check at the reported tag before quoting a line.
- The padding block is rewritten twice between `v0.12.0rc1` and `v0.13.0rc1`, so a reporter's wheel
  line number can sit on either shape — fingerprint the statement, not the line.
- `update_aclgraph_sizes` changes shape and address (`resources_per_graph`,
  `get_total_num_hidden_layers`, `utils.py:485` region by `v0.16.0rc1`) — cite the tag you read.
- §D's fix exists as three shas for one change; `git tag --contains` per sha, not per title, and check the
  release line the reporter is on (`v0.23.0.post1` / `v0.26.0rc1` / `v0.27.1rc1` are different lines).
- §E's key is a legacy spelling: `full_cuda_graph` is present only up to `v0.9.x` in the fork and deprecated
  upstream at `v0.10.1`; at any later ref the same intent arrives as `cudagraph_mode`, so re-run
  `git grep -n full_cuda_graph <ref> -- vllm_ascend` before quoting §E at a `v0.10+` report.
- §H's marker lists are newer than the failure they document: `acl_graph.py` grew
  `_is_stream_resource_capture_error` for `207008`, and `docs/source/faqs.md` grew its entry 17 over the
  same release range. Re-read both at the reported tag — a later tag may have widened the markers to the
  event family, which changes the answer from "the guidance did not fire" to "it fired and was ignored".
  The `update_aclgraph_sizes()` shrink heuristic exists only up to the release named in that tag's
  `Design_Documents/ACL_Graph.md`; do not offer it as a lever on a tag that documents its removal.
