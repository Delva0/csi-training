# Structured output / guided decoding on Ascend (grammar-bitmask path)

Scope: reports where the worker dies inside grammar-guided decoding — `response_format` with a
`json_schema`/`structural_tag`, `guided_json`, or a forced `tool_choice` when
`--enable-auto-tool-choice` is set. Verified by reading the clones
(`vllm-project/vllm`, `vllm-project/vllm-ascend`, `mlc-ai/xgrammar`) on 2026-11-01; nothing was
executed on an NPU. Line numbers are per tag — re-anchor with `git grep -n <symbol> <ref>`.

## Signature that identifies this class

```
vllm_ascend/worker/worker.py:342 sample_tokens
 -> vllm_ascend/worker/model_runner_v1.py:1564 sample_tokens
 -> vllm/v1/structured_output/utils.py:126 apply_grammar_bitmask
 -> xgrammar/matcher.py:142 apply_token_bitmask_inplace
 -> xgrammar/kernels/apply_token_bitmask_inplace_cpu.py:46 apply_token_bitmask_inplace_cpu
 -> _core.kernels.apply_token_bitmask_inplace_cpu(
TypeError: apply_token_bitmask_inplace_cpu(): incompatible function arguments.
  1. apply_token_bitmask_inplace_cpu(logits_ptr: int, logits_shape: tuple[int,int],
     logits_strides: ..., bitmask_ptr: int, ..., vocab_size: int,
     indices: collections.abc.Sequence[int] | None) -> None
Invoked with types: int, tuple, tuple, int, tuple, tuple, int, torch.Tensor
```

The client-facing error is `vllm.v1.engine.exceptions.EngineDeadError` after this worker
`TypeError`; every TP rank repeats it because each rank runs `apply_grammar_bitmask`.

## Why a CPU kernel runs on an NPU worker

xgrammar dispatches on `logits.device.type` (cpu -> CPU kernel, cuda -> triton, anything else ->
`apply_token_bitmask_inplace_torch_compile`). vllm-ascend deliberately moves the logits to CPU
first, in `vllm_ascend/worker/model_runner_v1.py::sample_tokens` — present at every tag checked
from `v0.11.0` through `v0.18.0`:

```python
if grammar_output is not None:
    # here we are different from gpu_model_runner,
    # the apply_grammar_bitmask uses torch.compile to optimize this, ascend does not support it now
    logits_dtype = logits.dtype
    logits = logits.to("cpu").float()
    apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
    logits = logits.to(self.device).to(logits_dtype)
```

So the CPU branch is the designed path for this plugin, not a misconfiguration. `main` replaces it
with a device-side kernel (`vllm_ascend/ops/triton/v2/apply_grammar_bitmask.py`,
`_apply_grammar_bitmask_kernel`, wired from `patch/worker/patch_triton.py`); check the reported tag
before quoting the CPU round-trip.

## Why it fires intermittently (mixed batches)

Upstream `vllm/v1/structured_output/utils.py::apply_grammar_bitmask` builds
`out_indices` from the requests in `grammar_output.structured_output_request_ids` and computes
`skip_out_indices = len(out_indices) == logits.shape[0]`. `indices` is passed **only** when the
constrained rows are a proper subset of the logits rows:

```python
index_tensor = None
if not skip_out_indices:
    index_tensor = torch.tensor(out_indices, dtype=torch.int32, device="cpu", pin_memory=True)
    index_tensor = index_tensor.to(logits.device, non_blocking=True)
xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
```

A batch where every row is constrained passes `indices=None` and cannot hit this crash; one forced
tool call arriving among free-form requests makes the subset non-empty. That is why an agent client
(tool-forced requests) on a shared server sees "once an hour at low concurrency" rather than a
constant failure — do not explain the rate with load.

## Root cause: xgrammar accepted a tensor in Python, rejected it in C++ (fixed upstream)

- `xgrammar/kernels/apply_token_bitmask_inplace_cpu.py` annotates
  `indices: Optional[Union[List[int], torch.Tensor]]` and forwards it unchanged.
- The nanobind binding it calls takes `std::optional<std::vector<int>> indices`
  (`cpp/nanobind/python_methods.cc::Kernels_ApplyTokenBitmaskInplaceCPU`), so a `torch.Tensor`
  raises exactly the `TypeError` above. This is true in the same release — it is not a mismatched
  wheel.
- Upstream `mlc-ai/xgrammar` issue #506, fix PR #507 (commit `365ea8b`, "Fix error
  `TypeError: apply_token_bitmask_inplace_cpu(): incompatible function arguments`") converts with
  `indices.tolist()`. First release tag containing it: **`v0.1.30`** — absent from `v0.1.28` and
  `v0.1.29`.
- Version map through the **upstream vLLM** repo's `requirements/common.txt` (verified in the
  `/repos/vllm-project/vllm` clone): vllm `v0.13.0` -> `xgrammar == 0.1.27`; `v0.15.0` ->
  `== 0.1.29` (still affected); `v0.18.0` -> `>= 0.1.32, < 1.0.0`; `v0.23.0` -> `>= 0.2.0, < 1.0.0`;
  `main` -> `== 0.2.8`. Do not read this from the vllm-ascend tree: that repo has only
  `requirements.txt` (plus `-dev`/`-lint`/`-nightly`), it never pins `vllm` at any tag from
  `v0.9.1` to `main`, and its xgrammar line appears only from `v0.14.0rc1` as the permissive
  `xgrammar>=0.1.30`.
- The plugin tag maps to the upstream tag through that tag's `Dockerfile`
  (`ARG VLLM_TAG`): `v0.13.0rc1 -> v0.13.0`, `v0.10.1rc1 -> v0.10.1.1`, `v0.11.0 -> v0.11.0`.
- Version check from the error text alone: the binding gains a trailing `logit_type: str`
  parameter from 0.1.28 on. A signature that ends at `indices` pins **<= 0.1.27**.

## Workarounds, cheapest first

1. Pin `xgrammar >= 0.1.30` in the image (or apply the three-line `.tolist()` to the installed
   `apply_token_bitmask_inplace_cpu.py`). This removes the cause and keeps guided decoding.
2. Remove the trigger surface for the affected build: no `response_format`/`guided_json`/
   `structural_tag`, and do not pair `--enable-auto-tool-choice` with clients that send
   `tool_choice: required` (most agent frameworks do). The engine stops dying; constrained output
   is lost.
3. Move to a plugin tag whose `ARG VLLM_TAG` is `>= v0.18.0` (xgrammar `>= 0.1.32`).

## Boundaries

- Source- and upstream-history-verified: the CPU round-trip, the subset condition, the xgrammar
  binding mismatch, the fix commit and its first tag. Not verified on hardware here.
- The report thread for this class can be closed as stale with no maintainer statement; a closed
  thread is not a fix, and the upstream xgrammar change is the evidence to quote.
- Keep this separate from sampler/top-k reports: `grammar_output is not None` is the proof the
  structured-output path ran, and no sampling knob changes this crash.

## Second class here: `AssertionError: (-1, '<req>', {'<req>': [-1, -1, -1]})` in `grammar_bitmask`

Signature (reporter log, image `vllm-ascend:0.13.0rc2`, whose `Dockerfile` pins
`ARG VLLM_TAG=v0.13.0`):

```
[... /project/cpp/grammar_matcher.cc:428: Warning: The token id -1 is out of range [0, 129280). Rejecting the token.]
ERROR [backend_xgrammar.py:180] Failed to advance FSM for request chatcmpl-... for tokens -1.
ERROR [patch_core.py:70] EngineCore encountered a fatal error.
  .../vllm/v1/core/sched/scheduler.py:1048 get_grammar_bitmask
  .../vllm/v1/structured_output/__init__.py:288 grammar_bitmask
    assert accepted, (token, req_id, scheduled_spec_decode_tokens)
AssertionError: (-1, 'chatcmpl-...', {'chatcmpl-...': [-1, -1, -1]})
```

The `-1`s are **scheduler placeholders for speculative tokens**, not drafts and not a grammar bug.
`StructuredOutputManager.grammar_bitmask` (serial path, vLLM `v0.13.0`
`vllm/v1/structured_output/__init__.py:268-288`) advances the grammar over *every* entry of
`scheduled_spec_decode_tokens`; xgrammar rejects `-1` as out of vocab (`backend_xgrammar.py:180`
logs and returns False, "Please file an issue"), the assert fires, and the EngineCore dies. The FSM
never advances for that request, so lowering concurrency or retrying cannot help.

Something pre-filled `request.spec_token_ids` with `-1`; which component did, decides the fix:

| injector | locator | precondition |
| --- | --- | --- |
| upstream `AsyncScheduler._update_after_schedule` — `request.spec_token_ids = [-1] * self.num_spec_tokens` | vLLM `v0.13.0` `vllm/v1/core/sched/async_scheduler.py:35-37` | `--async-scheduling` (default `False` at `v0.13.0`; explicitly allowed with EAGLE/MTP, `vllm/config/vllm.py:542-560`) |
| plugin `RecomputeScheduler.add_request` — `request.spec_token_ids = [PLACEHOLDER_TOKEN_ID] * self.num_spec_tokens` | plugin `v0.13.0rc2` `vllm_ascend/core/recompute_scheduler.py:101-111`; `PLACEHOLDER_TOKEN_ID = -1` at vLLM `v0.13.0` `vllm/v1/sample/rejection_sampler.py:22` | `additional_config.recompute_scheduler_enable` (plugin `platform.py:333-336` swaps in `RecomputeSchedulerConfig`) **and** `is_mtp_kv_consumer` = speculative config + `kv_transfer_config.is_kv_consumer` + `vllm_version_is('0.13.0')` — i.e. a PD **decode** node with MTP |

Both are opt-in, so the report needs the D node's serve line, its `--additional-config` and whether
`--async-scheduling` was passed; a body that pastes only a traceback does not identify the injector,
so say that and ask for those flags instead of picking one. The plugin injector has a *different*
symptom on the 0.20 line — the sampler indexes the `-1` row instead of rejecting it, giving
fluent-but-wrong output rather than this crash — in
[sampling-and-output-quality.md](sampling-and-output-quality.md).

Fix boundary, upstream: `43e3f8e4a9` ("[Misc] Various code simplifications (#31666)") added
`if token == -1: apply_bitmask = False` — the loop stops advancing the grammar at the padding token —
first tag **`v0.14.0rc1`/`v0.14.0`**; still present on `main` (`:497`, now as
`padding_or_grammar_invalid = True` with no raise on rejection). The tuple-shaped assert itself first
appears in `47604137a2` (#28298), first tag `v0.11.1`, so the message also dates the build
(`>= v0.11.1`). The plugin's `vllm_version_is('0.13.0')` gate means *that* placeholder fill is
specific to the 0.13.0 pair.

Workarounds, cheapest first:

1. Move the plugin/image to `v0.14.0rc1` or later, whose paired vLLM skips `-1` in the grammar
   advance. Nothing in the grammar path needs a patch.
2. Remove the placeholder source on 0.13.0: drop `recompute_scheduler_enable` /
   `additional_config` from the decode node, or drop `--async-scheduling`, or disable MTP there (the
   reporter's own workaround). Each costs what the placeholders exist to enable — full-graph
   compatibility on the first decode step, or MTP throughput.
3. No structured output on that node (correct but loses the feature).

Boundaries: verified in source at the tags above, not executed. The matching report thread
(0.13.0rc2, DeepSeek-V3.2 MTP + PD) was self-closed by the reporter as "Inconsistent configuration
between prefilling and decoding nodes" with no detail and an unanswered follow-up, and it carries no
launcher — that is neither a root cause nor evidence for one of the two injectors.

## No exception at all: a streaming hang, and only one of two deployments hangs

Scope guard: both classes above always leave a worker exception. When the symptom is "the client
prints nothing after `choice (Streaming):` and never returns", with no `EngineDeadError`, no worker
line and a server that still answers `/health`, the grammar-bitmask path is not implicated — do not
read `apply_grammar_bitmask`, `matcher.py` or the sampler. The only evidence in such a body is the
**deployment diff**, and the answer is an A/B one.

Report shape (`#8328`: `v0.18.0rc1` + vLLM 0.18.0, Qwen3.5-27B-W8A8-mtp, DP=1, body = the full
`collect_env.py` plus two `docker run`/`vllm serve` blocks, no log; thread = one collaborator
comment "Plz see this issue.", open, no diagnosis):

- **Subtract what both deployments share first.** Here that is `--async-scheduling`, MTP
  (`qwen3_5_mtp`, `num_speculative_tokens: 3`), `--compilation-config FULL_DECODE_ONLY`,
  `--quantization ascend` and `--max-num-seqs 128`. A shared flag cannot be the delta, so the
  async-scheduling / spec-token-placeholder material above is out of scope by construction — say
  that in the answer instead of re-deriving it.
- **Then filter the remaining names against the reported tag.** At `v0.18.0rc1` the A-only
  `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`, `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` and
  `VLLM_ASCEND_ENABLE_PREFETCH_MLP` have no hit in `vllm_ascend/` (no plugin reader at that tag;
  `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` is a retired 0.9.x name, and any unknown-name warning is
  cosmetic — [env-vars-and-additional-config.md](env-vars-and-additional-config.md)),
  `VLLM_ASCEND_ENABLE_NZ=2` is weight-format only (`envs.py:91`), and the one scheduling delta that
  survives is `VLLM_ASCEND_BALANCE_SCHEDULING=1` (`envs.py:105`).
- **`VLLM_ASCEND_BALANCE_SCHEDULING=1` substitutes the scheduler class.**
  `patch/platform/__init__.py:38` imports `patch_balance_schedule` only when it is set, and that
  file ends with `vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler` (`:706`) plus a
  replacement `EngineCoreProc.run_engine_core` (`:705`). `BalanceScheduler` is a copy of the
  upstream `Scheduler` with a per-DP-rank `balance_gather` (`:54-56`) filling `self.balance_queue`
  (`:49-53`, one CPU tensor per `data_parallel_size` entry).
- **At DP=1 the balancing logic is inert, so it is a poor root cause for a single-engine hang.**
  The replacement `run_engine_core` builds `BalanceDPEngineCoreProc` only
  `if parallel_config.data_parallel_size > 1 or dp_rank > 0` (`:676`) and plain `EngineCoreProc`
  otherwise (`:686`), so `balance_gather` never runs, `balance_queue` stays all-zero and
  `balance_flag = max(...) >= max_num_running_reqs - 1` (`:269`) is False. What is left is the
  frozen-copy risk — `Scheduler` *is* replaced at DP=1, so a divergence from the paired upstream
  `vllm/v1/core/sched/scheduler.py` is the only live path there; compare those two files at the two
  tags before calling it a cause.
- `HCCL_OP_EXPANSION_MODE="AIV"` (also A-only) *is* read by the plugin, but as an ACL-graph stream
  budget selector (`vllm_ascend/utils.py:506`, hint at `:554`), not by the sampling path; it changes
  which graph shapes get captured ([full-decode-only-graph-mode.md](full-decode-only-graph-mode.md)).
  The other A-only names (`MIES_USE_MB_SWAPPER`, `ATB_LLM_COMM_BACKEND`, `HCCL_BUFFSIZE`,
  `HCCL_OP_RETRY_ENABLE`, `LD_PRELOAD`, `CPU_AFFINITY_CONF`) are owned outside the plugin and stay
  unexamined, and `--mm-processor-cache-type shm` (upstream `v0.18.0`
  `vllm/config/model.py:305`) only touches multimodal preprocessing. Say so rather than implying
  they were cleared.

Answer shape: candidate = the one gated delta (balance scheduling), or a frozen-`Scheduler`
divergence if a diff shows one; workaround = run the same failing script with
`VLLM_ASCEND_BALANCE_SCHEDULING` unset (equivalently, deployment B's env set) — one variable — and
the confirming check is that rerun, plus the server log around the first request and the client's
read timeout if it still hangs.

Boundaries: all of the above is read from `v0.18.0rc1`/`v0.18.0` source; nothing ran on an NPU, and
the thread carries no maintainer statement, so the mechanism stays a candidate and the one-variable
rerun is the deliverable.
