# LoRA adapters on Ascend: TP>1 kernel contract, 0.9.x graph capture, and test coverage

Scope: reports where `--enable-lora`/`--lora-modules` fails to load, crashes in the dummy warmup,
dies during graph capture on 0.9.x, or hangs the device in decode, especially with
`--tensor-parallel-size > 1` and packed (`MergedColumn`/QKV) modules such as the GDN
`in_proj_ba`. Verified against the read-only clones at plugin tag `v0.23.0` and upstream
`v0.23.0` (2026-11-01); the 0.9.x graph-capture section at plugin `v0.9.1rc1`/`v0.10.1rc1`, the
device/rank kernel-binding section at `v0.10.1rc1`, `v0.13.0rc1` and `v0.18.0`.
Re-anchor line numbers with `git grep -n <symbol> <tag> -- <paths>` before quoting.

## V1 + ACL graph on 0.9.x: the `torch_ops` einsum guard

The 0.9.x line selects the PyTorch-native LoRA ops, not the AscendC kernels. At `v0.9.1rc1`
`vllm_ascend/lora/punica_wrapper/punica_npu.py:6-8` imports all six kernels from
`vllm.lora.ops.torch_ops`. V1's default `PIECEWISE` compilation leaves
`use_aclgraph = PIECEWISE and not enforce_eager` true with no LoRA guard
(`vllm_ascend/worker/model_runner_v1.py:325-327`), so graph capture traces the LoRA forward:
`_shrink_prefill` -> `sgmv_shrink` -> `bgmv_shrink` -> `torch.einsum("bi, boi -> bo", ...)`.
FakeTensor cannot guard the dynamic token size and the worker dies with:

```text
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: Could not guard on
  data-dependent expression Ne(u0, 1) (unhinted: Ne(u0, 1)).  (Size-like symbols: u0)
ATTENTION: guard_size_oblivious would fix the error, evaluating expression to True.
```

The outer log ends in `RuntimeError: Engine core initialization failed`; the `einsum` /
`bgmv_shrink` frame is the useful signature, not the `multiproc_executor` wrapper. The maintainer
answer on the thread (vllm-project/vllm-ascend#1629) is `--enforce-eager` (the thread spells it `--enable-eager`; the CLI flag is
`--enforce-eager`, `vllm/engine/arg_utils.py:500` at vLLM v0.9.1). It makes `use_aclgraph` false;
`capture_model()` then logs `Skipping NPU graph capture for eager mode.`

LoRA + aclgraph first ships in `v0.10.1rc1` through two commits, both absent from every v0.9.x tag:
#2113 (`c0f0b7081`) adds the `csrc/torch_binding_meta.cpp` Meta registration for custom ops (the
same fix behind the `_C::<op>: attempted to run this operator with Meta` class in
[log-signatures.md](log-signatures.md)), and #2325 (`3648d18e6`) adds the AscendC
`sgmv_shrink`/`sgmv_expand` kernels; at `v0.10.1rc1` `punica_npu.py:9-16` imports
`vllm_ascend.lora.punica_wrapper.lora_ops` except on 310P. #2325's own message reports about
100% TTFT/TPOT/throughput improvement on Qwen2.5-7B under aclgraph, so the same pair covers both
the start-up failure and the 2-3 tok/s V0 path: the slow path is the `torch_ops` implementation
itself, not graph capture. The LoRA precision fix #4141 (`ea8f544ce`) is a later, separate
follow-up first shipped in `v0.13.0rc1`.

## Slow LoRA decode: the plugin deliberately binds `torch_ops` on 310P and at rank ≥ 128

A "LoRA drops throughput to a few token/s" report is a kernel-binding question before it is a
kernel-performance question. `vllm_ascend/lora/punica_npu.py` chooses between upstream's
PyTorch `vllm.lora.ops.torch_ops` and the plugin's own `vllm_ascend.lora.lora_ops`, and the
upstream branch is exactly the unoptimized path #2325 was written to replace:

- `v0.10.1rc1`: `if is_310p(): from vllm.lora.ops.torch_ops import ...` else the plugin kernels
  (`:9-16`) — the first tag where the plugin kernels are imported at all, and then only off 310P.
- `v0.13.0rc1`: `if get_ascend_device_type() == AscendDeviceType._310P:` (`:9`) — 310P alone stays
  on `torch_ops`.
- `v0.18.0` and `v0.23.0`: the gate gained a second arm,
  `get_ascend_device_type() == AscendDeviceType._310P or (self.lora_config is not None and
  self.lora_config.max_lora_rank >= 128)` (`:25-26`); upstream's default `max_lora_rank` is 16
  (`vllm/config/lora.py:34` at v0.23.0), so a report that raised `--max-lora-rank` to ≥128 puts
  an A2/A3 run back on `torch_ops` too.

Consequence for triage: on 300I DUO / 310P the accelerated LoRA kernels are excluded **by
design at every tag checked**, so the fix for the 0.9.x slowness (upgrade past #2113/#2325,
`v0.10.1rc1`) does not buy a 310P deployment the AscendC path, and the one tip the tag's own
`docs/source/user_guide/feature_guide/lora.md` offers — "disable graph mode and try again"
(`v0.9.2rc1`) — has nothing to disable on 310P, whose tag docs state the 310I series "only
supports eager mode" ([310p-atlas-300i-duo.md](310p-atlas-300i-duo.md)); the recorded 310P
`v0.9.2rc1-310p` report says the same from the deployment side. Check the binding first, then the
rank gate (`--max-lora-rank`), then the hardware.

Boundary: the binding is read from the tags; no 310P LoRA throughput was measured here, so state
the mechanism and the excluded path, not a speedup number, and treat any "eager mode speeds it up"
claim as belonging to the graph-capture class above rather than to this one.

## Multimodal tower/connector LoRA at start-up: the dying ops are upstream torch, not the plugin kernels

Recorded in vllm-project/vllm-ascend#10626 (open; its only comment pings a maintainer, no root
cause): `quay.io/ascend/vllm-ascend:v0.19.1rc1`, Qwen2VL (`Qwen2VLForConditionalGeneration`),
three adapters, `--max-lora-rank 64 --max-loras 8`; both launches die during `profile_run`, i.e.
after the model and the adapters loaded and before serving. Two scenarios, two different ops:

| Launch | Failing op / code | Statement |
| --- | --- | --- |
| `--enable-lora` only (TP=1, `--enforce-eager`, fp16) | `aclnnUniqueConsecutive`, `507018` (AICPU) at `profile_run -> _dummy_run -> maybe_dummy_run_with_lora -> set_active_adapters -> compute_meta -> torch.unique_consecutive` | upstream `vllm/lora/punica_wrapper/utils.py:25-27`: `lora_indices_tensor, seq_length_tensor = torch.unique_consecutive(token_lora_tensor, return_counts=True)` |
| `+ --enable-tower-connector-lora --fully-sharded-loras` | `aclnnInplaceCopy`, `507057` plus "The DDR address of the MTE instruction is out of range" at `set_active_adapters -> activate_adapter -> set_lora -> lora_a_stacked.copy_` | upstream `vllm/lora/layers/base_linear.py:115-117`: `self.lora_a_stacked[0][index, 0, :lora_a.shape[0], :lora_a.shape[1]].copy_(lora_a, non_blocking=True)` |

- Pin and kernel selection at the reported tag: `v0.19.1rc1:Dockerfile:51` is
  `ARG VLLM_TAG=v0.19.1` (image base `quay.io/ascend/cann:8.5.1-910b-...`, `:18`), and
  `vllm_ascend/lora/punica_npu.py:21-44` binds `vllm.lora.ops.torch_ops` only on 310P or when
  `max_lora_rank >= 128`, otherwise the plugin's `vllm_ascend/lora/lora_ops.py`. At rank 64 on a
  910B the plugin's AscendC `bgmv_*/sgmv_*` set is selected, yet **neither reported failure is in
  those kernels**: both die in plain torch ops of the adapter-metadata/activation path, before any
  LoRA matmul. For this shape do not start from the `csrc/` assertions or the upstream
  `_mcp_apply` gather.
- Read the two warning lines as scope, not cause: `... supports adding LoRA to the tower modules`
  and `no matching PunicaWrapper found; visual.* will be ignored` come from upstream
  `vllm/lora/model_manager.py` (`:177` and `:373` at `v0.19.1`), so in scenario A the visual tower is not LoRA-wrapped and that
  crash is not a visual-LoRA bug; `LoRA for the tower and connector of multimodal models is
  experimental` is scenario B's own gate.
- Candidate checks, not verified here: for scenario A print `token_lora_tensor` (dtype, device,
  values; `convert_mapping` builds it) before blaming the plugin — the code has no dtype/device
  guard around `unique_consecutive`, and the reported run is `--enforce-eager`, so no graph is
  involved. For scenario B compare `lora_a.shape` with the destination slice of `lora_a_stacked`
  (sized from `max_lora_rank`/`tp_size`) and re-test without `--fully-sharded-loras`, which changes
  the stacked shapes; `507057` is a write past a buffer on an in-place copy. Both need the plog's
  `current working operator name` to confirm.

## Where the code lives at v0.23.0

- Plugin: `vllm_ascend/lora/{punica_npu,lora_ops,fused_moe,utils}.py`.
- `PunicaWrapperNPU.__init__` picks the kernel set by device/rank — `torch_ops` on
  `AscendDeviceType._310P` or at `max_lora_rank >= 128`, otherwise the plugin's AscendC
  `bgmv_*/sgmv_*` ops (per-tag lineage in the slow-decode section above). So "which kernel rejected
  the shape" depends on the rank in the launch, not only on the model.
- Expand dispatch inside `add_expand` → `_apply_expand` → `self.is_prefill ? _expand_slice_prefill
  : _expand_slice_decode`, i.e. **prefill/warmup reaches `sgmv_expand_slice`, decode reaches
  `bgmv_expand_slice`** (the non-slice `_expand_prefill`/`_expand_decode` use `sgmv_expand`/
  `bgmv_expand` for embedding-style callers). That mapping tells you which of the two C++ checks a
  report's stage can hit: crash during dummy warmup = sgmv, hang during decode = bgmv.

## The two assertions and the real contract

`csrc/torch_binding.cpp` at `v0.23.0`:

```text
:476  TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");   // bgmv_expand
:561  TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");   // sgmv_expand
```

The same functions set `int lora_rank = x.size(1);` and pass it as the reduction width, while the
weight is documented one line earlier as `[num_loras, hidden_out, hidden_in]`
(`:470-471`, `:429-430`). So the width handed in is used as the K of `x @ weightᵀ`: the contract the
kernel actually needs is *width == lora_B.hidden_in*, and `width <= slice_size` is only a weak sanity
bound that can pass while K over-reads the weight. `bgmv_expand`/`sgmv_expand` also take raw
`x.data_ptr()` into a custom handler, so a non-contiguous narrowed view is not safe there.

## The widening caller is upstream, not the plugin

Upstream `vllm/lora/layers/column_parallel_linear.py` at `v0.23.0`:

- `_mcp_apply()` builds `buffers` shaped `(n_slices, tokens, local_lora_rank)` and calls
  `tensor_model_parallel_all_gather(buffers)` (line 64, default `dim=-1`) → width `tp * local_rank`.
- Every MCP/QKV LoRA class in that file routes `apply` through `_mcp_apply`, both the
  `@_not_fully_sharded_can_replace` and `@_fully_sharded_can_replace` variants.
- `vllm_ascend/lora/punica_npu.py` contains **no** all-gather; it hands `x[slice_idx]` straight to
  `_apply_expand`. A report blaming the plugin's `add_expand` for the gather is describing the
  upstream caller.

Why the same buffer is fine on GPU: the Triton expand takes its reduction bound from the weight —
`vllm/lora/ops/triton_ops/lora_expand_op.py:206` `K = lora_b_weights[0].shape[-1]  # K= rank` — so the
redundant gathered columns are simply not read. The Ascend C++ kernel takes the bound from
`x.size(1)` instead. That divergence, not the gather itself, is the defect shape to report.

Shapes that matter for `--fully-sharded-loras` (`MergedColumnParallelLinearWithLoRA.create_lora_weights`):

| config | `lora_a_stacked` rank dim | `lora_b_stacked` last dim | gathered width |
| --- | --- | --- | --- |
| default | `max_lora_rank` | `max_lora_rank` | `tp * max_lora_rank` (too wide) |
| `--fully-sharded-loras` | `max_lora_rank / tp_size` | `max_lora_rank` | `max_lora_rank` (matches) |

So `--fully-sharded-loras` is the first thing to test on a TP>1 MCP LoRA crash: it makes the
all-gather produce exactly `lora_B`'s rank. Candidate workaround — the plugin-side kernel contract was
not exercised here, and it needs `max_lora_rank` divisible by TP. The flag's documented meaning
(`docs/source/user_guide/feature_guide/lora.md` on `main`: "Without this flag, tensor parallelism
shards only part of the LoRA computation") is also the reason the flag changes shapes at all.

Rank limits: `vllm/config/lora.py:26` at `v0.23.0` is
`MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]` (default 16). There is no 4, and
`max_lora_rank` must be ≥ the adapter rank, so an r=8 adapter on a layer whose per-rank slice is
6 dims has no smaller legal value to try.

## What the tag's docs and tests do and do not establish

- `docs/source/user_guide/feature_guide/lora.md` at `v0.23.0` says nothing about TP limits (grep for
  `tensor.parallel|limit|support` returns only usage text and the MoE/EP/quantization caveat). Silence
  is not "unsupported", and it is not support either. The current `main` doc adds the fused
  single-slot fast path and its gates (`--max-loras 1`, BF16, rank in {8,16,32,64}, ≤128 tokens) —
  do not project those back onto `v0.23.0`.
- e2e LoRA coverage at `v0.23.0`: `tests/e2e/pull_request/one_card/lora/` has
  `test_qwen35_densemodel_lora.py`, `test_olmoe_lora.py`, `test_lora_with_spec_decode.py`;
  `two_card/lora/` has `test_ilama_lora_tp2.py`, `test_llama32_lora_tp2.py`,
  `test_qwen3moe_lora_tp.py`. TP=2 LoRA is exercised for llama/Qwen3-MoE; there is no TP>1 test for
  the hybrid-GDN Qwen3.5 model, so a TP>1 Qwen3.5-27B LoRA report falls outside coverage — say that
  rather than "unsupported".
- `in_proj_ba` is a two-slice packed module (`vllm_ascend/quantization/modelslim_config.py:88` maps
  `in_proj_ba` → `["in_proj_b", "in_proj_a"]`) and GDN runs through `vllm_ascend/ops/gdn.py` at this
  tag, so a report's "MergedColumn [48,48]" is this layer and its per-rank output slice is 48/tp.

## Next verification steps for this class

1. Reproduce with `--fully-sharded-loras` and with TP=2 before TP=4/8, and log
   `x.shape`/`x.stride()` at `add_expand` entry to confirm the width is `tp*rank`.
2. Check the fix direction, not just the crash: the reporter's "narrow to `lora_B`'s `hidden_in` and
   `.contiguous()`" rests on `lora_A` being replicated so the gathered chunks are identical; confirm
   that per layer class, then compare TP=1 vs TP=4 logits — a narrowing that drops real columns
   fails the numerics check instead of the shape check.
3. Search forward before proposing a plugin fix:
   `git log --oneline v0.23.0..origin/main -- vllm_ascend/lora/punica_npu.py` (7 commits at the
   snapshot, the LoRA-relevant ones being #15884 performance and #16457 310P graph-mode/MRV2) and
   `git tag --contains <sha>` for each.

## Batch invariance does not cover the LoRA kernels

The batch-invariant dispatch table registered by `vllm_ascend/batch_invariant.py` overrides
`aten::mm`/`matmul`/`addmm`/`bmm`/`softmax`/`sum` (and `npu_fused_infer_attention_score`,
`npu_add_rms_norm`) only; adapter deltas run through
`torch.ops._C_ascend.{sgmv,bgmv}_{shrink,expand,expand_slice}`, which no Aten registration can
intercept. For a "same prompt, different tokens at higher concurrency with LoRA" report, read
batch-invariance.md first and present LoRA coverage as an unmeasured gap, not a proven cause.

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **A kernel-contract assertion can be a C++ `TORCH_CHECK` in `csrc/`, with the widening caller
  upstream.** `hidden in should be smaller than hidden out` is `csrc/torch_binding.cpp:476`/`:561`
  at `v0.23.0`, while the `tp*rank`-wide LoRA buffer is produced by upstream `_mcp_apply`'s
  all-gather and never by the plugin's `punica_npu.py`. Grep `csrc/` and the upstream clone before
  patching a Python wrapper.
