# A quant aux param allocated with a parallel factor but loaded unsharded: `The expanded size of the tensor (C//N) must match the existing size (C)` (W4A8 MoE `w2_scale_bias`)

Lane B, load phase, every rank, while the shard bar is still moving. The deepest
frames are the **weight loader**, not a kernel:

```text
(Worker_TP0 pid=96546) ERROR [multiproc_executor.py:772]   … worker/model_runner_v1.py:2341 load_model
(Worker_TP0 pid=96546) ERROR   … vllm/model_executor/models/qwen3_moe.py:628 load_weights
(Worker_TP0 pid=96546) ERROR   … vllm/model_executor/layers/fused_moe/layer.py:1310 weight_loader
(Worker_TP0 pid=96546) ERROR   … vllm/model_executor/layers/fused_moe/layer.py:979 _load_per_channel_weight_scale
(Worker_TP0 pid=96546) ERROR       expert_data.copy_(loaded_weight)
RuntimeError: The expanded size of the tensor (4) must match the existing size (16) at
non-singleton dimension 1.  Target sizes: [4096, 4].  Tensor sizes: [4096, 16]
```

Recorded #7088: vllm-ascend v0.15.0rc1 (`3d43ed997e71`) + vllm v0.15.0
(`f176443446f6`), 4×910B3, `vllm-ascend/Qwen3-235B-A22B-W4A8`, served with
`--tensor-parallel-size 4 --gpu-memory-utilization 0.95 --enable-chunked-prefill
--max-num-batched-tokens 8192 --max-num-seqs 2 --async-scheduling
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, the log carrying
`Using the vLLM Ascend modelslim Quantization now!`.

**Read the two size lists before anything else: `Target sizes` is the engine's own
parameter, `Tensor sizes` is the checkpoint tensor.** So the relation is
`param = checkpoint // N`; name `N` first. Here `16 // 4 == 4` and 4 is the parallel
size, and `Target sizes[0]` (4096) is the model's `hidden_size`, i.e. this is the w2
side scale — two arithmetic facts that place the owner before any file is opened.
The same `RuntimeError` text at **request** time is a different family (DP padding
`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md` §0, EAGLE draft width
`../spec-decode-mtp/draft-model-serial-hidden-size-crash.md`); classify on the phase
and on the failing line (`copy_` inside a `weight_loader`).

## What the pin does (verified at v0.15.0rc1 `3d43ed9` ↔ vllm v0.15.0 `f176443`)

1. **The allocation carries the divisor.** `AscendW4A8MoEMethod.get_dynamic_quant_param_modelslim`
   creates the two new-version scale biases as
   `w13_scale_bias = [E, 2*intermediate_size_per_partition, 1]` and
   `w2_scale_bias = [E, hidden_sizes, 16 // self.tp_size]`
   (`vllm_ascend/quantization/methods/w4a8.py:317-322`), reached only when
   `new_quant_version` — `quant_description["version"] == "1.0.0"` (`:194-196`).
   That `self.tp_size` is **not** `--tensor-parallel-size` verbatim:
   `self.tp_size = 1 if enable_expert_parallel else self.ep_group.world_size`
   (`:202`), and `:204-205` refuses `> 16` for the new version. On the reporter's
   TP4/DP1 launch the EP group is 4, hence the 4.
2. **The tag decides the loader, and the tag is set by a conditional.**
   `AscendFusedMoEMethod.create_weights` stamps **every** dynamic-quant param
   `quant_method = FusedMoeWeightScaleSupported.CHANNEL` and then re-tags only the
   names in `per_group_param` (`vllm_ascend/quantization/method_adapters.py:210`,
   `:211-215`, `:223-226`) — and that list is `["weight_scale_second",
   "weight_offset_second", "scale_bias", "weight_scale", "weight_offset"]` only when
   `group_size > 0`, else empty. `group_size` defaults to **256** when the
   description omits it and means per-channel only at `0` (`w4a8.py:191-193`).
3. **Upstream dispatches on that tag** (`vllm/model_executor/layers/fused_moe/layer.py:1309-1332`):
   `CHANNEL` → `_load_per_channel_weight_scale`, whose `shard_id == "w2"` branch is a
   bare `expert_data.copy_(loaded_weight)` with no narrow (`:969-981`); `GROUP`/`BLOCK`
   → `_load_model_weight_or_group_weight_scale` → `_load_w2`, which narrows the
   checkpoint tensor to the param's own width (`shard_size = expert_data.shape[shard_dim]`,
   `:1018-1035`). So the same checkpoint loads through the group path and cannot load
   through the per-channel path.
4. **Consequence:** reachable when the description carries `version: "1.0.0"` **and**
   `group_size: 0`. Any parallel size from 2 to 16 gives `16 // N < 16` and fails;
   `16 // 1 == 16` is the only value that passes. The tree's only W4A8 MoE e2e case is
   `tensor_parallel_size=2` on `vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w4a8`
   (`tests/e2e/multicard/2-cards/test_quantization.py:68-79`) — so the variable CI does
   not cover is the **description variant** (`version`/`group_size`), not the parallel
   size: that checkpoint's description cannot be the `1.0.0` + `0` pair.

## The one probe

The description in effect decides it, so read it instead of re-running the serve:

```bash
uv run --no-project python - <<'PY'
import json; d = json.load(open('/path/to/model/quant_model_description.json'))
print({k: d.get(k) for k in ('version', 'group_size', 'ascend_quant_method')})
PY
```

`version: "1.0.0"` + `group_size: 0` is the whole trigger set
(`new_quant_version` and `is_per_channel_weight` both true). `group_size: 256` (or
the key absent, same default) re-tags `scale_bias` as `GROUP` and this shape cannot
raise. `scripts/checkpoint-key-prefix.py --scales <model dir>` prints the same keys
beside the aux-tensor census.

## Ladder, cheapest first

1. The probe above — it separates "this export's description is per-channel" from
   every other load-phase shape error.
2. `--tensor-parallel-size 1`, or `--enable-expert-parallel`, makes that divisor 1
   (`w4a8.py:202`) and the `copy_` fits. Source-derived, and it is the arrangement the
   checkpoint's own layout assumes — one unsharded `[hidden, 16]` per expert, all 16
   summed in `update_bias` — so this is a candidate deploy, not only a diagnostic;
   whether it fits the box is a separate sizing question (235B W4A8 is ~120 GB of
   weights, EP splits experts across ranks and does not shard inside them).
3. Practical unblock: serve a **per-group** W4A8 export of the same model
   (`group_size = 256`); `scale_bias` then reaches `_load_w2` and is narrowed.
   Source-derived, not measured.
4. Code fix to evaluate: take `"scale_bias"` out of the `group_size > 0` conditional
   in `method_adapters.py:211-215` so both `w13_scale_bias` and `w2_scale_bias` always
   carry the `GROUP` tag. The w13 side is unchanged in effect (`_load_w13` narrows
   dim 0 by the same half-shard), but the w2 side then narrows 16 → `16 // N`.
   Acceptance is **one real request compared against a bf16 or W8A8 run**, never a
   healthy port.
5. Report it upstream of you: this is a vllm-ascend loader-contract defect, not a
   checkpoint corruption and not an upstream vLLM one (upstream only reads the tag).
   Which release fixed it is open — the snapshot holds this pin only.

**Do not drop `// self.tp_size` from the allocation.** A param of width 16 makes
`copy_` succeed, but `update_bias` folds that axis
(`w4a8.py:431-432`, `transpose(1, 2).contiguous().sum(axis=1)`), so the bias then
sums 16 terms per hidden channel where the code intends `N` — a silent numeric error
instead of a crash.

## Precedents and limits

- `#3214` (RFC "Support MOE w4a8 per-channel quantization") is the format this
  checkpoint uses: per-channel is the `group_size == 0` path, and the RFC's own text
  says only the scale **shapes** change — which is exactly where this contract broke.
  The other `expanded size … non-singleton` corpus hits (`#7365` EAGLE draft hidden
  size, `#4985` / `#3994` DP padding, DP+EP) have different owners.
- Sibling shape, same page family: a load-phase `KeyError: '<param>.weight_scale'` is
  an aux tensor with **no** parameter (`weight-scale-keyerror-quant-variant.md`); this
  page is the same family with the parameter present and mis-shaped. Neither is the
  quantization **method** gate (`quantization-method-gate.md`) — here
  `--quantization`/auto-detect worked and the modelslim line printed.
- Not verified locally: the reporter's `quant_model_description.json` was never
  pasted, so `version`/`group_size` are inferred from the error itself (a per-group
  description would have taken the `GROUP` branch and could not raise at `layer.py:979`).
  There is no later-rc evidence in this snapshot.
