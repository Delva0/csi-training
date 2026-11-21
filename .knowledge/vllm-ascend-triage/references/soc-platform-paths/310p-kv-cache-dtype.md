# 310P: a non-fp16 KV cache is refused at the first cache store

## Signature

The reporter's own serve line carries `--kv-cache-dtype fp8` (or `bfloat16`) on a
`-310p` image. The engine then comes up *completely* — weight load, profiling,
`Available KV cache memory: <N> GiB`, `GPU KV cache size: <T> tokens`, `init engine
(profile, …) took N seconds` — and the **first** request kills every rank in the same
second. Recorded: `#8284`, vllm-ascend **v0.18.0rc1** (`99e1ea0fe685`) + vllm v0.18.0,
image `v0.18.0rc1-310p-openeuler`, Qwen3-30B-A3B-Instruct-2507, TP4 + EP, `--dtype
float16 --enforce-eager --max-model-len 8192`:

```text
(APIServer …)   INFO  [config/cache.py:212] Using fp8 data type to store kv cache …
(Worker_TP0_EP0)INFO  [worker_310p.py:96] Available KV cache memory: 8.82 GiB
(EngineCore …)  INFO  [v1/core/kv_cache_utils.py:1316] GPU KV cache size: 768,384 tokens
(Worker_TPn …)  ERROR [v1/executor/multiproc_executor.py:932]   File ".../vllm_ascend/device/device_op.py", line 32, in reshape_and_cache
(Worker_TPn …)  ERROR [v1/executor/multiproc_executor.py:932]     torch_npu._npu_reshape_and_cache(
(Worker_TPn …)  ERROR [v1/executor/multiproc_executor.py:932]   File ".../torch/_ops.py", line 1255, in __call__
RuntimeError: ReshapeAndCacheOperation setup failed!
```

The deepest `vllm_ascend` frame is the **KV store**, not attention math: `attention_v1.py`
`reshape_and_cache` → `device/device_op.py:32` → `torch_npu._npu_reshape_and_cache`.

## The verdict, and it is a config verdict

At this pin the 310P document states the envelope outright —
`docs/source/tutorials/hardwares/310p.md:5`: *"the 310I series only supports eager mode
and the float16 data type"* — and **no** validated row in that file passes
`--kv-cache-dtype` (`rg -n 'kv-cache-dtype' docs/source/tutorials/hardwares/310p.md` is
empty at `99e1ea0`). So a non-fp16 KV cache on 310P is outside the supported set at the
reported release, and the cheapest next action is one flag: drop `--kv-cache-dtype`
(leave `--dtype float16`). Lead the answer with that sentence; the mechanism below is
supporting detail, not the deliverable.

## Why the refusal lands at the store (source-verified at v0.18.0rc1, `99e1ea0`)

| Fact | Where |
|---|---|
| Nothing validates the flag at config time | `rg -c kv_cache_dtype vllm_ascend/platform.py` → **0**; the paste's own `Using fp8 data type to store kv cache` (`config/cache.py:212`, upstream) proves the value reached `CacheConfig` |
| The attention impl accepts the string and never reads it | `attention_v1.py:358` (parameter), `:371` (`self.kv_cache_dtype = …`) — the only two occurrences in that file; `rg -n kv_cache_dtype vllm_ascend/_310p/` is **empty**, so no fp8 scale/dequant path exists on this SoC |
| The store is unconditional | `attention_v1.py:882-900` → `DeviceOperator.reshape_and_cache` → `device/device_op.py:29-36` `torch_npu._npu_reshape_and_cache(key=…, value=…, key_cache=…, value_cache=…, slot_indices=…)`, no dtype branch |
| 310P takes the **base** adaptor | `device/device_op.py:456-461`: `get_device_adaptor()` returns `A5DeviceAdaptor` only for `AscendDeviceType.A5`, `BaseDeviceAdaptor` otherwise |
| The cache it is handed is FRACTAL_NZ | `_310p/model_runner_310p.py:54` `self._acl_format = ACL_FORMAT_FRACTAL_NZ`; `utils.py:138-156` `_should_trans_nz` — `# 310P always converts to NZ` returns `True` before any dtype check |
| Nothing objects while the page size is computed in fp8 | `patch/platform/patch_mamba_config_310.py:35-49` derives `kv_cache_dtype` from `cache_config.cache_dtype` and sizes the block from it |

So the request reaching the failing op is *fp8 dtype × FRACTAL_NZ layout on 310P*, and
the op refuses at setup. `ReshapeAndCacheOperation::Setup` itself lives in
op-plugin/torch_npu, which this snapshot does not contain: `rg -l -i reshape_and_cache
/workspace/repos/<issue>/Ascend/op-plugin` is empty at the same pin, and `rg -l "setup
failed"` over `Ascend/`, `pytorch/`, `vllm-project/` returns only an unrelated vllm unit
test. **One grep, then stop** — which of dtype, layout or shape the setup rejected is
not citable here, and the report does not need it: the flag is the owner.

## What settles it, and the precedent

1. `--kv-cache-dtype auto` (flag absent) on the same command, same request. Serving ⇒ the
   flag was the cause, and the answer is "not supported on 310P at this release".
2. `#7685` is the same pair one release earlier on this SoC (vllm 0.17.0 image,
   DeepSeek-V3.2, `--kv-cache-dtype fp8 --dtype float16`): it surfaces one layer up as
   `AssertionError: Invalid kv_cache_dtype: float16. Valid values are: ('auto',
   'bfloat16', 'fp8', 'fp8_e4m3', 'fp8_e5m2', 'fp8_inc', 'fp8_ds_mla')`, with the
   reporter's own comment "但我指定了 --kv-cache-dtype 为 fp8，似乎没有传进去生效".
   Two different surfaces for one unsupported combination — so do not read either as a
   bug that a release has already fixed.
3. Only after the flag is removed does the *next* question appear: whether Qwen3-MoE
   itself runs on 300I Duo at this pin (`310p.md` validated table at `:71`, i.e. Qwen3-0.6B
   / Qwen2.5-7B at TP2 / Pangu-Pro-MoE-72B at 8 cards, and nothing 30B-A3B). Answer it as
   a coverage statement, not by re-reading the traceback.

## Delivery note (the recorded cost)

The session behind this page had the serve line and the `reshape_and_cache` traceback in
hand by its 5th assistant turn, then spent ~30 further turns grepping `setup failed` and
`kv_cache_dtype` across the tree, reading the 310P platform page and the 310P
`_310p/` listing, and ended **output-truncated with no message text at all** (1.85 M
cumulative prompt tokens). When a reporter's own serve line carries a dtype/format flag
that the pinned doc excludes for their SoC, the first message is that verdict plus the
one-flag ablation — the source chain above is what you add *after* it, if at all.

## Limits

- The rejected argument inside `Setup` is not verifiable here; the message string comes
  from the reporter's torch_npu build and may move between builds.
- One-commit snapshot: whether a later release adds an fp8/non-fp16 KV path on 310P is
  unanswerable — re-grep `_310p/` for `kv_cache_dtype` per version instead of assuming.
- `--kv-cache-dtype fp8` is not the only member of this family: a bf16 *checkpoint* on
  310P is the weight-side twin (`atlas-300i-310p.md`, the `TransData`/`EZ3002` section),
  and the quantised-checkpoint axis is `310p-quant-checkpoint-soc-match.md`.
