# KV budget charged from device-global HBM (the 2nd engine on one card starves)

## Signature

Two forms, one owner. Nothing crashes on the device; the number handed to the
cache config is simply wrong.

1. **310P, v0.22.1rc1** (`#14274`: `Qwen/Qwen3-0.6B`, fp16, TP1, `max_model_len`
   4096, `gpu_memory_utilization: 0.5`, `npu-smi` 20,687 / 44,278 MB used ≈ 23.06
   GiB free, and `No running processes found in NPU 0`):

   ```
   Total non KV cache memory: 1.65GiB
   Available KV cache memory: 0.09 GiB (halved for workspace)
   ValueError: To serve at least one request with the model's max seq len (4096),
   (0.44 GiB KV cache is needed, which is larger than the available KV cache
   memory (0.09 GiB). Based on the available memory, the estimated maximum model
   length is 768.
   ```

   `(halved for workspace)` is the 310P override's own format string
   (`vllm_ascend/_310p/worker_310p.py:134`); the generic worker logs
   `Available KV cache memory: %.2f GiB` without it — the suffix alone says which
   implementation produced the number.
2. **Any two co-tenants on one die before the generic fix** — `#7308` (v0.14.0rc1,
   two Qwen3-0.6B serves, both `--gpu-memory-utilization 0.3`): the second prints
   `Available memory: 0`, and on v0.15.0rc1 / v0.17.0rc1 `Available KV cache
   memory: -1.33GB`. 0.3 / 0.6 "starts both", and 0.8 / 0.85 gives the second
   instance an `Available memory` far above its 0.05 share — the giveaway: the
   number tracks **what is free**, not the requested share.

Both are the same answer, owed immediately: the device had the HBM, and the
worker charged the co-tenant's usage to this instance a second time. It is not
`kv-init-and-format-cast-alloc.md` (there the KV-init *allocation* fails with
`EL0004` at the format cast) and not a genuinely small card
(`../startup-failures/max-model-len-budget.md`).

## Which of the two multi-instance failures is it

Two texts, two owners, both on one card with several engines:

| Log | Owner |
|---|---|
| `ValueError: Free memory on device (27.18/60.96 GiB) on startup is less than desired GPU memory utilization (0.9, 54.86 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.` | the startup check working as designed — the **sum** of the co-tenants' ratios is infeasible (`#8621`: 0.2 + 0.45 + 0.70 + 0.9, 910B, 4th start). Sizing answer: `../deploy-sizing/index.md`, sum ≤ 1.0 |
| the check passed, then `Available memory: 0` / `Available KV cache memory: -1.33GB` / `… 0.09 GiB (halved for workspace)` and the KV arithmetic fails | this page: a passed check means `total × gmu ≤ free` at that moment, so the later 0/negative number is bookkeeping, not capacity |

Do not conflate them: `#8621` is the first row (a correct refusal), `#14274` and
`#7308` are the second.

## Finding the twin

The twin issues are titled by the **deployment shape**, not by the symptom, so a
symptom-word search misses them. One search over this corpus returns the family:
`bash <skill_root>/scripts/history-titles.sh -i '单卡|多实例|多服务'` →

- `#4404` `[Installation]: vllm-ascend单卡部署多模型报错` — the earliest form, with
  the old log line `Available memory: 0, total memory: 65452113920` while 34 GB was
  still free after the 8B instance started, on a *different* command for a 3B model.
  That log shape (`[worker_v1.py:NNN] Available memory: N, total memory: M`, the
  file renamed `worker.py` before v0.13) is this bug on an older ref, and the same
  sentence is still there at v0.13.0rc1 (`vllm_ascend/worker/worker.py:283-284`) —
  recognize it instead of treating the newer 310P wording as the only signature.
- `#5917` `单卡多vllm实例报错`, `#7308` `单卡多服务场景下 … HBM内存管理异常` (v0.14.0rc1,
  the version-tabulated one), `#8621` (the first row of the table above).
- `#9374` `[Doc]: 缺少 Ascend 910B 显存调优与多实例部署指南` — the docs gap the family
  keeps producing.

Once one of these is open, its body usually already carries the working/failing
gpu_memory_utilization pair (`0.3`/`0.3` vs `0.3`/`0.6`), which is the cheapest
thing to quote back to the reporter.

## One-call arithmetic that names the term

The 310P non-RC branch is `available = (requested − non_kv_cache − X) // 2`.
Solve for `X` and compare it with what other processes hold: on #14274,
`2 × 0.09 + 1.65` gives `X ≈ 19.8 GiB` against 20,687 MB total `npu-smi` usage,
i.e. roughly the co-tenant's share. Label it a **calculation**: those three
pasted numbers do not fully reconcile (19.8 GiB plus ≥ 1.2 GiB of fp16
Qwen3-0.6B weights exceeds the `npu-smi` reading), so quote the direction, not
the exact figure, and close it with `VLLM_LOGGING_LEVEL=DEBUG` (the
`Memory profiling takes …` line, `vllm/utils/mem_utils.py:174-184`) plus a second
`npu-smi info` during profiling.

## The mechanism at the pin

Verified at vllm-ascend **v0.22.1rc1** (`5f6faa0`) + vllm **v0.22.1** (`0decac0d`) unless
a row names another ref:

| Fact | Where |
|---|---|
| `free_memory, total_memory = torch.npu.mem_get_info()`; `torch_memory = torch.npu.memory_reserved()`; `non_torch_memory_before_empty_cache = total − free − reserved` is a **device-global** quantity | `vllm_ascend/_310p/worker_310p.py:96-102` |
| `non_torch_memory_cleared_by_empty_cache = non_torch_memory_before_empty_cache − self.non_torch_memory` mixes the global figure with the instance-local profiled increase | `:104-106` |
| the non-RC branch subtracts that mixed quantity **in addition to** `profile_result.non_kv_cache_memory` | `:128-130` |
| the generic Ascend worker subtracts only `non_kv_cache_memory` + the NPU-graph estimate — no device-global term | `vllm_ascend/worker/worker.py:547-549` |
| upstream reference implementation, same shape | `vllm/v1/worker/gpu_worker.py:447-451` |
| `non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights`, and `non_torch_increase = (after_profile − before_create).non_torch_memory` — already instance-local, so `non_kv_cache_memory` alone is the whole charge | `vllm/utils/mem_utils.py:257-275` |
| the startup check `requested = total_memory × gmu`; `free_memory < requested` → `ValueError` — the one place co-tenant usage legitimately belongs, and it should stay | `_310p/worker_310p.py:157-169` |
| the older **generic** form of the same charge (pre-`#7427`), clamped at 0 — the `Available memory: 0` signature | `vllm_ascend/worker/worker.py:275-284` at v0.13.0rc1 (`1b5d5ab`) |

History, so it is not re-derived. The **generic** worker carried the same
device-global charge at v0.13.0rc1 (`1b5d5ab`): `vllm_ascend/worker/worker.py:275-284`
computes `non_torch_allocations = (mem_get_info()[1] − mem_get_info()[0]) −
memory_stats()["allocated_bytes.all.current"]`, adds it to `peak_memory`, then
`available = total × gmu − peak_memory` clamped by `max(…, 0)` — which is why the
older reports read `Available memory: 0` (`#4404`) instead of a negative. That
generic term was dropped by `#7427` (present clean at v0.22.1rc1 above), but `#7372`
had already copied the old calculation into the new `_310p/` override, which never
inherited the fix — the recorded instance is `#14274`, with `#7308` the same symptom
on A2/A3 (v0.14.0rc1 `0`, v0.15.0rc1/0.17.0rc1 `-1.33GB`). The reporter of #14274
opened `#14275` with this patch — merge status is **unverified** here (PRs are
unavailable in this snapshot), so say "proposed", not "fixed".

## `gpu_memory_utilization` is a per-instance limit

Upstream says so in the option's own docstring: *"This is a per-instance limit,
and only applies to the current vLLM instance. It does not matter if you have
another vLLM instance running on the same GPU. For example, if you have two vLLM
instances running on the same GPU, you can set the GPU memory utilization to 0.5
for each instance."* (`vllm/config/cache.py:66-74` at v0.22.1).

So the recurring maintainer answer on this family — "the ratio includes memory
other processes/instances occupy, so set the second instance to 0.8" — contradicts
the documented contract and must be corrected, not repeated. It appears to work
only because the 310P startup check compares `total × gmu` against a
**device-free** number, so `gmu ≤ free/total` at that moment; on #14274's numbers
that ceiling is ≈ 0.53, i.e. a knife-edge window (0.5-0.53) in which the
double-charged budget happens to reach the 0.44 GiB a 4096-token request needs.
Cross-instance feasibility is the separate sum rule — the sum of
`gpu_memory_utilization` over processes sharing a device ≤ 1.0
(`../deploy-sizing/index.md`) — never a reason to inflate one instance.

## Patch, and the traps around it

```python
# non-RC branch only
self.available_kv_cache_memory_bytes = (
    self.requested_memory - profile_result.non_kv_cache_memory
) // 2
```

1. **Delete the dead locals** `free_memory` / `total_memory` / `torch_memory` /
   `non_torch_memory_before_empty_cache` / `non_torch_memory_cleared_by_empty_cache`
   (`:96-102,106`): nothing reads them afterwards and a dangling assignment trips
   ruff `F841` in CI. **Keep** `self.non_torch_memory` (`:104`) and
   `self.peak_activation_memory` (`:105`) — the inherited
   `compile_or_warm_up_model()` budget-suggestion log reads both
   (`worker.py:744-775`).
2. **Leave the RC branch and the startup check alone**: `:126`
   (`psutil.virtual_memory().used`) and `:157-169` are the intended shared-memory
   path for RC devices.
3. **`--kv-cache-memory` is inert on 310P.** The override has no
   `kv_cache_memory_bytes` fast path (no hit in `vllm_ascend/_310p/`) while the
   generic worker (`worker.py:477-490`) and upstream (`gpu_worker.py:370-388`)
   both have one — yet the inherited log at `worker.py:765-772` still recommends
   `--kv-cache-memory=…`. Never offer that flag as the workaround here; report it
   as a separate gap, since it deletes the manual escape hatch on the SoC where
   co-tenancy makes it most useful.
   The gap is not 310P-only at older refs: the **generic** Ascend worker also has no
   fast path at v0.13.0rc1 (#6299) and v0.18.0rc1 (#9372) — upstream reads the key
   only inside `vllm/v1/worker/gpu_worker.py` (`:362-380` at v0.18.0), which the
   Ascend `determine_available_memory` override replaces, and
   `rg -n 'kv_cache_memory' vllm_ascend/` returns only the profiling variable. Grep
   the ref before recommending the flag either way (`index.md` rule 5).
4. **No NPU/ACL-graph estimate term either** (the generic worker subtracts
   `npugraph_memory_estimate_applied`). Re-derive when 310P gains one — see
   `index.md`, `## Reading a v0.21+ startup memory block`.
5. **No unit test covers this**: `tests/ut/_310p/` has no worker test. Mirror
   `tests/ut/worker/a2/test_worker_v1.py:535-660` and assert
   `available == (requested − non_kv) // 2` with a stubbed `memory_profiling`
   whose `non_torch_increase` is far below the device-global usage — that fails on
   the unfixed code.

## When the failing method is an SoC override

`vllm_ascend/_310p/worker_310p.py` (and any later `_<soc>/` subpackage) inherits
from the generic `vllm_ascend/worker/worker.py` and re-implements a handful of
members — at v0.22.1rc1 `NPUWorker310` overrides exactly five
(`init_device:49`, `save_sharded_state:58`, `determine_available_memory:80`,
`_warm_up_atb:140`, `_init_device:144`). Everything else is inherited, so the
default expectation is "the generic fix applies"; a stale formula inside the
override is a **missing propagation**, and the method is to diff the override
against `worker/worker.py` and upstream, then look up the twin symptom that was
fixed in the generic path. Do that before theorising from the reporter's
traceback: it yields the cause, the one-line patch and the precedents at once.

## Recheck conditions

Verified only at `5f6faa0` (v0.22.1rc1). Re-derive if the 310P override is folded
back into `NPUWorker`, if `memory_profiling`'s `non_torch_increase` semantics
change, if a `kv_cache_memory_bytes` fast path or graph estimate reaches the
override, or if `#14275` lands (then the reporter's tree may already carry the
fix; quote their `sed -n '90,140p' vllm_ascend/_310p/worker_310p.py`).
