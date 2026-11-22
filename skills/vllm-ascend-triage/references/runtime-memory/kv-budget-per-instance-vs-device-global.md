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

   The oldest recorded ref is `#5917` (v0.12.0rc1 `42ceaf08a1ef`, 910B2C, one pod, two
   `vllm server --gpu-memory-utilization 0.4` on the same die): instance 1 logs
   `[worker_v1.py:270] Available memory: 23106324992, total memory: 65452113920` and
   reaches capture, instance 2 on the same card logs `Available memory: 0` and dies in
   `check_enough_kv_cache_memory` with `No available memory for the cache blocks` while
   `npu-smi` shows `VLLMEngineCor` at 25,515 MB and ~36 GiB free. Note the **file name**:
   `vllm_ascend/worker/worker_v1.py:261-271` before the rename to `worker.py` — the
   paste's own `[worker_v1.py:NNN]` prefix dates the ref.

Both are the same answer, owed immediately: the device had the HBM, and the
worker charged the co-tenant's usage to this instance a second time. It is not
`kv-init-and-format-cast-alloc.md` (there the KV-init *allocation* fails with
`EL0004` at the format cast), not a genuinely small card
(`../startup-failures/max-model-len-budget.md`), and not
`device-allocations-outside-profiling-window.md` (there the budget arithmetic is
correct and something allocates *after* it, so no co-tenant is involved).

## Which of the two multi-instance failures is it

Two texts, two owners, both on one card with several engines:

| Log | Owner |
|---|---|
| `ValueError: Free memory on device (27.18/60.96 GiB) on startup is less than desired GPU memory utilization (0.9, 54.86 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.` | the startup check working as designed — the **sum** of the co-tenants' ratios is infeasible (`#8621`: 0.2 + 0.45 + 0.70 + 0.9, 910B, 4th start). Sizing answer: `../deploy-sizing/index.md`, sum ≤ 1.0. With a **single** engine the same line means the die is held from outside your container — see the subsection below the table |
| the check passed, then `Available memory: 0` / `Available KV cache memory: -1.33GB` / `… 0.09 GiB (halved for workspace)` and the KV arithmetic fails | this page: a passed check means `total × gmu ≤ free` at that moment, so the later 0/negative number is bookkeeping, not capacity |

Do not conflate them: `#8621` is the first row (a correct refusal), `#14274` and
`#7308` are the second.

### What to do with the first row's refusal (the "I commented out the check" reply)

The check is upstream's own `request_memory()`
(`vllm/v1/worker/utils.py:216-235` at vllm v0.17.0 `b31e932`) inlined into the
Ascend worker (`worker.py:266-278` at v0.17.0rc1 `e20f0b1`), and the guard is
called **unconditionally** — `vllm/v1/worker/gpu_worker.py:285` at that ref, with no
`kv_cache_memory_bytes` gate — so at that ref `--kv-cache-memory-bytes` is not a way
past it; re-read that call site at the reporter's ref before offering the flag as a
bypass, because the v0.21+ startup text itself *suggests* a
`--kv-cache-memory=<bytes>` value (`index.md`, `## Reading a v0.21+ startup memory
block`). Deleting the guard is not a fix either: the
Ascend KV budget is derived from `requested_memory` (`worker.py:362-363`:
`available_kv_cache_memory_bytes = requested_memory − non_kv_cache_memory −
non_torch_memory_cleared_by_empty_cache`), so the instance then tries to allocate the
pool its ratio asked for out of memory that is not free, and the failure moves to the
KV allocation or the first request with a worse message. Say both, then give the
ratio sum (`../deploy-sizing/index.md`). Same ref, a second co-tenancy failure to
expect during the attempt: `assert self.init_snapshot.free_memory > free_gpu_memory`
(`worker.py:353-361`, "other processes sharing the same container release GPU memory
while vLLM is profiling") fires when a neighbour frees memory inside this instance's
profiling window.

**Do not assert the earlier instances were holding their caps.** `gpu_memory_utilization`
is a ceiling charged against `total`, and the guard only compares it with free HBM at
that instant, so a paste's ratio list and its one free-memory figure need not
reconcile (recorded #8621: `0.2 + 0.45 + 0.70` implies ≈82 GiB against a 60.96 GiB
card, yet the fourth start reported 27.18 GiB free, i.e. ≈33.8 GiB held). Recompute
per instance from its own `Available KV cache memory: X.XX GiB` line and the
`npu-smi` of each card at that moment — or treat "not all of them are on this die" as
open — instead of writing a contradiction into the answer.

### The first row without a co-tenant of yours: a shared die under a device plugin

Recorded #9428 (vllm-ascend `v0.17.0rc1` `e20f0b1`, vllm `v0.17.0`, 910B4 64 GB, a K8s
pod given two **non-first** cards by the Ascend device plugin, `--tensor-parallel-size 2
--gpu-memory-utilization 0.90`, Qwen3.5-27B):

```text
ValueError: Free memory on device (51.26/60.96 GiB) on startup is less than desired GPU
memory utilization (0.9, 54.86 GiB). Decrease GPU memory utilization or reduce GPU
memory used by other processes.
```

`54.86 = 60.96 × 0.9` is the same check, but there is **one** engine: the 9.7 GiB
missing from the named card belongs to something outside this container. Read three
things before proposing a config change:

- **The `npu-smi` process table is namespace-local.** In the same paste the container's
  `npu-smi info` listed `NPU 3 = 3202/65536` and `NPU 4 = 12824/65536` with
  `No running processes found` — a co-tenant pod on the same die has no PID in your
  namespace, so an empty table plus a non-zero used figure is *expected* and is not a
  "leak". Attribute on the **host**: `npu-smi info` there, then
  `cat /proc/<pid>/cgroup` + `tr '\0' ' ' < /proc/<pid>/cmdline` for the owner
  (the ladder is `../deploy-sizing/node-processes-and-device-attribution.md`).
- **The variable the plugin injects is not the one torch_npu reads.** #9428's env dump
  carried `ASCEND_VISIBLE_DEVICES=4,3`; torch_npu reads only
  `ASCEND_RT_VISIBLE_DEVICES` (`torch_npu/npu/__init__.py:341-366`), so that name never
  reaches the process — its reader is the Ascend Docker Runtime **prestart hook** at
  container creation, which mounts the named `/dev/davinciN` and appends their cgroup
  allow entries (`../deploy-sizing/card-budget-and-device-visibility.md`, two variables
  at two moments). The isolation that actually worked is therefore the **mount**, not a
  Python-side variable, so "the pod sees only NPUs 3 and 4" proves the mapping — and the
  torch ids are then `0,1` while `npu-smi` keeps the host labels. Do not open the answer
  with "the variable took effect", and do not read the plugin's descending list as a
  zero-device torch list: the hook's own parser splits and sorts it.
- **A per-container memory quota is invisible to `aclrtGetMemInfo`.** If the plugin
  shares the die by quota, the whole-card `free ≥ total × gmu` pre-check cannot be
  satisfied by any quota below `gmu`; lowering `gmu` (0.90 → 0.75 here) is the
  one-run confirmation, and an exclusive allocation or a free pair of cards is the
  durable one. Verify the reading first with the per-device loop in the pod
  (`torch.npu.set_device(i); torch.npu.mem_get_info()`), which also separates "the card
  is occupied" from "we are mapped to a different die".

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
- `#5707` `[Usage]: 请问如何正确的在一张npu加速卡上面跑多个vllm 程序以启动多个模型服务`
  — a **usage-shaped** title (no `[Bug]`, no symptom word), so a bug-titled search
  misses it; it is the family's v0.11.0 report (910B2, two `0.4` serves on one
  `davinci7`), the one that carries the reporter-confirmed `expandable_segments`
  non-fix below, and the one whose follow-up asks the support question outright.
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

What to offer instead, in the order that does not break the contract: **size the
co-tenant you control.** The failing instance's own log line is an equation —
`available₂ = total × gmu₂ − (own peak + co-tenant footprint)` — so the co-tenant has to
stay under `total × gmu₂ − own_peak`. On #5917's numbers (`total` 60.96 GiB, `gmu₂` 0.4 →
24.38 GiB, own peak 2.86 GiB from instance 1's own subtraction) that bound is ≈ 21.5 GiB,
i.e. instance 1 at `gmu₁ ≈ 0.32-0.35`, with `0.35 + 0.4 = 0.75` still inside the sum rule.
Give the inequality, the one-line change, and the check that proves it worked — instance 2
printing a **non-zero** `Available memory:` — rather than a ratio to set on the starving
instance. There is no startup guard to bound the alternative at the older refs: the Ascend
worker overrides `init_device` (`worker_v1.py:225-231` at v0.12.0rc1) and never runs
upstream's `Free memory on device … is less than desired GPU memory utilization` check
(`vllm/v1/worker/gpu_worker.py:240-243` at vllm v0.12.0), which is also why such a paste
carries no such line. Report the fix as **absent** at these refs and say so plainly: #5917's
reporter answered a "solved?" follow-up on 2026-01-22 with 「没有解决」, so the number is
still wrong a week later, and only the workaround is actionable.

### The identity that answers the paste in one line, and what it rules out

Two consequences of `available = total × gmu − peak` are worth deriving once,
because they turn this family into arithmetic instead of a source walk:

- the **first** engine on a card settles at **`total × gmu₁`**: its KV pool is sized
  to fill whatever the budget leaves (`peak + available`), so instance 1's share is
  not "its weights plus what it happens to cache", it is its whole ratio
  (#5917: 21.5 GiB of KV against a 24.38 GiB budget, the remaining 2.86 GiB its own
  peak);
- therefore instance 2 receives `available₂ ≈ total × (gmu₂ − gmu₁) − peak₂`, i.e.
  **whichever engine starts second needs the larger ratio** (#7308, reporter-run:
  `0.3`/`0.3` fails, `0.3`/`0.6` starts both) — and that instance's own footprint is
  `total × gmu₂ − total × gmu₁`, so the card as a whole ends at `total × gmu₂`:
  ordering the ratios this way cannot over-commit the die and the sum rule stays
  satisfied. Say that out loud when a maintainer's instinct is "you are
  oversubscribing". This points the same way as the maintainer's "set the second
  instance to 0.8", so when you use it give the honest reason — the extra ratio is
  spent *paying the co-tenant's charge*, not granted to instance 2, whose own KV is
  the **difference** — and prefer shrinking the co-tenant you control (the paragraph
  above) whenever you can, because that keeps every instance's ratio meaning what
  its docstring says.

### Recorded non-fixes, and how uncontested the bug is

- **`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is not a remedy here.**
  `#5707` (vllm-ascend image v0.11.0 + vllm 0.11.0, 910B2 64 GB, `--device
  /dev/davinci7`, two `vllm serve` at `--gpu-memory-utilization=0.4`: Qwen3-0.6B,
  then Qwen2.5-0.5B-Instruct): the reporter set the variable **inside the container
  before launching** — the maintainer's own caveat — and got the same
  `Available memory: 0, total memory: 65452113920` →
  `ValueError: No available memory for the cache blocks` (2026-01-08), plus the same
  failing KV arithmetic as every other ref in this family. Expected, not a mis-set
  variable: the charged term is `mem_get_info()[1] − mem_get_info()[0]`
  (`vllm_ascend/worker/worker_v1.py:247-251` at v0.11.0 `2f1aed98`, logged at
  `:257`), device-global used, which no `PYTORCH_NPU_ALLOC_CONF` value
  (`expandable_segments`, `max_split_size_mb`) can change, since none of them move
  the free/used reading a co-tenant holds. The flag belongs to the
  segment/unmap lane (`allocator-expandable-segments-unmap-crash.md`), a different
  symptom on the same card. So do not prescribe it; if the paste already shows the
  attempt, say the outcome is expected instead of offering another allocator value.
- `--kv-cache-memory` — inert at these refs, see trap 3 under *Patch*.

**Support status when the reporter asks whether co-tenancy is sanctioned.** Nothing
in the pinned docs says whether several engines may share one die; `#5707`'s
reporter asked exactly that after reading them (「不知道在一个显卡上运行多个vllm进程
是否是官方支持的用法 … 官方文档都没有提到」), and on 2026-01-29 relayed this
family's public state: a bug in the current version, several issues discussing it,
**no concrete fix plan**, pointing at `gpustack/gpustack#4325` (a downstream
project's issue for the same multi-engine-per-card failure). So ship the arithmetic
workaround, label any release fix absent/unverified, and cite `#9374` when the ask
is a document reference rather than a config.

**Solving the equation without instance 1's log.** `#5707` pasted only the failing
instance, and its own numbers still bound the co-tenant: `Loading model weights took
0.9275 GB` (≈0.86 GiB) plus its own activations against `total × 0.4` = 24.38 GiB
of the 60.96 GiB card leaves `footprint₁ ≳ 21 GiB`, i.e. instance 1 already held
~0.35 of the die — so `0.4` on the second engine could never fit. Quote it as a
**bound** (its peak is not printed) — enough to prescribe the ordering above, and
cheaper than asking for a log the reporter may no longer have.

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
   fast path at v0.12.0rc1 (`rg -n 'kv_cache_memory' vllm_ascend/` returns only the
   profiling locals at `worker_v1.py:266,269` — #5917), v0.13.0rc1 (#6299) and
   v0.18.0rc1 (#9372) — upstream reads the key
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
