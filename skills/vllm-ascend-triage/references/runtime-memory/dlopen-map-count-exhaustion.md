# `libkernel.so: failed to map segment from shared object`: the per-process mapping table ran out

## Scope

A worker that served normally for **days** dies in a Triton-Ascend kernel compile:

```
File ".../triton/backends/ascend/compiler.py", line 555, in linalg_to_bin_enable_npu_compile_A2_A3
    lib = ctypes.CDLL(callback_path)
File ".../ctypes/__init__.py", line 376, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /tmp/tmpXXXXXXXX/libkernel.so: failed to map segment from shared object
```

…followed by `MLIRCompilationError: [ConvertLinalgRToBinary] encounters error:` whose
detail line is that **same** `libkernel.so: failed to map segment …` string, and by
`RuntimeError: Worker failed with error '…'` on the EngineCore. Usually on a hybrid
MLA/DSA MoE model (DeepSeek-V4 class) with MTP, `--enable-chunked-prefill`, dynamic
prefill shapes and a W8A8 checkpoint, on one node of 8-16 cards.

This is a **host mapping-resource** failure. Not device memory, not free RAM, not disk,
not a compiler bug:

- `failed to map segment from shared object` **with no errno suffix** is glibc's
  `dlopen` reporting **ENOMEM from `mmap`**. A permission/`noexec` problem prints
  `: Permission denied`, an absent file `: cannot open shared object file`, a wrong
  ABI `: wrong ELF class`. Read the suffix before ranking anything.
- The compiler banner is the red herring: `[ConvertLinalgRToBinary] encounters error` is
  only how the Triton compile wrapper re-raises the `OSError`. In the worker's
  own traceback the `OSError` block comes **first**, then `During handling of the above
  exception, another exception occurred`. EngineCore never sees the frames — upstream
  `multiproc_executor.get_response` raises `RuntimeError: Worker failed with error
  '<str(e)>'`, so a paste that starts at the EngineCore line carries the compiler
  banner and the loader text with no `ctypes` frame. **Always read the per-rank worker
  traceback before the EngineCore re-raise.**

## Mechanism (reporter-confirmed, #10551)

Triton-Ascend's Ascend backend AOT-compiles each JIT'd kernel into a temporary
directory and, to read four metadata callbacks out of the produced library —
`_infer_task_type_function` → `bs_task_type`, `_infer_workspace_shape_function` →
`workspace_size`, `_infer_sync_block_lock_num_function` → `lock_num`,
`_infer_sync_block_lock_init_function` → `lock_init_val` — it does

```python
lib = ctypes.CDLL(callback_path)     # ascend/backend/compiler.py:493 (…_910_95) and :690 (…_A2_A3)
__get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
...
```

and never unloads it (`rg -c dlclose third_party/ascend/backend/compiler.py` → **0**;
`ctypes.CDLL` has no implicit `dlclose`). Each compilation therefore leaks the library's
mappings for the life of the process. Reporter's measurement on the affected node:
**7.00 VMAs per compilation** (210 over 30 compilations), ~1200 compilations/day because
dynamic prefill shapes keep missing the JIT cache, 8639 compilations over the container's
life → `8639 × 7.00 = 60 473` plus ~4 782 baseline VMAs ≈ 65 255 against the default
`vm.max_map_count` of **65 530**. The engine died when that ceiling was reached, i.e.
after days of uptime, and from then on *every* `dlopen` failed — the worker's
`execute_model` is simply where the next compile happened to be.

Upstream fix: triton-lang/triton-ascend PR #2328 releases the handle after the metadata
has been copied out (verified on 910B3: 7.00 → 0.02 VMAs per compilation, kernel results
and callback metadata unchanged). Until a release carries it, the leak is present.

## Step 1 — bound it by uptime, and rule the RAM explanation out with the paste

Two greps on the reporter's log decide whether this is the lane at all:

```bash
rg -n "failed to map segment" "$LOG" | head        # the loader line, and how many ranks
rg -n "Loading model weights took|Available KV cache memory|Starting vLLM|Provisioning model" "$LOG" | head -3
```

- The engine's start timestamp and the failure timestamp are **days** apart in the
  recorded case (start 06-08 18:02, first `OSError` 06-16 07:02). Minutes ⇒ some other
  lane; a fresh start that fails on the first request is not this page.
- Kill the RAM candidate with the reporter's own numbers rather than asking for more:
  in the recorded case the host had **979 GB** free, `/tmp` had **171 GB** free and was
  not `noexec`, and `dmesg` showed no OOM kill. Expect the same shape — if the paste does
  not carry them, ask for exactly these four (`free -g`, `df -h /tmp`,
  `findmnt -no OPTIONS /tmp`, `dmesg -T | grep -i 'killed process'`) in one line, not for
  another log.

## Step 2 — measure the mapping table, not the memory

```bash
sysctl vm.max_map_count                     # 65530 by default; the container inherits the host's value
wc -l /proc/<worker-pid>/maps               # a live EngineCore/worker pid
awk '{print $6}' /proc/<worker-pid>/maps | sort | uniq -c | sort -rn | head   # what is leaking
```

Run them **inside the container** (`docker exec … bash -c '…'`) or in the right PID
namespace: the process is the container's, and the sysctl is a host setting the container
inherits. A count within a few percent of the ceiling is the confirmation; the tail of the
second list being `libkernel.so`-style Triton temp paths is the identification, and the
`(deleted)` suffix on them is what a temporary directory leaves behind. Both read a *live*
process, so take them on a running deployment or right after a restart-and-replay, not
after the worker has exited. The arithmetic that predicts it from the log is
`compilations × 7 + baseline`, with compilations counted from the reporter's own
per-day rate — a number worth showing, never worth re-deriving from source.

## Step 3 — the answer

1. **Immediate, no restart, takes effect at once** (host-level, verified in the recorded
   case):
   ```bash
   sysctl -w vm.max_map_count=1048576
   echo 'vm.max_map_count=1048576' > /etc/sysctl.d/99-vllm.conf
   ```
   It buys headroom, it does not stop the leak — say both.
2. **Cut the compile rate** so the leak grows slower: persist `TRITON_CACHE_DIR` on a host
   volume (`triton-ascend python/triton/runtime/cache.py:65` reads `TRITON_CACHE_DIR`, else a default
   cache dir inside the container), so a recreated container does not recompile several
   thousand kernels on its first day; optionally reduce dynamic-shape churn (a fixed
   `--max-num-batched-tokens`, fewer distinct prefill lengths).
3. **Real fix**: the triton-ascend release that contains PR #2328 (or a local patch that
   `dlclose`s, or deletes, the callback library after the four callbacks). Two
   `rg` checks on the reporter's tree date it:
   `rg -c dlclose .../site-packages/triton/backends/ascend/compiler.py` (0 = leak
   present) and the line count of `linalg_to_bin_enable_npu_compile_A2_A3`.
4. **The container health check cannot see it**: in the recorded case the container stayed
   `Up` — only EngineCore and the workers became zombies while the API-server shell
   survived, so a container-level liveness probe reports healthy while serving is dead.
   Recommend a per-worker check (a request that must succeed, or `pgrep -f EngineCore`
   from inside the container) rather than `docker ps`.

## Adjacent lanes — do not confuse

| Paste also/actually carries | Lane |
|---|---|
| `[ConvertLinalgRToBinary] … operation's operand is unlinked` / `bishengir-compile … returned non-zero exit status 1`, no loader line | installed **wheel × compiler pair** mismatch — `../image-dependency-faults/triton-ascend-pair.md` |
| `AclrtUnmapMem … 507899` / `unmapHandles` during `empty_cache()` | allocator **unmap** failure — `allocator-expandable-segments-unmap-crash.md` |
| `Memory_Allocation_Failure(EL0004)` / `207001`, `NPU out of memory` | device HBM budget — `hbm-growth-and-oom.md`, `index.md` part 1-2 |
| container `container_memory_working_set_bytes` rising, service healthy | cgroup host memory — `container-working-set-growth.md` |

## Version scope

Recorded case (#10551): vllm-ascend v0.20.2rc1 image, vllm v0.18.0, triton-ascend
3.2.0.dev20260322, torch_npu 2.9.0.post1, openEuler 24.03 aarch64, 8× NPU, MTP + chunked
prefill. The leak site is *verified in the pinned tree* at
`triton-lang/triton-ascend third_party/ascend/backend/compiler.py:493` and `:690`
(`ctypes.CDLL(callback_path)` inside `linalg_to_bin_enable_npu_compile_910_95` / `…_A2_A3`, the
`callback_path` built at `:309`/`:517`, zero `dlclose` in the whole file) and the installed path is
`triton/backends/ascend/compiler.py` with the same `linalg_to_bin_enable_npu_compile_A2_A3`
function; the numeric leak rate (7.00 VMAs/compile), the ceiling arithmetic and the
`sysctl` remedy are **reporter measurements on that node**, not something a local tree can
reproduce. Recheck before quoting numbers on another release: the `dlclose` grep and
`vm.max_map_count` default.
