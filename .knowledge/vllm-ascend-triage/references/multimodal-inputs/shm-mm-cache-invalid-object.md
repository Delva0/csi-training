# The `shm` multimodal cache object is stale: `Data for address:id '<addr>:<id>' has been modified or is invalid.`

Scope: a VL serve launched with **`--mm_processor_cache_type shm`** dies on a request
whose only error is

```
ValueError: Data for address:id '4826649034:82' has been modified or is invalid.
```

raised by every TP rank in
`vllm/distributed/device_communicators/shm_object_storage.py::get`, wrapped by the
worker RPC as `RuntimeError: Worker failed with error 'Data for address:id …'`, then
`EngineCore encountered a fatal error` and `EngineDeadError: EngineCore encountered an
issue. See stack trace (above) for the root cause` at the API server. No `vllm_ascend`
frame (only `patch/platform/patch_core.py`'s `run_engine_core` loop). Owner: upstream
vLLM's SHM **multimodal feature cache**, not the plugin and not a device fault.

**Not a trigger: the flag on its own.** This class exists only when the error string
above appears on a **request**. `--mm_processor_cache_type shm` in a serve line whose
paste ends `WorkerProc failed to start.` (or any worker-start failure with no readable
traceback) routes to `../startup-failures/silent-worker-exit.md` and
`../startup-failures/weight-scale-keyerror-quant-variant.md` first: recorded #7744 ranked
this page's "drop the flag" ladder first while the failing phase was a load-phase
`KeyError: 'layers.37.mlp.experts.w2_weight_offset'`, and the reporter's resolution was
the newer tag, not the flag.

Verified at vllm v0.11.0 `b8b302cde434` ↔ vllm-ascend `01a13a9b` (recorded #5117) unless
a line is marked otherwise.

## The verdict the message already carries

| Site | Fact |
|---|---|
| `shm_object_storage.py:583-590` | `get(address, monotonic_id)` reads the slot's `[4-byte id][4-byte size][data]` prefix and raises when `buf_metadata[0] != monotonic_id` — i.e. **another object was written into that slot after the writer indexed this one** |
| `shm_object_storage.py:628-635` | `default_is_free_check`: a slot is recyclable only once `reader_count >= writer_count * n_readers`, so the buffer's design assumes every reader consumes every object exactly once |
| `shm_object_storage.py:609` / `vllm/multimodal/cache.py:399,410` | `n_readers = parallel_config.world_size` on the sender (`ShmObjectStoreSenderCache`, the "cache used on P0", API process) and carried in the handle to the workers |
| `vllm/multimodal/cache.py:600,676-697` | the readers are the **workers**: `ShmObjectStoreReceiverCache` built by `worker_receiver_cache_from_config` |
| `vllm/config/multimodal.py:42,51,54` | `mm_processor_cache_gb = 4` GiB ring buffer (the segment size), `mm_processor_cache_type = "lru"` by default, `mm_shm_cache_max_object_size_mb = 128` per object |
| `vllm/envs.py:198,1437-1439` | the segment **name is a constant** (`VLLM_OBJECT_STORAGE_SHM_BUFFER`), override `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` |
| `vllm/multimodal/cache.py:495-514` | the shm path needs `_enable_ipc_cache` (`_api_process_count == 1 and data_parallel_size == 1`, or external LB) **and** `mm_processor_cache_type == "shm"` |

So the line says one thing only: at read time the slot held a newer object. It does not
name who read late, and because the same `(address, id)` pair is what all ranks were
handed, per-rank plog / `npu-smi` / HCCL evidence cannot move it. `--shm-size` too small
is a *different* error (`MemoryError: Not enough space in the data buffer`,
raised at `shm_object_storage.py:206` when the segment cannot fit the object).

## Why this is not a vllm-ascend lane

- `rg -n 'mm_processor_cache|mm_receiver_cache|shm_object_storage' vllm_ascend/` returns
  nothing at that pin (source, tests, docs): the plugin neither sets nor reads the flag.
- The plugin's only multimodal patch is `_merge_multimodal_embeddings`
  (`vllm_ascend/patch/worker/patch_multimodal_merge.py:26-58`, installed at `:59`),
  which runs **after** the cache in the same worker RPC, and is not in the stack.
- The stack is upstream end to end: `v1/executor/utils.py:22` (`get_and_update_mm_cache`,
  called from `multiproc_executor.py:665`'s `worker_busy_loop`) →
  `multimodal/cache.py:555` (`get_and_update_features`) → `:644` (`get_and_update_item`)
  → `shm_object_storage.py:588`.

## Action ladder (cheapest first)

1. **Drop the flag.** `--mm_processor_cache_type` defaults to `lru`, and the object store
   only exists when it is `shm`; the LRU/preprocessor cache keeps working, so removing
   `--mm_processor_cache_type shm` removes this failure class and costs only the IPC
   shortcut. One restart, same request.
2. **`--mm-processor-cache-gb 0`** — disables the cache outright; use it to confirm the
   attribution, not as the running config.
3. **If shm must stay:** one engine per host per buffer name — the name is a constant, so
   a second instance on the same host reaches the same segment — give each its own
   `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`; keep `--mm-processor-cache-gb` above the
   request's feature working set; keep the container's `shm_size` above it too.
4. **Report upstream** (vllm), not vllm-ascend, with the full serve command, the
   `vllm`/`vllm-ascend` versions and the dumped `SchedulerOutput`.

## What to send back / what not to do

The dumped `SchedulerOutput` is the whole evidence: `mm_features=[MultiModalFeatureSpec(
data={'address': …, 'monotonic_id': …}, …)]` lists exactly the pairs the workers were told
to read, and the `identifier` hashes say whether two requests shared them. Paste it and
the serve command — nothing else is needed from the reporter.

Do **not** re-derive the ring-buffer arithmetic. Recorded #5117 spent its entire
16 384-token output cap on `allocate_buf`/`access_buf`/`free_buf`/`default_is_free_check`
internals — address wrap, per-slot id bytes, `ID_MAX` — while the one flag change was
already visible in the compose block it started from, and shipped **152 characters** of
message text. The config verdict is available on turn 1 from `--mm_processor_cache_type
shm` plus the upstream frame; the allocator give no evidence about *this* run's reader
schedule.

## Recheck conditions

`get()`'s message, `default_is_free_check`, the `n_readers` plumbing, the config defaults
and the env-var name all move between vLLM releases. Re-grep the sentence
(`rg -n 'has been modified or is invalid' <vllm>/vllm/`) at the reported pin, and if the
pin's `shm` path is gone say the path does not exist there and ask for the version.
