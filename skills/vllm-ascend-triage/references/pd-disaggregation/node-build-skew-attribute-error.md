# `'ReqMeta' object has no attribute 'block_ids'` in a KV-transfer thread: compare the two nodes' builds

Verified at vllm-ascend **v0.20.2rc1** (`367b8e62da79`) + vllm **v0.20.2**
(`bc150f502991`), recorded for **#10398** and **reporter-confirmed**: 「已解决，P和D节点
没注意统一代码版本」 — the two nodes were not on the same code version. The issue
carried a title and one log line only, with the `collect_env` block still reading
`Your output of above commands here`.

```text
[kv_transfer.py:84] Error in KVCacheTransferThread: 'ReqMeta' object has no attribute 'block_ids'
```

## Procedure — steps 1-2 *are* the answer

1. **Place the sentence before reading code.** Three sites print it:
   `kv_p2p/mooncake_connector.py:464` and `kv_p2p/mooncake_hybrid_connector.py:445`
   (the P2P lane) and the pool thread's catch-all
   `kv_pool/ascend_store/kv_transfer.py:91` in `KVTransferThread.run()` (`:24`).
   `rg --files -g '*kv_transfer*.py' vllm_ascend/` returns exactly one production
   file with that basename, so `[kv_transfer.py:84]` is the **KV-pool /
   AscendStore** lane. The number is already evidence: at v0.20.2rc1 that
   `logger.error` is `:91`, so `84` is an **earlier build of the same file** — not
   another module, and not a different code path.
2. **Ask the tree whether the tag can raise it — one grep, then stop.** At the pin
   the object *has* the attribute: `config_data.py:557 class ReqMeta`, field
   `block_ids_by_group: list[list[int]]` `:563`, and a **compat property**
   `block_ids` `:620-626` (getter returns group 0, setter normalizes through
   `normalize_block_ids_by_group`), duplicated on the sibling
   `LayerMultiBlockReqMeta` (`:697`, `:702`, `:731-737`). The `__init__` takes
   `block_ids_by_group` `:588` **and** a legacy `block_ids` kwarg `:600`. Every
   thread-side read uses `block_ids_by_group` (`kv_transfer.py:257`, `:391`,
   `:507`, `:558`) and `rg -n '\.block_ids\b' vllm_ascend/` finds no `ReqMeta`
   reader. So the line says *the reporter's `vllm_ascend` is not this commit*, and
   the deliverable is a build comparison. Both recorded forwards that walked the
   reader instead exhausted their output budget without shipping an answer.
3. **Compare builds per node, in one reply** (P, D and any pool/scheduler node run
   separate interpreters — `../startup-failures/environment-correspondence.md`
   §"one environment per worker process"):
   ```bash
   pip show vllm vllm-ascend | grep -E 'Name|Version'
   python -c "import vllm_ascend; print(vllm_ascend.__file__)"
   sed -n '555,565p;615,628p' "$(python -c 'import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))')/distributed/kv_transfer/kv_pool/ascend_store/config_data.py"
   ```
   Different second output on the two nodes ends the investigation: unify the
   `vllm_ascend` build (and its paired vllm) and re-run. Identical on both means
   this page's cause is out — then look for the reader in *their* tree
   (`rg -n '\.block_ids\b' <site-packages>/vllm_ascend/`), which names the file and
   line that owns it.
4. **Read the direction before naming it.** With the shim present at this ref the
   rename landed *inside* this rc window — the getter/setter plus unit tests that
   assign `new_req.block_ids = …`
   (`tests/ut/distributed/ascend_store/test_config_data.py:203`, `:216`, and
   `test_pool_scheduler.py:283`) exist to keep old callers alive. So two builds a
   few commits apart disagree whichever way they sit: an older reader names
   `block_ids` while the newer class only has `block_ids_by_group` (this message),
   and a newer reader names `block_ids_by_group` against an older peer
   (`'ReqMeta' object has no attribute 'block_ids_by_group'`, the mirror). Ask
   which node was installed/updated last rather than labelling either build stale.
5. **Report the second observable too, because it is a stall, not log spam.**
   `run()`'s handler (`kv_transfer.py:90-91`) calls only `logger.error` — no
   `task_done()`, no `set_finished_request()` — and the save path blocks in
   `pool_worker.wait_for_save` at `pool_worker.py:599`
   (`self.kv_send_thread.request_queue.join()`), which exists to make stores
   visible before the request is reported finished. So a save-side failure on the
   first request reads as *no progress plus one line per request*, and a recv-side
   one leaves the request unmarked. None of that is transport evidence: no plog,
   HCCL line, rank table or `device_ip` question applies until the builds match.

## Do not "fix" it with the shim

`block_ids` returns `block_ids_by_group[0] if self.block_ids_by_group else []`
(`config_data.py:620-621`), so a bare `.block_ids` read is **group 0 only** — on a
multi-group/hybrid KV cache it moves one group and silently drops the rest. Re-adding
the property to an older tree hides a contract mismatch between nodes that exchange
metadata built by the other's class; matching the builds is the fix, and the shim is
only why the *current* tree does not raise.

## Not this page

- `Error in KVCacheTransferThread. error=unhashable type: 'list'.` on a D node that
  keeps serving: a *masked* failure (grouped `BlockIds` fed to `set[int].update`) →
  `kv-transfer-thread-masked-error.md`. Here the message names its own exception, so
  there is nothing to unmask; note only that this catch-all prints it for *any*
  `_handle_request` failure.
- `IndexError: list index out of range` at `_get_kernel_block_ids`: a group-**count**
  mismatch between the sides → `kv-group-count-mismatch.md` (whose ranked cause 4 is
  the same peer-build question, reached from a different symptom).
- The reader is inside a *single* node's tree and both nodes match:
  `../upstream-downstream-drift/index.md` (a downstream field/API drift against its
  paired upstream) and `../startup-failures/connector-config-attributes.md`.
