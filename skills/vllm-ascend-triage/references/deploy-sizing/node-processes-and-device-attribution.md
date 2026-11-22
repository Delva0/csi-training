# An extra process on "another worker's" NPU: attribute the PID before the layout

## Scope

Nothing crashed and no counter is wrong, but the operator sees a process where it
does not belong: "an extra worker process from another card appears on a certain
NPU" (recorded #12338, releases/v0.23.0), `npu-smi` showing a PID on a card outside
that engine's slice, a second process on a card after the Nth benchmark round, or
the inverse — one engine's PID on **every** die of the node.

This is an **attribution** question about the reporter's node, so the deliverable is
the process identity plus the one command that decides who owns it, not a vLLM
source walk. It stays in the sizing/visibility topic because the answer is a card
map: which process may touch which die.

## Rule 1 — the PID is the first check, and it is one call

`npu-smi info` prints a *process table* keyed by PID; a PID appears against a die
as soon as that process created a context on it (`aclrtSetDevice` /
`aclrtCreateContext`), so the card a PID is listed under says **nothing** about
which engine launched it. Ask for:

```bash
npu-smi info                      # the process table: pid, name, per-die HBM
ps -o pid,ppid,lstart,cmd -p <PID>
tr '\0' '\n' < /proc/<PID>/environ | grep -E 'ASCEND_RT_VISIBLE_DEVICES|^VLLM|PROMETHEUS'
```

A PID whose `cmdline` is not a `vllm serve`/`VLLM::EngineCore` worker of that
engine ends the vllm-ascend part of the case. Two outcomes account for most such
reports: an **external monitoring/profiling agent** (next section) and a process
whose own `ASCEND_RT_VISIBLE_DEVICES` is unset or node-wide (rule 3-4). Do not
start from `cpu_binding` or the multiproc executor's env setter: the recorded run
walked both (and Mooncake's ADXL context code) and the resolution was neither —
the extra PID was not a vllm worker at all.

## Rule 2 — the recorded resolution: the monitoring platform, not the framework

#12338 was closed by its original reporter as **introduced by installing the
msservice-metrics monitoring platform**, fixed in
msserviceprofiler PR #419, with "the current vllm-ascend framework does not have
this problem" (reporter-confirmed, 2026-07-24). So on a box that runs an Ascend
monitoring stack, that stack is the *first* suspect for a stray PID on a die, and
the engine is exonerated by rule 1 rather than by argument.

What the snapshot carries about the platform (label: source-verified at
`Ascend/msserviceprofiler` `tag_MindStudio_26.0.0.B120_0012`, `d23012b`): the
metrics component is a **Python library injected into the engine process**
(`ms_service_metric/DESIGN.md` §1-2: dynamic hooks over YAML symbol config,
bytecode injection to read locals, a control channel over shared memory + `SIGUSR1`,
CLI `ms-service-metric on|off|status|restart`) and it expects
`PROMETHEUS_MULTIPROC_DIR` to be exported before the serve. The same repo family
also ships an ACL-profiling path. Whether the stray PID is the platform's own
collector, a hook inside a worker, or an `aclprof`/`mspti` attach is **not settled
here** — PR #419 is not in the snapshot — so phrase the mechanism as open and the
attribution as reporter-confirmed.

Checks that cost one line each: `pip show ms_service_metric` and
`ms-service-metric status` on the node; `PROMETHEUS_MULTIPROC_DIR` in the env dump;
the date the symptom started against the install date; and the control run — the
same launch with the platform uninstalled or `ms-service-metric off`, after which
the extra PID must be gone. A `collect_env.py` dump proves nothing here: the
injection needs no environment variable the engine reads.

## Rule 3 — one engine legitimately occupying several dies

Two documented reasons, both of which look exactly like "a foreign process on my
card":

- **The KV-transfer engine opens a context on every device it can see.**
  Mooncake's Ascend-direct (ADXL) transport enumerates
  `aclrtGetDeviceCount()` and calls `aclrtSetDevice(i)` for each visible device in
  `ContextManager::initialize()`
  (`mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/context_manager.cpp:44-80`
  at Mooncake `v0.3.11.post1` `e9c6107`), and `AscendDirectTransport::registerSegment`
  registers one engine per device in that set
  (`ascend_direct_transport.cpp:180-197`). The process that owns the transfer
  engine therefore shows on **every** die of its own `ASCEND_RT_VISIBLE_DEVICES`,
  and it does so at the first KV transfer (request time), not at startup — which is
  why such reports say "after the second round". A *visible set* that is unset or
  node-wide turns that into "another worker's process on my card".
- **`enable_cpu_binding` reads the node-wide process table.**
  `AscendCPUBinding.get_running_npus()` parses `npu-smi info` and then intersects
  with `ASCEND_RT_VISIBLE_DEVICES`
  (`vllm_ascend/cpu_binding.py:148-180` at vllm-ascend v0.22.1rc1 `5f6faa0`), so a
  foreign PID on one of *your own* dies silently changes your binding, and a PID on
  a die outside your set is invisible to you. Pin the visible set before reading any
  binding complaint.

## Rule 4 — then the layout: overlapping slices

Each `vllm serve` of an external-DP deployment is launched with its own
`ASCEND_RT_VISIBLE_DEVICES` (`--data-parallel-rank` / `--data-parallel-address`
spread one rank per process), and the launcher hands every invocation cards from 0.
Intersect the launches' visible sets before any port, HCCL or connector read —
`../pd-disaggregation/same-host-pd-device-partition.md` owns the collision form
(`Bind_Failed(EJ0003)`, `[HcclSocket][Listen] … port[16666] fail`,
`hcclCommInitRootInfoConfig … error code is 7`), and
`card-budget-and-device-visibility.md` owns the plan form and the
`ASCEND_RT_VISIBLE_DEVICES` traps (ascending order, the other spelling is the
container-creation hook's input, ids index `/dev/davinci*`).

## Answer shape

1. One line: is the PID an engine worker of *this* instance (then a card-map
   question for that engine's visible set) or a third party on the node (then not
   a vllm-ascend defect)?
2. The one command that settles it (rule 1), and the identity of what you find —
   with the reporter-confirmed precedent as one line: `#12338: msservice-metrics
   install, fixed in msserviceprofiler #419, framework unaffected`.
3. The legitimate multi-die cases (rule 3) *before* proposing a config change, so
   the reporter can recognise "my transfer-engine process on all 16 dies" or "my
   own process on 4 dies == my TP" without another round trip.
4. Scope: `Mooncake v0.3.11.post1`, vllm-ascend `v0.22.1rc1`; a later release may
   change `cpu_binding`'s device source or the ADXL context policy — re-grep
   `get_running_npus` and `ContextManager::initialize` at the reporter's pin.

## After the engine is gone the card still shows HBM ("显存未释放，起不了新服务")

Recorded #9373, second report (vllm-ascend **v0.18.0** `e18643f8a4d5`, vllm v0.18.0
`bcf2be961200`, 910B, `--gpu-memory-utilization 0.95 --max-model-len 32768`): the serve
died, `npu-smi info` still shows 40 GB+, and a new serve cannot take the card. The
reporter's asks are "register SIGTERM/SIGINT handlers so an abnormal exit releases ACL
resources" and a troubleshooting doc. Two corrections, then the ladder — this is the
same attribution question as above, with the engine *absent* instead of extra.

- **A row in the process table is a live context** (see Limits): the residual number is
  itself evidence that something still holds `/dev/davinci*` — an orphaned
  `VLLM::EngineCore`/`VLLM::Worker` child, a second engine or container, or a rank
  parked inside a device call. Run rule 1's three commands first; "leak" is only
  reachable once the table is empty and the number is still wrong.
- **There is no exit-time device release to add at this pin.** At v0.18.0
  `rg -n "aclrtDestroyContext|aclrtResetDevice|npu_shutdown" vllm_ascend/worker/
  vllm_ascend/platform.py` is empty; the package's only `atexit` unsubscribes print
  streams (`vllm_ascend/utils.py:134-135`, `_unregister_print_streams_on_exit`); the
  only `SIGTERM`/`SIGINT` handler is installed when `enable_npugraph_ex and
  enable_static_kernel` and its job is `uninstall_static_kernel()` before
  `raise SystemExit()` (`vllm_ascend/worker/worker.py:141-156`, `:158+`). A handler
  cannot run on `SIGKILL` and normal exit is already the runtime's own teardown, so
  "call ACL free on exit" does not change the reported case. The upstream half that
  exists is a **parent-death pipe**: `vllm/v1/executor/multiproc_executor.py:643-678`
  hands each child a `death_pipe` reader and keeps the writer in the parent ("when
  parent exits, death_reader in child will get EOFError"), and `:229-233` / `:444-447`
  close the writers on `shutdown()`. It fires only when the child is at a read point,
  so a child parked inside an ACL call outlives its parent — name that mechanism
  instead of a signal handler.

Ladder, cheapest first: (1) `ps -eo pid,ppid,stat,rss,cmd | grep -E 'vllm|EngineCore|VLLM'`
plus the **full** `npu-smi info` taken while the card is still occupied — the process
table names the owner; (2) if the owner is a child of the dead serve, kill the **process
group**, not the parent PID (`pkill -f 'vllm serve'`, systemd `KillMode=control-group`,
container `--init`); (3) `STAT=D`/`Dl` on that PID means it is inside a driver call and
will not die until the call returns — the next artefact is plog / `dmesg -T`, not
another kill; (4) `npu-smi set -t device-reset -i <id>` is the last resort, kills every
tenant of that card, and is documented nowhere in the pinned trees (`rg -n
"device-reset"` is empty at v0.18.0 — a plain `-n`, not the `-rn` `--replace` trap),
so never lead with it.

The doc finding is real: the pinned tree's only memory guidance is `npu-smi info` as a
liveness check (`docs/source/faqs.md:93`) and the installation prerequisites — there is
no orphan-process / HBM-cleanup page, which is the second thing the reporter asked for.

## Limits

- A PID in `npu-smi` is a *context* on a die, not a scheduling fact: it does not
  show which process is computing, and it disappears only when the context is torn
  down (process exit), so "it is still there after the engine exited" is a stale
  process, not a leak in the release.
- The monitoring platform's presence on the box is not readable from any
  `collect_env.py` output; only `pip show` / its own status command answers it.
- "The driver reclaims HBM when the last context closes" is the platform expectation the
  section above rests on, and it is **not verifiable in this snapshot** (no driver or CANN
  runtime export): state it as the expectation and let the process table be the proof.
- Per-process HBM attribution (`npu-smi info -t proc-mem …`) is a recalled sub-command:
  `proc-mem` occurs in no pinned tree, so have the reporter probe `npu-smi info -h` for the
  exact name rather than promising the invocation.
