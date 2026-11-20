# Which block in a pasted Ascend error report is this incident

Scope: a pasted plog / serve log where several error families arrive together —
`EE9999 ... rtMemcpy ... ErrCode=107030`, `EZ9999 ... chipId: ... fftsplus
aivector error`, `507011 aclrtSynchronizeEvent`, `ERR00100 PTA call acl api
failed`, `ERR02005 DIST internal error`, an HCCL watchdog terminate. This page
answers one question that must be settled before any mechanism: *which of those
blocks happened when the reporter says the service failed.* Inherit the rules in
`index.md` and SKILL.md.

## The report is a merge of three kinds of text

1. **Live lines**, written at failure time: the vLLM traceback, the
   `aclrtSynchronizeEvent` / `event wait to be complete failed` failures, the
   teardown (`ERR02005`, `HCCL watchdog thread terminated`).
2. **A previously recorded error block, re-printed with its own original
   timestamp** — observed in #9183 as
   `EE9999[PID: 1229] 2026-05-12-03:50:17.907.457 (EE9999): rtMemcpy execution
   failed ...` sitting inside a traceback whose own log prefix reads
   `05-13 15:29:07`, i.e. ~36 h earlier. That a re-print keeps the record's date is
   the observation; the exact runtime bookkeeping behind it is not documented in
   the snapshot, so describe it as a re-printed record, not as "the same crash
   happened again".
3. **Consequences** of (1) or (2): a stream already in an error state makes every
   later `aclrtSynchronizeEvent` / `rtMemcpy` / query fail with its own code.

So the sentence a reporter quotes in the title can be a day older than the
failure. Adopting it names the wrong owner *and* the wrong ablation set.

## Procedure

1. One call decides it —
   `bash <skill_root>/scripts/plog-timeline.sh --since <YYYY-MM-DD of the failure> <pasted-log>`.
   It prints, per error family, the span of the embedded device records and of the
   log-line prefixes, and how many lines carry a record older than their own log
   line, then names the replayed families and the first in-window error.
   Without `--since` it uses the newest log-line date in the paste, which is
   enough when the paste really does cover the failure minute.
2. If it reports *no* record inside the incident window, the paste does not
   contain the fault. Say so in one sentence, quote the older block only as
   history, and ask for the plog covering the failure minute
   (`~/ascend/log/debug/plog/plog-<pid>_<timestamp>.log`, every reporting rank) plus
   the vLLM log's first traceback. Do not build a mechanism from the older block —
   that is the recorded #9183 failure mode.
3. Search every rank for the device line: `grep -n 'EZ9999\|chipId' <log>`. The
   reporting rank is not necessarily the faulting chip
   (`collective-error-chain.md` step 4).
4. Order the in-window families by timestamp and classify the *first* one:
   - `EZ9999 ... (chipId:N, dieId:M) ... fftsplus aivector error ... error code =
     0x800000 ... The DDR address of the MTE instruction is out of range` →
     device-side kernel/pre-launch exception: `../custom-ops-runtime-faults/index.md`
     (and `device-health-checks.md` for the hardware/software split).
   - `507011` and `rtEventSynchronize ... reason=the model stream execute failed`
     → a synchronization on a stream that already failed. It is the messenger,
     never the owner.
   - `checkUceErrAndRepair: … NPU error, error code is 507899` with
     `rtGetMemInfoByType … invalid handle`, `rtGetMemUceInfo execution failed,
     reason=driver error:internal error`, `dsa task submit failed … 507053`,
     `launch failed for DSARandomNormal, errno:561000` → torch_npu's UCE
     (uncorrectable-error) *query/repair* path: the c10 exception names
     `checkUceErrAndRepair` itself, so these lines date the trap and report a
     device context the trap left broken; the random-number submit failure is a
     later submit on that context, **not** an RNG or `aclnnInplaceNormal` defect.
     Two readings stay open — a real HBM UCE, or the query failing because the
     trap killed the context — settled by `npu-smi info -t health -i <id>`, the
     plog's UCE record, and one rerun on another card.
   - the deepest Python frame sits in `profile_run` /
     `determine_available_memory` (e.g. `_dummy_sampler_run`'s
     `hidden_states[logit_indices]`, `vllm_ascend/worker/model_runner_v1.py:2158`
     at v0.14.0rc1) → a **sync point** of the last dummy forward, not the owner:
     the paste's own async caveat says the "current working operator" line names
     the last *submitted* op. Attribute with `--enforce-eager` +
     `ASCEND_LAUNCH_BLOCKING=1` before opening any kernel's shape code.
   - `ERR02005 DIST internal error`, `HCCL watchdog thread terminated`,
     `terminate called after throwing ...` → teardown after a process/device error.
   - `shm_broadcast.py:... No available shared memory broadcast block found in 60
     seconds` → an EngineCore has been waiting on a worker for ≥60 s, so it dates
     the stall relative to the device block; it says nothing about its cause
     (`../startup-failures/acl-graph-capture.md` step 2 reads the same line at
     startup).
5. Hardware vs software when several ranks fail: compare the device blocks. In
   #9183 three workers on three different chips (`chipId:3 dieId:1`,
   `chipId:5 dieId:0`, `chipId:1 dieId:1`) reported the same `error code =
   0x800000`, the same `mte error info` word and the same `pc start:
   0x12c0c3cec7b8`, with differing `blk:` — one kernel image executing an
   out-of-range address on every rank is a software/address-computation shape; a
   single card that keeps failing across runs is the hardware shape. Do not ask
   for a card replacement on the first shape.

## Where `107030` belongs

`rtMemcpy:ErrCode=107030, desc=[the current capture mode does not support this
operation]` with `synchronized memcpy failed, kind = 1` and small `cnt` (72 and
456 bytes in #9183) is a real capture-path family in the history — #6002 (v0.11.0,
DS-V3.1-w8a8, MTP + `FULL_DECODE_ONLY`), #4208, #8502 — where #6002's reporter
reads it as a framework-side blocking copy during capture and proposes a
capture-mode switch (reporter's reading; no confirmed fix in the record), and #8502
is reporter-confirmed to serve normally with `multistream_overlap_shared_expert`
off (`../startup-failures/acl-graph-capture.md`).

Two conditions before you adopt it:

- **It must be the in-window first error.** If its timestamps predate the failure
  (as in #9183), it is the previous crash.
- The same block prints `threadCaptureMode=0, exchangeCaptureMode=0,
  contextCaptureMode=0`, i.e. its own diagnostic fields show no capture in
  progress. The sentence therefore does not explain itself; ask for a re-run with
  `--enforce-eager` (graph off) rather than reasoning from the wording.

## Worked case #9183 — the answer that quoted the previous day's crash

vllm-ascend v0.18.0rc1 `99e1ea0`, vllm v0.18.0 `bcf2be9`; A3, 1P1D (2+2),
DeepSeek-V3.2-w8a8, failure during an HLE evaluation.

| Fact | Evidence |
|---|---|
| The reporter's first paste is a plog excerpt only: 400 lines, 45 timestamped lines, **all** `2026-05-12-03:50:22`, all `rtMemcpy`/`MemCopySync`/`107030`; no Python traceback, no second date | first comment (25 KB paste) |
| The recorded session ranked a capture-window H2D copy (sfa `.item()` loops, `multistream_overlap_shared_expert`, `--async-scheduling`, tight HBM) from that block, and asked for the plog | the run's answer |
| The plog the reporter then posted has **two** dates: 6 `EE9999` lines carrying `2026-05-12-03:50:17` records inside `2026-05-13 15:29:07` tracebacks, and the live block | comment `#issuecomment-4539074892` |
| In-window first error: `EZ9999 ... 2026-05-13-15:29:05.818 ... error from device(chipId:3, dieId:1) ... fftsplus aivector error ... The DDR address of the MTE instruction is out of range` | same |
| Then `507011` at 15:29:07 from `_get_valid_sampled_token_count` / `get_output`, then `ERR02005` + HCCL watchdog terminate; `shm_broadcast ... 60 seconds` at 15:28:39 on all four EngineCores | same |
| Three chips hit the same `pc start` in the same two seconds (see step 5) | same |

State of the conclusion: **unresolved** — no maintainer diagnosis or fix is in the
record. What the timestamps settle is the ownership split: the reported failure is
a device-side aivector/MTE-address exception inside the window, and the `107030`
capture-mode block is the previous day's crash and cannot anchor the diagnosis.
Next asks for the reporter, in this order: which kernel/op ran at 15:29:05 on that
chip (plog lines above the `EZ9999` block, tiling `I`-line if the op is an AscendC
custom op), the same failure with `--enforce-eager`, then with
`ASCEND_LAUNCH_BLOCKING=1` to attribute the op, and whether the three chips repeat
the same `pc start` on every attempt (software) or one chip does it alone on
re-runs (hardware).

## A two-node paste carries two environments (#6275)

Worked case #6275: vllm-ascend **v0.14.0rc1** (`52d4acf`) + vllm **v0.14.1**
(`d7de043`), 2×A2 910B2 64G, `DeepSeek-V3.2-Exp-w8a8`, DP2 × TP8 + EP + MTP
`num_speculative_tokens: 2`, `FULL_DECODE_ONLY`. The dump is the
`fftsplus aivector … MTE DDR out of range` trap during `profile_run`
(`Worker_DP0_TP0_EP0`, `determine_available_memory`), followed by the UCE-repair
block above.

Two findings that change the answer, both from the paste rather than from source:

- The launch command is **verbatim** the tree's own A2 multi-node block
  (`docs/source/tutorials/DeepSeek-V3.2.md`: TP=8, DP=2, `cudagraph_capture_sizes:
  [3,6,…,48]`, `num_speculative_tokens: 2`, `--data-parallel-rpc-port 13389`,
  `HCCL_INTRA_PCIE_ENABLE=1`). Config-diff route exhausted — "your command
  deviates from the tutorial" is a wrong answer for this report, and the
  remaining deviations to ablate are the framework ones (`--enforce-eager`,
  MTP off, `VLLM_ASCEND_ENABLE_MLAPO=0`).
- The two nodes' `collect_env` dumps disagree: master `npu-smi 24.1.rc3` on
  openEuler 5.10 versus worker `npu-smi 25.3.rc1` on Ubuntu 5.15 (both CANN
  8.5.0). On a 2-node HCCL deployment, diff the two dumps first and align
  driver/firmware/CANN/OS before any kernel theory — the containers mount the
  *host* driver (`-v /usr/local/Ascend/driver/lib64`, `npu-smi`,
  `version.info`), so the two ranks really are running different stacks.

The recorded session reached this list and shipped **no message text at all**
(rule 1: 14 tool turns, then the output cap).
