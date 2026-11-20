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
