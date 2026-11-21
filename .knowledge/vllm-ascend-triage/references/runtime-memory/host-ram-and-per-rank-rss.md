# "The machine crashed": host RAM and per-rank RSS

Symptom family: the report says the **host** stopped, not that a request failed —
"the 910C machine crashes", ssh drops, the box needs a reboot, `top`/`htop`
becomes unusable — while the engine log carries no device error, no plog
`EZ9999`/`507xxx`, no collective timeout and no Python traceback. One report of
this shape ends with the reporter pasting `top` and asking "so what's going wrong
here?"; the answer is in that paste, not in the NPU lanes.

Second member of the family, and the one to check first: the box is still up and
the title says **memory leak** ("memory used keeps increasing", "内存持续增长") with a
`top` whose `VLLM::Worker_TP` rows climb while `npu-smi` sits flat at the
`--gpu-memory-utilization` share. That paste has no crash to read, and the growth
may not be in vllm-ascend at all — §"Step 0" below answers it from the same
environment block, before any of the arithmetic in §"Steps".

## Worked observation (#3902, reporter-confirmed, not a mechanism)

`vllm serve /mnt/sfs_turbo/model/Qwen3-VL-235B-A22B-Instruct -tp 16` on one 910C
(A3) host, image `vllm-ascend:v0.11.0rc0-a3` + CANN 8.2.RC1. The reporter's later
`top` on that host:

```text
Tasks: 327 total,  17 running, 310 sleeping
MiB Mem : 2062270.+total, 459119.2 free, 1600621.+used,   2529.9 buff/cache
MiB Swap:      0.0 total,      0.0 free,      0.0 used. 455359.8 avail Mem
   6482 root 20 0 8326.4g  92.5g 703164 R 107.3   4.6   0:54.78 python3      (16 such rows)
```

Read it with `bash <skill_root>/scripts/host-ram-check.sh <paste|->`: **16 python3
ranks at 88-100 GiB RES each = 1472 GiB of 2014 GiB host RAM (73 %), swap 0**, load
average 69/131/422 rising. That is a single-host, 16-way deployment of a 235B
checkpoint: every rank's resident footprint is paid on the same 2 TB of host
memory, and nothing in the NPU tooling (`npu-smi`) reports it. Established by that
paste: the host is the constrained resource. Not established: whether the crash was
an OOM kill, swap thrash (there is no swap here), or an unrelated kernel fault —
`dmesg`/`journalctl -k -b -1` for the boot that died decides it.

Two reading rules: **quote RES, never VIRT** (`8326.4g` is torch_npu's
per-process virtual reservation and appears on healthy hosts), and do not quote
`%Cpu(s)` from a first sample (it is a since-boot average; this paste shows 94 %
idle beside a load average of 422 and 17 running tasks).

## When there is no `top` to sample: read the log's last line

A host that died during engine start leaves nothing to sample — ssh is gone by the
time the reporter returns. Take the bracket from the log instead: if the last
lines are the per-rank `rank N in world size M is assigned as DP rank ...` set and
no `Starting to load model` follows, the workers died inside
`NPUModelRunner.__init__` (before weight load), and one eager host allocation
there is large enough to kill the box on its own — the `max_model_len²`
attention mask, `attention-mask-host-allocation.md` (price it with
`scripts/attn-mask-budget.sh`). A long `--max-model-len` (128K-256K) is the
give-away; the reporter's narrative ("machine crashed", "313T", "cannot
restart") adds nothing.

Prefer that reading over this page's `top` route when the report has **no**
`top`/`free` paste and the log ends mid-startup: a real resident footprint cannot
be measured on the dead box, but an arithmetic peak can be computed from the
flags alone. Both routes end at the same levers (`free -g`, `dmesg -T | grep -iE
'oom|kill'`, fewer ranks per host, smaller `--max-model-len`).

## Step 0: the `torch_npu` (pta) line in the same paste, before the arithmetic

**One grep first: the pinned tree's own release notes name this leak and its owner.**
`rg -n -i 'leak' <pin>/docs/source/user_guide/release_notes.md` returns, at
v0.13.0rc1 (`1b5d5abf`), `:197` under the `## v0.11.0rc1` known-issue list: *"The
memory may be leaked and the service may be stuck after long time serving. This is
a bug from torch-npu, we'll upgrade and fix it soon."* That is the project's own
statement at the reported ref — source-verified, not a hypothesis — and it decides
the lane before any code read. Do this grep on every "memory leak" report and on
every long-run growth report; the same section carries scheduler/KV and accuracy
known issues with their owners.

**Rule: when the env dump carries `torch_npu==2.8.0`, the leak is the stack's
before it is vllm-ascend's.** A known pta defect leaks in 2.8.0 and is fixed in
**`2.8.0.post1` / `2.8.0.post2`**; the fix that ends the report is the install
plus a vllm-ascend rebuild, and no amount of vllm-ascend source reading can refute
the lane, because the fix is not in the snapshot: the case view carries one partial
commit of `Ascend/pytorch` (the pta tree) and a `pytorch/pytorch` pinned to a
*different* torch line than the reporter's (v2.9.1 vs `2.8.0` on #6299), and a
`.post` release is not a commit you can read.

### Recorded: #6299, "memory leak when running GLM-4.7-W8A8" (reporter-confirmed)

Report: 8×910B3, CANN 8.3.RC2, vllm 0.13.0 / vllm-ascend v0.13.0rc1, `torch==2.8.0+cpu`,
**`torch_npu==2.8.0`**, TP8 + `--enable-expert-parallel`, `--gpu-memory-utilization 0.9`,
`FULL_DECODE_ONLY` capture sizes, `--async-scheduling`. Evidence in the body: `npu-smi`
steady at `59718/65536 MB` on all eight chips (91 %, i.e. exactly the configured
share — a *device* leak is not in this paste), one process per chip at 56 373 MB, and
a `top` where every `VLLM::Worker_TP` row sits at **172.8-175.6 GB RES** (`%MEM` 8.6)
while `VLLM::EngineCore` is 822 MB. Presenting symptom: "Memory used keeps
increasing".

What settled it, in the reporter's own later comments:

1. `--no-enable-prefix-cache` → "problem still": prefix cache / APC is refuted for
   this shape, so do not rank it.
2. The maintainer's guidance on the same issue — "upgrade torch-npu to 2.8.0.post1 or
   2.8.0.post2. There is a known issue that 2.8.0 will lead mem leak."
3. "I have upgraded pta to 2.8.0.post2 and rebuilded vllm-ascend. Bug fixed."

So the ranked answer this report needed, at turn 1, was: *the versions are the
problem — `torch_npu==2.8.0` leaks, install `2.8.0.post2` and rebuild vllm-ascend;
the `npu-smi` numbers are the configured share, not growth, and the rebuild is part
of the fix because the C++ extension is compiled against the installed pta
(`setup.py:324-333` takes `-DTORCH_NPU_PATH` from `pip show torch-npu`).* The RSS
split of §"Steps" still decides leak vs large-flat-footprint and is worth sending in
the same message, but it does not outrank a known-bad pta line.

Two consequences worth carrying:

- **"Every version matches the release" is not a clearance here.** At v0.13.0rc1 the
  release's own `requirements.txt:31` pins `torch-npu==2.8.0`, so a reporter who
  installed this ref's requirements has the leaking pta *by construction*
  (`../startup-failures/environment-correspondence.md`). Read the pta line against
  the known-good patch releases, not against the release table.
- **The mechanism stays open even when the fix is confirmed.** Neither sample
  separates per-rank host RSS from device HBM after the upgrade, so say "pta-version
  defect, surface observed in worker host RSS, mechanism not isolated" rather than
  naming a leaking structure. A later report on a pta ≥ `.post1` is a different
  investigation.

The prior-month corpus cannot supply this: in the #6299 view none of the 794 earlier
issue bodies is a memory-leak report (21 mention "leak" in passing, none is one;
zero carry `泄漏`), and `scripts/history-titles.sh -i 'memory leak|泄漏' <root>`
returns **0 hits** — a corpus gap, not evidence against the lane. The version lane
comes from the release's own pin and the pinned release notes, not from precedent.

The lane closes — and the page returns to §"Steps" — when the paste's `torch_npu`
is already `2.8.0.post1`/`.post2`, or 2.9/2.10, with the growth still present; then
the leak is above pta and the RSS split decides it.

## Steps

1. Get the numbers, not the adjective: `free -g`, `nproc`, and
   `ps -eo pid,rss,nlwp,cmd --sort=-rss | head -25` (RSS in KiB), plus
   `dmesg -T | grep -iE 'oom|kill'`, `journalctl -k -b -1 | tail -50`,
   `cat /proc/meminfo | grep -E 'MemTotal|MemAvailable|Committed_AS|Huge'`.
   Feed the paste to `scripts/host-ram-check.sh`; it prints the worker sum, the
   mean per rank, the ratio and the verdict band (≥ 60 % = first-class cause).
2. Size it arithmetically before blaming anything: weights ≈ params × bytes /
   TP per rank, plus the per-rank host-side buffers, plus the API frontend, plus
   the mmap'ed checkpoint pages. All 16 ranks live in **one** RAM budget on a
   single host; on two hosts the same TP=16 halves it. `scripts/card-budget.sh`
   answers the *per-card HBM* question only — it says nothing about host RAM, so
   do not cite it for this.
3. One-axis ablations that change host RAM without changing the model: TP across
   two hosts (8+8) instead of 16 on one; a smaller `--max-model-len`; and a repeat
   `top`/`ps` sample during load. If the ratio drops below the band and the box
   stays up, the ranking is confirmed.
4. Keep the NPU lanes separate: if the same run also has a capture-phase or
   collective block, classify that block on its own text
   (`../startup-failures/acl-graph-capture.md`,
   `../hccl-and-hardware/collective-error-chain.md`) and say explicitly which
   artefact each cause rests on. A host-memory reading does not refute a device
   error, and a device error does not explain a `top` like the one above.

## Answer shape

- Direct answer: on this evidence the machine is running out of **host** memory
  (or host CPU), not of card memory — the number and the ratio first, then the
  cheapest lever (fewer ranks per host, smaller `--max-model-len`). When the paste
  carries `torch_npu==2.8.0`, the direct answer is instead the pta patch release
  (`.post2` + rebuild) with the ratio as the confirmation, because that is the only
  action that ends the report without a second sampling session.
- Label the phase: is the paste from weight load, from capture, or from a later
  steady state? Each has a different expected footprint, so ask for the time
  relative to `vllm serve` start.
- Keep the ranking honest when the reporter has not sent `dmesg`: "host memory
  pressure, mechanism open" is the claim, and `dmesg -T | grep -iE 'oom|kill'`
  plus `free -g` are what closes it.
