# "The machine crashed": host RAM and per-rank RSS

Symptom family: the report says the **host** stopped, not that a request failed —
"the 910C machine crashes", ssh drops, the box needs a reboot, `top`/`htop`
becomes unusable — while the engine log carries no device error, no plog
`EZ9999`/`507xxx`, no collective timeout and no Python traceback. One report of
this shape ends with the reporter pasting `top` and asking "so what's going wrong
here?"; the answer is in that paste, not in the NPU lanes.

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
  cheapest lever (fewer ranks per host, smaller `--max-model-len`).
- Label the phase: is the paste from weight load, from capture, or from a later
  steady state? Each has a different expected footprint, so ask for the time
  relative to `vllm serve` start.
- Keep the ranking honest when the reporter has not sent `dmesg`: "host memory
  pressure, mechanism open" is the claim, and `dmesg -T | grep -iE 'oom|kill'`
  plus `free -g` are what closes it.
