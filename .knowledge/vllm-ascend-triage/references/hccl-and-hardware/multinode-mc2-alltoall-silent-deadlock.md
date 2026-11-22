# Multi-node MC2 AlltoAll deadlock: a silent whole-service hang with AICore 0 %

Signature family — a P/D or plain multi-node MoE deployment serves healthily for
minutes, then stops answering **completely and silently**: the API server still
accepts TCP and listens, `curl` returns no bytes, `npu-smi` shows `AICore 0 %` on
every chip while HBM still holds the KV pool, and the worker logs simply stop. No
traceback, no device dump, no OOM. Recorded #8975: vllm-ascend v0.18.0rc1
(`99e1ea0`) + vllm v0.18.0, two decode nodes plus a prefill node, expert parallel,
MC2 dispatch/combine, "多次运行 benchmark 压测均在运行一段时间后触发".

Not this page: the startup/`profile_run` MC2 stall with `507014` / `107020`
(`mc2-dispatch-profile-run-hang.md`, `aiv-mc2-env-ablation.md`), the
graph-vs-单算子 mode split with no log at all (`graph-vs-single-op-comm-fault.md`),
or one in-flight request stalling under DP + ACL graph while concurrency is served
(`dp-aclgraph-single-request-hang.md`).

**One competing lane has a resolution and this one does not.** The same silence — serve
healthy for tens of minutes, log just ends, process dies hours later — was reported on
this same pin pair for a Qwen3.5-MoE P/D deployment whose launch carried an MTP drafter
and `--async-scheduling`, and the reporter closed it with **three sampling/MTP
cherry-picks** rather than anything in the collective path (#8916 — Step 3's "the
recorded case has no resolution" is about #8975, not about this shape in general). One
question splits the two before any plog is bought:
does the launch carry `--speculative-config` and `--async-scheduling`? If yes, read
`../spec-decode-mtp/mtp-sampler-silent-stall.md` and rank its ablation ladder beside
Steps 1-3 here — the two lanes' first checks are different artefacts (serve command and
one flag vs the plog), and only the plog evidence in Step 1 above can promote this page
back to first. **One answer flips the order: a build already carrying all three of those
patches** (`pip show vllm-ascend` plus its cherry-pick list) has used up the sampler
lane, so this page is first again on any release that ships them
(`../spec-decode-mtp/mtp-sampler-silent-stall.md`, consequence 3).

**A second competitor is decided by one line of the D-side log.** On a P/D deployment
whose decode node is the one that stops, this page competes with the KV-transfer
handshake lane: if D's log holds `Receive timeout` / `Failing N request(s) due to KV
load failure` / a `GPU KV cache usage` figure, D is up and failing per request and
`../pd-disaggregation/mooncake-handshake-timeout-wedge.md` owns it; total silence with
no line written at all is this page. A report with no log cannot choose between them,
so the first message asks for that log instead of ranking (recorded #10963 ranked this
lane first for a GLM-5.1 4×A3 1P1D 140K 压测 D-side 挂死 on the pin where #12383 — the
same model, hardware and topology — is the other candidate).

## Step 0 — `AICore 0 %` does not mean the host is stuck

That inference was the recorded run's first answer: 「AICore 利用率 0 是这次最有力的
一条证据：device 上一个 kernel 都没有跑。这排除了 … HCCL 数据面死锁这整条线」, and it
ranked the host-side DP barrier `dist.all_reduce(packed_tensor,
group=get_dp_group().cpu_group)` (`worker/model_runner_v1.py:512-552`, the barrier at
`:538` at v0.18.0rc1) as cause #1. The reporter's next paste was a **device-side**
deadlock with AICore still at 0 %.

A rank that entered an all-to-all and waits for a peer that never arrives executes no
AI-Core work, so 0 % is the state of *both* lanes. `AIVector` is not a safe test
either: it read 8-16 % in #13688's MC2 replay deadlock (`aiv-mc2-env-ablation.md`), so
a non-zero reading does not clear the device lane and a zero one does not enter it.
The discriminator is the **plog**, never the utilisation column — and that column is
not load evidence either (`../performance-throughput/index.md`). Keep the host-side
ranks ranked *below* the collective read, each with its own check: a `py-spy dump`/`gdb -p … 'py-bt'` of
the worker and EngineCore pids, and a gloo `Connection closed by peer` stack
(`dp-group-peer-close.md`, `worker-segfault-no-python-frames.md`) — those are the only
evidence that moves it back to the host.

## Step 1 — one artefact, four greps

Ask for the plog (`~/ascend/log/debug/plog/plog-<pid>_*.log`) from **every** node for
the hang window, then one call:

```bash
bash <skill_root>/scripts/hccl-stuck-timeline.sh plog-D0-*.log plog-D1-*.log plog-P-*.log
```

It prints the four lines that decide the case, and the cross-file verdict:

| Line in the plog | What it decides |
|---|---|
| `[Load] comm config of [group_name_N]: bufferSize[2097152000], deterministic[0], execTimeOut[1836]` | the group's loaded comm space and watchdog; `bufferSize` is **bytes** (2097152000 B = 2000 MiB), not the MB you exported — Step 3 |
| `[Communicator Key Info] identifier[…] rankSize[32] serverNum[2] …` | how wide the group really is and across how many servers — the EP / all-to-all width to compare with the launch flags |
| `[BuildRemoteResByTag] newtag[group_name_NNN_mc2RunAlltoAllDirectFullmesh_device]` | the **op family**: the substring after the group names the collective, `mc2…AlltoAllDirectFullmesh` is the MC2 MoE dispatch/combine all-to-all, and the `mc2` tag is vllm-ascend's own group (`distributed/parallel_state.py:96`, `group_name="mc2"`) |
| `[heartbeat.cc:1587] … count of currently executed operators:N` | whether that rank is **still stepping** — read the series, not the STUCK lines |

Reading it:

- **Equal `N` on two nodes at the same second** ⇒ both ranks are inside the same
  collective instance, each waiting for the others. Recorded: 2706 at `03:12:39` on
  both decode nodes, with `status[STUCK]` for the whole rank set inside 5 s.
- **A rising `N`** ⇒ that process is not stuck, whatever its STUCK lines say.
  Recorded: the P node printed STUCK at `03:34:52` and still climbed
  `110822 → 4191386` before freezing at `05:38`. Never forward a P-node STUCK line
  as evidence about the D-side group without this check.
- The `STUCK` records also give the topology the peers saw (`crimer rank
  [<ip>]/<rank>`), which is how you tell a whole-group stall from a subset.
- **A single `count` line is not a freeze.** The claim is that the series stopped
  moving, so ask for two or more samples across the hang window; the script labels a
  lone line `ONE SAMPLE` and refuses to build the cross-file verdict from it. Equal
  counts at the same **second** are what pair two nodes — compare to the second, not
  the millisecond, because the ranks' suffixes differ.

## Step 2 — is MC2 even the expected path?

The per-batch selector at the recorded pin is `select_moe_comm_method`
(`ascend_forward_context.py:203`): no `--enable-expert-parallel` or `ep == 1` ⇒
`ALLGATHER` (`:237-238`); A2 with EP ⇒ `MC2` only when
`num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <=
mc2_tokens_capacity` (`:239-248`), else `ALLGATHER`; A3 adds the
`VLLM_ASCEND_ENABLE_FUSED_MC2` / `dispatch_ffn_combine` split. The selector runs per
batch, so one deployment can use MC2 below `mc2_tokens_capacity` and `ALLGATHER`
above it — which is why Step 1 reads the op family out of the plog instead of
predicting it. A 32-rank A2 EP group satisfies the rule for any checkpoint with
≤ 768 experts (`24 × 32`), so state that the all-to-all *is* the selected path before
proposing anything, and make the ablation one axis that removes it
(`--enable-expert-parallel` off, or a one-node run) rather than a `cudagraph_mode`
change. If the run is `ALLGATHER`, this page does not apply.

## Step 3 — the comm space is the one lever the plog can falsify

`bufferSize[2097152000]` is exactly 2000 MiB. `HCCL_BUFFSIZE` is in **MB** and is
multiplied by 1024×1024 (`cann/hcomm/src/pub_inc/externalinput_pub.h:36,38` — default
200 MB; `src/platform/common/externalinput.cc:1401-1428`, which logs `HCCL_BUFFSIZE
set by environment to [N]M` at `:1427` or `set by default to [N]M` at `:1406`).
Recorded #8975 said 「HCCL buffer 设置的 2560」 while that group loaded 2000 MiB, so the
value had not reached that group: exported after the process started, on other nodes,
or in another launch. It matters more here than for any other group — vllm-ascend
passes **no** pg_options to a group whose name contains `mc2` (`utils.py:961-965`:
"Current mc2 operators only perform communication space partitioning based on
HCCL_BUFFSIZE configuration"), so the process env is its only source.

Three reads, all on the paste already in hand: the `HCCL_BUFFSIZE set by …` line in
the **same** plog window, `bufferSize[` for every rank of the group, and
`execTimeOut[`. A **different `bufferSize` across the ranks of one group** is a
candidate mechanism (`rg -n 'bufferSize\[' *.log | sort | uniq -c` decides it in one
call). Keep the cause open: the recorded case has no resolution.

## Step 4 — why there is no error text yet

`execTimeOut[1836]` is HCCL's **default** task/notify wait — `NOTIFY_DEFAULT_WAIT_TIME
= 27 * 68` (`cann/hcomm/src/pub_inc/hccl_common.h:84`, i.e. 1836 s), overridden only by
`HCCL_EXEC_TIMEOUT` (`:257-259`, the `SET_BY_ENV` state) — and it is the same number the
`fftsplus task timeout … stuck notify num:1, timeout=1836s` block eventually prints
(`error-report-timestamps.md`). So silence for the first half hour is expected, and
"死锁前无任何 ERROR 或 OOM 日志" is not a distinguishing fact; when the value in the
plog is not 1836 the reporter raised `HCCL_EXEC_TIMEOUT` and the wait is longer by
their own configuration. Ask for the plog
window *after* that expiry rather than a rerun; the `shm_broadcast … No available
shared memory broadcast block found in 60 seconds` and `RPC call to execute_model
timed out` lines are downstream of it (`rpc-watchdog-timeouts.md`).

## Delivery: the follow-up that carries the plog is the reversal

State the new lane in one sentence, give Steps 1-3 and the ablation, and keep the
source walk for afterwards — a delta turn, not a re-issue
(`../evidence-workspace/delivery-discipline.md`). Recorded #8975 published the
reversal ("这是 device 侧 MC2 集合通信死锁，不是我上一轮排第一的 DP host 侧 all_reduce")
and then spent the whole 8 192-token second segment re-deriving the MC2 selector and
the pg_options from source, with no message text on its last turn, so the action
ladder for the lane it had already named never shipped.

## Scope, and what to send back

Pins for every `path:line` above: vllm-ascend **v0.18.0rc1** (`99e1ea0`) and vllm
**v0.18.0** (`bcf2be96`); `select_moe_comm_method`'s line numbers and the pg_options
`FIXME` move between releases, so re-grep the function name and the comment before
quoting them (the A3 branch has already gained the fused-MC2 split at this pin). The
`mc2` **group name** and the fact that MC2 partitions by `HCCL_BUFFSIZE` are the parts
to re-verify first.

Send back: the plog from each node for the hang window (not a curated excerpt), the
`npu-smi info` taken *during* the hang (every utilisation column the reporter's
version prints — AICore alone does not separate the lanes), the launch
command of each node with its `HCCL_*` exports, and — if the deployment is PD — both
serve commands, because whether the two sides run the same EP/DP shape decides
whether they can even be in one group.
