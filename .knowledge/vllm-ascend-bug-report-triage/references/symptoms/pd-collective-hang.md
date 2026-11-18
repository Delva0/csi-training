# PD / multi-DP silent hang: HCCL STUCK and the divergent MoE collective

The service has been serving and then stops mid-run without an error. Distinguished from the two
neighbouring hangs: a startup stall has no served request yet (`startup-and-stall.md`), and a
pool/AICPU hang dies in the pooling path with `sample_tokens` timeouts (`c8-and-kv-layout.md`).

## Signature

- Port still listening, `curl` returns nothing, AICore 0 %, no new worker log line, and **no**
  ERROR/OOM/traceback — reporters call it 静默卡死 ("silent freeze").
- The decisive evidence is the CANN plog / HCCL heartbeat, not the vLLM log:
  `[heartbeat.cc:1314] ... status[STUCK]` on every rank, `[HCCL_TRACE]rank:<ip>/<n>, count of
  currently executed operators:<N>` with the **same N on all ranks**, and a line such as
  `[BuildRemoteResByTag] newtag[group_name_2338_mc2RunAlltoAllDirectFullmesh_device]` naming the
  communicator and the collective.
- Interpret it as: one group's members are not issuing the same collective in the same position, so
  the collective never completes and every participant blocks. The group identifier's rank count
  (e.g. 32 = DP16 × TP2) tells you it is the EP/MC2 MoE group, i.e. the mismatch is between **DP ranks
  inside one MoE collective group**. A P-side rank that later reports STUCK is waiting on that same
  group, not a second bug.

## First moves (cheap, before any code reading)

1. **Resolve the twin issue.** #8974's body was only a Feishu doc link; its closing comment said "you
   raised #8975 which contained more details", and #8975 carries the plog, the HCCL trace and the
   maintainer hypothesis. A thin body plus a duplicate pointer means the evidence is in the twin.
2. **Get the working-vs-failing deployment diff.** #8975's reporter eventually posted a full table of
   env vars and `vllm serve` args for the previously failing and the now-stable deployment. That diff
   localises the class far faster than reading the model code.

## Thread evidence (`reported`)

Issue #8975 (the detailed thread; #8974 was closed as its duplicate) — vLLM `0.18.0` / vllm-ascend
`0.18.0rc1`, CANN 8.5.1, A2, Mooncake layerwise connector, decode `dp_size=32 tp_size=1`,
`--additional-config` containing recompute + fuse_muls_add + cpu_binding, prefix caching off on P.

- Maintainer `linfeng-yuan` (2026-05-11): the deployment script sets `VLLM_ASCEND_BALANCE_SCHEDULING=1`,
  "a scheduler advancement used in non PD disaggregation deployments … it might lead to the problem
  that some DP ranks on D Node perform `All2AllV` communication op and others perform `MC2` op which
  obviously stuck vLLM's `execute_model`"; asked the reporter to remove it.
- Another maintainer agreed they have "never enabled these two schedulers at the same time" and asked
  for a check to be added.
- The reporter later reported >24 h stability after unsetting the flag **plus** ~20 other env/arg
  changes ("difficult to pinpoint the exact contribution of each"), no controlled A/B; then
  `wait-feedback` → `stale`. The thread gives a workaround to relay and a hypothesis to check,
  **not** a verified fix.
- The docs shipped at the reported tag say it outright: `docs/source/tutorials/models/DeepSeek-R1.md:123`
  (also `DeepSeek-V3.1.md:134`, `Kimi-K2.5.md:131` at `v0.18.0rc1`) — "enabling this feature is not
  recommended in scenarios where PD is separated" (`verified` doc line).

## Verified code chain at `v0.18.0rc1`

The DP all-reduce that keeps ranks in agreement can be switched off; when it is off, the MoE
collective type becomes a per-rank function of a per-rank token count.

1. `vllm_ascend/patch/platform/patch_balance_schedule.py` — `BalanceScheduler` is installed when
   `envs.VLLM_ASCEND_BALANCE_SCHEDULING` is truthy (`patch/platform/__init__.py:38`) and adds
   `balance_gather()` = `dist.all_gather(self.balance_queue, running_tensor, group=dp_group)` each
   step. Its purpose is to keep DP ranks' *running* counts close, which means per-rank token counts
   are allowed to differ step to step; `BalanceDPEngineCoreProc` additionally rebinds
   `vllm.v1.core.sched.scheduler.Scheduler` and `EngineCoreProc.run_engine_core`.
2. `worker/model_runner_v1.py:472` `_skip_all_reduce_across_dp_group()`: for a non-MoE (dense) model
   it returns True at once (no DP communication needed). For a MoE model it returns False unless
   `self.is_kv_consumer` is set (`:345`, true on PD decode ranks), and on those ranks it returns
   `decode_must_use_mc2 and (prefill_must_use_mc2 or ascend_config.recompute_scheduler_enable)`
   (function `:472-510`, the return at `:510`; its local `needs_mc2()` just calls
   `select_moe_comm_method`).
3. `_sync_metadata_across_dp` (`:512`) takes that branch at `:524-526` and returns the **local**
   `num_tokens` and `with_prefill`; `_sync_batch_across_dp` (`:1891-1893`) builds
   `torch.tensor([num_tokens_padded] * self.dp_size)` — the local value replicated `dp_size` times,
   not a real all-reduce. The synced branch (`:528-549`) is what all-reduces `num_tokens` **and**
   `with_prefill` across the DP group and returns the DP max — the only mechanism that made the ranks
   agree.
4. `ascend_forward_context.py:72` takes
   `max_num_tokens = int(num_tokens_across_dp.max().item())` only when that tensor is supplied, else
   the local `num_tokens`, and `:73` feeds it to `select_moe_comm_method` (`:203-282`), which maps a
   token threshold to a **different collective per SoC**:

   | SoC | MC2-ish branch | otherwise |
   | --- | --- | --- |
   | A2 | `MC2` if `num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity` | `ALLGATHER` |
   | A3 | `FUSED_MC2`/`MC2` below capacity (gated by `VLLM_ASCEND_ENABLE_FUSED_MC2`, EP size, draft model) | `ALLTOALL` |
   | A5 | `MC2` below capacity and `world_size_across_dp > 1` | `ALLTOALL` |
   | 310P | — | `ALLGATHER` |

   **Where the gate lives moved three times — read the reported ref's own copy.** The table above is
   the mid-release form (`ascend_forward_context.py:select_moe_comm_method`). At `v0.12.0rc1` it is
   instead `worker/model_runner_v1.py:_select_moe_comm_method` (`~:1402-1430`) with an inline per-SoC
   if/elif: EP off → `ALLGATHER`; EP on and SoC `_910B` and `num_tokens <= mc2_tokens_capacity` and
   `world_size_across_dp / pipeline_parallel_size >= 16` → `MC2`; otherwise `w4a8_dynamic` →
   `ALLTOALL`, everything else → `ALLGATHER`. At HEAD (`ascend_forward_context.py:388`) the decision
   is policy-driven: `get_current_hardware_profile().moe_comm_policy` selects one of
   `_select_capacity_and_expert_density_moe_comm_method`,
   `_select_fused_or_capacity_moe_comm_method`, `_select_capacity_and_world_size_moe_comm_method`
   (`:381-385`), with LoRA + EP forced to `ALLTOALL` (`:414-419`). So a report at an old tag cannot be
   answered from the HEAD condition list — #5295 (`v0.12.0rc1`, 910B, DP2 + TP8 + EP, crash inside
   `PrepareAndFinalizeWithAll2All.finalize` at `dist.all_gather`) is exactly this trap: at its tag the
   `world_size_across_dp // pp >= 16` branch is satisfied by 16 cards, so decode steps take MC2 while
   prefill-sized steps fall to `ALLGATHER` — verify the pair (token count, selected type) per rank
   before naming a mismatch.

   Capacity itself is small: `set_mc2_tokens_capacity` (`ascend_forward_context.py:170-182`) uses
   `min(max_num_reqs * uniform_decode_query_len, 512)` rounded up to TP, so a decode-sized step is
   MC2 while a large recompute/prefill-sized step on another rank is ALLTOALL.
5. Mechanism: once the sync is skipped, two DP ranks of one EP/MC2 group that straddle the capacity
   threshold — or that disagree on `with_prefill` (prefill vs decode branch) — issue different
   collectives at the same position. The result is exactly the observed shape: identical operator
   counts, all ranks STUCK, no error. This is the code-level form of the maintainer's attributed
   guess about `All2AllV` vs `MC2` above.

The skip condition is reachable through two independently configured schedulers, and nothing at this
ref rejects enabling both: `additional_config["recompute_scheduler_enable"]` (`ascend_config.py:89`)
switches `scheduler_config.scheduler_cls` to `RecomputeScheduler` (`platform.py:435-439`), while
`VLLM_ASCEND_BALANCE_SCHEDULING` (`envs.py:105`) installs the platform patch. Which scheduler class
is finally instantiated depends on that rebinding vs the `scheduler_cls` dotted path, so record which
one a reproduction actually ran instead of assuming the combination behaves like either alone.

The last step from step 5 to the report is **inference**: the code proves the alignment can be
bypassed on decode ranks, the reporter's plog proves a collective mismatch at one op index, but no
run recorded which ranks chose which type. State it that way.

## The invariant that must hold (useful at any ref)

MoE communication-method selection must be fed a token count that is **identical on every rank of the
EP group**. HEAD makes this explicit: `vllm_ascend/platform.py:728-733` ("NOTE: Must use
max_tokens_across_dp instead of num_tokens for MoE comm method selection to ensure consistent
communication method across all DP ranks"), fed from `dp_metadata.num_tokens_across_dp_cpu` and
falling back to the local `num_tokens` when `dp_metadata is None`; `_skip_all_reduce_across_dp_group`
no longer exists at HEAD. When triaging a hang at any ref, check (a) how the argument is derived and
(b) whether every rank can compute the same value — a local fallback on one rank is enough to break
it.

## Workaround (ordered; state which one you verified)

1. **Unset `VLLM_ASCEND_BALANCE_SCHEDULING` on the decode nodes.** This is the maintainer's advice in
   #8975 and is documented at the reporter's own tag (docs line above). It costs the output
   throughput / TPOT gain the reporter copied from a tutorial. Mind how the switch is spelled at each
   ref (`../vllm-ascend/config-flag-lifetimes.md` §7): env-only at `v0.18.0rc1`/`v0.19.1rc1` (no
   `additional_config` key, so a JSON opt-out there is **silently ignored**), config key from
   `v0.20.2rc1`, documented form `--additional-config '{"enable_balance_scheduling": true}'` at HEAD.
2. If the feature must stay, attack the skip condition itself: `recompute_scheduler_enable` off, and
   `--max-num-batched-tokens` above the MC2 capacity so `prefill_must_use_mc2` is false (both are
   needed — either one alone keeps the all-reduce skipped while `decode_must_use_mc2` holds). Then
   equalize the per-rank token counts: uniform `--max-num-seqs`, and no mixture of long-prefill and
   short-decode inputs on the same D node.
3. Reported-only and confounded (the reporter says >24 h stable, but changed many knobs at once):
   `HCCL_BUFFSIZE=2560`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, `MooncakeLayerwiseConnector` instead of
   `MooncakeConnectorV1`, decode topology dp32/tp1, `multistream_overlap:false`,
   `--no-enable-prefix-caching` on P, decode `--max-num-seqs 48` with 128 batched tokens, prefill
   `--max-num-seqs 16`. Never present one of these as an isolated fix.

## Status and limits

- #8974 was closed as a duplicate of #8975 (2026-05-12); #8975 is open, carries a `wait-feedback` →
  stale-bot comment (2026-05-30), and the reporter never ran the controlled experiment that would
  isolate the flag. No fix commit is claimed, and none was found in the clone's history.
- Verified: the code path, the skip condition, the capacity arithmetic and the documented "not
  recommended with PD" line. Hypothesis: that this exact report took that path.

## Check before reuse / falsifying check

- Verify the key exists at the reporter's ref — an unknown key is silently ignored
  (`../vllm-ascend/config-flag-lifetimes.md`). The switch moved: `9bef60ad5` (#14827, "Remove balance
  scheduling environment variable") first appears in `v0.27.1rc1`, after which it is
  `--additional-config '{"enable_balance_scheduling": true}'` /
  `scheduler_config.enable_balance_scheduling`; HEAD's docs still say it is not recommended when
  prefill and decode are separated.
- Dump the chosen `MoECommType` per rank per step (`select_moe_comm_method` input and result) on the
  decode node and diff across ranks, or log `num_tokens`, `with_prefill` and `moe_comm_type` for the
  steps before the hang: a divergence confirms the mechanism, agreement falsifies it.
- If `_skip_all_reduce_across_dp_group()` returns False at the reported ref, the DP all-reduce keeps
  the ranks aligned and the cause is elsewhere (`patch_sched_yield` idle-skip, a different connector).
  `git grep -n 'select_moe_comm_method' <ref>` plus the DP-sync call sites give the current
  derivation; do not reuse the v0.18 line numbers. A hang that disappears when all ranks do identical
  work is this class.

## Neighbouring shape: the thread names MTP cherry-picks and the log simply freezes (#8916)

Same observable (serving stops, nothing errors) but the evidence is a cherry-pick list, not a plog.
Report: **#8916**, Qwen3.5-MoE, PD disaggregation, `vllm-ascend base:0.18.0rc1` + `vllm 0.18.0`;
"stopped after ~1 h, the log stuck, hours later the server shut down" — body otherwise the template
(`collect_env` left as `Your output of above commands here`), no `vllm serve` line, no plog.

- **Thread:** two asks for details (one maintainer reports not reproducing it locally), then the
  reporter's own post-upgrade environment block and: *"this bug fixed with these three pr, but the
  accurancy bug still exists #8932"*. The three are #8531 (`696dcc926`, sample kernels with a
  non-static mask / invalid `draft_token_id`), #7904 (decoder MTP padding token id) and #8764
  (`9a0b786f2`, "Fix race in non-blocking `num_accepted_tokens`"). Issue closed `completed` with no
  further diagnosis.
- **Check `merged` before relaying a PR as a fix.** #7904 is **closed, unmerged** (`base: main`; its
  body: "padding `-1` causes sampling operator error when using qwen3.5 with sampling parameters …
  changes the padding token_id to the EOS token_id"). A fix list containing an unmerged PR means the
  reporter is carrying a local patch — "upgrade to get it" does not exist. The other two are release
  cherry-picks: both are ancestors of `v0.18.0` and of **no** `v0.18.0rc1`.
- **The one verified code fact is a downstream override dropping an upstream guard (`verified`).** At
  vLLM `v0.18.0rc1` *and* `v0.18.0`, `vllm/v1/worker/gpu_model_runner.py:794`/`:797` declares and
  creates `num_accepted_tokens_event`, `:1378-1379` records it right after the non-blocking D2H copy,
  and `:1925-1926` synchronizes before the CPU read. `NPUModelRunner(GPUModelRunner)`
  (`vllm_ascend/worker/model_runner_v1.py:223`) overrides `_build_attention_metadata`; at
  `v0.18.0rc1` its `:2045-2048` replaces the whole block with
  `self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]` +
  `copy_to_gpu()` — no event, no synchronize, and `git grep num_accepted_tokens_event v0.18.0rc1 --
  vllm_ascend` is empty. `v0.18.0` carries the guard at `:2043-2045`. So an rc1 build reads the
  accepted-token array on the host without any guarantee the copy landed; the runner's own comment at
  `:2611` says the path is selected through `MambaSpec`, i.e. a hybrid model under MTP, matching the
  report.
- **Levers:** an image/tree at `v0.18.0` or later (that tag's tree carries the sync), or drop MTP on
  the decode node first (cost: the speculative speedup; it also removes the sampler-side defects #8531
  and #7904 belong to). Any rc1 build needs the cherry-picks, and #7904 has no released form.
- **Status and limits:** the race is `verified` in code; that it is *this* freeze is `hypothesis` —
  there is no plog, no HCCL line and no traceback in the thread, and a stale accepted-token count is
  at least as likely to corrupt output as to block. The accuracy half is a separate open report
  (#8932: AIME −3.5 on `0.18.0` vs `v0.18.0rc1`, its lever list already tabulated in
  `issue-ledger/reports-later.md`); do not merge the two.
- **Falsifier:** the same workload with MTP off on the decode node still freezing after ~1 h → this is
  not the spec-decode bookkeeping family; go back to the plog/`STUCK` route above (which the thread
  never supplied). A freeze that starts at a fixed request count rather than a fixed time is the
  admission/blocking class, not this one.

## Neighbouring shape: the producer holds its blocks (legacy `llmdatadist`, `≤ v0.11`)

Not a collective hang — nothing blocks — but it is the other way a PD deployment "stops progressing" while
every process stays healthy, and the prefill log names the mechanism if you read it as arithmetic:

```text
Some requests in prefill node fail to receive KV Cache transfer done signal in 120s. If a greater mean
TTFT is acceptable, you can 'export VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=600' (10 minutes) to relax the
timeout condition.
```

`VLLM_LLMDD_ABORT_REQUEST_TIMEOUT` (default `120`, `envs.py:140-141`) is a **release** deadline on the
producer's KV blocks in `request_finished`, so the log's own advice lengthens the stall it advertises to
shorten; only a decode-side `ReqForFinished`, one per announced decode TP rank, frees the block earlier.
The module sits at `vllm_ascend/distributed/llmdatadist_c_mgr_connector.py` — **not** under
`distributed/kv_transfer/` — and is gone from `v0.12.0rc1` (`a77045f35`, #4780), so the durable lever is the
connector choice, not the timeout. Full chain, levers and falsifier: `pd-llmdatadist-timeout.md` (#3062).
