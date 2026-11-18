# Issue ledger: index of the analysed reports

134 rows, all from `vllm-project/vllm-ascend`, filed against an rc tag, a nightly build or a
released image and analysed from the local clone only (no NPU access). Each report gets one row with
the verified facts established for it, so a repeat report starts from that row instead of re-deriving
the code reading. Thread states were read through the GitHub API (2026-11-17); dates are the
comments' own.

Several rows are not vllm-ascend defects: #16237 and #8664 land in upstream vLLM, #1768 in the installed
transformers (v4.53.0's `from_args_and_dict`), #5299 is a `[Usage]` question whose answer is a build gate,
#12692's allocator is torch_npu's, #6529 in CANN's capture window (#107041), #6942 in the container runtime,
#18158 in the guide's own serve block,
#1316 is an expectation gap (the KV cache is sized to `gpu-memory-utilization`, so per-card memory says
nothing about TP), #2566 is not a bug report at all but a cross-release "why is the newer tag faster"
question, #11737 is a community-test task sheet with no failure to diagnose, #2169 is upstream vLLM's
engine selector, #16023's quoted frame is a teardown artefact and #5635's is a transport status code
rather than the cause, #3462/#13218/#17593 are real crashes whose owner is the opt-in scheduler, the
replay barrier and the graph-capture dummy batch respectively, #1696 is an image/CANN-base coverage gap (an
A2 kernel set on an A3 card) rather than a code defect, and #1114 is a build-artefact question the thread
answered with the installation docs. #8022 and #9659 are upstream-frontend questions (the response
field is `reasoning`, and only the client's round-trip keeps a prior turn's thinking), not plugin
defects. The batch-13 group (all seven rows are in `reports-newest.md`) is mostly of that kind: #8788 is
the build script's git-submodule fetch (offline source install), #10068 is a **harness measurement**
question the maintainers declined with nothing in the repo to localise, #1791 is an eval-command/DP-MoE
question whose parallel key was silently overridden, #15911 is a **bundled upstream** parser (whose file is
gone by `v0.23.1rc0`, so no plugin fix exists), and #5813 is the `deq_scale` twin of the
quantised-checkpoint-loaded-dense class whose flag never reached the process. Only #15985 (the one report in
the group with a maintainer-named, tag-verified fix — `v0.26.0rc2`) and #10963's two symptoms route into
plugin pages with a named mechanism, and #10963's thread supplied no fix at all.

The batch-14 group (rows also in `reports-newest.md`) adds eight of mixed kind. #17718 is a `[Doc]`
contributor-setup report whose hard fact is `setup.py`'s generated `vllm_ascend/_build_info.py` (a mount
shadows it); #12398 is a tag-leak report that **cannot be localised** because the body carries no
`vllm serve` command (parser selection is the answer); #6421 is a real upstream bitmask-loop defect whose
filer-confirmed lever is a P/D config mismatch (the assert on a `-1` placeholder, which vLLM `v0.14.0`
skips by construction); #7580 is #7682's `v0.17.0rc1` twin seen from the client as `Unterminated string`;
#5170 is a **verified** worker-level warm-up defect with a named fix tag (`v0.14.0rc1`) and a one-line
workaround on the reporter's build; #14440 is an open, measured per-engine concurrency cliff with probes
only; #7401 pairs a non-fatal Mooncake store-child failure with a `FULL_DECODE_ONLY` dummy-batch worker
death; and #3062 is a retired `llmdatadist` PD connector whose prefill holds blocks for
`VLLM_LLMDD_ABORT_REQUEST_TIMEOUT`. None of the eight has a maintainer-confirmed workaround, so every
lever in those rows is labelled.

The batch-15 group is four reports over three issues, and it is the place where one answer could be
checked twice: #7401 came back in a later pass and was answered, and #7448 turns out to
name #7401 as its twin — same GLM5/W4A8/`FULL_DECODE_ONLY` crash family, but a single-node DP2+TP8 shape
rather than PD, so the twin's thread is the evidence and the shape is a widening, not a second owner.
#12989 is the only maintainer-acknowledged *limit* in the group (a 310P capture-event ceiling, "needs a
future driver release") with a reporter-confirmed `--enforce-eager`; #13356 is the only one whose ask was
wrong (the behaviour it reported is upstream's own clamp, so the lever is lookup cost).

The batch-16 group (rows also in `reports-newest.md`) is eight reports. Two
are fully localised defects: **#11231** is a `could not broadcast input array from shape (29,) into shape
(31,)` in `update_tokens_for_pcp` once PCP meets MTP on a hybrid-attention model, whose shapes are the
arithmetic of a token count used as a request index and whose fix (#10487) first ships in `v0.23.0rc1` —
the reporter's "#10172 is already merged" is a different, earlier fix, checked by hunk; and **#8328**'s
guided-stream hang is an Ascend-side scheduler defect with a nameable fix (`VLLM_ASCEND_BALANCE_SCHEDULING=1`
strands every structured-output request in upstream's `skipped_waiting` queue; #10509/#10699/#10559, first
tags `v0.22.1rc1`/`v0.23.0rc1`), with the rest of its env delta dead at the tag and the torch_npu/CANN layer
declared as a gap. **#9666** is the §G padding class in its accuracy shape, with the version lever confirmed
by the reporter (`v0.18.0`). **#8223** is the MiniMax-M2/MiMo `--enable-expert-parallel` 乱码 class whose
reporter-confirmed lever is `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` and whose deciding branch is a custom op before
`v0.20`. **#5942** is **not a plugin defect** but a DP launch whose started ranks summed to less than
`--data-parallel-size` (a `logger.debug`-only `wait_for_engine_startup` poll), fixed by config by the
reporter; **#5188** is a maintainer "resolved in the latest version" close with no log to localise (lever
only, no provable mechanism across ~4 000 commits); **#11770** is a `[Community Test]` sheet whose only
question was how to send a request into a headless container and whose cited page does not exist at its own
tag; and **#3045** is a cross-release `init engine` timing question with an unfetchable env link and zero
comments. Each of them is answered by its lever plus at most one ref-pinned fact, never by a mechanism
the report never exhibited.

The batch-17 group (rows also in `reports-newest.md`) is seven distinct reports; five get a new row and the
two already-ledgered ones (#3062, #6678) added no mechanism, only a confirmed page. **#6250** is an image
whose own `Dockerfile` pip-installs a `torch-npu` `.post` build the tag's `pyproject.toml` never asked for,
so the first captured graph faults (§D of `../../vllm-ascend/dockerfile-env.md`); **#6765** is a
zero-argument GLM call that raises inside the **upstream** `glm4_moe` parser and is dropped, guarded from
vLLM `v0.15.0rc0` (`tool-call-streaming.md` §4a); **#3843** is the `available_memory <= 0` branch of the KV
check, i.e. the `profile_run` the platform's `max_num_batched_tokens → max_model_len` rewrite blew up
(`startup-and-stall.md` §3c, `../../vllm-ascend/config-flag-lifetimes.md` Trap D); **#9430** is a P/D + MTP
deployment answering fluent garbage, the connector's uncut block-id list, both fixes first tag
`v0.20.2rc1` (`mooncake-transfer-errors.md`, "The silent sibling"); and **#12891** is a startup death whose
only error is the `WorkerProc initialization failed … background process` wrapper — the child was killed
rather than raising, and the missing child log *is* the deliverable (`startup-and-stall.md` §8). #6678 was a
report whose row and page already settle the workaround, the status and an `[inference]` mechanism, and
re-deriving that mechanism instead of replying shipped nothing.

The batch-18 group (rows also in `reports-newest.md`) is seven reports, one of them already ledgered,
and most of the group was first attempted without a shipped reply — the *shape* of the answer mattered
more than the search. **#8213** hit the ledger (`model-type-support-floor.md`, the tokenizer-class twin)
and was answered from batch 1's thread plus the class page without a code read — the gate's intended
shape. **#2040** is not a vllm-ascend defect at all — the reporter's own script spells the request key
`multi_model_data`, and vLLM's prompt dict is passed through unvalidated
(`vllm/inputs/preprocess.py:418 _process_text` is the only reader of `multi_modal_data` at `v0.9.2`), so the
`<image>` placeholder decoded as text and produced `ca, \n  a A . γ`; a contributor's corrected script is the
lever (`multimodal-input-not-applied.md`). **#12339** is the batch's one fully localised plugin defect:
alternating good/bad answers on a 0.23 P/D hybrid deployment, because the Mooncake receive thread applied
its KV **reformat** per pull message instead of after all TP-offset pulls of the request — fixed by an
`releases/v0.23.0` backport (`86217325b`, #12371, first tags `v0.23.0`/`v0.23.0.post1`) that the reporter's
build `6cee2e163` predates by eight days and carries no `pending_reformat`, while the mainline original
`1db626510` (#12359) is only in `v0.27.1rc1`, the wrong tag to quote back to a 0.23 reporter — and #12359 is
the wrong PR to call this report's twin (`mooncake-transfer-errors.md`, "The TP-inequality shape"). **#7154** is a hang whose only error-looking line is upstream's 60 s
`shm_broadcast` warning: `VLLM_ASCEND_ENABLE_FUSED_MC2=1` cannot select the fused path on a BF16 checkpoint
(`ascend_forward_context.py:252` requires `w8a8_dynamic` through `v0.16.x`), the tag's own Qwen3-235B
tutorial calls the feature W8A8-only and names `=0` as the lever, the reporter patched that gate, and their
own follow-up pins the op's limit at M > 2048 (`moe-ep-dispatch.md` §C). **#3911** has nothing localisable:
a bare aivec `DDR address of the MTE instruction is out of range` in an eager, non-PD run, where the plog's
`fault kernel_name` is the only owner and the thread is a "use the latest release" close
(`device-kernel-faults.md`, the aivec sibling section). **#1955** is the environment/fabric class: the
plog's first error is the device-plane RDMA init, and the close ("network card issue", then "TorchAir is
deprecated") is what the lines say — with TorchAir removed from the tree at `v0.12.0rc1`, there is no
forward fix to hunt (`../../hccl/init-failures.md`). **#18315** has only a reporter-confirmed lever (unset
`PYTORCH_NPU_ALLOC_CONF` + `HCCL_OP_EXPANSION_MODE`) and a reference block that prescribes exactly that
pair, so it ships as doc-conformance plus an A/B, never as a derived mechanism
(`../../hccl/env-support-matrix.md`). **#8686** is a packaging contract, not a defect at a tag: `setup.py:522`'s
`find_packages` drops any subtree whose parent lacks `__init__.py`, the two directories do not exist at
`v0.13.0` at all (they arrive on the branch via #8648), and the fix #8729 (`58ea73f95`) is in **no tag**
(`../../vllm-ascend/install-and-image.md` §5).

The batch-19 group (the #8916 row is in `reports-early.md`, the other six in `reports-newest.md`) is
seven reports, and most of them are *lost replies*: in one minibatch only one of four runs shipped, and in
another three of four opened with a setup message, spent a fourth or fifth read and hit the output cap
while still reasoning — the ledger row is what makes those recoverable. **#8916** closes a gap the earlier
rows only pointed at: the 「stuck after ~1 h, then the server is down」 twin of #8920 is the
`num_accepted_tokens` D2H race — `v0.18.0rc1` has no `synchronize` at `worker/model_runner_v1.py:2045-2048`,
`9a0b786f2` (#8764) adds it, first tag `v0.18.0` — and its third PR (#7904) is **closed unmerged**, so the
reporter's three-PR list is two merged plus one not (`spec-decode-and-parallelism.md` §1b, including the
*fork-override-missed-the-upstream-fix* shape: paired vLLM `v0.18.0rc0` already had #36013). **#16022**'s
thread carries a community tester's reproduction on `v0.26.0rc2` with numbers (`44.5 → 46.0 GB`, then
stable, KV back to 0 %) — a plateau reading that *is* the deliverable; its two verified facts are that the
script's `-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` **suppresses** the fork's default
(`expandable_segments`, `v0.23.0rc1:platform.py:761-777`) and that the argv's `torchair_graph_config` has
been dead since `v0.12.0rc1`. **#12278** is the cleanest lever-not-mechanism case in the group: the
reporter's own 【原因定位】/【修复建议】 names `pass_config: {"enable_sp": true}` (remove it, or use
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`), the tag's docs *do* still document the key — so the reporter's "no
longer supported" is a `limits` line, not a cue to hunt — and what is left is `enable_sp_by_pass` versus
`enable_sp` (`flashcomm1-and-moe-reduce.md`). **#3398** has no lever at all (one maintainer question, then
the stale bot), and its two settled negatives are worth keeping: eagle3 *is* supported at
`v0.11.0rc0`/vLLM `v0.11.0` (the Baichuan-M2 `draft/` head declares `LlamaForCausalLMEagle3`, a registered
architecture), and the pasted `shm_broadcast.py:466` is vLLM `v0.11.0`'s **reader** branch, so the engine
published nothing for 60 s. **#15247**'s "dense W4A8 was removed on main" is a sunset, not a regression
(last tag with the linear scheme `v0.26.0rc2`, `quantization-load.md` §9); **#17991** is the first row owned
by the FA3 backend (a token count sliced into a request axis, fixed on `main` only — `fa3-backend.md`); and
**#8248** is a 310P MoE report whose title (「运行正常，无法输出」) contradicts its attachment (an OOM in
the Python routing path, EP unsupported — `310p-capability.md`). A tool note from that group: a `web_fetch`
on `huggingface.co/<org>/<repo>/raw/main/config.json` died at 120 s, while `curl -sSL` in `bash` returned
the same JSON.

## Pages

- `reports-early.md` — the reports that shaped the original notes: PD/Mooncake and KV-pool stalls,
  load-time and operator failures, graph/capture crashes, spec-decode and C8 output corruption; its newest
  row is **#8916**, the 「stuck after ~1 h, then the server is down」 twin of #8920 (the
  `num_accepted_tokens` D2H race: no synchronize at `v0.18.0rc1`, fixed by `9a0b786f2`, first tag `v0.18.0`,
  and the reporter's third PR #7904 is unmerged).
- `reports-later.md` — later triage: quantisation-description and transformers floors, the batch-12 group
  (a `[Usage]` question answered by the store backend's `-DUSE_ASCEND_DIRECT` gate, an OOM after a healthy
  start under the per-stream workspace allocator, a GLM-4.1V processor `TypeError` that is a transformers
  pin, a 310P twin of the slow-LoRA report, a `MultiConnector` composition assert, a `FULL` mode silently
  downgraded on a hybrid linear-attention model, a `llm.`-prefixed quant-description `KeyError`, and a
  `SchedulerDynamicBatch` signature drift), prefix-cache
  unit mismatches, the PCP/MTP field gates, tool-call streaming, the docs/benchmark rows, the seven
  rows of the previous group (a `[Community Test]` sheet, a teardown frame, a Mooncake connect code, the
  opt-in AscendScheduler, upstream's V1 oracle, the replay-sync OOM label and a graph-only garbage case)
  and the batch-10 group (a missing-`.so` install, an image/SoC kernel gap, a KV admission cliff, a
  first-call-only garbage run under sleep mode, a Mooncake allocator/`LD_PRELOAD` image defect and the
  open first-request `completion_tokens: 1` report).
- `reports-newest.md` — the rows added after `reports-later.md`: the **batch-19** group — #16022 (a memory
  plateau whose lever is the community tester's newer-rc reproduction plus two dead/suppressing env facts),
  #12278 (`pass_config.enable_sp` and the `enable_sp_by_pass` twin of `enable_sp`), #3398 (no lever; eagle3
  *is* supported at `v0.11.0rc0`, and the pasted `shm_broadcast` line is the reader branch), #15247 (the
  W4A8-linear sunset), #17991 (the FA3 mixed-batch `507015`, fixed on `main` only) and #8248 (310P expert
  parallelism) — then the **batch-18** group — #2040 (a
  misspelled `multi_modal_data` key, so the image never entered the model; a client-side answer),
  #12339 (alternating accuracy from a Mooncake KV **reformat** that ran before all TP pulls arrived —
  branch fix `86217325b`/`v0.23.0`, mainline `1db626510`/`v0.27.1rc1`), #7154 (a BF16
  `VLLM_ASCEND_ENABLE_FUSED_MC2` hang at the op's M > 2048 limit, with the 60 s `shm_broadcast` line as
  the symptom), #3911 (a bare aivec `DDR address` with no kernel name, so the plog is the ask), #1955 (a
  device-plane RDMA init failure that leaves the `MC2MultiServer` communicator without a transport; a
  deprecated-TorchAir close), #18315 (an env pair only the doc block prescribes — doc-conformance plus an
  A/B) and #8686 (a branch build whose wheel drops a package tree — `find_packages`) — the **batch-17**
  group — #6250
  (an image whose `Dockerfile` installs a `torch-npu` `.post` build the tag's `pyproject.toml` never asked
  for, so the first captured graph faults; lever is the pin, #6287, `v0.13.0rc3`), #6765 (a zero-argument
  GLM call that raises inside the **upstream** `glm4_moe` parser and is dropped; guard from vLLM
  `v0.15.0rc0`), #3843 (a hybrid model's per-rank `Available memory: 0` — the profile pass overshot the
  budget after the platform rewrote `max_num_batched_tokens`), #9430 (P/D + MTP answering fluent garbage;
  the block-id cut, first tag `v0.20.2rc1`, and "v0.20" means `v0.20.2rc1` because no `v0.20.0` tag
  exists) and #12891 (a startup death with no child traceback — the child was killed, so the missing log
  *is* the deliverable) — then the **batch-16** group (#11231's
  PCP+MTP broadcast, #9666's repeated token, #8328's two-deployment env-bundle diff and its
  `BALANCE_SCHEDULING` scheduler, #5188's no-log `dp+etp+fullgraph` hang, #8223's EP-only 乱码,
  #5942's rank-count stall, #11770's task sheet and #3045's cross-tag `init engine` time) —
  then the **batch-15** group — #12989 (a 310P
  DUO start that exhausts capture **events** at `capture_end`; the fork's handler knows only the stream
  code), #13356 (a pool whose lookups hit while every request loads 0 tokens — upstream's
  `need_to_allocate` clamp) and #7401's second appearance (delivered this time), with #7448 as the same
  body's duplicate pointer to #7401 — then the batch-14 group — #17718 (the
  generated `_build_info.py` a mount shadows), #12398 (markup in `content`, a parser-selection answer),
  #6421 (the `-1` placeholder meeting the grammar under MTP + PD), #7580 (the GLM5/MTP-3 tool-call
  `Unterminated string`), #5170 (a 310P start dying in the ATB warm-up, fixed by #5776 / `v0.14.0rc1`),
  #14440 (the open per-engine decode cliff, probes only), #7401 (a `MultiConnector` store-put half beside
  a `FULL_DECODE_ONLY` dummy-batch crash) and #3062 (pre-`v0.12` PD TTFT held by the `llmdatadist`
  producer-side block timeout) — plus #7448 (closed as a duplicate of #7401: same class, same reporter) and
  #11406 (a 310P `w8a8sc` pre-sharded checkpoint read by the default loader, `--load_format sharded_state`
  missing from the argv) and #13418 (a `MemcacheBackend._setup_store` bare assert under PP, mechanism open,
  the thread's PR list already in the tag) — then the whole batch-13 group — #8788 (an
  offline source install that dies on the `catlass` submodule), #10963 (a log-less PD decode-node hang
  paired with an MC2 OOM), #1791 (a gsm8k collapse with TP2/DP2 whose ETP key never applied), #10068 (an
  evalscope-vs-ais-bench mmlu-pro gap), #15985 (`507034`/`507035` on DeepSeek-V4 A3 PD, fixed in
  `v0.26.0rc2`), #15911 (a bundled upstream parser emitting a whitespace `content` delta) and #5813 (a
  W8A8 `deq_scale` `KeyError` whose flag never reached the process); it also carries the shared recheck
  conditions.
- `lessons-across-reports.md` — what the rows show in aggregate: recurring shapes, where the answer was
  *not*, negative results worth keeping, and how thread state and attribution should be read.

## Which report is where

Grouped by the detail page each row names (the page that carries the mechanism):

- `310p-capability.md` — #8248 (310P MoE with `--enable-expert-parallel`: the platform's Python `moe_init_routing` OOMs in the first forward at `_310p/fused_moe/token_dispatcher.py:113`, and #9105 both replaces that routing with an NPU op and rejects EP from `v0.21.0rc1`)
- `aclgraph-mode-and-capture.md` — #5076, #4326, #1183, #17593 (§D), #4180 (§E, the legacy `full_cuda_graph`
  boolean), #11947 (§F, `FULL` silently downgraded on a GDN model), #7401 (§G, the `num_reqs_padded`
  dummy-batch death on a 2-node PD deployment), #12989 (§H, a 310P capture-**event** exhaustion the
  handler's `207008` marker misses), #9666 (§G's repeated-token twin at `v0.17.0rc1`, gone by `v0.18.0`)
- `ascend-scheduler.md` — #3462 (§A); #18688 (§B, `SchedulerDynamicBatch` vs upstream `Scheduler.__init__`, `hash_block_size`); #8328 (§C, the env-gated `BalanceScheduler` that never reads upstream's `skipped_waiting`)
- `c8-and-kv-layout.md` — #12533, #15927, #13983, #12956
- `community-test-task-sheets.md` — #11737, #11770 (a tooling complaint and a cited path missing at the sheet's tag)
- `config-flag-lifetimes.md` — #11624, #8328 (§6's vendor-env-bundle liveness table + §7's `VLLM_ASCEND_BALANCE_SCHEDULING` patch and its stale `schedule()`/`run_engine_core` copies), #3843 (Trap D's profile-run consequence: the rewritten `max_num_batched_tokens` sizes `profile_run`), #12278 (Trap A on `pass_config.enable_sp`: the docs still ship the key at `v0.18.0`) and #16022 (§5's `torchair_graph_config` bullet: dead since `v0.12.0rc1`)
- `deepseek-v4-thinking-and-effort.md` — #10413
- `device-kernel-faults.md` — #17082, #1696 (the ACL-init / other-SoC sibling), #3911 (the bare aivec `DDR address` family with no kernel name and no plog: the plog request plus the release lever, no code read)
- `doc-guide-vs-capability.md` — #18158 (the table's trailing note also points at two further
  load-time reports, #1127 and #3494)
- `dockerfile-env.md` — #12059, #5609 (§C, the package the image must carry), #6250 (§D, a `torch-npu` `.post` build pip-installed over the tag's own pin)
- `dynamo-graph-break.md` — #8224
- `fa3-backend.md` — #17991 (the FA3 backend's token-vs-request slicing in a mixed prefill + spec-decode step: `507015` at a stream sync, fixed by #18068 on `main` only, no tag through `v0.30.0`)
- `flashcomm1-and-moe-reduce.md` — #10503, #8990, #10466, #8223 (§B, the `≤ v0.19.x` op-level guard and the documented EP/FC1 pairing), #12278 (§D, `pass_config.enable_sp` alone — no env var — re-routing the MoE communication through `enable_sp_by_pass()` and truncating the answer mid-sentence)
- `grammar-bitmask-and-dep-skew.md` — #5894 (§A), #6421 (§D, the `-1` placeholder under MTP + structured output)
- `graph-capture-and-custom-ops.md` — #1895, #1383, #6529, #13218 (§C2), #5170 (§B2, the 310P warm-up ATB op)
- `../../hccl/index.md` — #1955 (the device-plane RDMA init that leaves the `MC2MultiServer` communicator without a transport, and the TorchAir timeline; `init-failures.md`), #18315 (the env support matrix behind the AIV/expandable-segments pair; `env-support-matrix.md`)
- `hybrid-gdn-pd.md` — #8920; #12339 is listed under `mooncake-transfer-errors.md` (a PD accuracy alternation whose fix is the Mooncake reformat timing, not the hybrid cache)
- `benchmark-accuracy.md` — #10068 (§1, harness-vs-harness on MMLU-Pro), #1791 (§2, the DP>1 collapse, twin #1289), #2040 (§4, an image prompt answered with garbage because the dict key is not `multi_modal_data`; the class page is `multimodal-input-not-applied.md`)
- `install-and-image.md` — #1114 (§2, the `vllm_ascend_C` / `libvllm_ascend_kernels.so` pair), #8788 (§3, the `catlass` submodule an offline source install fetches), #17718 (§4, the contributor mount over the image's editable install), #8686 (§5, `find_packages` drops a subtree whose parent lacks `__init__.py`)
- `kernel-launch-limits.md` — #3971
- `kv-admission-cliff.md` — #16365 (twin #15984)
- `kv-connector-registration.md` — #7971, #6678
- `kv-pool-store-hang.md` — #8808, #6512, #16342 (§C, `load_async=False` in-line load), #5299 (§D, the store backend's single `protocol: "ascend"`), #13356 (§E, lookups that hit while `need to load` stays 0)
- `lora-and-tp.md` — #16586, #1629, #1812 (the 310P twin)
- `memory-sizing-and-parallelism.md` — #1316
- `metrics-counters.md` — #8664
- `missing-downstream-feature.md` — #4505
- `model-type-support-floor.md` — #8315, #8213 (the tokenizer-class twin), #1768 (the processor twin: transformers 4.53.0 `from_args_and_dict`)
- `moe-ep-dispatch.md` — #10944, #2567, #7154 (§C, `VLLM_ASCEND_ENABLE_FUSED_MC2=1` on BF16 cannot select the fused path, and the op hangs above M 2048), #18315 (the dispatch op failing only with the reference block's env pair; reporter-confirmed lever, mechanism open)
- `mooncake-transfer-errors.md` — #12339 (the receive-side reformat timing: "The TP-inequality shape"), #5635 (the store-backend/`-DUSE_ASCEND_DIRECT` half of #5299 is `kv-pool-store-hang.md` §D), #7401 (the store child of a `MultiConnector` failing to PUT), #9430 ("The silent sibling": P/D + MTP answers fluent garbage because the block ids are not cut to the prompt, #8908/#9810, first tag `v0.20.2rc1`)
- `multimodal-input-not-applied.md` — #2040 (the `multi_modal_data` key, the placeholder token, the encoder budget: a VLM answering as if the image were not there)
- `multi-connector-composition.md` — #7085 (`MultiConnector` starving its layerwise child)
- `npu-device-visibility.md` — #6942
- `pcp-dcp-and-mtp.md` — #8926, #13750, #8578 / #8402; #11231 (§E, the PCP+MTP `could not broadcast` shape mismatch); #14440 (§D, the open throughput cliff)
- `pd-collective-hang.md` — #8974 → #8975, #5295; #8916 (the MTP-cherry-pick variant: a downstream `_build_attention_metadata` override that dropped upstream's `num_accepted_tokens_event` sync, and a fix list containing an **unmerged** PR); #10963's hang half is probes only; #3062 is the neighbouring producer-block-hold pointer; #5188 (probes only — a maintainer "latest release/rc" close with no log)
- `pd-llmdatadist-timeout.md` — #3062 (the pre-`v0.12` `LLMDataDistCMgrConnector` block hold)
- `pd-proxy-examples.md` — #15389
- `pooling-and-embedding.md` — #7192
- `post-startup-oom.md` — #12692 (allocator half verified; the owner split and the size of the climb open), #16022 (lever 7: a user `max_split_size_mb` suppresses the fork's `expandable_segments` default; the tester's newer-version plateau), #10963's OOM half (probes only)
- `prefill-dense-mask-a2-a3.md` — #3357
- `prefix-cache-granularity.md` — #10365
- `quantization-load.md` — #12675, #15836, #5603, #4313, #979, #569 (§2c, description file never read on `≤ v0.7.3rc2`), #7507 (§7, a weight variant the ref's tutorial does not list), #3867 (§2d, a nested `llm.` prefix the loader does not strip), #5813 (§3, a `deq_scale` `KeyError` because `--quantization` never reached the process), #15247 (§9, a scheme retired between releases — the registry miss, not a checkpoint defect)
- `reasoning-parser-output.md` — #8022, #9659 (§"Common rules"–§"Apparent TTFT", plus the `KeyError` registry split)
- `sleep-mode-and-allocator.md` — #15560
- `spec-decode-and-parallelism.md` — #10495, #15939, #10405, #15985 (case: `507034`/`507035` on A3 PD with DSpark, fixed in `v0.26.0rc2`), #8916 (§1b, the P/D + MTP stall whose accepted-token mirror was unsynchronised on `v0.18.0rc1`, guard first tag `v0.18.0`)
- `startup-and-stall.md` — #3308; #3843 (§3c, `Available memory: 0` — the profile pass overshot, and the log line dates the image before that rewrite was removed); #5942 (§6, the multi-node DP rank shortfall); #3045 (§7, a cross-tag `init engine` time regression); #12891 (§8, the killed child behind `WorkerProc initialization failed … background process`)
- `startup-port-conflicts.md` — #16237
- `tokenizer-and-chat-template.md` — #6345
- `tool-call-streaming.md` — #11788, #7682, #15911 (§4b, a whitespace-only `content` delta before the first tool call, from the bundled legacy `qwen3_xml` parser), #7580 (#7682's twin, seen from the client as `Unterminated string`), #12398 (§4c, the model's own markup left in `content`/`function.name`), #6765 (§4a, the upstream `glm4_moe` parser's swallowed `TypeError` on a zero-argument call)
- `v1-engine-selector.md` — #2169
- `version-pinning.md` — #2566

Two rows carry no verified mechanism, only probes. **#9355** (`reports-later.md`) is an open,
unsettled first-request-only `completion_tokens: 1` report — the row carries the pinned log line.
**#10963** (`reports-newest.md`) has a maintainer "known issue" and no log at all; its two symptoms (a PD
hang on the decode node, an MC2 OOM on a single A3) are separate levers, each with the flag that carries it
named in the row, and the probes live in `pd-collective-hang.md` and `post-startup-oom.md`.

Three of the batch-11 rows are not vllm-ascend defects either: #8213 is a transformers pin floor on the
checkpoint's tokenizer class (answered by image/checkpoint choice), #8022 is a client reading a field name
the release no longer emits, and #9659 is an open prompt-side question whose parser is stateless. The other
three are answered by a range/ancestry (#8932), a documented artefact (#7507) and a proposed connector flag
with an unsettled native cause (#16342).

`reports-early.md` also gained the two older-era rows of this batch: #569 (§2c of `quantization-load.md`)
and #4180 (§E of `aclgraph-mode-and-capture.md`).

The batch-13 rows: #8788, #10068, #1791, #15911 and #5813 are not plugin defects (build script, harness
measurement, eval command plus parallel algebra, bundled upstream parser, and a flag that never reached the
process), #15985 is a verified release fix, and #10963 is two unrelated symptoms with no localising evidence.
The pages that carry a mechanism for the group are `benchmark-accuracy.md` (#10068, #1791),
`quantization-load.md` §3 (#5813), `tool-call-streaming.md` §4b (#15911), `install-and-image.md` §3 (#8788)
and `spec-decode-and-parallelism.md` (#15985); #10963 points at probes only.

The batch-12 rows: #5299 and #1768 are **not** defects — a `[Usage]` question answered by a build gate and a
dependency floor/version skew answered by an upgrade — while #12692 is a torch_npu-allocator class whose code
half is verified and whose magnitude is open, #1812 repeats #1629 on 310P with the version lever already
known, #7085 is a composition defect fixed from `v0.16.0rc1`, #11947 is a support statement rather than a
bug (the mode is downgraded by upstream), #3867 is a verified `llm.`-prefix lookup whose description shape is
the reporter's check, and #18688 is a release-scope gap closed by #11933.

Cross-topic rows the table points at: #8932 (a same-line `rc` → final accuracy regression) has no class
page — its detail is `../../workflow/fix-discovery.md`, "When the report is a same-line regression". The
teardown frame of #16023 is in `../../workflow/log-signatures.md`; #8579 (PCP without EP) is in
`spec-decode-and-parallelism.md` §2 and #10872 (310P long-context stall) in
`attention-mask/310p-dense-mask.md`.

The batch-14 rows split four ways by what they settle: #5170 and #6421 are **verified at the ref** with a
named fix or a named lever (#5776 / `v0.14.0rc1`; the recompute-scheduler key or a provider whose vLLM is
`>= v0.14.0`); #17718 is a verified build artefact behind a doc gap; #7580 and #12398 are **class**
answers with a `hypothesis` mechanism because neither body names its parser flags — that flag is the one
fact to ask for; #14440 is measured and unowned (probes only); #7401's two halves have different owners and
the maintainer attribution is `reported` because the reporter saw it again after `v0.18.0rc1`; #3062 is a
retired connector whose gating code is verified and whose thread carries no lever. None has a
maintainer-confirmed workaround, so every lever in those rows is labelled.

## Recheck conditions

Re-verify before reusing: the issue states, the tag list (`git tag --list`), and the presence of the
cited commits (`merge-base --is-ancestor`). Release branches exist only as remote refs here
(`origin/releases/vX.Y.Z*`); a bare `releases/v0.23.0` is an unknown revision, so quote the `origin/`
form in commands. The clone advances; a later report may already contain the fix that was absent when
these rows were written. Line numbers and flag gating drift with the clone and are marked for recheck
throughout the reference pages.
