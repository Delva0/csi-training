---
name: vllm-ascend-triage
description: 'Triage vLLM-Ascend issue reports; load alone first'
---

# vLLM-Ascend issue, deployment and support triage

## Before the first shell call

**Your first tool block holds exactly one call: `skill_view` on this skill, by name**
(`skill_view(name='vllm-ascend-triage')`; a **filtered** `skills_list` returns `[]` even
though the skill is installed, while the unfiltered call lists it — so an empty list is
a tool quirk, not "no skill for this task"). Anything batched beside that call — `ls /workspace`, `ls
/workspace/history`, `cat SOURCES.json` — executes before this body is in context, so no
rule here can stop it, and its output is re-sent on every later turn: six recorded runs
paid 39 KB / 14 KB (both forwards of one minibatch, #16365 among them), 54 KB / 28 KB
(#14731, #12102), and 37 KB of directory names from `ls /workspace/history` batched beside
this call in #16581 — which ended output-truncated at 1.1 M cumulative prompt
tokens — and 4.5 KB in #13007, from `ls /workspace/repos` plus the first 100 lines
of `SOURCES.json` batched beside this call (the line cap is the only reason it
stayed that small). That is why the frontmatter description says "load alone first": the
description is the only
text the agent reads *before* this body. If something has to share it, let it be
`bash <skill_root>/scripts/sources-summary.sh` (~10 lines). Then orient with
`search_files (target='files')` on one level of
`/workspace` plus that same summary and nothing
else: never `cat /workspace/repos/SOURCES.json`, never enumerate a history directory,
never `cat /workspace/history/*.md`. Search by token (`rg -l`,
`scripts/history-titles.sh`), never by listing — a recursive `search_files
(target='files')` glob is a listing too: `pattern='*Qwen3*'` over
`/workspace/repos` returned 51 paths, nearly all of them vllm `tests/evals/gsm8k`
configs, while the wanted page was one `rg --files -g '*Qwen3.5*'` run inside
`docs/source/tutorials/models` of the plugin repo. **Attachments are the one exception**:
open the specific files the issue links (`ATTACHMENTS.txt` lists names and SHA256s
only, so a grep of it is not an existence test) — rule 7. Costs and traps: rule 7.

## Problem class this covers

Ten entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter. The last shapes deliver a *decision* (sizing),
a *document review*, a *capability verdict* or a *change plan* instead of a root
cause, but the same rules apply — answer first, evidence pinned to the reported
version, gaps stated.

- **Issue report** (template-shaped: environment dump + logs + narrative) about
  something that went wrong on an Ascend NPU host — a device-side operator crash, an
  accuracy/score gap, garbled or truncated output, a long-run OOM, a container metric
  that climbs for days, a performance claim, a flag or endpoint that does nothing, or
  a `python -c "import <pkg>"` that aborts in a shipped library. When the task is only
  "explain/answer this issue", the answer is still evidentiary, not a summary.
- **Install / pip-resolution report**: no engine, no worker, no NPU traceback — two
  `pip install` lines and what they printed (`Could not find a version that
  satisfies the requirement …`, `No matching distribution found for <pkg>`,
  "unmatched torch version between vllm=X and vllm-ascend=Y", an `ImportError:
  Please install vllm[bench] …` inside a shipped image), or an install that finished
  and only **warned** (`requires torch==2.10.0`). The deliverable is *which* artefact
  pip could not see — a mirror-only pin the metadata cannot carry, an unsatisfiable
  torch/torch-npu/torchvision set, an air-gapped container with no reachable index —
  never a release survey — `references/image-dependency-faults/pip-index-visibility.md`
  and `pip-resolver-warning-block.md`.
- **Deployment failure**: engine start or a running worker fails — `EngineCore failed
  to start`, `WorkerProc initialization failed`, silent worker death, allreduce /
  `ERR01100` / `ERR00100`, OOM, "the model failed to load", an `AttributeError` /
  `TypeError` from a patched forward or a KV connector, or a `ValueError` from
  upstream `arg_utils` rejecting the parallel/DP flags.
- **Support / version question**: "does vllm-ascend <version> deploy <model>?" — a
  version gate with a one-line answer and a registry-level proof, not a bug hunt.
  When the follow-up is "支持计划是什么样的 / when will F work for M on my SoC", answer
  in three separate levels — the base model family at that SoC (release-note line,
  support-matrix row, the SoC fallback `TODO` that bounds it), whether the *feature's*
  code branch exists (one grep), and whether anything covers the combination (tests,
  tutorials) — then the prerequisites and the grep whose result would change the answer,
  and no date (`references/feature-enable-recipes/index.md` item 9).
- **Feature-enable / capability question**: "如何正确开启 <feature>", "F 单独开均有不同的
  报错，能否给出能跑通的示例", "does F work on model M on this SoC", usually with no log
  and two errors in prose. The deliverable is the validation verdict at the reported
  ref plus the constraints the code enforces, not a launcher assembled from them —
  `references/feature-enable-recipes/index.md`.
- **PD / EPD disaggregation**: the deployment is up but P and D cannot move KV
  (`HIXL Connect failed | 103900`, HCCL `RanktableCheck` / `'device_ip' is not set
  correctly`, `wait_for_save failed`, `Failed to put N keys out of N`, a recipe that
  needs `hostNetwork: true`), or KV moves and the decode node emits wrong/alternating
  output on a hybrid linear-attention model (GDN/Mamba — but a **DSA/MLA** model such
  as GLM-5.2 that repeats one phrase to the token cap is a recipe-contract question
  first: `references/pd-disaggregation/recipe-contract-diff.md`), or a KV pool is attached and the *first*
  transfer fails (`503900` / `HcclCommPrepare ret: 0x13`, the pool's
  `global_segment_size` the cause), or the put never lands (`HcclBatchPut,
  hccl_result = 4`, `PutEnd` 0), or the decoder never receives the request
  (`proxy通信超时`) — an endpoint-count question before any connector read; transport
  identity and rank tables are a topology question, a pool deployment a config
  question — `references/pd-disaggregation/index.md`.
  When the transport lines are Mooncake's (`ascend_direct_transport.cpp:820 Failed to
  connect to target … 103900`, `mooncake_connector.py:410 Mooncake transfer failed …
  ret: -1`) rather than HIXL's, the decoder's plog is the first request, not a
  port/`hostNetwork` test: cross-node P/D is already multi-server, so the requirement is
  a `device_ip`, and `use_ascend_direct` (default `True` at v0.11.0rc2, engine-name
  only) is not the fix — `references/pd-disaggregation/mooncake-adxl-connect-103900.md`.
  If instead the engines start and the **first request** kills one side's EngineCore
  behind a bare `EngineDeadError: … See stack trace (above)` on a deployment setting
  `--decode-context-parallel-size` > 1: DCP is a *matched* parameter (a 1M recipe's
  `8` reduced on one side fails the connector's divisibility assertion), so compare
  both serve commands before any source walk —
  `references/feature-enable-recipes/context-parallel-pcp-dcp.md` §3.
  If instead requests *are* served and TTFT is slow while the prefiller prints
  `Some requests in prefill node fail to receive KV Cache transfer done signal in 120s`
  (`VLLM_LLMDD_ABORT_REQUEST_TIMEOUT`, default 120): that line is expiry bookkeeping
  emitted by design on the non-zero P `tp_rank`s — the advertised RPC port carries no
  `tp_rank` — so it is not the slow-TTFT mechanism. Settle TTFT with the concurrency and
  first-request discriminators before any connector or rank-table read —
  `references/pd-disaggregation/llmdatadist-done-signal-and-ttft.md`.
  If instead the **decode node logs `Error in KVCacheTransferThread. error=unhashable
  type: 'list'.`** and the deployment keeps serving: that line is the recv thread's catch-all
  printed *in place of* the real failure — downstream `_mark_failed_recv_request` feeds
  `set[int].update(<list[list[int]]>)` a grouped `BlockIds` before the `logger.exception` that
  would have named it — so the paste carries no transport information; unmask (flatten, or the
  layerwise connector, which already flattens) before any HCCL, plog or rank-table read
  (`references/pd-disaggregation/kv-transfer-thread-masked-error.md`).
  If instead the same catch-all carries a **bare** `AttributeError` on a downstream
  metadata field — `Error in KVCacheTransferThread: 'ReqMeta' object has no attribute
  'block_ids'`, with no environment block in the issue — the pin's own class **has** the
  field (a compat property over the renamed `block_ids_by_group`), so the line reports
  the reporter's *build*, not a reader bug: diff `vllm_ascend`'s install path and that
  one class's source on P and D before any source walk (reporter-confirmed #10398,
  「P和D节点没注意统一代码版本」) —
  `references/pd-disaggregation/node-build-skew-attribute-error.md`.
  If instead **both engines run on one machine** and the second one to start never gets
  past worker init — `Bind_Failed(EJ0003) … port have been bound already` +
  `hcclCommInitRootInfoConfig … error code is 7` out of `w8a8_dynamic.py`'s
  `get_hccl_comm_name`, plog `[HcclSocket][Listen] … port[16666] fail`, "start P first,
  D crashes; start D first, P crashes" — the port is real and the collision is a shared
  **card**: intersect the two launches' device sets (the external-DP launcher numbers
  cards from 0 every invocation) before any `HCCL_IF_BASE_PORT` / control-plane
  arithmetic — `references/pd-disaggregation/same-host-pd-device-partition.md`.
  If instead **one side's EngineCore dies on its own bookkeeping** and the other keeps
  serving, name the side and the owning structure from the last three frames before any
  transport read: a prefiller bare `AssertionError` in upstream
  `v1/core/sched/scheduler.py` `_update_from_kv_xfer_finished`
  (`assert req_id in self.requests`) is a send-completion for a request the scheduler
  already freed, and `step_with_batch_queue` in the same stack proves
  `--async-scheduling` (or PP>1) — one launch-flag ablation, no connector walk
  (`references/pd-disaggregation/prefill-scheduler-kv-xfer-assert.md`); a decode
  `assert … chosen_rank_list(…) does not match num_group_pulls(…)` on P TP≠D TP with an
  MLA/DSA model is one pull-count derivation skipping the MLA single-pull rule
  (`references/pd-disaggregation/asymmetric-tp-pull-count.md`); and a decode
  `IndexError: list index out of range` inside `_get_kernel_block_ids`
  (`meta.remote_block_ids[kv_cache_group_id]`, `_get_kv_split_metadata` above it) is
  a KV-cache-**group-count** mismatch — the decode side indexes the prefiller's list
  with its own group ids — so diff the two `--speculative-config` blocks (a draft
  model on one side only adds a group) before any connector or transport read
  (`references/pd-disaggregation/kv-group-count-mismatch.md`, reporter-confirmed #16177).
  If instead the pair **serves** and then individual *decode* ranks keep their
  requests `Running` while generation and completion counters stop — health 200, no
  traceback, one rank taking all the new work, affected ranks recovering hours
  later — the KV blocks are being freed and reused under an in-flight step on an
  `--async-scheduling` + hybrid-KV (`--no-disable-hybrid-kv-cache-manager`) serve:
  `references/scheduler-admission/kv-block-lifetime-async-scheduling.md`
  (#13964 → vllm-ascend #13518), not a collective and not the throughput topic.
- **Deployment sizing / stage layout**: the question is a plan — "does this model
  fit on 2×64 GB", "the GPU docs do a 2-device stage deploy and mine does not come up
  on 2 NPUs", a bundled `platforms.<vendor>` yaml with more stages than the reporter
  has cards, "how do I pin stage 1 to one card" — **or** two serve commands that
  differ by one resource knob (`--tensor-parallel-size`, `--max-model-len`,
  `--gpu-memory-utilization`) where the smaller one comes up and the larger one OOMs
  at engine start — **or** a KV number against the reporter's own arithmetic ("27
  KB/token/card, I compute ~7.4 KB/token from the fp8 KV layout") and a context
  target ("a path to ≥512K on A2?"). Answer with per-card arithmetic plus the
  device-visibility variable, and for the KV half reconcile the cache-kind term list
  and quote the tutorial's per-SoC envelope before accepting a dtype-promotion
  hypothesis — `references/deploy-sizing/index.md`, `references/deploy-sizing/dsv4-kv-footprint-terms.md`.
  A report that the **whole model is loaded on every card instead of being split** is
  the same topic and is *not* answered by `npu-smi`: the per-rank weight-footprint line
  against `params × bytes ÷ TP` decides it —
  `references/deploy-sizing/sharding-not-applied-full-model-per-card.md`.
- **Community-test / doc-verification task** (`[Community Test]`, a 众测任务书 row
  with 蓝区镜像 / 权重链接 / 卡数 / 交付件 测试报告, or a `[Doc]` report that a
  tutorial step does not work as written): the deliverable is a *report about a
  document*, graded on completeness/usability/correctness without the reporter's
  hardware. The evidence is the pinned tree — which command no longer parses, which
  path is absent from the image, which baseline predates the image, whether the card
  count matches the weights — `references/tutorial-verification/index.md`.
  When the reported defect is an **absence** instead ("the Quick Start should
  mention a mirror / link the FAQ", "the contributor docs never state a
  prerequisite"), three greps decide it — does the content exist, does any page
  reach it, do the sibling pages already carry it — and the reporter's own URL
  anchor is checked against the pin before it is quoted
  (`references/tutorial-verification/missing-guidance-and-contributor-docs.md`).
- **Release known-issues bulletin**: a maintainer's notice for one release —
  `[Bug][vX.Y.Zrc1]: Known Issues for <model A>/<model B> service hang on A3`, an
  impact table (version / models / hardware / scenario), one or two tracked issue
  ids with error codes, "reply in the comments instead of opening a new issue" —
  or a user quoting that notice and asking whether their run is the same thing.
  No log, and none is owed: the deliverable is a **claim-check at the pinned
  release** — which signatures the notice pins and what they localize, what they
  cannot settle (mechanism, event ordering), whether each code is even
  model-specific, whether the reported config is inside the release's validated
  envelope — plus one ranked list per tracked issue and the reply template
  (plog `[AIC_INFO] fault kernel_name=` block, both serve commands).
  `references/release-known-issues/index.md`.
- **Code-change / cleanup request**: the body asks for an *edit*, not a diagnosis —
  "remove `VLLM_ASCEND_ENABLE_<X>` before <release> final release", "delete the
  temporary switch added in PR #N", "deprecate this flag", often with nothing but two
  links and no log. First question: does the named identifier exist at the pinned ref
  (absence dates the snapshot against the PR and is itself the finding); deliverable:
  the removal sweep with `path:line` anchors plus the behaviour change a default flip
  implies — `references/code-change-requests/index.md`, then `env-flag-retirement.md`.
  The delivery gates apply here too: no crash to report, so the absence verdict goes in
  the first message (recorded #1262 replay: first message text on turn 15).

Recorded cases, per-issue index and how each session ended:
`references/evidence-workspace/recorded-cases.md`; the load-bearing
delivery pattern is rule 1.

## Rules that apply whenever this skill is used

1. **Say the answer in a message, early, and obey the delivery gates.** A long
   session can be cut off mid-sentence, so the deliverable has to survive
   truncation: what decides the outcome is not how much evidence was gathered but
   whether the message exists. The runs that shipped **no answer at all** each had
   a usable lead in hand (#6784, #13853, #11454, #11095, #17991, #3107, #6266, #5571, #527,
   #587, #11680, #1579, #11771, #7947, #7508, #10522, #1255, #5772, #6488, #7996 — #1255 and
   #5772 with the owning line already in the paste (#1255's plog names the kernel and its owner is one grep away;
   #5772's dumped `SchedulerOutput` carries the two counts, and the deciding fact is a
   model-directory log line); #2807 and #15394 held every verified fact of a complete
   ranked answer and spent the remainder on a host detail no local file settles, i.e. the
   `quant_mode`/`chunk_finish` arithmetic their own pages now replace with one print and one
   DEBUG line; per-run table `delivery-discipline.md`), while #4296
   answered in ten turns from one traceback. The two thinnest are also the most recent:
   #6488 wrote **81 characters** of message text over 21 turns / 39 tool calls against
   40.6 K characters of reasoning, and #7996 1 540 characters over 11 turns against
   48.9 K — each ending on a reasoning block with no accompanying message and no tool call,
   i.e. the cap landed on reasoning that had the shape of the deliverable.
   Hard stops:
   - **G1 — 3rd assistant turn:** message text with the direct answer to the
     reporter's literal question plus the track/lane classification. A pasted
     traceback, one config value or one registry listing is enough; do not hold
     the answer for the mechanism.
   - **G2 — 8th assistant turn:** the full ranked answer (each cause with its one
     check, cheapest action, version scope, missing artefacts). Afterwards a tool
     call is allowed only to confirm or correct a named claim: one per turn, a
     couple of lines of reasoning.
   - **G3 — 12th assistant turn:** answer text only; no reasoning, no tool call,
     no "one more thing".
   - **The gates cap cumulative output (~45-50 K characters of reasoning+text), and
     the turn index is the weak part: your own count drifts 2-3x low.** #6529 wrote
     "I'm at turn 7" on its 15th turn, #6243 "I'm at 6" on its 14th, so G1-G3 never
     fired as written. Trigger on evidence instead of the number: the turn you can
     state a cause with its one check is the turn the answer is owed; if you do
     count, count the `tool` results in the transcript, not from memory. A verdict
     line is a *hypothesis*: the ranked deliverable is still owed as one message, and
     a block that restates the verdict, numbers the fixes or lays out the evidence
     table *is* this turn's message: send it, then one tool call
     (`references/evidence-workspace/delivery-discipline.md`).
   - **Reasoning is a scratchpad, a few lines per turn:** what the last result
     proved, the single next check, go. It spends the **same output budget** as the
     answer, before any of it becomes text (ids in `delivery-discipline.md`). If a
     block enumerates the answer's sections, counts turns against G1/G2 ("this is
     turn 5, G2 is turn 8"), or says "I have enough" / "let me write the answer",
     that block *is* the answer — and a turn that reasons "the gate has passed, I
     must send the deliverable" while sending empty text and another tool call is
     the recorded way to ship nothing (#6243: 15 turns, 67 characters of text, cut
     inside a 14 K character block that held the whole ranked answer). The
     #14731/#12102 pair shows the arithmetic: 42.9 K and 39.4 K characters of
     reasoning against 1.3 K and 2.7 K characters of delivered answer, both cut at
     the 16 384-token cap — the bigger run spent an 11.8 K character block deriving
     a parser mid-session and then re-wrote the whole ranked answer in a 6.2 K block
     before typing it. Both forwards of one later minibatch did it again with the
     answer already in hand — #7150 at 51.8 K characters of reasoning against 756 of
     text, #18754 at 59.1 K against 1.7 K, each ending on an **empty** final message.
   - **Budget in characters per block, not turns** — the count-free form of the rule
     above. #13527 hit the cap with no draft at all: 16 turns, 26 `tool` results,
     **43.8 K characters of reasoning against 2.2 K of message text**, nine of its turns
     opening with a self-dated count of its own turns ("I'm at turn 5", "turn 8",
     "turn 10") that was 2-3 low and closing with two more tool calls. #587 spent 52.6 K
     characters in **7** assistant turns and shipped 800 characters of text, #11680
     51.2 K in 10 turns and 2.8 K: G1/G2 never fired because the turn index was low
     while single blocks ran 8-12 K characters. So test the characters, not the count:
     **message text ≥ a third of the reasoning you have written so far**, and keep each
     reasoning block short (~1.5-4 K characters) — what the last result proved, the one
     next check, and this turn's answer section typed rather than "one more check".
     That last ending is recorded twice more: #11680 held its decisive datum on turn 8
     (the pinned tree's own `tests/e2e/weekly/single_node/configs/GLM-5_1-W8A8_A3_weekly.yaml`
     matching the reporter's launch on every flag except `TP16/DP1` + `--async-scheduling`)
     and wrote "One last check on the replay-ordering guard" instead of the answer; #587
     had the owning line (`glm4.py:203` assign-a-tuple) on turn 4 and spent three more
     turns re-deriving the corrected math. Once a message can name the discriminator, the
     **next** message is the ranked answer built on it — a further read is a turn whose
     text, if there is one, cannot carry the answer after a truncation.
   - **The paste can already contain the whole answer surface.** An argparse error
     prints the complete valid choice list; a log line can name its own thread and
     owner. When the evidence is one such line, the verdict is owed at G1 with no
     source read: #4796 had `--task rerank` vs the printed `score` by its 4th turn,
     spent 32 turns and 1.77 M cumulative prompt tokens before the cap, and emitted
     **no message text at all**; the same issue's pooling release gate (v0.9.1 🟡
     Planned, v0.9.2rc1 #1359) was two greps from the same paste. A traceback can
     also name its own owner and rule: `'DummyModelLoader' object has no attribute
     'get_all_weights'` (#2346) names the loader class *and* the method that only
     `DefaultModelLoader` has, and the MLA `queries per kv … only support {32, 64,
     128}` assert (#1451) states the rule and tells the reporter to check the ratio
     after the TP split — two greps plus arithmetic close both. Its minibatch
     sibling #10962 reasoned the whole source-verified answer and typed it on the
     last turn, cut mid-sentence. Test at every turn: could I write the first
     paragraph now? — if yes, the tool call goes in a message that already carries it.
   - **A new hypothesis that needs a new code area is not this session's job.**
     Put it in the answer as an open question with the one command that decides
     it; #14506 delivered only after a final block that took half the session, and
     #2934 read the `-inf` id pattern that decides its own lane on turn 8, then spent the
     rest of the budget on how a mask reaches the logprobs and shipped nothing
     (`references/accuracy-benchmarks/topk-logprob-negative-inf.md`). For
     an accuracy/corruption or topology report, a ranked list plus an ablation
     matrix already *is* the full answer. The same applies to a **mechanism the
     paste cannot carry**: *which* scheduler step mixed decode with prefill, or
     *which* phase refused a communication resource, is a fact about the reporter's
     run that local source reading cannot produce. Both forwards of one recorded
     minibatch spent their entire budget there after the answer existed — #1369
     held the pinned tree's own `only work without chunked-prefill` doc and CI
     comments while it derived mixed-batch reachability (46 messages, 13 K- and
     19 K-character blocks), #5947's paste carried no Python frame and no
     `LoadGraph failed` block at all while it re-derived the MC2 capacity — and
     neither shipped message text. The same shape with a *confound the report
     never carried*: #8350 held its two serve flags by turn 3 and then treated the
     reporter's `--block-size 128 --enable-prefix-caching` as a hybrid KV-cache
     experiment — one command, no works-vs-fails control, so there was nothing to
     grade — and shipped 185 characters of text in 16 turns
     (`references/accuracy-benchmarks/reasoning-output-routing.md`, #8350 section).
   - **A paste that stops in silence is answered, not investigated.** When the log
     simply ends (no traceback, last line inside init), the cause is decided by
     artefacts on the reporter's machine — process count, per-rank logs, device
     mapping — never by more source reading, so the ranked list plus "send these
     three" *is* the message and is owed on the turn the shape is classified.
     #1496 spent 12 turns and ~49 K characters of reasoning across four mechanisms
     (HCCL env, the gloo `cpu_group`, a `all_reduce`, "all ranks picked npu:0") and
     shipped **zero** characters before the cap;
     `references/startup-failures/multiprocess-launch-hang.md` holds the cheap
     discriminators that replace that walk.
   - **An ownership question is answered by naming the owner.** "The flag did
     nothing" and "the log line is missing" are gate checklists on the owning
     component — usually the upstream vLLM OpenAI frontend, not vllm-ascend — and
     take 3-4 calls; #10985 and #9583 each spent 20-30 calls inside the owning
     module with the decisive check in hand.
   - **When the deliverable is a document, the document is the gate.** Open the
     report file on the 2nd-3rd turn with the skeleton plus the first verified
     section, then append **2-3 K characters per write** and make the final
     message the condensed report (path, defects, open items). Never compose the
     whole report in one heredoc: #11739 wrote no file, #11759 lost a finished 5.7 K
     report to the cap, #11742 opened its heredoc on turn 9 of 10 and shipped **no
     text** (`community-test-report.md`). The file body and the message share one
     budget — heredoc characters are tool-call arguments, i.e. output tokens: #11751
     wrote a 15.4 K-character report in three appends (16.1 K characters of tool
     arguments, 36 % of the cap) and then had its **message** cut inside the defect
     table, losing the three required 完整性/易用性/正确性 scores and the FAQ. Keep
     the file ≤ ~8 K characters and lead the message with the 交付件 items (verdict,
     report path, the three scores, one line per defect) instead of re-pasting the
     deploy script or the tables.
   - **`/workspace/scratch/findings.md` is the answer draft, not an extra step:**
     write it at G2 *in place of* the reasoning block; once it exists the next
     turn's text is the answer (`delivery-discipline.md` §draft).
   - **A missing or mismatched artefact is itself the finding** — an empty
     environment block, a file absent from the pinned tree, a version the snapshot
     does not carry. State the gap, give the one command that closes it, stop.
   - **A byte count that "matches" is not the answer.** Re-deriving a mechanism from
     an allocation size is the most expensive way to spend the budget, because the
     decode needs inputs the report does not carry (a `config.json`, a group size,
     an expert count). #7123 had 30 tool results, four assistant turns with **no
     message text at all**, and died at the 16 384-token cap while re-deriving
     upstream DP/EP group algebra to explain a 674 MiB request — the two log lines
     that settle it (`[EP Rank r/N] … Local/global number of experts`,
     `rank … in world size … assigned as … EP rank …`) were never asked for, and what
     the reporter later published as the resolution was a serve line (the cross-node
     `--data-parallel-*` flags + `gpu_memory_utilization`), not a mechanism. Ask for
     the line; send the ranked answer
     (`references/deploy-sizing/moe-load-oom-expert-shard.md`). **The same holds for a
     tensor shape quoted from a debugger** (`变量 x.shape [5040, 1, 1152]`): it is a
     gap in the paste, not the failing allocation, and which line produces it is a
     confirmation, not the answer. #7419 spent 19 tool turns (punica `bgmv`/`sgmv` and vision-attention
     reads) to locate the producing `.unsqueeze(1)` and shipped **zero**
     message text at the cap, while the two facts that *were* the answer — upstream
     `vllm/lora/models.py:613-624` drops vision-tower adapters, and the middle `1` is
     the Ascend VL tower's own batch axis — came from two greps
     (`references/lora-adapters/multimodal-vit-lora-ignored.md`).
   - **Truncation eats the tail, not the head — and the body has a size too.** One minibatch showed both halves (#1255 typed no message text across 24
     tool turns and shipped nothing; #3170 kept its verdict and ranked causes but lost the
     action ladder's last commands), so type the ladder before the scope block.
     Each action directly after its cause, the closing scope block 3-5 lines. With
     no log pasted: 3-4 causes with one check each, one ladder of ≤6 lines,
     precedents as one line of `#id: fact`, 3-5 lines of scope (≈3-4 K characters);
     a fifth cause is an open question with its one command.
   - **Every assistant turn carries message text, not only a tool call.** A turn
     whose whole output is a tool call leaves nothing for a truncation to keep.
     #5295 emitted *zero* message text on all 16 turns and #16563 on all 20; #17797
     and #5525 emitted only status notes and never a verdict; #17616 ran 26 turns of
     a three-question report (42.6 K characters of reasoning, 25 tool calls) and
     shipped **79** characters of text on its first turn only, cut at the
     16 384-token cap with every fact it needed already collected. Both forwards of one
     recorded minibatch repeated the shape: #2349 reasoned the entire ranked answer
     into a 6.3 K-character block and shipped 386 characters before the cap, and
     #6018 emitted nothing on its first 13 turns and wrote the answer on its last
     one. Carry one or two sentences of the
     current answer draft in the same message as every tool call — verdict line
     first, then the evidence being checked — so G1/G2 need no special turn. A
     *status note* is not a draft: the sentence must state the current verdict even
     when the next call may overturn it. #8370 is the pure form: 14 turns, ~49.5 K
     characters of reasoning, zero verdict text - every message a next-action
     narration ("Now orienting with…", "Let me pin both sides", "Two source-verified
     facts now…") while its last turn held the answer. The mechanical test at any
     turn: no message text yet names a cause plus a `path:line` or `#id`, so nothing
     has been drafted.
2. **Classify before investigating.** For a startup log, place it in lane A/B/C;
   for a runtime report, decide whether it is a device-op fault, an accuracy/score
   gap, a corruption report or a performance claim. The signature index
   (`references/first-checks/signature-routes/index.md`) routes to the page holding the
   row for a specific signature; the track-level routes are the table in
   `references/dispatch/classified-shapes.md` (*Support entry points*). The
   classification decides which code and which evidence even matter. Do not open
   a deep source walk when a config value, one re-run flag or a health tool can
   settle the question.
3. **The deliverable is a ranked diagnosis the reporter can act on**, not a code
   walk: per hypothesis, the evidence you have, the one command or log line that
   confirms or refutes it, the cheapest fix to try, and what to send back.
   **A repair that only removes an exception is not a fix.** For a downstream copy of a
   library body, a `getattr`/`try`/`except` guard starts the service and leaves the
   numerics unvalidated (reporter-confirmed #11991: the `attention_type` guard made
   `vllm serve` come up and broke image recognition), so name the structural repair and
   state the acceptance check — one real request whose output is compared with a working
   run — rather than offering the guard
   (`references/upstream-downstream-drift/patched-forward-api-drift.md`, rule 6 for the
   claim label).
4. **The reporter's machine is the only environment here.** You cannot inspect
   it, and local NPU state (healthy, absent, a different card) is not evidence
   about it. Prefer checks the reporter can run — config value, env var, health
   tool, one re-run with a flag — and when the report is hardware-shaped, name
   the artefact that settles it (plog, tiling log, raw request/response dumps,
   `npu-smi info`). When that artefact **is** a process on a card ("an extra
   worker process from another card appears on a certain NPU"), it is the PID's own
   `cmdline` and `/proc/<pid>/environ`: a stray PID is attributed before it is
   explained, and an engine that keeps serving is exonerated by the attribution
   rather than by an argument (`references/deploy-sizing/node-processes-and-device-attribution.md`,
   reporter-confirmed #12338).
5. **Pin evidence to the reported version.** Read `/workspace/repos/SOURCES.json`
   before citing code: it records repo, ref, commit, basis and coverage cutoffs.
   Every code claim gets `path:line` plus the commit/ref it came from. State
   recheck conditions when a constant, table or log line is version-specific. A
   report citing `docs.vllm.ai/.../latest/…#<section>` is quoting a doc that may
   postdate the pinned release: check that section exists in the pinned tree,
   answer inside the pinned version, and if it is absent say so and ask for the
   text rather than assuming what it says. `latest` is the Read the Docs build of
   main (`READTHEDOCS_VERSION`) and each release is `/en/vX.Y.Z/`, while the tag a
   page prints is substituted at build time from `mkdocs.yml extra.*` — so a
   tag/flag disagreement is graded against the docs build with those two greps,
   not by hunting the upstream `Literal` that owns the value
   (`references/tutorial-verification/docs-version-space-and-image-tag.md`,
   #16954). **When the title names one release and
   the body's launch log another** (recorded #11231: title v0.21.0rc1, body
   v0.22.1rc1; #18315: title v0.27.1RC1, env v0.26.0rc1), say which one you
   verified and give the one command the reporter runs on *their* tree
   (`sed -n 'A,Bp' <file>` around the cited line) instead of asking for another
   launch — and when a **newer pin in the snapshot already carries the repair**,
   quote it: that settles "is it fixed upstream" without PR or comment access.
   A reference page states the pins it was
   verified at: when those equal the refs in `SOURCES.json`, its `path:line` are
   citable as they stand — re-grep only what the page itself marks as
   version-specific, and spend the saved turns on the delivery.
   **A dependency's patch release is part of the version contract.** `torch_npu`
   (pta) `2.8.0` leaks memory and `2.8.0.post1`/`.post2` fix it — the pinned tree
   says so itself (`docs/source/user_guide/release_notes.md:197` at v0.13.0rc1,
   under `## v0.11.0rc1`: "The memory may be leaked … This is a bug from torch-npu"),
   reporter-confirmed on #6299 (`--no-enable-prefix-cache` changed nothing, the
   `.post2` install plus a vllm-ascend rebuild ended it) — while v0.13.0rc1's **own**
   `requirements.txt:31` pins `torch-npu==2.8.0`, so an install that matches the
   release table exactly can still carry the defect. Read the env dump's `torch_npu==` against the patch
   releases before attributing a symptom to vllm-ascend code, and keep the mechanism
   open when only the fix is confirmed
   (`references/runtime-memory/host-ram-and-per-rank-rss.md` §Step 0).
   **A self-built vllm beside an rc vllm-ascend is a version-contract defect even
   when the torch line matches.** #488 pasted `vllm: 0.7.2.dev` + `vllm_ascend:
   0.7.1rc1` (installed from an older page's manual-compile recipe) while the rc's
   own pair is vllm `v0.7.1` (`docs/source/conf.py:65`); the recorded answer
   excluded the version lane from the matching `torch_npu` wheel, ranked a device
   fault first, and the issue closed on the maintainer's "you are using a old
   version vllm / vllm-ascend / torch_npu … install 0.7.3rc2" plus the reporter's
   "i tried latest version (vLLM Ascend 0.7.3rc2) and the problem solved." With a
   mixed or several-rcs-old trio, make the install-and-re-run **rung 1-2** of the
   action ladder and keep the mechanism below it — a snapshot pinning one commit
   cannot hold a later fix, and the upgrade is the only check that can see one
   (`references/startup-failures/environment-correspondence.md` §*A self-built
   vllm beside an rc vllm-ascend*).

   **The same applies to an old image tag with no mismatch in it.** When the tag in
   the report is two or more rcs (or ~a month) older than the analysis date, the
   snapshot predates any fix by construction, so for a **garbage/degenerate output or
   a crash on a recently registered model** the tag is rung 1: one line in the first
   message — "re-run the same body on the current image; `<tag>` predates the fix" —
   and the ranked mechanism below it for the case where the re-run still fails.
   #3806 pinned v0.11.0rc0 while rc2/rc4 were current; the same `curl` body on
   `quay.io/ascend/vllm-ascend:latest` returned the correct table (reporter-confirmed),
   and the recorded session never offered the upgrade
   (`references/accuracy-benchmarks/vl-vision-tower-degeneracy.md` §*Fourth pin*).
   Do not read this as licence to demote a real mechanism: quote what the pinned tree
   *does* carry (the registration, the dead lever, the coverage absence) and say plainly
   that "is it fixed" cannot be answered from the snapshot.

   **Read the direction of the comparison first — the tag rung above is one-directional.**
   It applies when the **older** tag is the broken one. When the reporter's own control
   runs the other way (old tag structures the document correctly, new tag garbles it, same
   weights/card/command — #4730: `v0.11.0rc0` good, `v0.11.0rc1` "html结构都错乱"),
   upgrading cannot help: they are already on the newer side. #4730's public follow-up
   (2025-12-05) also has rc2 **and** rc3 failing, so "the next rc fixes it" is not a rung
   either. Then the first message owes three things instead: the zero-restart request
   arithmetic (`finish_reason` / `usage.prompt_tokens` against the tutorial's image-token
   envelope), the one-command stack dump taken **inside each image** (a tag moves vllm /
   torch_npu / CANN / transformers together, so the delta is not yet a vllm-ascend claim),
   and the within-image one-restart rows (`USE_OPTIMIZED_MODEL=0`, `--enforce-eager`, a
   text-only request, the checkpoint's dtype against the engine's) —
   `references/accuracy-benchmarks/harness-vs-engine-attribution.md` §*When the older image
   is the good one*, `references/accuracy-benchmarks/vl-vision-tower-degeneracy.md`
   §*Fifth pin*.

   **A matching trio does not close the version lane when the pin is an rc and the
   trigger is a feature crossing that pin never tests.** An accuracy/numeric report
   whose own controls are "feature A off: fine, feature B off: fine" points at the
   intersection, and a fix that lands after the pin is invisible in the snapshot —
   so the *next rc/final* is rung 1, not the mechanism. Recorded #12060
   (`v0.20.2rc1` image, PP2+TP2 + chunked prefill, 首token wrong) closed on the
   reporter's "tried the latest version, it works fine", while the recorded run
   spent its whole budget deriving the PP token-id path
   (`references/pipeline-parallel/pp-chunked-prefill-numeric.md`).
6. **Label the strength of every claim**: reporter-confirmed > verified in the
   pinned source (`path:line` at commit X) > precedent in the issue history
   (quote the issue id) > unverified hypothesis / recalled tool default. Never
   present a recalled default (sampling params, tool flags, CANN behaviour) or a
   matching-looking number (tensor count, hostname, line number) as proof. A
   maintainer's clean run on different hardware is a data point, not a
   refutation of a reporter-confirmed fix on theirs; say what was verified on
   which hardware.
7. **Keep reads small and exact.** Use `rg -n -C2 <symbol>` and
   `sed -n 'A,Bp' file`; reserve whole-file reads for files under ~150 lines. A
   20-40 KB full-file read of an AscendC kernel header buys one fact and costs
   the budget for several conclusions. Two traps that each cost a recorded run
   whole steps:
   - `rg -r` is `--replace`, not "recursive": in any short cluster whose first
     letter is `r`, the rest of the cluster is the replacement string, so `-rn`,
     `-rl` **and `-rln`** rewrite every match to `n` / `l` / `ln`. Output then
     looks like corrupted source (`self.n`, `decoder_layer.l`) or, worse, like a
     real upstream defect: #14423's `rg -rn "daocloud|nju.edu.cn"` turned
     `m.daocloud.io` into `m.n.io`, and #17718's `rg -rln 'set_env.sh'` turned
     `source …/ascend-toolkit/set_env.sh` into `source …/ascend-toolkit/ln` in
     `faqs.md` — which it nearly published as a doc typo and dropped only by
     luck, blaming a "stale .po". **The tell is one short token spliced
     uniformly into every match across unrelated files**; re-run as
     `rg -n '<pat>' <dir>` before quoting any matched text. Recursion is implicit
     on a directory: use `rg -n "pat" .` or `grep -rn`. **An empty replacement is
     the same flag with a worse tell:** a replace flag prints whole matching *lines*,
     so `rg -n "30566" -r '' .` returned **63 KB** — one minified
     `tests/models/fixtures/*.json` line, matched by accident inside a number — and
     that run (#8036) ended output-truncated with no answer typed. A "hit" whose
     result is a single gigantic line is this trap, not a finding: quote only the
     `path:line` you asked for, with `-l` or `--no-messages -c` when you only need
     to know whether it exists.
     An **empty pattern matches every line of every file**: a probe typed while
     unsure how to recurse — `rg -rn ""`, `rg -n "" .`, `grep -rn "" .` — dumped
     65 KB into one recorded session (#6240, mid-investigation, buying nothing).
     To list files use `rg -ln '<pat>' <dir>`; with no pattern in mind use
     `rg --files -g '<glob>' <dir>` (or `find <dir> -maxdepth 1 -name '<glob>'`) —
     never an empty pattern, which is still what #549 typed while looking for an
     upstream selector and cost itself 65 KB in one result.
   - **A pasted message is usually split across source lines.** C++ `OP_LOGE` and
     Python messages are emitted as adjacent literals (`"… default chat template is no
     longer "` / `"allowed …"`), so `grep -rn "<whole sentence>"` returns nothing and reads
     as "not in this tree" — grep the shortest distinctive clause, or add `-A1`. #6345
     spent three calls that way, on the exact sentence it was looking for.
   - **A `--glob '!<dir>/**'` exclusion is matched against the path you passed,
     not against the search root.** `rg -l 'x' docs/source --glob '!locale/**'`
     still searches `docs/source/locale/**` (verified at v0.22.1rc1: one .po hit
     survives), while `--glob '!**/locale/**'` or `cd docs/source` first excludes
     it. A failed exclusion floods the result with gettext `msgid`/`msgstr`
     lines that read exactly like doc prose — and `.po` files are machine
     translations whose `#: ../../source/x.md:N` references lag the source, so
     neither their text nor their line numbers are citable.
   - **A piped `sed -n 'A,Bp' file | rg -n '<pat>'` numbers the stream, not the file.**
     `rg` counts its stdin from 1, so every match line is `A + n - 1` — an off-by-one
     that lands as a wrong `path:line` in the answer. Either cross-check one match
     against an absolute `rg -n '<pat>' file` before quoting the rest, or keep the
     file's own numbers: `rg -n '<pat>' file | awk -F: '$1>=A && $1<=B'`. Recorded
     #12905 re-derived a tutorial block's anchors twice (`sed | rg -n` then `1199 +
     index`) before one absolute grep confirmed them; the window form would have
     answered in the same call.
   - **Scope a grep to one repo.** A case view holds 20-40 exported trees under
     `/workspace/repos/<issue>/`; an unscoped `rg -n '<string>' .` there takes
     seconds and returns hits from unrelated products (torchair, metadef,
     ascend-transformer-boost). #11454 ran three of them looking for one log
     string, and the finding was that the string is in **no** export — state that
     as the gap (rule 10) instead of widening the pattern: `rg -n '<pat>'
     /workspace/repos/<issue>/<owner>/<repo>` per candidate, then stop.
   - **Never list, never dump** — the recipe at the top of this file is the rule; this
     is what breaking it costs. A listing was ~24-28 KB once (#13699: 2857 entries) and
     13-104 KB per later session (#9335, #9183, #11739, #8984, #11454 — the last in the
     same block as `skill_view`), `cat SOURCES.json` likewise, `cat
     /workspace/history/*.md` 7 KB for one issue because those files are flat whole
     bodies (#3107). A missing `/workspace/attachments` also makes a bundled listing
     exit 2, which reads like a failed orientation. One recorded two-line
     orientation block (`ls /workspace`, `ls scratch`, `cat SOURCES.json`,
     `ls history`) came back as a single **62 K-character** tool result (#15985:
     `SOURCES.json` alone is 25 KB at v0.26.0rc1, the history listing ~800 names)
     and that run ended at 1.1 M cumulative prompt tokens, output-truncated; #5687's
     same block cost 3 KB only because `SOURCES.json` was piped through `head -50`.
   - **Attachments are the exception: a manifest grep is not an existence test.**
     `ATTACHMENTS.txt` lists names and SHA256s only, so a grep for the issue's
     tokens proves nothing (#12658 read that as "no attachments", skipped
     `raw_completion.json`, and burned the session in the parser). Search the
     attachments view by the file names the issue links (`search_files pattern
     '*.json'`) and open them — they are often machine-readable dumps.
8. **The prior-month corpus is not evidence of absence, and a lexical accident is
   not "no precedent".** Search titles with **one** distinctive token — an op name,
   an error string, a numeric code, a fault kernel, a model id — never the failure
   category and never a word the environment dump already carries (`suffix` ←
   `ATB_SHARE_MEMORY_NAME_SUFFIX=`, `mtp`, `ngram`, `spec`, `npu`, `acl`) or a
   mechanism phrase (`graph capture|capture size|图捕获`). Two recorded forms: #9278
   searched `-i 'suffix|507035|aivec|MPU address'`, got "207 hits / 207 distinct
   issue ids", read the whole result as env-var noise and stopped — while #488 in
   that same output carries the reporter's own `error code:507035`, and the
   titles-only search returns eight real precedents (`suffix` +
   `full_decode_only` among them); #16581's mechanism words returned 442 hits of
   file names, while the paste's own code tokens (`507015|MTE|DDR address`)
   returned 91 and named #3263/#2424. When a pattern hits ~every file, split the
   alternation instead of discarding the survivors.
   `history/*.md` and
   lexical issue search cover only earlier issues; a current issue is never in
   them, and the ids this skill's own pages cite as recorded cases are usually
   newer than the case's corpus (a #6533 view ends at 6436 while the pages cite
   8590, 7689, 13591 …). So a `not in this corpus` from `history-body.sh` weakens
   nothing: quote the page's pin and facts instead of re-opening the id, and never
   drop the script's only message with `2>/dev/null | head` — #6533 read the empty
   stdout as an empty body, then spent a GitHub call on a 404 and disowned the
   same-pin #8590 precedent it had. Make one grep for the distinctive token (op
   name, error string, dataset
   name), read 2-3 hits, then stop: **for a corruption report the distinctive token
   is the pasted output itself** — the repeated character run (`-b '!!!'`) or the
   exact answer text — not the category word. #3547 searched titles for
   `乱码|garbled|重复|感叹号|!!!` and got three V0-era hits; `-b '!!!'` also returns
   #1754, whose **title names the mechanism** ("The ACLGraph graph model does not
   support pipeline parallelism") while its body is `Output: '!!!!!!!!!!!!!!!!'` for
   every prompt — the neighbour a symptom-word title search can never reach. Precedent supports a hypothesis, it does not
   replace reading the source at the pinned commit. Reach
   for the `github` issue search only when the corpus token has **no** hit: it returns
   whole issue bodies
   paginated (17 K characters in #9986 with `display_complete: false`), so read the
   first page and stop. **Pick the bodies by whether
   the title names a mechanism, not by whether it repeats your symptom.** A title
   like "多卡ViT qkv部分权重为0导致精度异常" carries the file, the check and often the
   version flip ("rollback to 0.7.3rc2 works"); four titles that only say
   "<model>-7B 的问题" are four empty shells. Recorded #2586 read the empty shells
   and cited #976 from its title line alone — the one body that named the exact
   `qwen2_5_vl.py` weight-padding mechanism the answer was guessing at.
9. **Closed components exist, but the split is not where it used to be.** The
   HCCL collectives, opbase and pipeline sources are exported under **`cann/hcomm`**
   (e.g. `src/framework/op_base/src/`, `src/algorithm/impl/operator/`,
   `src/platform/common/externalinput.cc` holds the env parsing), so grep there
   before calling a behaviour undocumented; `cann/hccl` itself is mostly headers,
   examples and the API surface. The genuinely absent pieces are named in
   `SOURCES.json` under `unavailable` (commonly `cann/hixl`, `cann/opbase`,
   `cann/ge`, the ops-* repos). Even when a tree is exported, `.cc` files that
   emit a given error string may be missing (the rank-table checkers under
   `cann/hcomm/src/framework/common/src/topo/` exist only as headers). A tree
   `sources-summary.sh` prints as `files=1478/4849 TRUNCATED` can also lack the whole
   module you came for (a `transformers` export with no `tokenization_utils_base.py`,
   exit 2 on the grep): check the path exists before reading its absence as "the string
   is not there", and prefer the caller's tree — vLLM re-raises its dependency's message
   verbatim. One grep pass per tree, then use the plog, the version correspondence and
   the hardware path.
10. **Report gaps instead of filling them.** Missing shapes, dtype, batch
    regime, server flags, harness config and filtered log lines are the usual
    reason a report is unanswerable or a lane misclassified; naming them is part
    of the answer, and the task environment asks you to flag stale/incomplete
    knowledge. Missing lines are a data gap, not a fact.
11. **A log sentence is not a diagnosis: some are fixed wrapper text.** vllm-ascend
    logs canned advice inside `except` blocks and re-raises, so the sentence names a
    cause the code never checked — the `ACLgraph has insufficient available streams
    … Recommended solutions` block prints for *any* capture exception
    (`model_runner_v1.py:3553-3565` at v0.11.0rc0, gone by v0.20.2rc1: re-grep per
    version). Before adopting a log's own explanation, `rg -n '<fragment>'
    vllm_ascend/` and read the emitting site: an `except Exception` that appends
    `{str(e)}` and re-raises carries the inner error as evidence, not the wrapper.
    Same family as procedure step 3's sync-point trap: where an error surfaced is
    not necessarily its owner. CANN prints the same kind of wrapper under the code:
    `[Error]: Failed to apply for memory.` / `Check the remaining storage space in the
    hardware environment` + `EL0006 … resources are insufficient` is the canned text of
    a *resource-alloc* refusal (`207005`), so read the `[SqCqManage]Alloc sq cq fail …`
    line under it before treating such a paste as an HBM, disk or container-size report
    (`references/startup-failures/capture-stream-table-exhaustion.md`).
    The rule covers *advice* too: `Cannot run aclop operators during NPU graph capture …
    please try to set torch.npu.config.allow_internal_format = False` is torch_npu's
    capture guard quoting a suggestion, and vllm-ascend assigns that option `True` at
    import (`worker/model_runner_v1.py:138` at v0.11.0rc0, `:154` at v0.18.0rc1), so it
    cannot be applied from the launch shell; the op named in the message is the subject
    and the FX trace beside it names the owner
    (`references/startup-failures/aclop-op-during-graph-capture.md`).
12. **A crash inside a native module the image installed is an environment
    question, and it is answerable without that module's source.** Three forms — a
    `python -c "import <pkg>"` glibc allocator abort, `std::logic_error` /
    `basic_string::_S_construct null not valid` at engine start, and
    `ImportError: …/_ops*.so: undefined symbol: …Ss` from a prebuilt wheel
    (`std::string` ABI split, decided by the env dump's `cxx_abi_{0,1}` root,
    reporter-confirmed on #1144) — all route from
    `references/image-dependency-faults/index.md`; do not walk the module's source.
    Cap ~6 calls (#5609, #10943 burned the session there). The fourth form is an
    **ACL/GE bring-up failure** at device open — `SetPrecisionMode … AclSetCompileopt(
    aclCompileOpt::ACL_PRECISION_MODE …) error code is 500001` +
    `PlatformInfoInit … SocVersion <name>` + `GEInitialize failed`, ending
    `ERR99999` with no `vllm_ascend` frame. Owner, in this order: a **containerized**
    run whose `docker run` already carries the documented `--device`/mount list takes
    that same command plus `--privileged=true` as its first re-run (reporter-confirmed
    #302: 「已解决，加个--privileged=true即可」, driver unchanged, one variable); then the
    driver/firmware × image-CANN pairing, with the printed SoC name read against
    `npu-smi`. No precision/dtype flag is ever the cause →
    `references/startup-failures/acl-init-500001-and-soc-version.md`.
13. **A pasted error block is not necessarily this failure.** Ascend re-prints a
    recorded device/runtime error block inside a fresh traceback and keeps that
    block's original date, so the title's sentence can be a day older than the
    incident (#9183 named the wrong owner that way) — while a record from *this*
    process's own `init engine (...)`/capture/warmup window is this incident
    reported late (#10700). One call decides both:
    `bash <skill_root>/scripts/plog-timeline.sh --since <report date> <paste>`; a
    **constant whole-hour** lag is a timezone, not a replay, so re-run at the cluster
    nearest the reported time before declaring the fault absent. Formats, the
    `W<MMDD>` blind spot and both worked cases:
    `references/hccl-and-hardware/error-report-timestamps.md`.
14. **A "how do I enable F" question is a capability question, and the verdict comes
    first.** When the report asks how to turn a feature on for a model/hardware pair
    and pastes no log, the reporter is asking whether it is *known to work*: give
    yes / no / partially in the first sentence, with the evidence class — documented
    as supported at the pinned ref, measured on that hardware, covered by
    `tests/e2e/**` (and the nightly yaml's `npu_per_node`), or only by an older
    precedent issue. "No validated recipe for this combination" is an answer;
    a launcher synthesized from the constraints the code enforces is not, and it
    provokes the follow-up "但有人在 A2 上成功跑通过吗?" (#8984). Then map each
    enforced constraint to the error text it raises, so the reporter can place their
    own two errors, and stage the experiment one axis per launch. Three turns,
    `references/feature-enable-recipes/index.md`. Two traps: a pinned doc's narrower
    hardware scope is the project's position for that version (quote it, do not
    overturn it with an older issue), and a comment saying a sub-feature is unsupported
    is often a branch the code already takes (`sfa_cp.py` vs the table's MLAPO ✅).
    A **weight-layout capability** report is the same shape with the premise inverted:
    "910B3 是否支持权重 nd 到 nz 的转换，提高 matmul 的性能？0.11.0 的代码好像只支持 310P"
    — `is_310p()` guards read both ways in this tree (`if not is_310p():` runs the NZ
    cast on 910 and skips 310P), so classify the polarity per site and give the
    per-path verdict before any patch, and grep `envs.py`/`ascend_config.py` at the
    reported ref because `weight_nz_mode` does not exist before ~v0.20
    (`references/feature-enable-recipes/weight-nz-layout-by-soc.md`).
    The same table settles the **"<model> 离线推理报错 / 跑不通"** shape when the
    release is named and the only evidence is screenshots: one `rg -n -i '<family>'
    docs/source/user_guide` (the matrix path moved — `support_matrix/supported_models.md`
    at v0.9.2rc1, a flat `supported_models.md` at older refs), and a `❌` row with a
    tracking issue *is* the verdict, while `| Need test |` is a coverage gap, not a
    refusal. Say the op fault is the manifestation and that the reporter's next
    attempt will surface *different* errors (recorded #6257: four more screenshots
    titled 其他错误 after the ❌-row answer), so the ladder must not read "then it
    will work" — `references/feature-enable-recipes/unsupported-model-row.md`.
15. **A working sibling path in the reporter's own evidence constrains the cause.**
    When the same box fails on one entry point and works on another — offline
    `LLM()` vs `vllm serve`, one flag, one model, one card — a mechanism that also
    predicts the working path fails is not yet the answer: name the difference
    first and rank the mechanism below it, with the log line or ablation that
    separates them. #1534 ranked "mindie_turbo replaces RMSNorm" above the working
    `vllm serve` path and left the follow-up ("是因为在线推理默认没有开启 acl graph 吗?")
    unanswered. Two log lines settle which path compiled and which process loaded
    the plugin — `PIECEWISE compilation enabled on NPU` / `Compilation disabled,
    using eager mode by default`, plus the wheel's INFO line
    (`references/image-dependency-faults/mindie-turbo-graph-break.md`);
    `enforce_eager=True` on the failing path is the one-line ablation.

    **Your own ablations obey the same rule: an ablation only rules out a phase it
    actually removes, and its own log line has to be in the paste.** `--enforce-eager`
    is neither universal nor free: at vllm-ascend v0.22.1rc1 it removes torch.compile,
    the ACL graph and even the `Using OOT custom backend for compilation` banner
    (`vllm/config/vllm.py:1032-1038` → `decorators.py:390` → `platform.py:590-592`),
    while the weight load runs identically in both lanes — so it cannot test a
    load-phase stall, and a reporter's "tried it, still stuck" eliminates nothing
    unless the paste carries that flag's own lines (`Compilation disabled, using
    eager mode by default`, `Cudagraph is disabled under eager mode`). Pick the
    ablation that removes the suspected phase, name the line that proves it ran
    (`references/startup-failures/compile-backend-banner-then-silence.md`). The same
    trap sits inside `--speculative-config`: with `--async-scheduling` on, `method:
    mtp` forces the drafter eager regardless of `enforce_eager`
    (`spec_decode/eagle_proposer.py:136-142` at v0.18.0rc1), so "toggled
    `enforce_eager`, both crash" is one path run twice — check the serve command for
    `--async-scheduling` before crediting that ablation
    (`references/spec-decode-mtp/empty-first-response.md` Step 6b).
    On one family that knob is not weakened but **erased**: for `deepseek_v32` with
    `method: mtp` upstream vLLM sets `speculative_config.enforce_eager = True`
    (`vllm/config/speculative.py:254-257`, commented "cudgraph with v32 MTP is not
    supported") and `vllm_ascend/platform.py:359-369` force-clears it (`:504-517` at
    v0.18.0rc1), so `"enforce_eager": true` in `--speculative-config` is a dead lever
    there — global `--enforce-eager` still works, and the plugin, not upstream, owns the
    support claim (`references/hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`).

    **A workaround named in the report's own prose is the first ablation, not
    background reading.** When the narrative quotes a neighbouring issue's comments
    ("comments mention disabling async scheduling or MTP, or adding sampling-stream
    synchronization as workarounds"), the reporter has already said which phase to
    remove: take it as rung 1 and rank the mechanism it points at, ahead of any
    theory that needs a code path the paste cannot show. #13964 is the recorded
    cost — the body carries exactly that line for a progressive per-rank decode
    stall (`Running` > 0, flat generation counters, no error, one rank still
    serving), and the session spent its whole output budget on a DP
    cudagraph-mode-sync / skipped-all-reduce argument that the report's own
    "DP1 finished 116 requests" sentence refutes, shipping nothing. The resolution
    was KV-block reuse under `--async-scheduling`
    (`references/scheduler-admission/kv-block-lifetime-async-scheduling.md`).

    **The strongest form of this evidence is a tested-configuration matrix in the body**
    (`DP=1,PP=1,TP=32 → works` / `PP=2,TP=8 → works` / `PP=4,TP=8 → fails` / `DP=2 → fails`):
    each failing row then has a named owner and one check, so the matrix answer — one
    verdict line per row — *is* the G1 message, and the rows that work are the controls
    that close the transport, capacity and labelling lanes. #10910 walked 45 turns and
    966 K prompt tokens past that matrix and shipped **no** answer text at all. Check each
    row against the code's own gates before crediting it as a control: on a
    sparse-attention model a row that toggles `--decode-context-parallel-size` also
    moves `tp` (`platform.py:723-731` asserts
    `dcp == tp` at v0.26.0rc1), so "dcp=8 fails, dcp=16 works" compares two recipe rows,
    not one variable (`references/dsa-sparse-attention/dcp-c8-accuracy-garble.md`).

    The same test applies when the reporter's *control* is another platform: "nv 卡多次
    一致，npu 多次不一致" makes sampling randomness (`没有 seed`, `temperature > 0`) a
    refuted cause, because it would randomise NV too — so it can be a one-line
    confirmation but not the leading rank. #7180 led with it and the reporter's next
    comment pasted `temperature: 0, top_p: 1, top_k: 1` — reproducibility between
    repeated runs, not accuracy (`references/accuracy-benchmarks/repeated-run-nondeterminism.md`).
16. **A release candidate's experimental mode is a version question before it is a
    mechanism question.** When the mode printed its own banner
    (`FULL_DECODE_ONLY compilation enabled on NPU`, `PIECEWISE …`), every rank finished
    capture, and the *first* request then stalls into the 300 s watchdog, the paste
    has already closed the capture lane and the "setting never applied" chain; what
    remains is model×mode support at that ref or a defect fixed after it. Three
    checks, then deliver: (a) which models the pinned tree validates for that mode —
    `rg -n 'cudagraph_mode.*FULL_DECODE_ONLY' <ascend>/tests/`, then confirm a
    `.github/workflows` file actually runs that test (at v0.11.0rc1 the single
    FULL-mode e2e file was not wired anywhere while the wired ACL-graph runs were
    PIECEWISE at `max_model_len=1024` on Qwen3-0.6B / DeepSeek-V2-Lite / Qwen3-30B-A3B,
    so no VL, 72B or 16 K case was covered); (b) whether the reporter's own control
    works — a default-mode success on the same command makes the *mode* the lever;
    (c) whether the tree carries any mode-specific handling for their family (absence
    is quotable as coverage, never as the cause). A fix that lands in the next rc **is
    not in the snapshot**, so a source walk cannot find it: #4178 walked 22 turns and
    48 K characters of reasoning on that path, emitted zero message text, and was cut
    before answering, while the coverage verdict was three greps and its reporter
    later reported the bug fixed on `0.11.0rc2`
    (`references/hccl-and-hardware/aclgraph-full-decode-runtime-hang.md`).

## Classify the evidence

Two passes. First the **track**, from the dispatch table under *Support entry points*
below (`references/dispatch/classified-shapes.md`); for a startup log that means lane
A (worker died without a per-rank traceback), lane B (worker raised a Python error) or
lane C (collective or device fault), and for a report with no symptom that means one of
the D tracks (sizing, document, capability, code-change). **An `[Installation]` /
安装 report that fails before any engine exists** — a compiler or linker log, a pip
resolution error, a missing library at import — is its own track: the paste's last
frames name the failing *stage* and the stage names the owner (vllm-ascend's
`CMakeLists.txt`/`setup.py` vs the CANN toolkit vs the image), the reporter's
CANN/torch/torch-npu trio is checked against the pin's own requirements table before
any version advice, and the cheapest working install is rung 1. A `multiple
definition` at the AscendC kernel relink is one source under two path spellings, not a
kernel defect: `references/image-dependency-faults/custom-op-build-stage.md`. Then, when the signature is specific — a log
fragment, an error string, a flag that did nothing, a title that names a mechanism —
the exact row in `references/first-checks/signature-routes/index.md` (five pages split by
the shape's home track): each row carries the lane or track it
belongs to and the page that owns it. Read the row's middle cell before opening the
page; it carries the decisive local rule. A token you cannot place: one
`rg -n '<fragment>' <skill_root>/references/` searches every row **and** every page
body, which is the version that pays. **A traceback whose deepest frame is in
`torch/`, `torch_npu/` or `vllm/` rather than in `vllm_ascend/` is a routing
signature: search the helper's name or the exception text in the same call as the
history search, before any source read.** Those trees are frequently absent from the
export (this snapshot's pytorch is `files=1733/19378`, no `torch/_dynamo/`), so no
source walk can confirm such a frame and the library is the only local evidence
there is — recorded #7947 spent its entire 16 384-token budget re-deriving
`AttributeError: 'NoneType' object has no attribute 'size'` from `qwen3_5.py` →
`qwen3_next.py` → `torch/_dynamo/utils.py::call_size` while
`references/accuracy-benchmarks/first-generate-degenerate.md` already carried that
stack, its ruling-out and its fix attribution. The same applies one level deeper: an
`IndexError: list index out of range` whose deepest frame is
`vllm/model_executor/models/<model>.py::get_mrope_input_positions`
(`image_grid_thw[image_index]`, called from `_init_mrope_positions` in
`v1/worker/gpu_model_runner.py`) is a **model-directory** question — the image count
is read off the checkpoint's `vision_start_token_id` while the features come from
`mm_features` — so the decisive lines are the server's own `Detected the chat template
content format to be '…'` and the ids in `config.json`, not a walk through
`paddleocr_vl.py` and the mm caches: recorded #5772 spent 20 turns and 50 K
characters of reasoning there and shipped nothing, while the reporter's own later
reply (a working run of the same tag and image, fix = re-download the model files)
was the answer (`references/multimodal-inputs/mrope-image-index-error.md`).

## Procedure

1. One level of `/workspace` (`search_files (target='files')`) — that level only,
   never the history or attachments directory (rule 7) — then run
   `bash <skill_root>/scripts/sources-summary.sh` for the `SOURCES.json` version
   contract; note which repos/refs are present and which are partial exports. The
   layout differs per case (flat `/workspace/history/<issue>.md` vs
   `/workspace/history/<issue>/` + `/workspace/repos/<issue>/`); adapt after
   looking, do not assume.
2. Read the pasted log and place it in a track/lane. Note what is *missing* (no
   worker prefixes, no `Starting to load model`, no version block) — reporters
   paste filtered logs, and missing lines are a data gap. When the body carries no
   log at all, the attachments under `/workspace/attachments/<issue>/` **are**
   the evidence: an image's `.ocr.txt` is the log paste, a referenced `*.json` is
   usually a machine-readable dump (request body, rendered prompt, tokenized
   request, completion) whose fields decide the case before any source read
   (`references/evidence-workspace/index.md`). Not only logs: a screenshot of a **vendor
   docs page** is the vendor's own table — a capability/image question ("可以用
   `quay.io/ascend/vllm-omni:vX` 部署 <model> 么") is often settled by its OCR's
   `Hardware` columns in one read, while reconstructing tag conventions from other
   issues costs a dozen calls and yields weaker evidence (#10303 never opened it).
   **A bare CANN/HCCL code in a paste (`EJ0001`, `EI0006`, `EL0004`, `507xxx`, a numeric
   `HcclResult`) is one call, not a source walk**: `bash <skill_root>/scripts/cann-error-lookup.sh <code>`
   prints the vendor's own `errTitle`, message and `Possible Cause`/`Solution` from
   `cann/metadef/error_manager/error_code.json` / `cann/hcomm/…/error_code.cc`, or the enum
   name from any exported `hccl_types.h`. Two of its answers change the lane outright:
   `7 = HCCL_E_UNAVAIL` beside `EJ0001` means **the previous process is still alive**
   (sleep mode, an engine that never exited), not a parameter or NIC fault; and `EI0006`'s
   solution text names `HCCL_CONNECT_TIMEOUT` before any link theory. When
   the reported title is the end of a
   chain (`Failed to infer device type`, `Failed to load plugin ascend`), the owning
   line is two or three above it —
   `references/image-dependency-faults/plugin-load-module-missing.md`.

3. Take the track's cheap checks before source archaeology:
   - Lane A/B: effective `max_model_len`, then the diff between the reporter's
     command/env and the matching tutorial or CI test
     (`references/startup-failures/max-model-len-budget.md`).
   - Lane C: re-run with `--enforce-eager`, then `ASCEND_LAUNCH_BLOCKING=1` to
     restore error attribution; read the plog; only then judge whether the
     Python frame named in the traceback is the failing op
     (`references/hccl-and-hardware/collective-error-chain.md`). If the
     collective failure survives re-runs, check health per device and correlate
     the failing device id (`references/hccl-and-hardware/device-health-checks.md`).
     An `npu_*` op on a 2×4/2×64 tensor inside `torch.compile` is a sync point by
     construction: go to the plog and the per-rank `EZ9999`/`chipId` line.
   - Any count or list derived in the log (capture sizes, block counts, chunk
     sizes) that does not match the upstream default: vllm-ascend is probably
     rewriting it. `rg -n '<config field>' vllm_ascend/` (platform.py,
     ascend_config.py, utils.py), read the log's own `Calculated ...` / `N → M`
     lines, and reproduce the number with
     `bash <skill_root>/scripts/aclgraph-size-budget.sh` instead of reading the
     formula by eye. The same holds for **behaviour switches**: downstream
     `check_and_update_config` may turn a feature on that the reporter never
     requested (at v0.11.0rc0 `platform.py:139-150` force-enables the
     `AscendScheduler` fork for a default dense serve and overwrites an explicit
     `enabled: false`), so grep the config field in `vllm_ascend/platform.py`
     before asking which flags they passed (recorded #3979).
   - Then the per-track checks: `references/first-checks/index.md` routes to
     `cheap-checks-by-track.md`, the full list. Open the entry that matches your
     classification instead of reading the whole list, and take its named script
     or check before any source walk.
4. **Write the answer now, then keep going** (rule 1). With the lane/track
   classified and the owning path or decisive check named, emit the ranked
   diagnosis in the message. When the report lacks what the track needs (the two
   serve commands of a PD deployment, prompt length, per-rank log), that first
   answer is the ablation plan plus the list of missing artefacts — already a
   complete deliverable. Continue into source work afterwards as corrections.
5. Fix the version correspondence: image tag → vllm/vllm-ascend commits →
   torch_npu/CANN/driver
   (`references/startup-failures/environment-correspondence.md`). Read the
   release's own `tests/vllm_ascend_v<ref>_test_conclusion.md` when the tree has
   one: it names the models that release validated and its unresolved known
   issues with issue ids. Claim "model unsupported" only after checking
   `vllm_ascend/models/__init__.py` registration and the upstream model file, and
   claim a *method or feature* unsupported only after the consumer grep plus the
   tree's own test for it (`rg -n '<feature>' tests/`) — a
   `pytest.skip("Not current support for the test.")` on exactly that feature is
   the project's own statement at that ref and citable as such (#2071). For the
   recognition-error form, answer from
   `references/startup-failures/model-type-not-recognized.md` before any deeper
   walk.
6. One grep of `history/` for the distinctive symptom token, 2-3 hits, then stop
   (rule 8); `bash <skill_root>/scripts/history-titles.sh [-i] '<regex>'` gives the
   titles of all hits in one call without opening the bodies (`-i` when the
   corpus capitalises the token, e.g. `reranker` vs `Reranker`; bodies are searched
   only with `-b`, because `suffix`/`mtp`/`ngram`/`spec` occur in every pasted
   `collect_env.py` and would return the whole corpus). **A token that is a code
   identifier or an exception string is the exception — run it with `-b` in the
   same call**, because such tokens live in tracebacks and launch scripts, never in
   the env dump: `-b 'TRITON_ALL_BLOCKS|putenv|marked as skipped'` returned 8 hits
   where the title search returned 1 (#8224) and named the `export
   TRITON_ALL_BLOCKS_PARALLEL=1` line five other reporters already carry in their
   launch block. A title-only search that comes back empty is **not** "no
   precedent" until the distinctive identifier has been body-searched once. Then
   read those 2-3 bodies with `bash <skill_root>/scripts/history-body.sh <id> [<id> …]`, not a shell
   prefix read of their first lines: the template puts a 100-250 line
   `collect_env.py` paste before the symptom section (#3963's bug heading is line
   148), and the script collapses that dump to a marker (`--env` restores it,
   `-g REGEX` pulls just the traceback).
7. Locate the owning code path at the pinned commit and walk it from the entry
   point (kernel launch site, harness config, model/feature flag) to the line
   that produces the symptom. Prefer computing the decision (a tiling branch, an
   aggregate metric) with a small `uv run --no-project python` snippet over
   reading code by eye.
8. Cross-check upstream vs downstream: vllm-ascend frequently re-implements or
   bypasses upstream vLLM (custom AscendC op vs `torch_npu` fallback, upstream
   reasoning parsers vs downstream template kwargs, docs baselines vs
   third-party harness, flat `hf_config` reads vs `model_config.hf_text_config`
   accessors, a copied forward body vs upstream `super().forward()`). Record
   which layer owns the behaviour and for which version, plus what would
   invalidate the claim. One level down when the frame is in an SoC subpackage
   (`vllm_ascend/_310p/`, a later `_<soc>/`): diff the override against the
   generic `vllm_ascend` implementation and expect a **missing propagation** of a
   fix the generic path already carries, not a new defect
   (`references/soc-platform-paths/index.md` rule 4).
9. Re-emit the answer in the shape below with the corrections, then stop.

## Answer shape

- The direct answer first: yes/no for the reporter's literal question, the
  version that works or the version where it breaks, and whether the reported
  crash is real. One sentence, before any analysis, in the language the report was
  filed in. **The cheapest next action belongs in that same first paragraph**, not
  below the evidence table: a session can be cut wherever it happens to be, and
  #7580 was cut inside its table, after the verdict and before its ablation plan.
- **A decisive fact already in hand is the answer.** Once one grep has settled the
  mechanism — the pin's own class carries the attribute the paste calls missing, the
  owning branch or call site is named, the one config line that differs is identified
  — send the verdict in the next message and spend nothing more on confirmation. Two
  recorded runs (#10398, #2239) each held exactly that fact and kept searching; neither
  shipped an answer at all, so the reporter got nothing from work that had already
  succeeded.
- Then the one-line classification: lane A/B/C and the phase that failed (load,
  memory profiling, graph capture, first collective); or, for a runtime report,
  the primary cause plus its track.
- Ranked causes, each with the evidence you have, the exact check that confirms
  or refutes it, and its confidence label (reporter-confirmed / source-verified
  at commit X / hypothesis).
- An evidence table (`path:line`) for code claims, with precedent issue ids
  quoted as `#<id>: <one-line fact>`.
- The action ladder, cheapest first: **the request's own sampling value**
  (`temperature`/`top_p` — on a reasoning-distill checkpoint a ≤0.1 temperature is
  itself a degeneration, so the rung *raises* it; `references/accuracy-benchmarks/
  short-prompt-corruption.md`), an explicit `--max-model-len`, a re-run with
  `--enforce-eager` / `ASCEND_LAUNCH_BLOCKING=1`, `return_token_ids: true` on the
  same request body, one `num_speculative_tokens` step, `asys health -d=$i`, a
  version correction — before any patch proposal.
- Version and hardware scope of the conclusion, plus what to send back (full
  per-rank log, plog, tiling I-line, raw request/response dumps, `npu-smi info`,
  `asys health` output, `dmesg` OOM lines, both PD serve commands).
- What you could not verify locally and which questions remain open, stated as
  such.

## Support entry points

Open **one** topic entry: every `references/<topic>/index.md` states
its scope, inherits these rules and routes to its own pages, so page-level triggers
live where they are used. Two procedure helpers come first:
`references/first-checks/index.md` (the per-track cheap checks of step 3,
`cheap-checks-by-track.md`) and `references/evidence-workspace/index.md` (case
layout, `SOURCES.json`, history search, attachment/OCR caveat;
`delivery-discipline.md` holds rule 1's G1/G2/G3 gates and the reasoning patterns
that break them, `recorded-cases.md` the per-issue index).

**The recorded-shape → topic-entry table** — one row per routing decision, written in
the reporters' own symptom wording, each row carrying the owner and the page to open — is
`references/dispatch/classified-shapes.md`; `references/dispatch/index.md` states how it
is used (classify first, match wording not title words, open one topic entry). Take the
classification from *Classify the evidence* above, then find the row. The coarse families
it covers, so a first guess needs no read: **accuracy and output**
(`accuracy-benchmarks`, `multimodal-inputs`, `serving-frontend-faults`,
`tool-call-streaming`, `pooling-embedding`); **startup and load** (`startup-failures`,
`image-dependency-faults`, `soc-platform-paths`); **runtime device, memory and custom
ops** (`hccl-and-hardware`, `runtime-memory`, `custom-ops-runtime-faults`,
`mixed-batch-splits`, `spec-decode-mtp`, `pipeline-parallel`, `lora-adapters`,
`observability-and-logs`); **PD, KV pool and sparse attention** (`pd-disaggregation`,
`kv-pool-store`, `dsa-sparse-attention`); **capability, sizing, perf and docs**
(`feature-enable-recipes`, `deploy-sizing`, `performance-throughput`,
`scheduler-admission`, `release-known-issues`, `code-change-requests`,
`upstream-downstream-drift`, `tutorial-verification`, `evidence-workspace`,
`first-checks`). A row that says *not this topic* is a route away from it — use it.

Scripts are re-runnable; `<skill_root>` is the path `skill_view` reports and each
header carries its flags. Run `.sh` with `bash` and `.py` with
`uv run --no-project python` — the executable bit is not durable. Their arguments, output and originating issue are in
`references/first-checks/scripts-catalogue.md`; open it rather than guessing a name,
and the topical pages name the ones they use. `templates/` holds starter files to copy
and edit rather than retell — currently `templates/omni-310p-deploy-config.yaml` (a
two-stage fp16 `--deploy-config` for one 310P die, cited by
`references/soc-platform-paths/310p-omni-model-deploy.md`). The per-signature
classification index (log fragment → track → owning page) is
`references/first-checks/signature-routes/index.md`.

