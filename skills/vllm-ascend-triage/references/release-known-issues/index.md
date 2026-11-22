# Release known-issues bulletins

## Scope

A **maintainer-filed bulletin** for one release: `[Bug][vX.Y.Zrc1]: Known Issues
for <model A>/<model B> service hang on A3`, "Issue for tracking: … known
issues", a table of affected versions / models / hardware / scenario, one or two
**tracked issue ids** with error codes, "evaluate the risk before adopting" and
"reply in the comments instead of opening a new issue". The follow-on shape is a
user quoting such a notice and asking whether their own run is the same thing.

The ask is not "diagnose this from the paste" — the paste is a headline with no
log. It is *is this notice accurate, what does it establish, and what should the
people replying attach*. Distinct from an accuracy/performance report (there is
no measured output here) and from `references/tutorial-verification/index.md`
(there the graded object is a document that a user must follow; here it is a
release claim that a user must weigh).

**Second shape this topic owns:** a user quotes the release's own **highlights
bullet** for a feature ("Support EAGLE-3 for speculative decoding. [#1032]", under
`## v0.9.1rc1`) and reports it missing from the later rcs, asking whether the final
release will carry it — no log either, and the graded object is a capability claim
about one release line. That one needs the feature's consumer at the pin plus the
tree's own test and CI wiring, not the bulletin procedure: route to
`highlights-bullet-vs-pinned-tree.md`.

## Rules

1. **Claim-check, not a second root cause.** Deliver: which signatures the notice
   pins (error code + named fault kernel + phase + hardware), what that
   localizes, what it cannot settle (mechanism, event ordering, the P/D split),
   and the one artefact that would settle it. Then one ranked list per tracked
   issue, each cause with its single confirming check — not a fresh investigation
   of both issues in depth.
2. **The release's own records come first, and they are one read.**
   `tests/vllm_ascend_v<ref>_test_conclusion.md` §1 (which models/hardware were
   validated at all; at v0.26.0rc1 `:5` names four models, table `:9-12`), §3
   (unresolved known issues *with issue ids*; `:53` onward), then the
   named model's tutorial recipe for the validated envelope. The validated
   numbers are the cheapest falsifier of "it is a bug": a reporter at 8× the
   tutorial's `--max-model-len` is outside the envelope whatever the notice says.
   Commands: `references/evidence-workspace/case-layout.md` §"release's own
   validation record".
3. **An error code in a bulletin is not a unique signature.** Ascend codes like
   `507034` (`AclrtSynchronizeStreamWithTimeout`, "vector core timeout") are
   generic sync/collective timeouts; `507035`/`EZ9999`/`507015` all recur across
   unrelated scenarios. One `rg -n '<code>'` over the release's own §3 table and
   the named model's tutorial decides whether the code is model-specific — at
   v0.26.0rc1 the same `507034` is attached in §3 to an unrelated Ascend 950
   C16 + `dsa_cp` start hang, so the code alone localizes nothing. Grep the **code**,
   not the id — that table cites #15268 in two rows (a perf regression, then the
   `507034` hang), so an id search reads the first hit as a contradiction. Read the
   whole table (`sed -n '53,80p'`), never a window around its head: at v0.26.0rc1 the
   row that supports a D-node sync-timeout report (#14911, `AclrtSynchronizeStreamWithTimeout`
   507001) is row 16 at its tail.
4. **Tracked ids are usually newer than the history corpus.** `/workspace/history`
   holds earlier-month issues only, so a `MISSING` for the ids the notice cites
   is expected coverage, not a failed search (rule 8) — say that, then map each
   signature onto this library's topic pages instead of hunting the issue. One direct
   `/repos/vllm-project/vllm-ascend/issues/<id>` read turns that from an assumption
   into a citation: a newer id answers 404 `snapshot_unavailable`.
5. **Say which axes the notice leaves out**: the two serve commands, per-agent
   node roles, env (`HCCL_BUFFSIZE`, `mc2_comm_alg`, spec config), and the plog
   `[AIC_INFO] fault kernel_name=` block. Those are the reply template; the
   notice itself usually names none of them.
6. **The release notes' own `Known issue` sections are the bulletin's cheapest
   source, and they name owners.** `rg -n -i 'known issue|leak|bug from'
   <pin>/docs/source/user_guide/release_notes.md` lists, per release section, the
   statements that already assign a component — at v0.13.0rc1 `:197` "The memory may
   be leaked … This is a bug from torch-npu, we'll upgrade and fix it soon", `:198`
   the qwen2.5-VL accuracy bug "lead by CANN", `:199` the long-sequence KV wedge "a
   bug for scheduler". Quote them as the project's position at that ref (and check
   the section heading: a v0.11.0rc1 bullet still stands in the v0.13.0rc1 tree), then
   say what changed since — a later release that no longer lists it (dropping a row
   is not itself a fix: decide that from source or ask), or a pta patch release that
   fixes it (`../runtime-memory/host-ram-and-per-rank-rss.md` §Step 0).
   **A row is scoped to the symptom its wording names.** "Prefix cache feature does not
   work with the Ascend Scheduler but without chunked prefill enabled" (v0.9.1rc1
   `release_notes.md:37`, #1350) is a *functionality* statement about the cache path; the
   same flag cell also produced a chance-level MMLU that survived with prefix caching turned
   off (`0.2585` vs `0.2584`, recorded #1553), so the row was not that report's cause. Quote
   the row, then test it by removing the one feature it names — a row whose trigger you
   removed while the symptom stayed is a control, not a diagnosis
   (`../feature-enable-recipes/torchair-mla-graph-and-chunked-prefill.md` §*chance-level*).
   **A row that quotes a *log line* instead of a symptom is the weakest citation of all.**
   v0.7.1rc1 `release_notes.md:33` says logs "like `No platform deteced, vLLM is running on
   UnspecifiedPlatform` or `Failed to import from vllm._C …`" do not affect functionality, and
   the very same `No platform detected` line is step three of that release pairing's own
   `Failed to infer device type` chain — so quote such a row for the follow-up it actually names
   (the `vllm._C` warning after a serve that works) and establish that the run reaches
   `/v1/models` before calling the line cosmetic
   (`../image-dependency-faults/platform-plugin-register-raised.md`, recorded #130, where the
   maintainer's answer to the follow-up was this page and the reporter still expected v0.7.3).
   On an ordinary **crash** report this section is the *first* read, not a
   bulletin-only step: when the command sets an experimental/opt-in mode, the reported
   release's own row can already name the model family, the mode and the tracked fix —
   v0.12.0rc1 `:42` "DeepSeek 3/3.1 and Qwen3 doesn't work with FULL_DECODE_ONLY graph
   mode … #4990" answered #5076 before any source read (reporter-confirmed fixed in
   0.13.0rc1; `../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md` §0). The
   same read answers a **GLM-5.x** graph crash: v0.21.0rc1 lists GLM-5.1 + Expert
   Parallel + FULL mode with a workaround (`release_notes.md:181` at v0.22.1rc1),
   so a `--enable-expert-parallel` + `FULL_DECODE_ONLY` launch is a declared
   combination before it is a mechanism — and read the row's own `## v…` heading,
   since the file is newest-first and the newest section does not restate it.

7. **Budget one message per tracked issue.** A two-issue bulletin at the 16 K
   completion cap was cut mid-ladder (#15985: verdict + both ranked lists
   survived, the action ladder did not), so lead with the verdict and the
   cheapest next action for both issues in the first paragraph, then cap each
   ranked list at 3 causes with one check apiece.

Route: `bulletin-claim-check.md` for the worked case and its verified anchors;
`highlights-bullet-vs-pinned-tree.md` for a feature a release note claims (three
co-signs in the tree, then a dated answer — #2562 EAGLE-3 on v0.9.1rc3).
