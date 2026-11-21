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
   C16 + `dsa_cp` start hang, so the code alone localizes nothing.
4. **Tracked ids are usually newer than the history corpus.** `/workspace/history`
   holds earlier-month issues only, so a `MISSING` for the ids the notice cites
   is expected coverage, not a failed search (rule 8) — say that, then map each
   signature onto this library's topic pages instead of hunting the issue.
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
   say what changed since — a later release that no longer lists it, or a pta patch
   release that fixes it (`../runtime-memory/host-ram-and-per-rank-rss.md` §Step 0).
7. **Budget one message per tracked issue.** A two-issue bulletin at the 16 K
   completion cap was cut mid-ladder (#15985: verdict + both ranked lists
   survived, the action ladder did not), so lead with the verdict and the
   cheapest next action for both issues in the first paragraph, then cap each
   ranked list at 3 causes with one check apiece.

Route: `bulletin-claim-check.md` for the worked case and its verified anchors.
