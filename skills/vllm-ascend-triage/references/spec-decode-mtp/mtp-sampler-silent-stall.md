# A silent whole-service stall on an MTP + `--async-scheduling` serve

Symptom class: the deployment serves normally for tens of minutes to hours, then
**stops**: the worker log simply ends mid-stream, no traceback, no `ERROR`, no device
dump, `npu-smi` still shows the KV pool resident, and the process/API server only dies
much later — "run 1x hour, it will be Stuck. the log will stuck, and many hours later it
shows the server shut down". The launch carries an MTP drafter
(`--speculative_config` with `method: qwen3_5_mtp` / `mtp`) and almost always
`--async-scheduling`.

This is the **second** lane for the shape
`../hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md` owns, and on the 0.18.0
line it is the one with a reporter-confirmed fix. Take it seriously before spending the
segment on a plog timeline.

## Reporter-confirmed resolution — recorded #8916

Environment, supplied by the reporter after the filed body (whose `collect_env.py` was
the untouched template, `Your output of above commands here`): torch `2.9.0+cpu`,
torch_npu `2.9.0.post1+gitdc51c2d`, transformers `4.57.6`, triton-ascend
`3.2.0.dev20260322`, vllm `0.18.0`, and **`vllm-ascend` `0.18.0rc1` base plus three
cherry-picks**:

| PR | title as the reporter filed it |
|---|---|
| #8531 | `[Bugfix][0.18.0] fix kernels in sample when mask is not static or draft_token_id is invalid` |
| #7904 | `[P/D][BugFix][Core] Replace decoder MTP padding token_id` |
| #8764 | `[bugfix][0.18.0] Fix race in non-blocking num_accepted_tokens` |

Their verdict: 「this bug fixed with these three pr , but the accurancy bug still exists
[#8932]」 — i.e. the stall and the accuracy gap are **two** defects, and the stall is
fixed by the sampling/MTP set. Which of the three removed the stall is **not
attributable**: they were installed together, so do not tell a reporter that one of them
is "the fix".

Two consequences that change how the report is read:

1. **The installed tree is not the tag.** `0.18.0rc1` + cherry-picks means no
   `path:line` from the pinned `v0.18.0rc1` tree describes the reporter's process, and
   a source-verified claim about their build is unsupportable. Ask for `pip show vllm
   vllm-ascend` plus the cherry-pick list, and label anything read from the export as
   "at the tag, not necessarily yours" (same class as
   `../pd-disaggregation/node-build-skew-attribute-error.md`).
2. **A stall fix is not "all clear".** Ask for a correctness spot-check on the same
   recipe afterwards; on this case the accuracy defect survived the stall fix.

## Step 1 — the serve command decides whether this page applies

One artefact, before any plog or source read: both roles' launch commands. Then check
three flags:

- `--speculative_config` — MTP on (`qwen3_5_mtp`, `mtp`) ⇒ this page is live; absent ⇒
  a stall is the MC2 / KV-transfer / OOM lanes only.
- `--async-scheduling` — the third fix names the non-blocking bookkeeping this flag
  enables, and it is the axis of the cheapest ablation below.
- `--compilation-config '{"cudagraph_mode": …}'` — `FULL_DECODE_ONLY` + MTP is the
  documented pair, and a *graph* fault is a different page
  (`draft-graph-capture-device-fault.md`).

**"I used the docs config" already answers this.** The pin's own Qwen3.5-397B tutorial is
an MTP launch: `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:101`
(`export HCCL_BUFFSIZE=512`), `:121`
(`--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3,
"enforce_eager": true}'`), `:122`
(`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`), `:124`
(`--async-scheduling`), at vllm-ascend **v0.18.0rc1** (`99e1ea0`) / vllm v0.18.0
(`bcf2be96`). A reporter pointing at that page (the #8916 body did, verbatim:
`docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html`)
has MTP `n=3` and async scheduling on, and their env block will never show it.

## Step 2 — the candidate sites at the tag

The fix titles name nothing directly readable, but the sites they point at are:

| site at v0.18.0rc1 | what it is |
|---|---|
| `vllm_ascend/ops/triton/reject_sample.py:55` (`mask = offset < vec_len`), `:102-111` (`if is_greedy_ptr is None: … else: tl.load(is_greedy_ptr …)`) | the greedy rejection-sample kernels; `vec_len` is `do_not_specialize`'d (`:88`), so the mask is runtime — "mask is not static" lands here. `draft_token_ids` are loaded **unmasked** by index inside the per-request loop (`:122`, `:176`), so an out-of-vocab / padding draft id is read as a real one — "draft_token_id is invalid" lands here |
| `vllm_ascend/sample/rejection_sampler.py:82` `def rejection_sample`, `:116` (`using_block_verify = max_spec_len >= 3 and draft_probs is not None`, consumed at `:195`), `:126-129` (`all_greedy` ⇒ `is_greedy = None`, else a per-request temperature tensor) | the host side that chooses those kernels and passes `is_greedy=None`/a tensor; `n=3` takes block verify, `n<3` does not |
| `vllm_ascend/worker/model_runner_v1.py:2046-2048` (`self.num_accepted_tokens.np[…] = …num_accepted_tokens_cpu[…]; …copy_to_gpu()`), `:2157` (fed into the attention metadata) | the CPU→device mirror of `num_accepted_tokens` — the value whose copy the third title calls non-blocking. It sits on the `use_async_scheduling` path (`:1153-1159`, `:1622-1627`), which is the axis Step 3 toggles first |

Reading these is for the *patch proposal*; the diagnosis does not need them.

## Step 3 — ablations, one axis each and cheapest first

1. `--async-scheduling` off, everything else identical: it is the flag that enables the
   non-blocking bookkeeping the third fix names, and it costs one flag. Note that with
   `method: mtp` **and** async scheduling on the drafter runs eager regardless of
   `enforce_eager` (`spec_decode/eagle_proposer.py:136-141` at v0.18.0rc1,
   `empty-first-response.md` Step 6b), so a draft-side `enforce_eager` A/B is one path
   run twice — do not spend the run on it.
2. MTP off (`--speculative-config` removed). It removes the rejection-sampler and
   proposer entirely. If the stall persists, this page is out and the MC2 page's plog
   route is back.
3. Only then a version/cherry-pick move: all three PRs are tagged for the `0.18.0`
   line, so a build cut after them is the version-side rung — date their merges against
   the tag instead of assuming a final release carries them, and remember that an rc
   plus a patch set is not "the release" either.
4. Keep a `py-spy dump` of the worker and EngineCore pids at the stall if the reporter
   can catch it — it separates "blocked in a transfer/collective" from "blocked inside
   the sampler" for whichever lane survives 1-2.

## Not this page

- plog shows `[BuildRemoteResByTag] … mc2RunAlltoAllDirectFullmesh` and equal
  `count of currently executed operators` on two nodes in the same second:
  `../hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md`.
- `Receive failed. error=Receive timeout` / handshake retries in the decode node's own
  log: `../pd-disaggregation/mooncake-handshake-timeout-wedge.md`.
- `dmesg` OOM-kill lines, or per-rank RSS climbing for hours: `../runtime-memory/index.md`.
- the pair `qwen3_5_mtp` + `--enable-prefix-caching` with an error or with "abnormal
  shut down": `hybrid-mtp-prefix-cache-unvalidated.md`.
- visible output symptoms (empty first response, acceptance 0.000, one repeated token):
  `empty-first-response.md`, `acceptance-collapse.md` — those carry text, this page
  carries silence.

## Scope and open questions

Every `path:line` above is verified in the vllm-ascend **v0.18.0rc1** export
(`99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f`) and the tutorial's flag block on the same
ref; `rejection_sample` and the model-runner line numbers move between releases, so
re-grep the function names. Open: which of #8531/#7904/#8764 removed the stall, whether
the same stall exists on an unpatched `0.18.0` final, and what fails on later refs —
none of that is in this snapshot.
