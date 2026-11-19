# Issue ledger: what the rows show in aggregate

Scope: the ledger is `index.md`; this page is the reflection over it, not a second copy of the rows.

## Why these entries are useful

- The later seven add three classes that are not hardware at all: a wrong **response body** produced by
  a downstream monkeypatch over the OpenAI serving layer (#11788) and streaming payload loss in the
  parser/serving split (#7682) — both `tool-call-streaming.md`; a connector that dies at KV-cache
  registration because it filters the wrong layer class (#7971, `kv-connector-registration.md`); and
  two operator-owned refusals — HCCP QP exhaustion under EP (#10944) plus a captured-stream host sync
  (#2567), both `moe-ep-dispatch.md`, and an int4/FRACTAL_NZ `group_size` contract (#5603,
  `quantization-load.md` §5). #10365 is the DeepSeek-V4 version of the cache-unit class
  (`prefix-cache-granularity.md`).
- They show the recurring shapes: rc tag (or image tag) + PD disaggregation + one optional feature
  (sparse C8, MTP placeholder handling, DSpark, balance scheduling); the older rc-tag-at-default-config
  OOM family (#3308, #3357) where the answer is a memory-budget term rather than a parallel-feature
  interaction; and, in the later reports, a hardware-specific kernel fault, an upstream core race, an
  upstream/plugin env-var mismatch, a checkpoint/model quantisation mismatch, a pooling/prefix-caching
  accuracy defect whose fix commit is known (#7192), a Triton launch-geometry ceiling (#3971), two
  reduction-plumbing classes (#8990, #10466), an image/`ENV` propagation defect (#12059), a connector
  stall whose honest answer is a version range (#6512), an rc-tag gap in the KV-pool Put (#8808), a
  LoRA/TP kernel-contract abort (#16586), a hybrid-model PD report closed without any diagnosis
  (#8920), and, in the latest reports, an upstream metrics-accounting crash (#8664), an installed-wheel
  version skew (#5894) and a truncated weight shard (#15836).
- Two rows localise *lifecycle* steps rather than maths: #8808 is a downstream `execute_model` fork that
  dropped one upstream connector call (so nothing is PUT and the pool fills), and #1895/#1383 are
  toolchain/dispatch questions — a custom op with no meta kernel, an op the SoC does not carry. In both
  families the decisive artifact is a log line or a `csrc/` listing, not a model file.
- Two rows are about *branch disagreement* rather than arithmetic: #8926 and #8578 both crash because a
  field is produced on one branch of the PCP/async path and read on another (`seq_lens_cpu` set to
  `None` under async scheduling, `mtp_slot_pad` created only on the decode pre-allocation). #13750 is
  the mirror image: a "broken feature" that is a unit mismatch (`block_size × dcp × pcp`), answerable
  from two lines of code and the reporter's own closing comment. When a traceback names a metadata
  attribute rather than a shape, find the assignment site and compare its guard with the reader's guard.
- Four rows have a **documentation or requirement-pin root cause** rather than a code one: a tutorial
  that pairs the proxy with the wrong connector (#15389), a loose `transformers>=` floor that shipped an
  image predating the model family (#8315), a cudagraph mode the maintainers had already declared
  unusable for the family at that tag (#4326), and a `[Doc]` guide whose A2 tab advertises 1M context
  while the same page says the A2 configuration is unrevalidated (#18158 — and note that the *governed
  ref* there is the docs tree, not the tag the reporter names). Each was answerable from the thread plus
  one ref-pinned read of a doc page, `requirements.txt` or `platform.py` — the expensive mistake is
  hunting a `vllm_ascend/` frame that does not exist.
- Four rows answer a **"does this even exist at my tag"** question, and each one is a single command:
  #4180 asks for a switch (`full_cuda_graph`) that only the 0.9 line has, so the report is on the one
  release where it does something and every newer serve block that copies it is inert; #15560's thread
  names two fixes (#12071, #12255) that `git tag --contains` puts *inside* both failing tags; #1114 is a
  build artefact whose producing rule is in `setup.py`/`CMakeLists.txt` while the package-manager half is in
  no clone; #5609 is an image that omits one runtime dependency its successor ships. In all four the lever
  is available in the first batch — quote the tag's own doc line, banner or commit list, and keep the
  unopenable half (site-packages, engine ordering, Mooncake internals) in `status and limits`.
- Three rows are about **capability plumbing rather than arithmetic**, and each answers in one batch from
  a page: per-card memory is KV sized to `gpu_memory_utilization` on every rank, so "the whole model on
  each card" is expected and the pid table decides the shape (#1316, `memory-sizing-and-parallelism.md`);
  a request input the downstream dataclass never got, with its support landing in a named tag (#4505 →
  `216fc0e8e` `#3026`, first tag `v0.12.0rc1`, `missing-downstream-feature.md`); and a docs block that
  exceeds the hardware (#18158, `doc-guide-vs-capability.md`). The first two also show the two ways a
  "missing feature" claim goes wrong: reading an expectation as a defect, and grepping an old file path
  at HEAD after the code moved (`worker/model_runner_v1.py` → `worker/v1/model_runner.py`).
- Five rows are not vllm-ascend code at all: #8664 (upstream metrics accounting), #5894 (the installed
  xgrammar binding against the vLLM call site), #15836 (a truncated weight shard), #7971 (the upstream
  connector's layer-class filter) and #10944 (HCCP QP allocation under EP); #5603 and #2567 are operator
  contracts — an argument the weight-layout conversion contradicts, and a host scalar inside a captured
  stream. In these rows the deliverable is a version/layout statement plus the selector line, and the
  code read exists to prove the boundary — do not invent a `vllm_ascend/` mechanism for them.
- Two rows are **streaming/serving protocol** defects rather than NPU compute: #11788 (the terminal
  remaining-args delta of the `patch_glm*` backport, #8327) and #7682 (delta construction in the vLLM
  parser plus the serving finish backfill). Their value is the tag window — no GLM patch at
  `v0.17.0rc1`, #7710 at `v0.18.0rc1`, #8832 at `v0.19.1rc1`, #9410/#9787/#9788 and #9901/#9902 from
  `v0.20.2rc1`/`v0.21.0rc1`, `c4a26bbce` at `v0.21.0rc1` — and the non-streaming request is the free
  discriminator.
- The reduction-plumbing rows are the ones that present *without* a traceback worth reading: a `copy_`
  mismatch whose two row counts differ by exactly the TP factor (#8990), and a doubled TP all-reduce
  that only makes the text nonsense (#10466). Count the shapes and count the reductions.
- They show where the answer was *not*: several threads contain no technical reply at all, and a
  maintainer's "fixed by upgrading" often names a runtime dependency rather than a code change
  (#3357 needed CANN 8.3 even though the fix commit shipped in the reporter's own tag).
- **A negative result is also a row.** #13983's value is that two plausible October C8 fixes (#19064,
  #17594) live in code that did not exist in the reported 0.23.0 line, which is why the next pass can
  skip them; #3971's is that the fix shipped one rc later than the reporter's tag and the issue thread
  never said so.
- **Not every report is a code defect, and not every report is a bug.** #12059 is an image/`ENV`
  propagation defect in `Dockerfile*` whose fix state has two halves (jemalloc removed,
  `LD_LIBRARY_PATH` still open) — check both before calling it fixed or unfixed; #2566 asks why a
  newer release is faster, so answer with the range's performance commits and defaults in a few lines
  and spend no exploration on it.
- They show that the reporter's own material is a claim to verify, not evidence: #16237's quoted
  `_used_ports` code exists in no ref of either repo, and #3308's closing summary names two PRs that
  were never merged.
- The closing behaviour matters: a stale-bot close (#12533, #3971) is not a fix, and reporting it as
  one would be wrong. A `resolved` label (#11624) can also mean "expected behaviour explained", not
  "code changed".
- Attribution matters as much as content: #17082's fix statement is a maintainer sentence about CANN
  and the retest that "worked" changed two variables at once; #8975's flag advice is a maintainer
  hypothesis that still needs the per-rank check; #8990's "not supported for 27B" is a model-family
  carve-out that appears in no doc or code gate. Report both, with the caveat.
- They carry the two most reusable version facts in this stack: a runtime gate can be a **CANN version
  string** (#3357) or a **`hasattr(torch_npu, ...)`** check (#10872), and an image tag hides the CANN
  base — read `Dockerfile`/`Dockerfile.310p` at the tag before attributing a fix to vllm-ascend. A
  development version string's `g<sha>` suffix (#12059) pins the built commit directly, which is what
  to use when the image tag and `collect_env` disagree.

## Verification order: three shapes worth naming

- **A hardware-support question is answered from the tag's Dockerfiles, not from a code path (#1696).**
  "Does image X support 910C?" is `Dockerfile`/`Dockerfile.a3`/`.310p`/`.a5` at that ref plus the install
  page's kernel package; the failure (`AclOpKernelInit failed opType`) is CANN kernel selection, and the
  maintainer's base-image swap is the lever. `Dockerfile.a3` exists from `v0.10.0rc1`, so a v0.7.x image
  has no A3 target at all.
- **A flag that names a lifecycle can be only an allocator (#15560).** `enable_sleep_mode=True` with no
  `sleep()` call still corrupts the first offline call on `v0.18.0`, and the flag changes
  `PYTORCH_NPU_ALLOC_CONF` and the allocator rather than only sleep/wake. The reusable artefact is the
  reporter-runnable matrix (`enable_sleep_mode` × graph mode × TP), not a code line.
- **A hang with no traceback can be admission, not a collective (#16365/#15984).** `allocate_slots`
  failing leaves the request in `waiting` and head-of-line blocks the queue; the arithmetic that matters
  is blocks (pool size × the *smallest* group's `block_size`) against the
  `max_admission_blocks_per_request` cap, and the cap scales with `max_num_batched_tokens`, which makes
  that flag both lever and confirmation test.
- **Tree presence beats `tag --contains` in both directions (#15560, #16365).** A fix the thread cites can
  already be *inside* the failing tags (#12071/#12255/#13865 are all in `v0.23.0`), and a helper whose
  `tag --contains` starts at `v0.24.0rc1` (#12079 `copy_snapshot_to_gpu`) can still sit in `v0.23.0`'s tree
  as a release-branch cherry-pick. So neither the tag list nor its absence decides: read the tag's tree
  (`git grep -c <symbol> <tag> -- vllm_ascend`).
- **An *alternating* symptom is a timing race, and the fix is usually a deferral — and when the thread names no
  fix, the reporter's release-branch build sha is the discriminator (#12339).** #12339's P/D deployment answers
  well, then badly, then well: the receive thread applied the KV reformat as soon as one pull group ended instead
  of after all TP-offset pulls of that shard, so the losing order left the ranks disagreeing; the fix defers the
  reformat per (request, shard). A body pasted as `vLLM Ascend releases/v0.23.0 6cee2e163…` anchors the answer: `git merge-base
  --is-ancestor <branch-fix-sha> 6cee2e163` said *not an ancestor* while the same sha is in tag
  `v0.23.0`, i.e. an eight-day-old branch build and a rebuild as the lever. One change shipped three ways
  (#12371 `86217325b` on the branch, #12382 `b6826ec33` for 0.24, #12359 `1db626510` mainline,
  `v0.27.1rc1`) — quoting only the mainline first tag would have told a 0.23 user to wait for 0.27.
  The 「已解决」 date (2026-08-03) sitting *before* the mainline merge (2026-08-16) is what made the branch
  copy the candidate, so read comment dates against merge dates — and **a neighbouring PR number is not the
  twin**: #12359 is the other branch's half. The check is always `git merge-base --is-ancestor <fix> <the
  reporter's sha>`, never the tag number in the title.
- **When the reporter's own script builds the model input, grep it before any code window (#2040).** The
  `0.9.2rc1` caption-garbage report was one letter (`multi_model_data`), and vLLM accepts any dict with a
  `"prompt"` key while reading only `multi_modal_data`, so the `<image>` token decoded as text and nothing
  warned. A `[Usage]`-shaped defect whose evidence is *in the body* costs no clone read: the thread's own
  「测试脚本有问题」 plus one `file:line` is the whole answer, the reviewer's leverage is to ask for the
  script rather than derive a vision-tower theory, and five further batches on the exact drop site ship
  nothing.
- **A hang whose only error-looking line belongs to another subsystem is still one owner (#7154).**
  `No available shared memory broadcast block found in 60 seconds` at `shm_broadcast.py:542` is upstream
  vLLM's reader warning: the writer (NPU worker) never produced a block because it sat in a fused MoE op.
  Reading the line as a communication defect is the trap the `log-signatures.md` table exists to prevent.

## Ownership first, measurement second

- **Before blaming the plugin's monkeypatches, check whether the file is bundled upstream (#15911).** A
  whitespace-only `content` delta came from `vllm/tool_parsers/qwen3xml_tool_parser.py` at the image's
  `VLLM_TAG`; `git grep -ln qwen3xml <ascend-ref> -- vllm_ascend` is empty, and the file is gone from the
  next tag, so no `0.23.x` release can carry a fix. The check is one command and it decides the owner.
- **Two harnesses disagreeing is a measurement row, not a defect row (#10068).** Maintainers declined it,
  the harness team's own answer is "compare against the model, not against another tool", and the three
  named causes (chat vs completions + system prompt, default sampling, metrics) are what the reply ships.
  Never let a score delta become a code-level mechanism, and read a foreign harness file with the `github`
  tool (neither harness is cloned here).
- **A parallelism key in the reporter's command may have been replaced before the worker saw it (#1791).**
  With `--enable-expert-parallel`, `platform.py` at `v0.9.2rc1` overwrites
  `additional_config.expert_tensor_parallel_size` with `1` and derives `ep_size` from the whole world, DP
  replicas included. The thread's "fix" moved two variables at once (release bump + lm_eval prompt flags),
  so the honest row is the differential to run, not a cause.
- **A named fix lives on an rfc branch with a tag prefix and sometimes a different PR number (#15985).**
  `git log --all --grep='<PR title words>'` found the `v0.26.0rc2` cherry-picks (#16712→`155f68974`,
  #16714→`69e54c8b7`) that `git log --grep='#<N>'` and mainline ancestry both miss.

## Two arithmetic/queue mechanisms, and reports with nothing to find

- **Both mechanisms are verified in code, and both are "the same value in two domains".** #11231 indexes a *token* count into a
  *per-request* array (`num_scheduled_tokens[self.num_decode_tokens :]`, short by
  `num_decode_reqs × num_speculative_tokens`), and #8328 lets a request's *birth status*
  (`WAITING_FOR_FSM`) decide which queue it lands in while the selected scheduler subclass reads only the
  other one. Neither is a hardware, kernel or model question, and both are invisible in a healthy run:
  the first needs a mixed prefill+decode step plus speculation, the second needs one guided request.
- **The reported shape is the cheapest verification there is — do it first.** `(29,)` into `(31,)` plus
  `1 generation requests, 3 generation tokens` plus `Running: 32 reqs` fixes `num_reqs`, `num_decode_reqs`
  and `k` uniquely; that arithmetic *was* the proof, and it is what lets the reply go out immediately instead
  of after an ancestry hunt.
- **A `patch/platform/*` file that copies an upstream method is stale exactly where the copy is silent.**
  `BalanceScheduler.schedule` reproduces `Scheduler.schedule` line for line from an older vLLM and therefore
  reads no `self.skipped_waiting`; the test is mechanical — list the names the paired base class references
  around the copied method and `grep -c` each one in the patch. The fix family is three cherry-picks of one
  change, so the *first tag* differs per release line (`v0.22.1rc1`, `v0.23.0rc1`, and a 0.18-branch commit
  in no tag here).
- **Some reports have no mechanism to find, and hunting one is the trap.** A thread
  with a single maintainer "resolved in the latest version" line (#5188) and a cross-release timing number
  with an unfetchable env link and zero comments (#3045) both end at a labelled lever plus one ref-pinned
  fact (or a probe list); anything past that is invention, and neither the tag's code nor a PR title can
  supply a mechanism the report never exhibited. #3911 is the same shape one step harder: a bare aivec
  `DDR address of the MTE instruction is out of range` in an eager, non-PD run, where the message is a
  family (57 fork issues), the owner is a compiled CANN kernel, and the artifact that would name it — the
  plog's `fault kernel_name=…` — is not in the report. The reply is that plog ask plus the thread's
  attributed "use the latest release/rc"; four tool batches of `vllm_ascend/` greps and a tutorial read
  produced nothing.
- **A packaging report is about the ref it names, and the ref may be a branch rather than a tag.** #8686
  lists directories missing `__init__.py`; at the `v0.13.0` tag those directories do not exist at all
  (they arrive on `releases/v0.13.0` with #8648 after the tag), so "is the tag broken?" is the wrong
  question and the branch is the ref to read. The class rule from `setup.py:522` — `find_packages` needs
  `__init__.py` in *every* parent — is what makes one empty file lose a whole subtree.

