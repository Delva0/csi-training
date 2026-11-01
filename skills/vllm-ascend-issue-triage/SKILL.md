---
name: vllm-ascend-issue-triage
description: Triage vLLM Ascend bug, accuracy and deployment reports.
---
# vLLM Ascend issue triage

Turn a pasted `vllm-project/vllm-ascend` report (accuracy degradation, garbled output,
start-up `KeyError`, hang, crash, "is X supported") into an evidence-labelled finding: the code
path or documented constraint at the revision the reporter ran, a workaround, and the boundary
of what stays unverified. The reporter's machine is remote; what can be verified here is
tag-pinned source, that version's own docs and release notes, and the issue thread.

Clones are read-only under `/repos/<owner>/<repo>` (`vllm-project/vllm`,
`vllm-project/vllm-ascend`, and when the stack needs it `huggingface/transformers`,
`Ascend/pytorch` (torch_npu), `Ascend/op-plugin`, `cann/*`, `kvcache-ai/Mooncake`), with `main`
checked out — usually many releases ahead of the reported build. The report's thread and the tagged sources are the primary evidence, never the clone's
`main`.

## Read efficiently

- **Load this file once, then choose references by class.** The body stays in context,
  but it is not charged to the generated-token budget. Choose the reference with the
  table at the end, `skill_view` it with `file_path: 'references/<topic>.md'`, or read a single
  table out of it with `grep -n`/`sed -n`. `log-signatures.md`, `pd-disaggregation.md`,
  `quantization-startup-keyerrors.md`, `image-and-build-reports.md` and `kv-pool-ascend-store.md`
  are the largest (each ≥45 KB) and several others pass 20 KB,
  so prefer a grep/section window over loading one whole — but do not skip a decisively matching
  reference to save context.
- **Start from the report's own issue number before its symptom words.** These bodies are usually a
  verbatim copy of a GitHub issue and often end with its URL, so one
  `grep -rln '<number>' <skills>/vllm-ascend-issue-triage/references/` beside `skill_view` finds a
  recorded entry in a single call. A symptom-first route crosses references that share the wording
  ("garbled/mixed-language" matches the sampler material *and* the KV-lifetime section) and can walk
  past the entry; when the number is new, route by the phase signature instead. A bare hit is not
  always the entry: `10390` also matches `ascend_direct_transport ... status: 103902` and `HIXL Error
  Code: 103900`, so confirm it is the issue reference (`#<number>`, or the model name beside it)
  before reading that file as this report's record. A hit that is only a Boundary/caveat line is not
  the entry either — print the enclosing heading
  (`awk '/^## /{h=$0} /<number>/{print h}' <file>`) and read that section; when a new number is
  recorded, put it in the section that answers it, not only in the caveat.
- **The budget is the *total* generated output, reasoning included, so a truncated argument costs a
  whole step and executes nothing** (`Truncated call was not executed`). Long reasoning immediately
  before a big command is the usual trip (one recorded step spent ~4k characters of reasoning and
  lost both calls), as is two long compound bash commands in one step (a `for` loop plus two
  `git show | sed` pipelines). Keep each call short, split the work, and after a truncation re-issue
  a shorter call, not the same one. Unspent budget is not recovered either: a forward that reasons
  4-6k characters per step ends on `output_tokens` with **no answer written**. Measured across the
  recorded forwards, a whole forward gets ~11-13k characters (~4k tokens) of generated text
  (content + reasoning + tool arguments) over 6-9 rounds, i.e. ~1.5k characters per round
  *including* the call's arguments (`forward_budget.config.output_tokens`); one 4-8k-character
  reasoning burst therefore spends a third of it or more before any source is read. The wasted
  reasoning is usually speculation about a file never opened (4-6k on what evalscope's `mmlu_pro`
  prompt contains, 10k on how chat-template kwargs flow): fetch that one file — the benchmark's
  `_meta/<name>.json`, the model's tokenizer or patch at the tag, the harness YAML the reporter
  ran — then reason about the finding, not the library. Budget ~1.5k characters per step and
  reserve ~400-600 tokens (roughly the same 1.5k characters) for the finding, leading with the
  workaround and the boundary: the recorded forwards behind this calibration all ended on
  `output_tokens`, most with an empty answer while their findings were already in hand, and one
  that did write lost its final sentence mid-word past this reserve.
- **When a reference already matches the signature, verify only the lines the answer will cite**
  (two or three greps at the tag), then write. Once the raising statement, its guard and the
  documented workaround are verified, continue only for a named unknown that would change the
  answer: re-verifying a whole documented section spends the step budget the answer needs. A
  reference entry that already names the tag, file and statement *is* the
  verification record — re-running it to see whether it still holds is the most common way this
  budget is lost. Re-check only the lines the finding will quote, and only when the reported version
  differs from the recorded one. Put them in **one short call** and write the finding in the next
  message: a per-tag loop beyond the reported tag and the single boundary tag is re-deriving the
  reference's table, not verifying it (recorded forwards whose first line read "reference already
  matches this exact report class, verifying only the lines I will cite" then spent eight source
  rounds re-probing the per-tag table and delivered nothing).
- Use focused source windows: `git show <ref>:<path> | sed -n 'A,Bp' |
  cut -c1-160`, or `grep -n` plus a ~40-line window. Two renumbering traps: `sed -n '/def
  f/,/pat/p'` prints **to EOF** when the end address never matches (one such range returned 64k
  characters), and `nl -ba` renumbers from 1 — pass `nl -ba -v<start>`; `grep -n` applied to a
  `sed` window also numbers from the window's first line, so grep the whole `git show` output
  when the line number is going into the finding.
- **A missing ref reads as `0`, so assert it before a version boundary is closed.** When
  `git show <tag>:<path>` fails, `grep -c` still prints `0` and the result is indistinguishable from
  "the line was removed" — one recorded probe reported the 310P eager-only note absent at
  `v0.17.1rc1`, a tag that does not exist, and the boundary entered the reference a release early
  until a later pass re-ran it. Before reading a `0`, check `git rev-parse -q --verify
  refs/tags/<t>` and, when the file path itself was guessed, `git cat-file -e <t>:<path>`; a
  path-fallback chain (`cat-file -e A || f=B; ...`) otherwise reports the counts of whichever file it
  fell back to, labelled with the tag you asked about.
- Count before listing: `git log --oneline A..B | wc -l` first, then filter with `--grep`.
  Dumping a 167-commit range or 80 lines of release notes costs a round and context rather than
  budget. Put `-i`/`-E` **before** `--grep=<pat>`: `git log --grep -i -E 'a|b'` makes `-i` the
  pattern and `'a|b'` a revision, so it dies with `fatal: ambiguous argument`, and with
  `2>/dev/null` that prints as an empty match list. Treat a silently empty `--grep` as a syntax
  failure until stderr says otherwise.
- A network fetch over the 1 s bash timeout becomes a background job: pass a larger `timeoutMs`
  and put the download *and* the inspection in one
  `uv run --no-project python` call.
- **Deliver as soon as the finding is defensible; never hold it for a final round.** Recorded
  forwards that had already reached the decisive source lines still returned nothing — they ended
  on `output_tokens` mid-verification with an empty answer. Once the raising statement (or the
  capability/support gate), its guard and the workaround are named, hold them within three or four
  tool steps at most and write the answer in that same message; a replied partial finding with an
  explicit next check is the target, not a complete investigation. When the request asks for a root
  cause but the thread and the tag support only a class, write the class as the deliverable — the
  raising statement or documented boundary, the mechanism labelled as a candidate, and the one check
  that would settle it. For a start-up config check or a Python assertion, the raising statement
  plus the value the reporter's launcher supplies already *is* the finding — why that value differs
  from the tutorial's belongs in the answer as a labelled candidate, not in another round of source
  reading. Continuing to derive an untestable mechanism from first principles is how forwards that
  already held the answer end empty.
- **Only `content` is delivered; `reasoning_content` is discarded at the cutoff.** A forward whose
  message text stays at "I'll start by loading the skill" while the thread, the tag's docs and the
  decisive regex are all in its reasoning returns an *empty* answer even though every fact was
  already verified. Put the finding-so-far in `content` on the same step as the next tool call
  (two or three labelled lines: what is verified, at which ref, what the next check is), then keep
  going — so a budget trip still hands over the finding, and the next step only extends it. Write
  that line on **every** step after the first verification, not once: recorded forwards that put a
  single content message mid-run reverted to reasoning-only steps afterwards and were cut off with
  only that stale half-finding delivered.
  In the message that carries the answer, write `content` **only, with no reasoning**: the one
  recorded forward that did answer spent ~3.4k characters of reasoning in that same message and
  delivered 110 characters of the finding (a heading plus "Pinned environment:" before the cut), so
  the reasoning consumed the residue before the answer began. Open with the workaround sentence and
  the pinned ref, not with a title or a plan.

## Work in this order

1. **Extract the report's coordinates.** vllm-ascend and vLLM version from `collect_env.py`
   (not the title), model and backend type (dense / MoE / hybrid GDN+Mamba / VL /
   pooling-embedding), hardware (310P = Atlas 300I DUO, A2/A3, A5), `npu-smi`/CANN/HDK, topology
   (PD split, DP/TP/PCP/DCP), the complete `vllm serve` line, and the failure stage (start-up,
   first request, steady state). When two configs are given, the delta is the hypothesis and its
   flag names are the search keys; subtract what both share first (a shared flag cannot be the
   delta), then keep only the names the reported tag actually reads — a `VLLM_ASCEND_*` name with no
   hit in that tag's `vllm_ascend/` has no plugin reader and cannot be the delta (upstream's
   unknown-env warning is cosmetic), while `HCCL_*`/`ATB_LLM_*`/`MIES_*` belong
   to CANN/torch_npu/ATB/MindIE and a plugin-tree miss proves nothing
   ([env-vars-and-additional-config.md](references/env-vars-and-additional-config.md)). Check that
   the model path, title and flags agree — maintainers
   flag that first. Pasted chat-completion JSON is evidence: `system_fingerprint` and the request
   parameters show the serving version and parallel mode actually used. When the image tag and
   `collect_env.py` disagree (often a `...dev<N>+g<sha>` build string), keep both and pin the
   image, which is what the reporter ran. The two versions must also pair: the plugin tag's
   `Dockerfile` states the upstream release in `ARG VLLM_TAG` (e.g. `v0.23.0rc1` -> `v0.23.0`,
   `v0.10.1rc1` -> `v0.10.1.1`), normally the matching `vX.Y` line; on the 0.9 line it is the
   identical string (`v0.9.1` -> `ARG VLLM_TAG=v0.9.1`). A `vllm-ascend 0.23.0rc1`
   reported with `vllm 0.11.x` is a mixed install — say so first, because every code pin below is
   then only half-anchored. A comparison against `main` moves this pin too — the plugin `main` of
   2025-09-19 pins `v0.10.2` while `v0.9.1` pins `v0.9.1` — so resolve `VLLM_TAG` at the `main`
   commit as well, or a difference between the two runs has two unseparated causes. The CANN/HDK
   the title pairs with the plugin is usually the image's own
   pin (`FROM quay.io/ascend/cann:9.0.0-910b-...` in the `v0.22.1rc1` Dockerfile), so a report
   naming CANN 9.0.0 is not a version mismatch — check the `FROM` line before calling one.
2. **Read the issue and its comments before any code, in one call** — the issue object with *no*
   `json_pointer` (`"/"` comes back `invalid_view`), the comments array as its own projection
   (`json_pointer: ""` plus a `fields` map; `/items` is only for the `search` arrays) — never add
   `paginate` to that `fields` read, which returns empty `data` with `missing_projection_field`;
   drop `fields` and read compact with a higher `max_chars` instead.
   Maintainer comments are the
   cheapest workaround and often name the mechanism; reporter follow-ups usually hold the
   workaround that worked. A `not_planned`/stale close, a "more info needed" reply or silence
   means **no maintainer root cause exists** — use their questions as a checklist, never as a
   confirmed fix — and "this is a known issue, we'll fix it asap" plus a stale close is the same
   shape: a maintainer's acknowledgement, no mechanism. Projection, pagination, duplicate-redirect
   chains, 众测 task-book threads and
   attachment limits are in [issue-thread-reading.md](references/issue-thread-reading.md); get
   the shapes right the first time, because a mis-projected read costs a round.
3. **Pin the reported version to an exact ref.** Release name → tag
   (`git tag --list | grep -i '<version>'`) and release line (`git branch -a | grep releases/`);
   rc builds normally have an exact tag, and from the 0.13 line an `origin/releases/vX.Y.Z`
   branch too — 0.9-0.12 exist as tags only. Image suffixes `-310p`, `-a3`, `-a5` are variants of
   the bare version and are registry-only (no such git tag); the `-a3` family starts at
   `v0.10.0rc1`, so a 910C host on an earlier bare tag runs the 910B base image
   ([model-tutorial-recipes.md](references/model-tutorial-recipes.md)).
   `org.opencontainers.image.revision` names the baked-in commit when a
   registry query is available (`quay-image-versions` skill). Prove the pin by reproducing the
   traceback's file, **statement** and line, not the version string; a lone log line can pin too.
   If the tree lacks the tag, use the nearest tag and state the residual uncertainty. A model whose
   plugin code exists in no plausible tag is a dev/branch image: date it, answer from the earliest tag
   that has the feature, and retire a `system_fingerprint` hex with one `git cat-file` probe. Procedure,
   tag/commit map, the two-statement loop and `dev<N>+g<sha>` resolution:
   [revision-pinning.md](references/revision-pinning.md).
4. **Read that tag's own docs, then diff the serve line against the documented recipe.** Docs in
   the tree are version-accurate; the published `latest` site is not.
   `docs/source/user_guide/release_notes.md` (per-release Known Issues and Dependencies),
   `docs/source/tutorials/models/<Model>.md` and
   `docs/source/user_guide/configuration/additional_config.md` answer a large share of reports on
   their own. Grep the tag's own file for the issue number first (`git show
   <tag>:docs/source/user_guide/release_notes.md | grep -n '#10963'`): a tag's own file often does
   carry its own section and can list the report verbatim under Known Issues (`v0.22.1rc1:44`
   reproduces that title, "With CANN 9.0.0" included), which is the whole "documented at that tag"
   finding. **Put that grep in the same call that pins the tag, before any `csrc/` or runtime-code
   read** — `v0.23.0:release_notes.md:143` already names two custom ops, the `rtMemcpy` 107030 plog
   noise and "does not affect inference", and forwards that opened the ops first spent the whole
   budget re-deriving a documented answer
   ([full-decode-only-graph-mode.md](references/full-decode-only-graph-mode.md)). Only when it
   greps clean look one tag **ahead**: some sections first appear in the next
   file (`v0.20.2rc1`'s own tag is clean; its section sits in `v0.21.0rc1`). **Read the enclosing
   `## vX.Y.Z` header of every hit**: the file accumulates all older sections, so a
   `grep -i <model>` hit can belong to a release several versions back. At `v0.8.5rc1` the
   "garbled code ... change the generation config" line (`:117`) sits in the `## v0.7.3rc1` section
   and that tag's own section carries no Known Issues; at `v0.20.2rc1` a Kimi hit at `:521` is the
   `## v0.15.0rc1` section's note that *upstream vLLM 0.15.0* had a Kimi-K2.5 issue, while that
   tag's `Dockerfile:51` is `ARG VLLM_TAG=v0.20.2`. Attribute a hit only to the section containing
   it, and check any upstream version it names against the tag's `ARG VLLM_TAG`. A
   `Highlights`/`Features` bullet is not availability proof, though: a feature PR can be reverted
   before the cut and stay advertised,
   so grep its code path at the tag and check the revert's ancestry before recommending it
   (`v0.17.0rc1` advertises W8A8C8 for PD-mix while `#7029` was reverted by `#7288` inside that
   tag; [c8-dsa-deployment-flags.md](references/c8-dsa-deployment-flags.md)). A serve line
   byte-identical to the tutorial points at a documented-path bug or a stale doc. Contradictory
   flags are themselves the finding: `--enforce-eager` with
   `--compilation-config '{"cudagraph_mode": ...}'` logs
   "Enforce eager set, disabling torch.compile and CUDAGraphs", so graph behaviour cannot be
   inferred from that run. A maintainer-named suspect flag: grep the tag's docs for it before
   reading code. Grep the same file for the serve line's `--additional-config` keys
   (`git grep -n '"<key>"' <tag> -- docs`): a retired key prints "... doesn't work anymore" in
   the release that removed it (`torchair_graph_config` and `ascend_scheduler` in `v0.12.0rc1`),
   and the run then used the default mode — do not build a mechanism out of how that key used
   to behave (`torchair_graph_config` on any `>= v0.12.0rc1` tag is ACLGraph, not a cached
   torchair graph).
5. **Trace the raising statement, then read each reported flag's guard at that tag.** Name the
   statement that raised, not just the frame; with no traceback but a pasted log line, grep its
   *format string* in both trees (`git grep -n 'KV pool load spec created' <ref>`) — the emitting
   function names the component, its placeholders (`load_async=%s`) the setting. A small offset
   means the installed tree is close to but not identical with the tag, and a snippet absent at the
   tag is the reporter's paraphrase.
   The plugin subclasses and patches upstream, so grep **both** trees: the traceback may name
   `vllm/...` while the executed override lives in `vllm_ascend/`. For a model-specific failure,
   list the model's hooks first (`git ls-tree -r --name-only <tag> | grep -i 'patch_<model>'`).
   Read the guard, default and layering of every flag or `--additional-config` key at the pinned
   ref; `vllm_version_is(...)` marks exact upstream layout boundaries and therefore pins the
   layout; `git log -S'<statement>'` dates a behaviour change. Environment gates decide branches
   the same way (a CANN-version check chose the attention mask at `v0.11.0rc0`), so read the gate
   together with the reported runtime. An **environment variable** a launcher stops exporting is
   not necessarily absent: the plugin re-applies some defaults itself
   (`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` unless `enable_sleep_mode` or the launcher
   already sets `max_split_size_mb`/`garbage_collection_threshold`, which suppresses the default), so "I unset X
   and it started" can be a no-op delta, and a report that drops two exports has isolated
   neither. Configuration surfaces and lifecycles:
   [env-vars-and-additional-config.md](references/env-vars-and-additional-config.md).
6. **Search forward for a fix and prove where it landed.** Restrict to the line that continued
   the report: `git log --oneline <tag>..origin/releases/<ver> -- <paths>` and
   `git log --all --oneline --grep='<flag-or-issue>'`. For each candidate ask both questions:
   `git merge-base --is-ancestor <sha> <ref>` (in the build?) and `git tag --contains <sha>` (in
   a release?). A commit reachable only from `main` is not an upgrade path; a fix only on a
   release branch is a backport, not proof that `main` is unfixed. Read the candidate's message:
   its testing section bounds what it establishes ("stability and output-consistency checks, not
   a formal accuracy evaluation" cannot be cited as the cause of wrong output). Resolve every
   number a comment cites to a commit before repeating it
   ([revision-pinning.md](references/revision-pinning.md)).
   **Search the owning file, not only the issue number.** `git log --all --oneline --
   <the file whose statement raised>` finds the commit whose *body* says `Fixes #<n>` even when no
   subject or `--grep` hit names the issue — two recorded hits: `attention/fa3_v1.py` →
   `5e1647cf9` ("Fixes #17991"), `ops/triton/bincount.py` → `f95a31ec2`/#10138. When the fix is
   main-only (`git tag --contains` empty), say that and offer the code-path workaround; an "upgrade
   fixes it" claim needs a tag containing the commit.
7. **Write the answer in a fixed shape, as soon as the above holds.** A few hundred words, not
   a report: symptom →
   mechanism at the pinned ref with `file:line` → documented recipe, workaround or flag
   replacement → one evidence label per line → the single check that would confirm or refute the
   candidate. Label each statement: **verified in the tag's source**, **documented at that tag**,
   **stated in the thread** (maintainer vs reporter), **candidate needing a runtime check**,
   **availability only**. Name what refutes each candidate (same prompt with the flag off, a
   numerics comparison, the worker-side log, a `hasattr`/device-memory check). Repository pins
   are intended versions; installed versions need in-container collection (`pip show`, source
   path, SHA, dirty state). Deepen only for a *named* unknown that changes the answer.

## Rules that apply to every report

- **`finish_reason: "stop"` with a token id in `stop_reason` is not a length cap, whatever the title
  says.** "Output is truncated by default" bodies route to
  [served-context-length.md](references/served-context-length.md) only when `finish_reason` is
  `length`; a `stop` at a `completion_tokens` well under the cap, with `content` breaking
  mid-phrase, is the sampler stopping on a token id, not the context window. Then read the launch
  script before any sampler theory: a `--compilation-config` entry
  `"pass_config": {"enable_sp": true}` without `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` is one recorded
  trigger with a reporter-verified fix (delete the entry, or export the env var); the host-side
  mechanism is still open
  ([env-vars-and-additional-config.md](references/env-vars-and-additional-config.md)).
- **A "first request / first `generate()` only" failure is its own class — read the first call's
  `token_ids` and `finish_reason` before any source.** Empty `content` with `completion_tokens: 1` is
  an EOS, an all-id-0 first run is a garbage fill, and a first-decode crash whose tail is
  `AttributeError: 'NoneType' object has no attribute 'size'` at `torch/_dynamo/utils.py call_size` is
  the `--language-model-only` `_dummy_run` warm-up mismatch (fixed on `main` by #8556, 0.18.0 backport
  open) rather than the ACL graph. `enable_sleep_mode` is the thread-verified trigger for the offline
  wrong-token/id-0 form, and `enforce_eager` inside `--speculative_config` only gates the MTP
  **drafter**. Never report the MTP method *name* (`qwen3_next_mtp` vs `qwen3_5_mtp`) as a defect —
  both normalize to `mtp` at the reported ref:
  [first-request-only-output.md](references/first-request-only-output.md).
- **A repeated sleep/wake device-memory-growth report is settled by one `git tag --contains` on the
  commit that added reclamation, not by reading the allocator.** Reclamation lives at the end of
  `CaMemAllocator.sleep` (`vllm_ascend/device_allocator/camem.py`), and `gc.collect()` +
  `torch.npu.empty_cache()` are appended there by #7709 (`e3c2d42e4`), first tag `v0.19.1rc1`; so a
  reporter on `v0.18.0` who believes they already have #7709 is contradicted by the tag. Give the
  first tag with the fix plus the reporter-confirmed `torch.npu.empty_cache()`-after-`sleep`
  workaround, and keep the residual "N MiB reappears after `wake_up`" as a candidate
  ([npu-workspace-memory.md](references/npu-workspace-memory.md)).
- **A usage report that asks *whether* something is supported — or whether two connectors can be
  enabled together — is answered from locators before any code read.** Three calls: the tag's own doc
  for the configuration it prescribes, the connector registry at that tag
  (`vllm_ascend/distributed/__init__.py`, later `distributed/kv_transfer/__init__.py`) for the names
  that resolve and what they alias, and `git tag --contains` on the commit that enables the
  combination (a release-note bullet can lead or lag it). Open the raising file only to pin the
  reported line, not to explain the feature. Recorded "is X supported" routes burned their entire
  output budget inside 1000-line transfer connectors after the thread and the tag's docs had already
  answered, and wrote no finding at all
  ([kv-pool-ascend-store.md](references/kv-pool-ascend-store.md)).
- **A MemCache-backed KV-pool report (`MemcacheBackend`, `ock.mmc.*`, `memcache-hybrid`,
  `MMC_LOCAL_CONFIG_PATH`) is decided by *which init path* the reported model and protocol take,
  not by the pasted HCCL log.** `_lazy_init` is `lazy_init and _is_device_sdma()`, and `lazy_init`
  means all models before `v0.23.0` but only DSV4-compress models after it: so a DSV4-compress model
  on `device_sdma` defers store init into the first `get`/`put` (crash on request 1-2) while a
  non-compress model or a `host_*`/`device_rdma` protocol initializes eagerly in `__init__`
  (start-up failure at the bare `assert res == 0`, which has no message — the reason is in the SDK
  log). Then check two commits rather than the connector
  code: `c970e5892`/#16287 (device id from the process-group local rank instead of the bound NPU;
  first tag `v0.28.0.rc1`) and `93c49251f`/#17289 with cherry-pick `832e63c85`/#17308 (DP-memcache
  init barrier, first tag `v0.27.1rc1`, reverted on `main` by `9ccd283e5`/#19370, which instead
  requires `memcache-hybrid`/`memfabric-hybrid` >= 1.3.0). Quote the protocol/path mapping and
  `git tag --contains` for the named commit; the thread usually has only pings
  ([kv-pool-ascend-store.md](references/kv-pool-ascend-store.md)).
- **A "memcache barely helps / slower than the Mooncake-only baseline" report with no error is a
  plugin version boundary, not a scheduler defect:** the local-prefix overlap it describes is
  `9e45da191` (#12783) and it is in `v0.23.0`, not `v0.23.0rc1` — check `git tag --contains`
  ([kv-pool-ascend-store.md](references/kv-pool-ascend-store.md)).
- **A reported capability, determinism or accuracy gap is a feature/configuration gate before it
  is a kernel question** (an availability question with no failing run takes the locator route
  above instead). Check whether the feature is documented, whether it is on by default,
  what it requires at build time and runtime, and which inputs its own docs exclude. Examples:
  [batch-invariance.md](references/batch-invariance.md) (`VLLM_BATCH_INVARIANT=1` at build *and*
  runtime plus `cudagraph_mode: PIECEWISE`), the reasoning-mode default of a tokenizer mode
  ([deepseek-v4-flash.md](references/deepseek-v4-flash.md)), and a `FULL` graph request on a hybrid
  linear-attention model, which the attention backends' minimum capability downgrades so prefill is
  never captured ([full-decode-only-graph-mode.md](references/full-decode-only-graph-mode.md)).
- **An EAGLE/EAGLE3 draft model that hangs "at the draft model stage" is first the draft-TP
  configuration gate.** The paired upstream vLLM `docs/features/spec_decode.md` says EAGLE draft
  models must run without tensor parallelism, and `_verify_and_get_draft_tp` defaults
  `draft_tensor_parallel_size` to the **target's** TP for every method except `mlp_speculator`, so a
  `tp=2` eagle3 serve enters the unsupported path and raises nothing (the plugin loads the draft
  with the target's parallel config). Offer `"draft_tensor_parallel_size": 1` before reading the
  proposer, and do not answer "eagle3 unsupported at that tag" when the tag's notes say it is back
  ([spec-decode-metadata-failures.md](references/spec-decode-metadata-failures.md)).
- **A score that is lower than another harness or than the published table has no failing run,
  so the measurement is the suspect.** Diff the two harness configs — prompt template and
  system prompt, whether and with which kwargs the chat template is applied, few-shot count,
  sampling defaults, answer extraction, truncation — then replay the same prompts through both
  templates; a model-level finding needs a prompt whose output visibly differs
  ([accuracy-eval-harness-gaps.md](references/accuracy-eval-harness-gaps.md)). **An rc→final
  comparison of the same deployment is the reverse case: grep the later tag's own `### Known Issue`
  block for the model family and topology first** (`v0.18.0` names Qwen3.x PD-disaggregation accuracy,
  #8421, and the FUSED_MC2 multi-DP padding caveat, #8320); the rc tag often has no section for that
  release at all, and a maintainer's "fixed by #N" still needs the ancestry check
  ([revision-pinning.md](references/revision-pinning.md)).
- **An accuracy collapse that appears only when `data_parallel_size`, `enable_expert_parallel`
  or `expert_tensor_parallel_size` is added is a MoE communication question.** Two MoE forwards
  coexist in the 0.9 line and only the plugin-owned one carries the DP/TP/EP collectives:
  establish which one the model's expert blocks instantiate, then whether that path
  communicates ([moe-dp-ep-accuracy.md](references/moe-dp-ep-accuracy.md)). From 0.17 on, check
  the flag pairing first: garbled **only** with `--enable-expert-parallel` is the plugin's own
  EP + `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` case, since `enable_sp()` is what moves the ALLGATHER MoE
  prepare/finalize onto the EP group — after confirming which strategy
  `select_moe_comm_method` picks for that hardware and topology.
- **"One request good, the next bad" on a quantized KV cache in PD is the C8/MXFP8 KV-scale
  transfer class, fixed only in `v0.30.0`** (`e6c0f4b7d`/#19064 on `main`, `14c6c3f43`/#19017 in the
  tag, absent from `v0.30.0rc1`; V-scale restore tag-only). C8 is declared by the checkpoint
  (`kv_cache_type == "C8"`), not by a CLI flag, and its attention file does not exist before
  `v0.30.0rc1`, so an older tag's alternating report is a different defect
  ([pd-disaggregation.md](references/pd-disaggregation.md)).
- **A VL checkpoint's text that is garbage — or evidence of "qkv weight/bias are all zeros" — only
  at TP>1 is the vision tower's padded qkv, not the decoder, sampler or checkpoint.** Head dims in
  `(64,128)` are padded to 128 and the inherited upstream `split_qkv` all-gathers/re-splits that
  padded tensor across TP (fixed `v0.9.0rc1`+, `patch/worker/patch_qwen2_5_vl.py` from
  `v0.12.0rc1`); `platform.py` also refuses Flash Comm V1 for VL models, so the EP-mode workaround
  above does not apply to them
  ([vision-tower-accuracy.md](references/vision-tower-accuracy.md)).
- **Before any vision-tower reading, check that the image was passed.** An offline `LLM.generate`
  dict prompt is accepted on `"prompt"` alone (`vllm/inputs/parse.py` at `v0.9.2`…) and only
  `multi_modal_data` is propagated (`llm.py:625-627`), so a misspelled key runs text-only with the
  `<image>` id as an ordinary token — client-side, not the plugin
  ([vision-tower-accuracy.md](references/vision-tower-accuracy.md)).
- **A "reasoning parser broken / `reasoning_content` null / thinking lost" report is an API-contract
  question first.** vLLM emits the split thinking under its own `reasoning` field at every tag checked
  (`v0.18.0`-`v0.30.0`), and accepts `reasoning_content` on *input* only from `v0.22.0rc1`, so a
  client (or an echoed history) using that key shows no thinking while the parser works;
  `thinking_token_budget` is a sampler-side holder, silently absent on Ascend until `v0.27.1rc1`:
  [serving-toolcall-reasoning-streaming.md](references/serving-toolcall-reasoning-streaming.md).
- **A "quantized but no faster / even slower" report is an expectation and maturity question before
  it is a defect.** Check the flag, whether the layers were actually quantized (description values,
  NZ gate), the size of the model and the tag's own maturity statement before reading a kernel;
  small dense models can legitimately gain nothing and upstream prints its "not fully optimized
  yet" warning for `ascend` by design
  ([quantized-accuracy-and-quarot.md](references/quantized-accuracy-and-quarot.md)).
- **A warning with no emitter in either tagged tree is a runtime-shim message, not a code lead.**
  The `torch.save with "_use_new_zipfile_serialization = False" is not recommended for npu tensor`
  line comes from `torch_npu/utils/serialization.py` wrapping `torch.serialization._legacy_save`,
  reached through Torch's storage `__reduce__` when an NPU tensor is pickled across a process
  boundary — vLLM `v0.11.0` and the plugin at that tag contain no such call, and recorded forwards
  lost their budget grepping the repos for it. Read it as "which tensor crossed which process"
  ([pooling-embedding-rerank.md](references/pooling-embedding-rerank.md)).
- **A BERT-family pooling report (`bge-*`, XLM-RoBERTa, `--task embed|score`) dies in an engine or
  scheduler support gate — or hangs with no exception while the access log says `200 OK` — before
  any kernel is the suspect**, in one of four terminal shapes (V1 `SupportsV0Only` oracle,
  `core/scheduler.py:209` force-routing into `AscendScheduler`, argparse `invalid choice: 'rerank'`
  because the task name is `score`, the `/v1/embeddings` client hang beside an NPU-tensor
  serialization warning; a Qwen3-Embedding serve on 310P fails in weight loading with
  `ValueError: There is no module or parameter named 'layers' in CustomQwen3ForCausalLM` until
  `v0.20.2rc1`, #8846). State the release *and* the engine
  (V0/V1), plus the first tag containing the fix:
  [pooling-embedding-rerank.md](references/pooling-embedding-rerank.md).
- **A start-up `TypeError: <Model>Processor.__init__() got multiple values for argument 'tokenizer'`
  is a transformers processor-attribute bug, not the checkpoint, the plugin or a mixed install.**
  Only transformers 4.53.0 builds the positional and keyword `tokenizer` twice (fixed by #39125,
  first tag `v4.53.1`), and vLLM's own skip compares the version string for exact equality, so a
  source or dev build of the same tree walks past the guard; pin `transformers==4.53.1`/`4.53.3`
  and reproduce the two statements rather than the reported version:
  [model-type-recognition.md](references/model-type-recognition.md).
- **A fork can simply lack an upstream request-input path.** When the request carries a field
  upstream v1 supports (`prompt_embeds`, new sampling metadata, encoder inputs) and the engine
  dies inside `npu_input_batch.py` / `_update_states`, the plugin's copied state is older than
  the paired upstream file: zero hits for the symbol in `vllm_ascend/` is the signature
  ([prompt-embeds-and-input-batch.md](references/prompt-embeds-and-input-batch.md)).
- **An upstream method with the same name and a different contract is a plugin/upstream seam, not
  a kernel fault.** When upstream adds or reshapes a method in the paired release while a
  `vllm_ascend/patch/` subclass keeps the older signature, the caller unpacks the old shape and
  raises a bare `TypeError` (e.g. `max(per_group_hits)` on an `int`,
  [pd-disaggregation.md](references/pd-disaggregation.md)). At start-up the same seam names the
  *upstream* keyword: `TypeError: <Subclass>.__init__() got an unexpected keyword argument '<kw>'`
  from `vllm/v1/engine/core.py` means a plugin `scheduler_cls` target enumerates its ctor while
  upstream added a parameter; grep every `Scheduler` subclass and read the sibling that already
  carries the optional-kwarg shim, which dates the upstream change
  ([env-vars-and-additional-config.md](references/env-vars-and-additional-config.md)). Compare
  the method's return annotation in both trees at the paired tags before reading attention or
  KV-cache code, and ask which release the plugin text is copied from.
- **Do not carry `main` findings back to an older report** (paths are renamed and split between
  releases), and do not carry a reported tag's behaviour forward as `main`'s. Test at the
  reported ref.
- **A defect in a file the released image bundles from upstream vLLM is answered from three
  locators, not a plugin hunt**: the tag's `Dockerfile` (`ARG VLLM_TAG=<upstream tag>`, cloned into
  `/vllm-workspace/vllm`), the tag's `vllm_ascend/patch/` subset for that component, and upstream
  `<tag>..<next-tag>` history (`git log --diff-filter=D`, `git ls-tree`, `git tag --contains`). No
  patch in that list means the answer is "upstream file at vX, removed/fixed by <sha>, first tag Y"
  plus an image (bundled vLLM) bump, not a plugin flag
  ([serving-toolcall-reasoning-streaming.md](references/serving-toolcall-reasoning-streaming.md),
  [image-and-build-reports.md](references/image-and-build-reports.md)).
- **`AssertionError: (-1, '<req>', {'<req>': [-1, -1, -1]})` from
  `vllm/v1/structured_output/__init__.py::grammar_bitmask` is a speculative-token placeholder, not a
  grammar defect.** `grammar_bitmask` advances the grammar over every entry of
  `scheduled_spec_decode_tokens`, and xgrammar rejects `-1`; whatever pre-filled
  `request.spec_token_ids` — upstream `AsyncScheduler` under `--async-scheduling`, or the plugin's
  `RecomputeScheduler.add_request` under `additional_config.recompute_scheduler_enable` on a PD
  decode node with MTP — is the owner. Both are opt-in, so a log-only body cannot choose between
  them: ask for the D node's launcher flags. Upstream stops advancing at `-1` from `v0.14.0rc1`
  ([structured-output-guidance.md](references/structured-output-guidance.md)).
- **A reporter's own patch is an unvalidated experiment, and "the fix PR was merged" is a
  claim.** Check `state`/`merged` and `git log --oneline <tag>..origin/releases/<ver> -- <path>`;
  an on-site patched build leaves the branch untouched. A maintainer's "not a known issue" means
  the evidence is still missing, not that the report is wrong. **A cited PR also only applies if
  its preconditions are in the report's own command and request** — speculative decoding for the
  grammar / reasoning-end bug, a `response_format` (not merely a `tool_choice`) for anything
  xgrammar-related at 0.18, PCP for a CP metadata crash. Check them before tracing that PR's code,
  or the whole line of investigation is spent on a configuration the reporter never ran.
- **A screenshot-only report has no readable exception; do not manufacture one.** With a body of
  `<img ...>` and no log, take the class from the thread's follow-ups, the tag's known issues and
  the call site the reporter can name, label it a candidate, and answer with that class.
- **A report that is log-free, or whose only evidence is a plog warning the tag's notes call
  harmless, and that matches a release-note Known Issue — or a reference's own
  retirement/boundary section — is answered from that entry, not by a mechanism hunt.** One grep at
  the tag plus the thread gives the symptom, the scope the entry names (CANN/model/topology) and
  the boundary (no root cause, no fix tag): write that, name the configuration that avoids the
  affected path, and label the mechanism unknown. Recorded
  forwards on such a report (`v0.22.1rc1` + CANN 9.0.0, a 1P1D long-context D-node hang plus a
  single-node MC2 OOM) found the entry within three steps, then spent the remaining budget grepping
  MoE dispatch paths for a cause the log-free body cannot pin, and delivered nothing. "D node hang"
  and "single curl OOM" with no traceback, plog or serve line do not license a kernel-level root
  cause. A body that ends at `EngineDeadError: EngineCore encountered an issue. See stack trace
  (above)` has the engine-side traceback outside the paste: name the class and stop. The same
  holds for `Exception: WorkerProc initialization failed due to an exception in a background
  process. See stack trace for root cause.` — `wait_for_ready` raises that fixed text `from None`
  on `EOFError`, so the worker's traceback never reached the engine. Read the thread's later
  reporter comments **once** before stopping: that is where the worker paste usually lands (a
  recorded `Engine core initialization failed` body carried the `KeyError` and its
  reporter-confirmed config fix only in comment 3;
  [quantization-startup-keyerrors.md](references/quantization-startup-keyerrors.md)). A retired
  connector likewise logs its transfer exceptions instead of re-raising them into the engine step
  ([pd-disaggregation.md](references/pd-disaggregation.md)).
- **PD and deployment reports often paste no configuration at all** (a body of `collect_env.py`,
  `npu-smi` and a Feishu/QR link is normal here). Do not reconstruct the serve command from it
  and do not read the absence as a defect; work from the tagged source and the thread.
- **A start-up `ValueError: KV transfer '<role>' config has a conflicting data parallel size.
  Expected 1, but got N` is the plugin validating `kv_connector_extra_config`, not the connector.**
  `vllm_ascend/utils.py::check_kv_extra_config` compares each `prefill`/`decode` sub-dict against the
  *engine-local* parallel config, and upstream resets DP to 1 in the engine process for a **non-MoE**
  model — so a MoE tutorial's `dp_size` fails on the first dense PD node. Fix the deployment
  (`dp_size: 1`, keep both keys: the mooncake connector asserts them), not the check
  ([pd-disaggregation.md](references/pd-disaggregation.md)).
- **A 404 on `/v1/messages`, `/v1/models` or `/metrics` behind the PD example proxy is the proxy's
  route list, not the engine.** Enumerate `@app.get/@app.post` in
  `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` at the tag: `v0.18.0` has
  only completions/chat/healthcheck/instances, `/v1/models` first ships in `v0.27.1rc1` and
  `/v1/messages` still has no route, while the paired vLLM attaches the Anthropic router
  unconditionally. Advise calling a backend port or a gateway rule
  ([pd-disaggregation.md](references/pd-disaggregation.md)).
- **A start-up `torch._dynamo.exc.Unsupported` that names a Python builtin is a host side effect
  inside the compiled forward, not a model or CANN fault.** vLLM compiles the language model
  with `fullgraph=True`, so a skipped builtin raises instead of graph-breaking. Read the
  traceback's `from user code:` tail; the recorded instance is
  `os.environ["TRITON_ALL_BLOCKS_PARALLEL"]` in `AscendMRotaryEmbedding.forward_triton` once a
  step exceeds 65535 tokens, unfixed through `main`
  ([torch-compile-graph-breaks.md](references/torch-compile-graph-breaks.md)).
- **A start-up ending in `Failed to compile .../launcher_cxx11abi1.cxx` (or
  `use of undeclared identifier '<aclType>'`) is the torch-npu wheel's own header, whatever the
  model.** It arrives as `Worker failed with error 'NPUModelRunner init failed, error is ...'` under
  a `compile_or_warm_up_model` RPC, while the reporter's visible paste is often a `torch.fx`
  compiled-forward (`<eval_with_key>`) traceback that reads like a graph-capture crash with the real
  error only in an attached log (`## log detail as attachment`). Fetch that attachment and grep its
  tail before entering a capture class; wheel version table and fix in
  [deepseek-v3.2.md](references/deepseek-v3.2.md).
- **A native import that aborts inside a shipped image is a Dockerfile delta, not serving code.** If
  the failing command never imports the plugin (`python -c "from mooncake.store import ..."` →
  `corrupted size vs. prev_size`), diff the image's build step across the reported tag and the next
  release, and quote the reporter's probe
  ([image-and-build-reports.md](references/image-and-build-reports.md)).
- **A start-up `ModuleNotFoundError: No module named 'triton'` behind `Model architectures [...]
  failed to be inspected` is the image removing a package that a model module imports at module
  scope.** vLLM inspects the architecture in a subprocess, so the failure surfaces as
  `CalledProcessError` → `ModelConfig` `ValidationError` and the outer line never names the
  package — read the last `ModuleNotFoundError` in that subprocess traceback. On the 0.10 line a
  GDN/Mamba hybrid imports the plugin's conv1d/fla files at import time
  (`vllm_ascend/ops/casual_conv1d.py` — the tag's misspelling — and `ops/fla.py`), while the tag's
  `Dockerfile` runs `pip uninstall -y triton` on purpose. Answer with that tag tutorial's "Install
  Triton Ascend" step (Ascend BiSheng toolkit + the pinned `triton_ascend-*.whl`; a mismatched
  wheel breaks `torch_npu` import with `cannot import name 'OutOfResources'`), never `pip install
  triton`. Bound it: the inspection failure is 0.10-line (from `v0.11.0rc1` those imports go
  through `vllm.triton_utils`' placeholder), and the image installs triton-ascend from
  `v0.18.0rc1`, so a later tag means a hand-built/mounted tree. Recorded case and tag boundaries:
  [qwen3-next-family.md](references/qwen3-next-family.md); mechanism and wheel timeline:
  [image-and-build-reports.md](references/image-and-build-reports.md).
- **Compare a launcher's `VLLM_VERSION` with the tag's `ARG VLLM_TAG` before reading any
  version-gated code.** `vllm_version_is()` prefers that environment value over `vllm.__version__`,
  and a shipped tutorial can export an older one — `v0.23.0rc1`'s `GLM5.2.md` exports
  `VLLM_VERSION=0.21.0` while its `Dockerfile` pins `ARG VLLM_TAG=v0.23.0`, which flips all 48
  `vllm_version_is("0.23.0")` sites (0.23.0-only patches skipped, `patch_v2.*` main-only set
  imported, `_V2_MODEL_RUNNER_SUPPORTED=True`). Have the reporter unset it and re-read the
  traceback; only then is a start-up `ImportError`/`AttributeError` at plugin import a
  plugin-vs-upstream seam
  ([env-vars-and-additional-config.md](references/env-vars-and-additional-config.md)).
- **`ModuleNotFoundError: No module named 'vllm_ascend.<sub>'` from a pip install is a packaging
  gate.** A directory present at the tag but lacking `__init__.py` never enters the
  `find_packages()` wheel, though a source checkout imports it as a namespace package. Answer with
  the first tag that adds the file or a source install — but establish first whether the reporter
  pip-installed a **release-branch tip** (`0.13.0rc4.dev3+g<sha>`): the fix for such a build can
  live only on that branch, `git tag --contains` empty, while the shipped tag predates the
  directory and is unaffected. Decide release impact from `git ls-tree <tag> -- <path>`, never
  from branch ancestry ([image-and-build-reports.md](references/image-and-build-reports.md)).
- **A from-source build that fails only in a network-isolated pod is a build-script dependency
  fetch, not a compiler, mirror or pip-index problem.** Grep the tag's build entry points
  (`setup.py`, `csrc/*.sh`, `.gitmodules`) for `git submodule`/`pip install`/`FetchContent` and name
  the directory to pre-seed; the tag's own installation doc requires
  `git submodule update --init --recursive` from `v0.12.0rc1` on, and the recorded fetch is
  `csrc/third_party/catlass` from gitcode.com. `COMPILE_CUSTOM_KERNELS=0` avoids it only by dropping
  every custom op, so state that cost
  ([image-and-build-reports.md](references/image-and-build-reports.md)).
- **Start-up `OSError: Could not load this library: .../libtorchcodec_image.so`
  (`libnvrtc.so.13` missing) is a CUDA-linked `torchcodec` wheel, not the model or checkpoint.** No
  `vllm_ascend/` frame is on the stack: vLLM `v0.25.0`-`v0.28.0` import `torchcodec.decoders` at
  module level in `vllm/multimodal/video.py` in a `try` that never catches `OSError`, so an absent
  package is tolerated while a CUDA-bound one aborts start-up, and the plugin's own
  `VLLM_TARGET_DEVICE=empty` install does not pull it. Uninstall it or install the `+cpu` build
  ([image-and-build-reports.md](references/image-and-build-reports.md)).
- **A start-up `RuntimeError: LinearOperation CreateOperation failed!` under `_warm_up_atb` is the
  310P support boundary, not a kernel hunt.** The plugin worker's `compile_or_warm_up_model`
  (`vllm_ascend/worker/worker_v1.py` before the `v0.13.0rc1` rename) ends with an unconditional
  `self._warm_up_atb()`, whose body calls `torch_npu._npu_matmul_add_fp32` — an ATB
  *sample* op that the plugin's own 310P code calls unsupported — so a 310P start-up reaching warm-up
  can raise there, with nothing in `envs.py` to switch it off. The 310P worker subclass skips it from
  `v0.14.0rc1` (`_310p/worker_310p.py`, `1ffca8673`/#5776, the same PR that adds the `_310p/`
  package); before that tag the workarounds are that upgrade or a local no-op override, and whether
  every 310P box fails depends on its NNAL/ATB install. Do not chase the op through
  `Ascend/op-plugin` or `Ascend/pytorch` once the plugin-side fix tag and the thread's "sample op, no
  device" statement are in hand ([310p-atlas-300i-duo.md](references/310p-atlas-300i-duo.md)).
- **`aclnnMatmulWeightNz` + `EZ1001: Not support mat2 n = 1 or k = 1 ... FRACTAL_NZ` at weight load
  is the NZ singleton-dim guard, not a kernel fault.** Before `v0.28.0.rc1` (`d43f2475a`/#16058)
  every eligible weight was NZ-cast, so a MoE shared-expert gate `ReplicatedLinear(hidden, 1)` fails
  with `error code is 161002`; the default `weight_nz_mode` `1` does not cast BF16 while mode `2`
  (a launcher's `VLLM_ASCEND_ENABLE_NZ=2`) does — dropping that export is the workaround, and
  `multistream_overlap_shared_expert` only makes it surface at load instead of the first forward
  ([env-vars-and-additional-config.md](references/env-vars-and-additional-config.md)).
- **A 310P MoE that serves but answers badly ("overthinking", malformed JSON) is the fused routing
  op before the sampler.** `npu_moe_gating_top_k_softmax` returned invalid routing above 1024 tokens
  per call until `0dc58f8e8` (#11391), first tag `v0.23.0rc1`; it later moves to
  `_310p/fused_moe/grouped_topk_router.py`, so a missing `experts_selector.py` on `main` is a
  refactor. Check `--reasoning-parser` in the serve line as a separate cause
  ([310p-atlas-300i-duo.md](references/310p-atlas-300i-duo.md),
  [serving-toolcall-reasoning-streaming.md](references/serving-toolcall-reasoning-streaming.md)).
- **A `507034`/`AclrtSynchronizeStreamWithTimeout` crash whose frames pass through
  `execute_dummy_batch` happened in the DP idle-rank dummy pass, not in the served request.** That
  method runs `_dummy_run(uniform_decode=True)` when one DP engine has no ready request while
  another does, so the failing op saw a synthetic `1 + num_spec_tokens` batch; read upward for the
  first device error (`507035` `IndexCheck_int64_high_performance_0`) rather than tying it to the
  request length or KV transfer. There is no disable env at vLLM `v0.26.0`
  ([log-signatures.md](references/log-signatures.md)).
- **A `[Doc]` "the guide omits X" report is answered by grepping `X` across `docs/` at the ref before
  agreeing.** The guidance usually lives on a sibling page, another tab or an include fragment (at
  `main`, `installation.md:136-145` carries `COMPILE_CUSTOM_KERNELS`/`SOC_VERSION`/
  `--no-build-isolation` and `:163` already cross-references the testing guide, while only
  `testing.md`'s Single-card tab lacks them), and a per-tab claim must name the tab the reporter
  followed. Then separate the genuinely missing piece from a discoverability request:
  [contributor-dev-environment-docs.md](references/contributor-dev-environment-docs.md).
- **A `[Performance]` report with a threshold and no error is a per-engine-state question before it
  is a kernel one.** When per-request decode time steps up at a specific number of *running*
  sequences with `Waiting: 0`, idle NPUs and no warning, establish (i) whether the threshold moves
  with the per-engine sequence count when you re-shard TP/DP, (ii) that KV is not exhausted, and
  (iii) that the collective in the profile is fast standalone — then stop ruling out toggles and
  look for state sized by the running-request count. A py-spy frame inside a collective only marks
  where the ordered device stream stalls. Recorded exclusions, the re-shard workaround and the
  `maybe_all_gather_and_maybe_unpad` call-site/implementation split:
  [throughput-and-concurrency-cliffs.md](references/throughput-and-concurrency-cliffs.md).
- **"Online serving works, offline `LLM()` fails" is not a patch-application asymmetry.** The global
  patch set is imported from `current_platform.pre_register_and_update()`, which
  `EngineArgs.create_engine_config` calls for every path including offline `LLM(...)` (verified at
  vLLM `v0.13.0`, `arg_utils.py:1312`, against `vllm_ascend/platform.py:128-129`); the CLI-only call
  at `arg_utils.py:2014` only widens the `--quantization` choices. Check the env gate and what the
  offline script omits relative to the documented serve tab
  ([deepseek-v4-flash.md](references/deepseek-v4-flash.md) has the `USE_MULTI_BLOCK_POOL` /
  `VLLM_ASCEND_APPLY_DSV4_PATCH` gates and the rc3 tutorial that carries no offline example).
- **A `[Doc]` report claiming the guide recommends something the hardware cannot do is answered
  from the tutorial's per-hardware tabs, not from a code mechanism.** Tab-set tutorials
  interleave A2/A3/A5 blocks with identical flag names, so a bare `--max-model-len 1048576` hit
  proves nothing: map the `^=== "` headings against the values, quote the doc's own
  non-revalidation or limitation sentence and the sibling recipes, and label the reporter's
  proposed mechanism a candidate
  ([model-tutorial-recipes.md](references/model-tutorial-recipes.md)).
- **Doc and deployment reports (an image tag, "why two containers", "which image do I pull") are
  documentation-drift questions.** The decisive pair is the substitution
  value the quoted URL renders and the form beside it: release-wide pages render
  `|vllm_ascend_version|` / `{{ vllm_ascend_version }}` from `docs/source/conf.py`
  `myst_substitutions` (Sphinx era) or `mkdocs.yml` `extra:` (MkDocs era), while a tutorial that
  writes a literal `quay.io/ascend/vllm-ascend:vX.Y.Z` freezes at authoring time — model tutorials
  are absent from the release bump list, so the frozen literal is the finding.
  Locators, the variant matrix and the duplicated-container case:
  [model-tutorial-recipes.md](references/model-tutorial-recipes.md).
- **Read the doc URL's version segment before pinning a ref.** `/en/latest/` (or `/zh-cn/latest/`)
  renders from `main`; `/en/vX.Y.Z/` corresponds to that release's branch, whose substitution
  value is bumped at cut-down while the tag's own copy can still hold the previous release's (the
  branch/page link is inferred from the two `conf.py` values — the reference labels it). A report
  quoting `latest` is compared against `main`-era docs, not against the tag it names.
- **The clones are read-only.** `git checkout`, `git worktree add` and any write into `/repos`
  fail and leave you on `main` (confirm with `git rev-parse`). Read with
  `git show <ref>:<path>`, search with `git grep -n <pat> <ref> -- <paths>`, and unpack a tree
  with `git archive <tag> | tar -x -C "$TMPDIR/<dir>"` for multi-file greps. A recursive
  `grep -rn /repos` runs long enough to be promoted to a background job; poll
  `job_output(wait=true)` before reading its first, partial output as "no match".
- **Load skills by name.** `skills_list` with a free-text query can return `[]` while the skill
  exists (recorded); call `skill_view(name)` directly.

## Reference map

Start with the reference that matches the signature, and read additional relevant references as needed.

| Report shape (first identifier) | Reference |
|---|---|
| proving a tag/commit, line-number mismatch, `dev<N>+g<sha>`, tag ordering, first tag containing a fix, cherry-picks that `tag --contains` misses, release-vs-rc ancestry, read-only clone commands | [revision-pinning.md](references/revision-pinning.md) |
| reading the issue thread: projection shapes, `invalid_view`, duplicate redirects, 众测 task-book threads, attachment windows, when the thread already answers | [issue-thread-reading.md](references/issue-thread-reading.md) |
| model type not recognized by transformers / `AutoConfig`; `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported` (checkpoint saved by transformers 5.x, image pins 4.x); `TypeError: <Model>Processor.__init__() got multiple values for argument 'tokenizer'` (4.53.0 only, exact-string vLLM guard, source-build trap); the per-vLLM-tag transformers pin table; Qwen3.5 tag boundaries, `Qwen3.6` titles that resolve to the Qwen3.5 tutorials | [model-type-recognition.md](references/model-type-recognition.md) |
| `max_model_len`, context-length 400s, launcher `MAX_SEQ_LEN`, gateway vs vLLM validation; a request that is never admitted waits instead of erroring; "truncated by default" with `finish_reason: "stop"` | [served-context-length.md](references/served-context-length.md) |
| `"pass_config": {"enable_sp": true}` in `--compilation-config`, output stopping mid-sentence with `finish_reason: "stop"`, FlashComm1/`enable_sp` switchboards | [env-vars-and-additional-config.md](references/env-vars-and-additional-config.md) |
| OOM in `attention_mask.py`, quadratic mask, chunk size, `splitfuse` vs dense mask, rc3 line pins, `#3048` already inside the tag that still OOMs | [attention-mask-oom.md](references/attention-mask-oom.md) |
| determinism / "same input, different tokens at higher concurrency" / `VLLM_BATCH_INVARIANT` / PIECEWISE | [batch-invariance.md](references/batch-invariance.md) |
| `attention_config {"backend": "FLASH_ATTN"}` / `fa3_v1.py` / `flash_attn_npu_v3` / "training-inference consistency": FA3 selection needs `VLLM_BATCH_INVARIANT=1` too (else it logs and falls back), dense only; mixed prefill + spec-decode batch faulting in `forward_impl` (prefill `block_tables` sliced by token count, offsets unrebased, decode `causal=False`), fix `5e1647cf9`/#18068 "Fixes #17991" on `main` only, remaining strided-page bug `#18729` | [fa3-attention-backend.md](references/fa3-attention-backend.md) |
| score below another harness or the published table: evalscope `_meta/<bench>.json` (MMLU-Pro 5-shot CoT, `ANSWER: [LETTER]`), lm_eval `--apply_chat_template`/`--fewshot_as_multiturn`, ais-bench case YAML, extraction and `max_tokens` truncation, closure with a candidate list | [accuracy-eval-harness-gaps.md](references/accuracy-eval-harness-gaps.md) |
| MoE accuracy wrong only with DP/EP/ETP > 1: 0.9.x unquantized `forward_oot` without DP collectives, `AscendFusedMoE` vs patched upstream `FusedMoE` selection, `get_fused_moe_state` All2All/MC2, first communicating tag `v0.10.1rc1` (`#2125`/`#2469`), DP comm in `ops/moe/fused_moe_prepare_and_finalize.py` from `v0.11.0rc1`; garbled only when `--enable-expert-parallel` is added (EP + `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` pairing, `select_moe_comm_method` strategy first, VL exception) | [moe-dp-ep-accuracy.md](references/moe-dp-ep-accuracy.md) |
| `prompt_embeds`, `TypeError: object of type 'NoneType' has no len()` in `npu_input_batch.py` | [prompt-embeds-and-input-batch.md](references/prompt-embeds-and-input-batch.md) |
| `VLLM_USE_V1=1 is not supported with ['BertModel']`, `--task rerank` / `--task score` / `--task embed`, `tuple index out of range` at `core/scheduler.py:209`, per-release and per-platform pooling support (`bge-*`; Qwen3-Embedding/pooling on 310P from `v0.20.2rc1`, #8846), a `/v1/embeddings` client hang beside a `200 OK` access log and the torch_npu `_use_new_zipfile_serialization` warning (no caller in either tagged tree; Torch's device-storage pickle path) | [pooling-embedding-rerank.md](references/pooling-embedding-rerank.md) |
| start-up quant `KeyError`, `.deq_scale` on a fused `gate_up_proj` shard name (static W8A8 param never created), `invalid choice: 'ascend'`, `KeyError: 'lm_head.weight'` on dense Qwen3-VL W8A8 (description remapped `lm_head` -> `language_model.lm_head` while the layer prefix stays `lm_head`; fix `8e5b20b85` first in `v0.19.1rc1`), wrapper-prefix namespace misses (`llm.model.embed_tokens.weight` on MiniCPM-V, `language_model`/`visual` prefixes, `quant_prefix_mapper`), FP8/W4A8 `161002`, `antiquantGroupSize`, retired `W4A8_DYNAMIC`, `WorkerProc failed to start` wrapper, the `v0.7.3rc2` no-description-file era and the quantization-guide KeyError FAQ; reporter-confirmed fix that merges `quant_model_description_w8a8_dynamic.json` into `config.json`'s `quantization_config` for a cause-A `q_a_proj` KeyError (#2015, `v0.9.2rc1:137`) | [quantization-startup-keyerrors.md](references/quantization-startup-keyerrors.md) |
| quantized accuracy (w4a16/w4a8/w4a4), QuaRot + EAGLE3 rotation, which precision lead belongs to which release; "quantized but no speedup" (small model, `VLLM_ASCEND_ENABLE_NZ`/`is_enable_nz`, `"FLOAT"` layers), model/version-table locators, W8A8_DYNAMIC-MLA + torchair fusion-pass caveat | [quantized-accuracy-and-quarot.md](references/quantized-accuracy-and-quarot.md) |
| MC2 vs FUSED_MC2 (automatic `MoECommType.MC2` with EP vs `enable_fused_mc2` 0/1/2 and its W8A8 / EP<=32 / non-MTP prerequisites, the `enable_mc2_hierarchy_comm` start-up `ValueError`, `dispatch_ffn_combine`'s `max_output_size=131072`; BF16 `dispatch_ffn_combine_bf16` hang on the 60-s `shm_broadcast` loop is a source hack — `quant_type == "w8a8_dynamic"` gate, reporter-verified M<=2048 cap); `compilation_config.custom_ops` forced to `["all"]` on non-310P while `enabled_custom_ops`/`disabled_custom_ops` are derived Counters, so a user's `custom_ops: []` is expected to be discarded, and the plugin's fx fusion passes are switched by `additional_config.ascend_compilation_config` (`fuse_norm_quant` etc.), not by `pass_config` (only `enable_sp` bridges); env var vs `--additional-config`, `Unknown vLLM environment variable`, `ascend_scheduler_config`, `SLO_limits_for_dynamic_batch` -> `SchedulerDynamicBatch` ctor `TypeError ... unexpected keyword argument 'hash_block_size'` (key gone from `v0.24.0rc1`), `weight_nz_mode`/FLASHCOMM1 lifecycles, `enable_mlapo` default-`true` but decode-only / W8A8-only / DSA-CP-excluded gates, disabled pass-through patch; a launcher's `VLLM_VERSION` overriding `vllm_version_is()` (the GLM-5.2 tutorial exports `0.21.0` against a `VLLM_TAG=v0.23.0` image); `torchair_graph_config`/`ascend_scheduler` removed at `v0.12.0rc1` (ACLGraph afterwards), `aclnnMatmulWeightNz` k=1/n=1 NZ guard `d43f2475a`/`#16058` first in `v0.28.0.rc1`, `PYTORCH_NPU_ALLOC_CONF` re-applied by `platform.py` so unsetting it is a no-op (only `enable_sleep_mode` skips the block, the RL config path disables expandable segments instead, and a launcher's `max_split_size_mb` suppresses the default); `HCCL_OP_EXPANSION_MODE` read sites per tag (`AIV` auto-set only for MiniMax-M2 at `v0.27.1rc1`; Kimi-K3 says leave it unset for prefill) | [env-vars-and-additional-config.md](references/env-vars-and-additional-config.md) |
| `FULL_DECODE_ONLY`/full-decode aclgraph crash **or log-free start-up hang**, `query_start_loc` pad, release-note model boundary (note first in `v0.12.0rc1`, `42ceaf08a`/`#4995`; `v0.11.0rc2` has no own notes section, `#5188`), ACL-graph capture-list budget not applied to `FULL_DECODE_ONLY`, FlashComm1 + capture; harmless plog-only `rtMemcpy` 107030 from `mla_preprocess`/`batch_matmul_transpose` host tiling (documented at `v0.23.0`; `batch_matmul_transpose` removed in `v0.29.0.rc1`, `mla_preprocess` migrated in `v0.30.0rc1`), distinct from the `TransposeKvCacheByBlock` crash form in this row; the `107030` that *does* kill the service at `v0.13.0` with DeepSeek-V3.2 `deepseek_mtp` + `additional_config.layer_sharding` (maintainer: the in-graph copy is not the cause; thread fix = `v0.18.0` with `layer_sharding` off); `cudagraph_mode: FULL` with PCP (`decode_meta` None in `attention_cp.update_graph_params`, "FULL does not support PCP"); `FULL` set but prefill never captured / gaps between operators on a hybrid GDN+full-attention model (min `AttentionCGSupport`, `linear_attn.py`, the downgrade warning) and the `ASCEND_LAUNCH_BLOCKING=1` incompatibility; legacy `full_cuda_graph: true` hang at `graph_task_update_end`, capture-size warning, prefix-caching silently disabled; `rtMemcpy ... capture mode` 107030/`587011`/`507011` after a `TransposeKvCacheByBlock` input2 init error (PD dummy run with `deepseek_mtp`, or non-PD replay at `ACLGraphWrapper.__call__`; GLM5 W4A8 `#7448`/`#7401`, #7235/#7290/#7460 + #7506 first in `v0.18.0rc1`, frame pins vs line drift) | [full-decode-only-graph-mode.md](references/full-decode-only-graph-mode.md) |
| first request / first `llm.generate()` empty, one-token stop, all-id-0 run, or a first-decode crash during graph replay (`None.size()`), `enable_sleep_mode`, `--language-model-only` warm-up mismatch, MTP method-name normalization | [first-request-only-output.md](references/first-request-only-output.md) |
| `mtp=2`/`mtp=3` slower than `mtp=1` in PD (`v0.13.0rc1`, mixed FULL-graph/eager decode ranks, `#5786`/`895d32c0e` first in `v0.13.0rc2`); PD `eagle3` draft-model acceptance vs colocated (PD tutorials prescribe only `deepseek_mtp`, k=1); `Running > 0` with flat generation counters, PD split, proxy choice by `kv_connector`, `unhashable type: 'list'` transfer thread, HIXL `engine_id`, ZMQ race, D-node `rejection_sampler.py:278`/`SUSPECT REMOTE ERROR` surface; first-request `TypeError: 'int' object is not iterable` at `max(per_group_hits)` in a hybrid-Mamba P/D run; `use_ascend_direct` placement; Mooncake `Sync batch data transfer timeout`; `RuntimeError: SUSPECT REMOTE ERROR` = ACL `507057`, remote device memory / SDMA link (a victim-surface code, not the local frame); `ValueError: KV transfer '<role>' config has a conflicting data parallel size. Expected 1, but got N` (plugin check vs engine-local DP, non-MoE DP reset to 1); example-proxy route inventory (404 on `/v1/messages`, `/v1/models`, `/metrics`; `/v1/models` only from `v0.27.1rc1`); retired `llmdatadist_c_mgr_connector.py` (0.9-0.11 only, `ThreadPoolExecutor(1)` pulls, the 120 s P-side block-hold warning, transfer exceptions logged only, so a worker-frame-less `EngineDeadError` is not sourced there); premature sliding-window KV free under `--async-scheduling` reported as stray/mixed-language tokens with a `(prompt+prefix) % block_size == block_size-1` first-anomaly boundary (`#13518`/`42a70b7ba` in `v0.23.0`, upstream `#47728` first in `v0.26.0`, and the reported-sha-before-the-fix trap); intermittent C8/MXFP8 KV corruption under concurrent transfer (#19064 on `main`, backport `14c6c3f43` in `v0.30.0` only; checkpoint-declared `kv_cache_type: C8`) | [pd-disaggregation.md](references/pd-disaggregation.md) |
| PD proxy decode-recompute retry, token-history reconstruction, fix only in `v0.26.0rc1`/`rc2` (absent through `v0.30.0` and `main`) | [pd-proxy-recompute-retries.md](references/pd-proxy-recompute-retries.md) |
| KV pool / mooncake store connector: is it supported and since which tag, `MooncakeConnectorStoreV1` vs `AscendStoreConnector` aliases and module paths, which connector pair each tag's `kv_pool.md` prescribes, `MooncakeLayerwiseConnector` inside `MultiConnector` (block-slice assert, `AscendMultiConnector` from `v0.16.0rc1`), whether `-DUSE_ASCEND_DIRECT=ON` is required; repeated `pool_scheduler.py:85` lookup freeze, `KV pool load spec created ... load_async=False` then a worker death (inline store `get`), pool never Puts, P-side backlog with mamba-align refcount leak, `Counters can only be incremented`, `OffloadingConnector` + MLA `KeyError`; MemCache backend (`ock.mmc.*`) start-up `AssertionError` from the bare `assert res == 0` (SDK `init()` status, no message), which init path a model+protocol takes (`lazy_init = use_compress and device_sdma`), device id from a process-group local rank (#16287/`c970e5892`, first tag `v0.28.0.rc1`), DP init barrier (#17289/`93c49251f`, cherry-pick #17308 in `v0.27.1rc1`, reverted by #19370); external hits inside the local prefix → `need to load: 0` and a still-paid lookup (#12783/`9e45da191` in `v0.23.0`, not rc1) | [kv-pool-ascend-store.md](references/kv-pool-ascend-store.md) |
| TTFT at concurrency 2 with "disabling chunked prefill changed nothing" (flag-effective checks, plugin force-enable gates); 0% prefix-cache hits, DCP/PCP block multiplier, DeepSeek-V4 compression LCM, pooling accuracy on cache hits, `--mamba-cache-mode align` | [prefix-cache-and-chunked-prefill.md](references/prefix-cache-and-chunked-prefill.md) |
| after serving: `torch.OutOfMemoryError: synchronize ... 107020`/`107027` at the `acl_graph.py` replay synchronize | [log-signatures.md](references/log-signatures.md) |
| `NPUWorkspaceAllocator tried to allocate ... MiB` + `EL0004` + `current working operator name is SelfAttentionOperation`, workspace growth on long diverse loads, whether `expandable_segments` applies, `torch.npu.empty_cache()`; sleep/wake memory growth (`enable_sleep_mode`), whether the reported tag's `CaMemAllocator.sleep` calls `empty_cache` (#7709 first in `v0.19.1rc1`), the sleep-replay race (#13757 in `v0.27.1rc1`) | [npu-workspace-memory.md](references/npu-workspace-memory.md) |
| `init engine (profile, create kv cache, warmup model) took N seconds` slow on `main` vs a release, turn on/off capture, which start-up phase grew | [log-signatures.md](references/log-signatures.md) |
| `[Performance]` decode cliff at a running-sequence threshold (`Waiting: 0`, idle NPUs, no error), per-engine vs per-node discriminator, re-shard workaround, DSA-CP `maybe_all_gather_and_maybe_unpad` call site vs implementation, the `256` literal dead end | [throughput-and-concurrency-cliffs.md](references/throughput-and-concurrency-cliffs.md) |
| any other traceback/timeout/hang signature: plog `[ASSERT] .../opp/.../ascendc/<op>_base.h` + `dex <i> out of range[0 N)` (CANN built-in op), `sample_tokens` timeout, `aicore execution times out` + HCCL watchdog abort, `shm_broadcast` stall, `RuntimeError: cancelled` (= executor teardown after a worker died), `WorkerProc initialization failed ... See stack trace for root cause` + `wait_for_ready`/`raise e from None` + `TBE Subprocess[task_distribute] ... main process disappeared` (worker died before READY; the tail is teardown), DP rank-count hang, the DP `num_tokens_across_dp` all_reduce that keeps idle DP ranks in step even with `--no-enable-expert-parallel` (single-request replay hang, then `RPC call to execute_model timed out`), `Aborted request`, start-up `KeyError: 'model.layers.<N>.self_attn.attn'` in `initialize_attn_backend` (PP>1), free-list assert, STUCK plog, `107027`/`stream is captured` at capture, `507034` + `execute_dummy_batch` frames (DP idle dummy pass), `KeyError: None`, teardown-log ordering, DP Gloo `Connection closed by peer`, 0.9.x LoRA `GuardOnDataDependentSymNode`, a bare `The DDR address of the MTE instruction is out of range` (shared phrase; classify by the named kernel/op — `fault kernel_name=token_bin_counts_and_mask_kernel_0` is the penalty Triton kernel's 65535 grid cap, #10138 first in `v0.21.0rc1`), RL weight-sync `HCCL error: not found specific resource` (trl `v0.19.0` aliases `PyHcclCommunicator` as `PyNcclCommunicator`, so the `pyhccl.py` frame needs no plugin patch; 0.10.2rc1/0.11.0rc* affected, 0.8.5/0.9.2 images reported OK), `107020` from an aclnn op with `event wait timeout`/`task_timeout` | [log-signatures.md](references/log-signatures.md) |
| `torch._dynamo.exc.Unsupported` / `Attempted to call function marked as skipped`, `posix.putenv`, `TRITON_ALL_BLOCKS_PARALLEL`, `--max-num-batched-tokens` crossing 65535 | [torch-compile-graph-breaks.md](references/torch-compile-graph-breaks.md) |
| spec-decode metadata: `AttributeError`/`NoneType` in `*_proposer.py`, `seq_lens_cpu`, `mtp_slot_pad`, FIA `561002`, draft-graph fields; EAGLE3 draft hang at the draft stage with TP>1 (paired upstream's "without tensor parallelism" note, `draft_tensor_parallel_size: 1`) | [spec-decode-metadata-failures.md](references/spec-decode-metadata-failures.md) |
| GDN/MTP state and acceptance plumbing, 310P `index_copy_`, acceptance collapse | [gdn-mtp-spec-decode.md](references/gdn-mtp-spec-decode.md) |
| CP/PCP paths, EP-gated FlashComm/SP helpers, `aclnnMoeDistributeDispatch` `561000`, QP exhaustion; MoE + MTP + FlashComm1 `4096 vs 8192` at `final_hidden_states[0] + final_hidden_states[1]` (draft-VL all-reduce bypass in `register_custom_ops._maybe_pad_and_reduce_impl`, `#7683`/`#8004`, regression fixed by `#16081`); PCP padding pushing `num_actual_tokens_pcp_padded` past `max_num_batched_tokens` (prefill `IndexCheck`), `pcp_utils.py` (<=`v0.24.0rc1`) vs `worker/v2/pcp_manager.py` (>=`v0.26.0rc1`, neither file at `v0.25.1rc1`); PCP+MTP mixed-batch `could not broadcast input array from shape (29,) into shape (31,)` in `update_tokens_for_pcp` (`num_decode_tokens` used as a request offset, fixed `v0.23.0rc1` by #10487 — not by #10172); `HcclAllocComResource ... ret 4` from a dead host NIC (`Getting socket times out` / `EJ0002 The network port is down`, `MC2MultiServer` comm created while a torchair graph loads, plog `ra rdma init failed devid[N]` -> `listen ... fail` -> `Level1rdma[1]`); `aclnnMoeDistributeDispatchV4 ... 107020` beside a 230 s task timeout (GLM-5.3; the tag's `v0.21.0rc1` Known Issue #9503 workaround is no EP with FULL graph, or PIECEWISE/eager); PCP-without-EP accuracy: `enable_sp()` is env-gated, not EP-gated, so the delta is `select_moe_comm_method` → ALLGATHER and `mc2_tokens_capacity` is `max_cudagraph_capture_size` or `max_num_reqs × query_len` (capped 512); DSv3.2 PCP/DCP + FlashComm1 is a documented `v0.18.0rc1` Known Issue | [context-parallel-moe-paths.md](references/context-parallel-moe-paths.md) |
| W8A8C8/LI-C8 for DSV3.2/GLM5 in PD-mix missing from `v0.17.0rc1` despite its release note (`#7029` reverted by `#7288`, restored by `#7294`; `enable_sparse_c8` from `v0.18.0rc1`); GLM-5.x sparse-C8 + DSA-CP + KV-pool flag cluster, `sfa_c8` accuracy known issues, `c8_enable_reshape_optim` lifecycle, GLM-5.3-Flash `glm5next` tag availability and the `index_kpool`/tail-ring numbers behind a long-prompt hang | [c8-dsa-deployment-flags.md](references/c8-dsa-deployment-flags.md) |
| 310P/Atlas 300I DUO: VL/OCR accuracy via the rope overrides (`AscendApplyRotaryEmb` ignored `is_neox_style` through `v0.29.0.rc1`, fixed `v0.30.0rc1`; `_310p` MRoPE bridge), `socversion ... does not match the current device` from the wrong image variant, `[max_model_len, max_model_len]` mask, graph limits incl. capture dying with `Insufficient_Event_Resources(EL0008)`/`rtEventRecord ... no event resource 207007` surfacing as `capture_end ... 507903` under TP>1, a pre-sharded `TP<N>` checkpoint without `--load_format sharded_state` dying in the `vocab_parallel_embedding` `org_vocab_size` assert, w8a8sc, 507018/507015, `VLLM_USE_V2_MODEL_RUNNER` map, which models the tag's tutorial actually lists (dense Qwen3-32B only from `v0.18.0`, w8a8sc TP4; no `_310p/` code before `v0.14.0rc1`), tutorial path per release (`single_node_300i.md` / `310p.md` / `hardwares/310p.md`), eager-only note through `v0.18.0rc1` (graph/quant note first at `v0.18.0`, and no `v0.17.1rc1` tag exists), the 310P `custom_ops` gate, MoE routing `npu_moe_gating_top_k_softmax` >1024-token chunk guard (`0dc58f8e8`/`#11391`, first `v0.23.0rc1`; later `grouped_topk_router.py`), 310P MoE dispatcher OOM at `token_dispatcher.py:113`/`:116` `arange`/`bincount` allocations 3-4 orders above a 16-sequence batch (#8248), start-up `LinearOperation CreateOperation failed!` in `_warm_up_atb` (`torch_npu._npu_matmul_add_fp32`, skipped by `_310p/worker_310p.py` from `v0.14.0rc1`) | [310p-atlas-300i-duo.md](references/310p-atlas-300i-duo.md) |
| Gemma3/Gemma4: `num_hidden_layers` AttributeError, PA/FIA arity (`expected 13, got 9`), Gemma4 support boundary | [gemma-family.md](references/gemma-family.md) |
| VL/multimodal accuracy blamed on the ViT: "qkv weight/bias all zeros" under TP>1 (head-dim padding to 128, inherited `split_qkv` all-gather/re-split, #805 first in `v0.9.0rc1`, `patch/worker/patch_qwen2_5_vl.py` from `v0.12.0rc1`), Flash Comm V1 refused for VL models at `v0.17.0rc1`, where 310P rope and Qwen3-VL patch binding reports go; first rule out an image that never reached the model (dict prompt accepted on `"prompt"` alone; the key must be `multi_modal_data`) | [vision-tower-accuracy.md](references/vision-tower-accuracy.md) |
| GLM-4.5/4.5-Air output quality, rope defect, DP+EP shared-expert reduce, GLM-5.x stale `slot_mapping`; any DSA/sparse-attention model garbled only under graph capture with DP — `_dummy_run` slot invalidation and its ordering fix | [glm-moe-accuracy.md](references/glm-moe-accuracy.md) |
| `[Performance]` `ApplyTopKTopPCustom` slow path (`top_p < 1.0` + `top_k >= vocab`, 0.18.x layout, kernel gone from `v0.23.0`), loops / repetition / garbled or mixed-language output, sampler surface per tag (incl. `v0.7.1rc1`/`v0.7.3` rows), `<pad>`/id 0 only when `temperature > 0` (`q.record_stream` boundary), version-pair regression under `temperature > 0` from a pre-`dev20250307` torch-npu, MTP accepted-token race, long-context repetition/divergence with `--async-scheduling` + MTP before `v0.19.1rc1` (#8461) | [sampling-and-output-quality.md](references/sampling-and-output-quality.md) |
| MiniMax-M2 hooks, double TP all-reduce, `</parameter>` leak in streaming tool calls, `invalid tool parameters` on 0.18 (0.17 → 0.18 streaming-parser rewrite; parsers exist from 0.17.1, v0.24.0 is only the engine), `</think>` lost only when a tool call is parsed (`minimax_m2_append_think` + `minimax_m2`, engine era ≥ `v0.24.0`) | [minimax-m2-family.md](references/minimax-m2-family.md) |
| reasoning/tool-call parsing: no `--reasoning-parser` leaves the thinking in `content` (a literal `</think>`, `reasoning: null`); `reasoning_content` always null or `reasoning` vs `reasoning_content` per tag, the legacy `qwen3xml` parser's whitespace-only `content` delta (upstream `v0.23.0`, file removed at `v0.23.1rc0`, `qwen3_xml` alias re-pointed to `qwen3_coder` then to the `v0.24.0` engine), thinking lost from turn 2 of a tool-call loop, `<think>` in the prompt (Qwen3.5/Qwen3.6), `prompt_is_reasoning_end`, `thinking_token_budget` ignored on NPU (holder hardcoded `None` until `v0.27.1rc1`, `VLLM_USE_V2_MODEL_RUNNER` warning), missing `id`, empty `arguments` (`""` streaming diff vs `"{}"` non-streaming zero-arg, and the `glm47`-vs-`glm4_moe` regex that decides whether such a call parses at all, including the `findall(None)` `TypeError` a zero-arg `glm47` call raises before `v0.15.0rc0`), client-side `Unterminated string` from accumulated `arguments` fragments (a mid-word delta split, not token loss), `</think>` leak, parser inventory per tag (incl. the 1084-line `v0.18.0` patch that replaces `chat_completion_stream_generator`), `vllm/parser/` arriving before the `v0.24.0` engine (`ParserState`/`Transition`/`EventType`) | [serving-toolcall-reasoning-streaming.md](references/serving-toolcall-reasoning-streaming.md) |
| structured output / guided decoding, xgrammar `TypeError`, CPU bitmask round-trip; `AssertionError: (-1, req, {...[-1,-1,-1]})` in `grammar_bitmask` (placeholder draft tokens: upstream `AsyncScheduler` vs plugin `RecomputeScheduler`, opt-in flags, upstream fix `v0.14.0rc1`); a streaming hang with **no** exception on one of two deployments (A/B env-delta filter, `VLLM_ASCEND_BALANCE_SCHEDULING` Scheduler substitution at `v0.18.0rc1`, balancing inert at DP=1) | [structured-output-guidance.md](references/structured-output-guidance.md) |
| LoRA shape asserts, `csrc` `TORCH_CHECK`, `_mcp_apply` gather, rank limits, 0.9.x ACL-graph `GuardOnDataDependentSymNode`, slow LoRA decode on 310P or with `--max-lora-rank` ≥ 128 (the `torch_ops` binding), multimodal tower/connector LoRA dying during `profile_run` in upstream `compute_meta` (`507018`) or `set_lora` (`507057`), batch-invariance coverage | [lora-adapters.md](references/lora-adapters.md) |
| image/build reports: Dockerfile `~/.bashrc` exports, device init `500001`, `--device /dev/davinciN`, `liboptiling.so`, pybind11 build break, non-root UID vs `HwHiAiUser` device nodes, CANN 9.2.1 entrypoint permission, wheel built against a newer glibc (`GLIBC_2.32 not found`, usually only the sleep-mode warning), the separate missing-`_C.weak_ref_tensor` load failure, start-up `!!!!!!! Segfault encountered !!!!!!!` / `Failed core proc(s): {'EngineCore': -11}` on a hand-built plugin (`+empty` vLLM, pip-list vs the tag image's triton/triton-ascend lines, patched `build_aclnn.sh` install path, no gdb), a from-source reinstall with no `libvllm_ascend_kernels.so` (`COMPILE_CUSTOM_KERNELS`, `$ORIGIN/lib`, 310P/A5 skipping it by design), Mooncake `corrupted size vs. prev_size` on `from mooncake.store import ...` (jemalloc delta `v0.12.0rc1`→`v0.13.0rc1`), a from-source build that fails only without public network (`csrc/build_aclnn.sh` -> `git submodule update --init` for `csrc/third_party/catlass`, from `v0.12.0rc1`), start-up `OSError: Could not load this library: .../libtorchcodec_image.so` (`libnvrtc.so.13` missing; CUDA-linked `torchcodec`, vLLM `v0.25.0`-`v0.28.0`), `ModuleNotFoundError: No module named 'vllm_ascend.compilation'` from a pip wheel (missing `__init__.py`, fixed in `v0.9.0rc2`) and the release-branch variant with `models/layer` + `transformers_utils` (fix `58ea73f95` only on `origin/releases/v0.13.0`, `471f0590b`/`v0.20.2rc1` on main; the `v0.13.0` tag predates both directories), and a start-up `No module named 'triton'` behind architecture inspection on the 0.10 line (the `Dockerfile` uninstalls it; the tag tutorial's triton-ascend + BiSheng recipe is the fix) | [image-and-build-reports.md](references/image-and-build-reports.md) |
| DeepSeek-V3.2/-Exp: chat-template `ValueError` + `--tokenizer-mode deepseek_v32`, torch-npu wheel header, 1P1D `rtMemcpy` capture | [deepseek-v3.2.md](references/deepseek-v3.2.md) |
| Qwen3-Next-80B-A3B (`Qwen3NextForCausalLM`, hybrid GDN+MoE): 0.10-line start-up `No module named 'triton'` during architecture inspection (image uninstalls triton, tutorial installs triton-ascend + BiSheng; 0.11 import-path boundary); v0.11.0 `Available memory: 0` then `No available memory for the cache blocks`; steady-state `reshape_and_cache` Aicore fault with `507057`/MTE DDR range | [qwen3-next-family.md](references/qwen3-next-family.md) |
| DeepSeek-V4 family: which tags contain it, documented recipes, `dsa_v1.py` MTP assert, truncated safetensors, `deepseekv4` image drift, thinking-mode default, `reasoning_effort` clamping, V4.1 separate package/parsers | [deepseek-v4-flash.md](references/deepseek-v4-flash.md) |
| Kimi-K2.7-Code tag boundary (no doc/config before `v0.23.0`) and its W4A8 PD-divergent-DP e2e recipe; what each tag's model tutorial prescribes (serve lines, `tests/e2e` configs, 众测 and TTFHW doc tickets, reporter deltas), the `quick_start.md` -> `getting_started/` move at `v0.27.1rc1` and `ModelScope` history, per-hardware tab matrices for `[Doc]` reports, image-tag substitution drift, duplicated `docker run` blocks, V4.1-Flash A2-at-1M contradiction, 910C on a pre-`v0.10.0rc1` image (no `-a3` family yet), Qwen3.5-397B-A17B BF16-vs-w8a8 footprint and the MP DP-across-nodes / Ray TP-per-node recipe at `v0.18.0`; a 众测 task-book URL that names a tutorial the tag lacks or names differently (`Qwen3.5-27B-Qwen3.6-27B.md` absent at `v0.18.0`, which carries `Qwen3.5-27B.md`; the merged file lives only on `origin/releases/v0.18.0`); "can't use Postman inside the container" (host-side client over `--net=host`, in-container `curl`/SDK, `-p 8000:8000` on the eval pages, no `curl` in the Dockerfile), and a 4-card ticket against a TP2/DP1 recipe; GLM-5.2 at `v0.23.0rc1` (section index, the A2 w8a8 multi-node block vs the A3/single-node W4A8C8 block, `glm5.2`/`glm5.2-a3` image literals) | [model-tutorial-recipes.md](references/model-tutorial-recipes.md) |
| [Doc]/[Usage] contributor dev environment: `testing.md` per-tab inventory (Local CPU vs Single-card), mounting a checkout over the image's plugin and the generated/gitignored `_build_info.py` (`ImportError`, reinstall with `pip install -e . --no-build-isolation`), `bash format.sh ci` = pre-commit lint, `COMPILE_CUSTOM_KERNELS=0` for UT, install pages' substitution drift | [contributor-dev-environment-docs.md](references/contributor-dev-environment-docs.md) |
| "how do I see KV-cache/HBM usage", `/metrics` field names, `num_gpu_blocks="None"`, why every TP rank looks full (per-rank weight line) | [kv-cache-metrics-and-hbm-usage.md](references/kv-cache-metrics-and-hbm-usage.md) |
