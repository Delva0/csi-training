# Evidence sources: issue thread, attachments, corpus, fix history

The report text is only half the evidence. Maintainers and other reporters supply the rest, and it
is cheaper than reading code. Every route here is read-only.

## The issue thread: one `github` call, two requests

```json
{"requests": [
  {"path": "/repos/vllm-project/vllm-ascend/issues/8579",
   "fields": {"title": "/title", "state": "/state", "comments": "/comments", "labels": "/labels"}},
  {"path": "/repos/vllm-project/vllm-ascend/issues/8579/comments",
   "params": {"per_page": 100},
   "fields": {"user": "/user/login", "created": "/created_at", "body": "/body"}}
]}
```

- **The plain form is the one that works** (verified on #13750: seven comments, 2 773 characters, one
  call). `/issues/N/comments` is a bare JSON array: do **not** add `paginate` with a hand-set
  `items_pointer` or a top-level `json_pointer`. `items_pointer: ""` returns `invalid_view` ("This
  pagination collection exposes its items at json_pointer "/items""), `items_pointer: "/items"` then
  returns "json_pointer does not identify a value in this response", `paginate` + `json_pointer: "/0"`
  fails with `invalid_view`, and `json_pointer: "/items/0"` alone raises `ValueError`. The same
  `invalid_view` ("json_pointer does not identify a value in this response") is what a `json_pointer`
  on the **plain** request returns — an array has no `/items` — so leave `json_pointer` and
  `items_pointer` out entirely rather than "projecting" the array. One recorded run burned three calls
  and two whole messages cycling those spellings before reading a single comment; another lost a
  batch-1 call to `json_pointer: "/items"` on #8920. Pick the shape above and keep it; `params:
  {"per_page": 100}` already returns up to 100 comments, and for a longer thread use
  `offset`/`max_chars` windows on the same plain call.
- Add `/body` to the issue request only when the prompt lacks it (see the next bullet).
- Project the fields rather than dumping: `fields` applies per array item here;
  `display_complete: false` with `omission_count` only means the projection dropped keys (node_id,
  avatar URLs), not that content is missing.
- The task prompt often already carries the issue body, log and `collect_env` dump: fetch
  `/comments` first and re-fetch the body only for a part the prompt lacks. Bodies can be 30–70 KB,
  and the tool returns a window (~16 KB) with `next_offset`; **`max_chars` accepts up to 40 000**, so a
  17–35 K thread arrives in one call when the batch-1 read carries `max_chars: 40000` (a 45.9 K thread
  still needs a second window, and that second one is the *tail*, not the middle); project with
  `json_pointer`/`fields`, or use the tool's own `start_line`/`context_lines` window around a literal
  match (a tool parameter, not a shell utility) to read the decisive part (start command, environment,
  traceback). Do not dump a whole body into context and re-fetch it later.
- **Read the report body's tail before the code — the lever is often the last line.** Reporters append
  their own tests and conclusions *after* the pasted log, where a reader who jumps to the traceback
  never goes: #4326 ends with "after removed `--compilation-config '{"cudagraph_mode":
  "FULL_DECODE_ONLY"}'` it will work", and #3308 carries its `--gpu-memory-utilization` answer in a
  comment below the body. Skim the last ~1 000 characters of the prompt body together with the comment
  tail in batch 1; a reporter-confirmed config change settles the workaround and the "did the fix
  work" status, leaving only the mechanism for batch 2 (`answer-and-claims.md`,
  `../symptoms/aclgraph-mode-and-capture.md` §A).
- **Read the comment thread to the last comment.** A truncated `context`/`data` window with
  `next_offset` means the reply you need may still be unread: on #3308 the whole body is a
  `collect_env` dump plus log, while the workaround (`--gpu-memory-utilization 0.5`, the
  `PYTORCH_NPU_ALLOC_CONF` settings) and a changelog-style closing note sat in comments beyond the
  16 KB window. Project the comments to `user`/`created`/`body` and, if the window still cuts, re-read
  with `offset: <next_offset>` before reasoning about the case.
- **A long thread that is mostly pasted logs: read the tail, not the sequence.** `/comments` reports
  `total_chars`; when it is tens of KB with a handful of comments, the diagnosis or the closure note is
  in the last window. Request `offset: <total_chars - 8000>` with a matching `max_chars` instead of
  walking forward from the first window (a perf thread of nine comments can be 56 KB of metric dumps
  with no technical reply at all — conclude that and stop).
- **A traceback that stops at the call site is not an invitation to enumerate exceptions.** When the
  report body is cut mid-traceback (the window ends inside the frame that called the failing function,
  so the exception type and message are missing), do not rank candidate raises in prose: look for a
  sibling thread with the same stack, or state the limit and answer with the localisable half (frame,
  ref, gating condition). One report spent two turns listing candidate asserts inside a connector
  function whose exception it never saw.
- **A log line with only `file.py:NNN` is ambiguous.** Reporters paste vLLM log prefixes such as
  `engine.py:298`, and the same basename exists in several modules; resolve it by grepping the message
  text at the reported ref (`log-signatures.md`), never by guessing from the basename.
- Fields worth capturing verbatim for the write-up: the reporter's launch command, their version
  lines, and the maintainer's advice (quote the original, translate if needed).
- Labels carry triage state (`bug`, `triaged`, `wait-feedback`, `stale`, hardware labels such as
  `310p`, feature labels such as `long-seq`, `advanced-features`). A bot "stale → closed" comment
  after `wait-feedback` means the issue was never resolved.
- Issue numbers that appear inside a report ("https://github.com/.../issues/3494") are the upstream
  thread of the same bug — fetch that thread, not only the local one.
- **Copy the number from the trailing URL; never type it from the title.** Issues and PRs share one
  number space, so the neighbouring number is often a pull request: a recorded run asked for `#1630` on
  a report whose last line was `issues/1629`, got a PR back, and spent a whole work batch reading a
  codecov comment and three review pings. The fingerprint is unmistakable — `codecov[bot]`,
  `github-actions[bot]` conflict notices, "please help review" — and a `title` that does not match the
  report. Verify `title` on the first fetch and re-read the report's last line before re-filing the call.
- **A closing summary that names PRs is a claim, not evidence.** Check each number with
  `/repos/O/R/pulls/<n>`: on #3308 the final automated comment said the OOM was fixed in
  `v0.11.0rc1` by #3305 and #3557, and both are closed, unmerged (a config fix, a doc update). Relay
  such a sentence as attributed-and-unverified, or leave it out.
- **Check for a sibling thread in the first fetch.** Reporters file twice, and the task's issue number
  is not always the detailed one: #8974 held two comments that only said "same issue as #8975, which
  has more details", while #8975 carried the maintainer diagnosis and the HCCL/`plog` evidence. Read
  the title/state/comments count of the fetched issue first; when it is a duplicate marker or shows
  fewer comments than the diagnosis needs, fetch the sibling before touching code. Treat the twin as
  the primary source and record both numbers so the duplicate chain is visible in the answer.
- **When the sibling is already in this skill's ledger, its row plus its class page *is* batch 2.** On
  #7448 ("This seems like a duplicate of #7401") a run fetched #7401's thread, then re-derived the class
  from `csrc/`, `model_runner_v1.py` and the three cited PR numbers, and reached the cut with no reply —
  while `../symptoms/issue-ledger/reports-newest.md`'s `#7401` row and `aclgraph-mode-and-capture.md` §G
  already carried every fact it was re-deriving. Search the ledger for the twin's number before opening a
  code window.

## Why not scrape the HTML page

Fetching `https://github.com/<owner>/<repo>/issues/<n>` returns navigation boilerplate plus a
`[TRUNCATED: N characters]` notice, and a model-assisted extraction of the same URL can return
only the "Description" section — losing exactly the comments that contain the answer. A
comment-permalink fetch returned "You can't perform that action at this time." Use the REST route
through the `github` tool; keep HTML fetching as a fallback only when the REST route is
unavailable.

`curl https://api.github.com/...` from bash works but is unauthenticated (low rate limit, `403`
after a few dozen calls) and dumps full JSON; a shell API call also gets promoted to a background
job mid-turn. Reserve curl for attachment files.

## Attachment logs

Attachment URL forms:

- `https://github.com/user-attachments/files/<id>/<name>` — text payloads: startup logs,
  `docker run` scripts, `collect_env` dumps, `npu-smi` output. `curl -sSL` with no auth works for
  public issues.
- `https://github.com/user-attachments/assets/<uuid>` — images (screenshots of a terminal or an IDE).
  Not text, and reading them is not a step you can take; if the decisive evidence exists only there,
  say so instead of guessing — and read the thread tail first (next paragraph).
- Private-user-images URLs with JWT query strings appear when someone quotes an image; they expire,
  so prefer the `user-attachments` form.

**A screenshot-only report is still answerable.** Recorded case (vllm-ascend #6257): the body and the
reporter's follow-up are five images, but a later maintainer comment types the failing strings while
re-testing ("未出现原 issue 中的 `AclrtSynchronizeStreamWithTimeout` 或 `captured-stream` 报错") and names
the release where they stopped (`v0.15.0rc1`), which is enough for a class plus a version lever, with the
unread traceback listed as a limit. Do: read every comment for quoted log text and for the version that
worked; take the launch command from any comment that pastes one; then state "the traceback itself is
only in the screenshots" in `limits`. Do not: try to infer the cause from the model family
(`git grep -i <model> <ref> -- vllm_ascend` is empty for most families because the model class lives in
the `vllm` clone), and do not spend a batch guessing at image contents.

**The title's symptom can be the wrong symptom — read the attachment before classifying.** #8248's body
is the `[Installation]` template with an empty environment and one `3.txt` link, and its title says
「运行正常，无法输出」 ("runs normally, cannot output"), which reads like an empty reply or a parser problem.
The attachment is a death: `RuntimeError: NPU out of memory. Tried to allocate 20.21 GiB (NPU 2; 43.24 GiB
total capacity; 39.06 GiB already allocated …)` with the deepest `vllm_ascend/` frame at
`vllm_ascend/_310p/fused_moe/token_dispatcher.py:113` (`moe_init_routing`, a 310P × Qwen3-30B-A3B
deployment), the workers shutting down and the frontend simply never answering. So an "engine is up but
every request returns nothing" report whose only evidence is an attachment is one `web_fetch` from a
different class: quote the OOM line, correct the premise in the reply, and ask for the log immediately
before it (first request vs `profile_run`) plus `max-num-batched-tokens` — the request is 20 GiB on a
43 GiB card that is already 90 % allocated, so which stage asked decides the lever. The reporter closed
that issue `completed` with zero comments, so no lever comes from the thread.

Use the script, not a raw dump:

```bash
bash scripts/fetch_issue_evidence.sh /workspace/scratch/issue-evidence \
  https://github.com/user-attachments/files/29382159/default.txt \
  https://github.com/user-attachments/files/29382160/docker.txt
```

It prints per-file size/line count, the first exception/stall markers, and the last 8 lines of each
file. The tail is the highest-value window: a startup stall stops producing worker lines while the
frontend keeps repeating one message. A 44 KB startup log fetched this way ended at the exact stage
where the engine stalled (draft-model load / mask allocation), which no amount of code reading
reproduces as quickly.

### Reading a startup log

1. Find the last line authored by a worker/rank process (`Worker_TP*`, `pid=`); that is where
   execution stopped.
2. Interpret frontend repetitions: `shm_broadcast.py ... No available shared memory broadcast
   block found in 60 seconds` means the engine core is waiting on workers stuck in weight load,
   dummy profiling, mask building or a collective — a symptom, never the cause.
3. Cross-check memory: the `npu-smi` block in the same attachment shows per-device used/total. A
   per-card usage near the total plus a stall right after weight loading supports an
   allocation-pressure explanation; a healthy total argues against it.
4. Note provenance: attachments are sometimes posted by a *different* reporter than the issue
   author, with a different model path, max length or image tag. Keep the difference in the
   write-up instead of merging two configurations.

## Issue search

Endpoint `/search/issues` with `params.q`. Search is lexical and AND-ed per token, so one
distinctive identifier wins:

- `repo:vllm-project/vllm-ascend is_layer_skipped_ascend` → 29 hits, including
  "[Bug]: 910b部署kimi-k2.5 w4a8量化版本，报错：KeyError: 'embed_tokens.weight'" (#7350) and a
  GLM-4.7 `KeyError: 'model.layers.92.head.weight'` (#5794);
- `repo:vllm-project/vllm-ascend KeyError embed_tokens.weight quant_description` → 0 hits.

Refinements: `in:title`, `is:open` / `is:closed`, `created:>2026-01-01`, `author:`. Reports and
maintainer answers on this tracker are often Chinese; search identifiers, symbols and config keys
rather than prose. Reuse maintainer answers, but re-check them against the code at the relevant ref
before repeating them, and quote them verbatim when relaying.

## The shipped nightly/e2e configs: the deployment the project actually tests

When a report asks "is this flag/env required, or is my config wrong?", the answer is often versioned
in the repo, not in the docs. Before reading Python, list the CI configs at the reported tag:

```bash
git -C /workspace/repos/vllm-project/vllm-ascend ls-tree -r --name-only <ref> \
  -- tests/e2e/nightly/multi_node | grep -i config
git -C /workspace/repos/vllm-project/vllm-ascend show \
  <ref>:tests/e2e/nightly/multi_node/internal_dp/config/Kimi-K2_5-W4A8-A2-dual-nodes.yaml
```

Each YAML is one model + quantisation + hardware + topology (`env_common`, then a per-role `envs` and
`server_cmd`). It is the strongest available evidence for "the project's own tested configuration sets
`X: 1`", which upgrades a commenter's suggestion to *reported+corroborated* while the mechanism stays a
hypothesis (#10503: this file sets `VLLM_ASCEND_ENABLE_FLASHCOMM1: 1` and
`--no-enable-prefix-caching` for Kimi-K2.5 W4A8 on two A2 nodes). Also read the tutorial under
`docs/source/tutorials/models/<model>.md` at the same ref; when the two disagree, say which you used.

**Documentation as evidence has its own ref rule.** A `[Doc]` report — or any report quoting the guide —
is about the page the user read: `…/en/latest/` is the **default branch's** docs tree (read
`docs/source/…` from the clone's working tree), while a versioned URL (`…/en/v0.23.0/…`,
`…/en/v0.7.3.post1/…`) is that tag's tree. Check whether the page existed at the release the report
names (`git ls-tree -r --name-only <tag> docs/source/tutorials/models | grep -i <model>`): a
`latest`-only page means the guide post-dates their build (#15836), and a page that changed under them is
a doc defect, not a code path (`../symptoms/doc-guide-vs-capability.md`).

## Fixes, PRs and CI

- `git log --all --oneline --grep='<issue or PR number>'` / `--grep='<title fragment>'` finds the
  commit in the clone. Commit messages usually paste the PR description, which states the mechanism
  and the tests added — often the cleanest root-cause wording available.
- `github` `/repos/O/R/pulls/<n>` and `/pulls/<n>/files` for the diff when the clone lacks it;
  `/search/commits?q=repo:<owner>/<repo>+<term>` when only a title is known.
- `/pulls/<n>/checks` or `ci_logs` only when the report is about a CI failure; for a user bug
  report, commit archaeology plus code reading is the evidence that matters.
- An issue number or a merged PR alone does not prove a fix: resolve the linked commit and check
  ancestry (`version-pinning.md`).
- **A third-party file (an eval harness's benchmark meta, a default config) is one `github` call, not a
  web fetch.** `/repos/<owner>/<repo>/contents/<path>` with `accept: application/vnd.github.raw+json` and
  the default view returns the text in `content` (verified on
  `modelscope/evalscope/…/_meta/mmlu_pro.json`, 19 K characters, `next_offset` paging); a `json_pointer`
  still selects one key of that JSON. Two failure spellings cost a work batch each: `view: "text"` is
  rejected for anything that is not base64 blob data (`view='text' requires base64 file/blob content; use
  view='full' or request the native raw Accept type`), and `web_fetch` on
  `raw.githubusercontent.com` returned `Web fetch timed out after 120s (URL safety, DNS and backend
  included)` — the same timeout hits a model's own config on a model hub
  (`huggingface.co/<org>/<model>/raw/main/config.json`, recorded), so a model's architecture is **not**
  evidence obtainable this way: read the ref's tutorial page or the fork's model registry, and leave
  "what the upstream checkpoint is based on" in `limits` instead of reasoning about it. Keep `web_fetch`
  for prose pages and issue attachments (`user-attachments/files/…` downloads work) and read repository
  files through the `github` tool.
  A **paste/diff host is not a document**: a report whose whole environment block is
  `https://example.invalid/<id>` (also pastebin, a WeChat/GitCode attachment page) has no fetchable
  text — the recorded call returned `Web fetch timed out after 120s` and nothing else (#3045). Treat that
  environment diff as missing evidence, answer from the tag range, and put the absent diff in `limits`.

## Shell and job mechanics that have burned this analysis

- The shell tool promotes any command over roughly a second into a background job, returning
  `"kind": "promoted"` with a `jobId` and only partial or empty `output`. **A promoted call is not
  a finished call**: collect it with `job_output(wait: true)` and read the result before
  concluding "no matches". An empty pickaxe result has been reported as "the string never existed
  in history" when the search was merely still running.
- Pass a larger `timeoutMs` (30000–60000) for `curl`, `git archive`, `git log -S --all` and
  whole-tree greps; bound long logs with `-n`/`| head` rather than re-running the query.
- One `sed -n` range per call: `sed -n '20,60p;40,170p'` prints the overlap twice and doubles the
  transcript for the same lines. Chain a `grep -n` for the anchor first, then read one window.
- **Every `bash` call needs both `command` and `description`.** Omitting `description` fails
  validation before anything runs (`'description' is a required property`) and that call — with its
  share of the batch — is spent: #9355 lost its `open_answer.sh` invocation this way, so the ref facts
  never arrived. Treat the pair as the shape of the call, not as optional metadata.
- `git log --grep=<token>` (message search) and `git log -S<token>` (content search) answer
  different questions; a config key often appears only in messages, its implementation in content.
  `git grep <token> <ref>` searches the tree at a ref without checkout and does not see
  worktree-only edits. A pickaxe can come back empty while the string is plainly present at a tag
  (merge-only introduction, pathspec or revision restricting the walk) — switch to grep rather than
  iterating on pickaxe:

  ```bash
  bash scripts/ref_grep.sh /workspace/repos/vllm-project/vllm-ascend c8_enable_reshape_optim \
    v0.23.0 v0.26.0rc1 HEAD
  ```

## Where not to spend time

- `skills_list` may be empty, and the repo-local `.agents/skills/*` are release/model-onboarding
  skills, not triage knowledge: do not read them end to end (see
  `../vllm-ascend/index.md`).
- Do not enumerate `/workspace/repos/*` repeatedly. Confirm the two or three repos the traceback
  names — usually `vllm-project/vllm-ascend` plus `vllm-project/vllm` — and go straight in.
- Do not re-derive the report's own facts with wider searches when the body already states them.
