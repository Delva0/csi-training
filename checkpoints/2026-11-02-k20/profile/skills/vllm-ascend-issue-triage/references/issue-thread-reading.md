# Reading the issue thread with the `github` tool

The thread is the cheapest source of workaround and mechanism. Everything below was exercised on
`vllm-project/vllm-ascend` issues and is tool behaviour, not repository knowledge.

## Two shapes, two projections

- `/repos/O/R/issues/N` is an **object**: omit `json_pointer` (or pass `""`). A lone
  `json_pointer: "/"` is rejected — `invalid_view`, `path: /json_pointer` — and costs a round;
  the recorded #18030 read paid for exactly that.
- A `fields` map cannot ride on a non-empty `json_pointer` (`/title` + `fields` returns
  `invalid_view`, "fields requires an object or array of objects after json_pointer"): either
  project the whole object with `fields`, or take one string with `json_pointer` alone.
- `/repos/O/R/issues/N/comments` is a **bare JSON array**: use `json_pointer: ""` plus a
  `fields` map (`/user/login`, `/created_at`, `/body`). `json_pointer: "/items"` is not a
  pointer into a bare array and returns `invalid_view`; `json_pointer: "items"` without the
  leading slash is rejected by the pointer parser itself (`ToolInputError`, "Use a JSON Pointer
  beginning with /"). (`/search/issues` is the mirror case: its array really is under `/items`.)
- Do **not** combine `paginate` with `fields` on the comments read: the recorded combination
  returned empty `data` plus `missing_projection_field` entries, while the same request without
  `fields` (compact view, `max_chars` raised) returned every body.
- Project `/title`, `/state`, `/labels`, `/comments` (the count) and read the comments array
  separately. Re-fetching `/issues/N` in full when the body is already in the prompt repeats text
  you have.
- Comment bodies are windowed: re-read with `offset` until `next_offset` is null. A page that
  stops mid-log is not the end of the thread. Attachments and long tracebacks truncate in the
  middle at the default budget — raise `max_chars`, because the dropped window is usually the
  raising frame (recovered that way in #3494).
- **When one comment is a wall of raw logs, enumerate by id instead of paging by offset.** The
  array read is one concatenated projection, so a character `offset` keeps landing inside the first
  huge body — a recorded read spent four requests and never left comment 1. The working shape is two
  calls: project only `/id` (with `/user/login`, `/created_at`) for the array, then read the comments
  that matter one at a time from `/repos/O/R/issues/comments/<id>` with a `/body` projection. Choose
  by author and date from the index, not by paging the text.

## Classify the thread before paging it

- A handful of comments is one call. Hundreds are usually an auto-generated 众测 task-book or
  delivery-verification ticket whose comments are per-tester "任务完成" posts with screenshots
  and benchmark tables: classify from page 1 (and `pagination.last` for the tail), then take the
  evidence from the reported tag's tutorial, release notes and `tests/e2e` config. Paging nine
  such pages yields nothing.
- A `not_planned`/stale close, a "please provide more info" reply, or silence means **no
  maintainer root cause exists**. Use their questions as the diagnostic checklist, never as a
  confirmed fix.
- Follow redirects as a chain, not one hop: a content-free report can point at a sibling that
  itself only says "accuracy bug still exists #M", and #M is where the fix PRs are named. Stop
  when a thread carries PR numbers, labels or a maintainer mechanism.
- Read what the reporter changed after the advice and how many variables at once — that decides
  whether a workaround is established or a co-variate.
- **A pasted log that simply stops — no exception, last lines `world_size=... rank=...` and
  `[Gloo] Rank ... connected` — is not a hang.** Reporters split long logs into parts ("第二部分") or
  append the rest as a comment, and the traceback is usually in the next part: in #11406 the body
  ended after the Gloo/`parallel_state` lines, two recorded forwards classified it as a stall and
  reasoned about HCCL/transport, and the first comment opened with "第二部分" and carried the real
  `AssertionError` from the weight loader. Read the comments before naming a mechanism for a
  no-error log, and say which log part the finding comes from.

## When to stop reading

- A maintainer comment that names the mechanism, the flag or the replacement spelling can support
  a draft answer; continue only for a named remaining unknown.
- When the thread also names the fixing PR, prove the raising statement at the tag, then
  `git tag --contains <fix-sha>`. A thread can carry the mechanism and the PR but not the
  release (reporters ask "which tag" and get no answer), which is why the tag check comes from
  the clone.
- **A `user-attachments/files/<id>/<name>` link in the body is a fetchable log, not a dead end.**
  `curl -sL --max-time 60 -o "$TMPDIR/a.txt" '<url>'` returns the raw file with no auth: #8248's
  521 KB `3.txt` held the whole `vllm serve` line, every restart and three distinct tracebacks, so
  the failing frames and the flag set were readable from a body that itself carried no command and
  no log. Fetch it before classifying; `max_chars` windows on the issue body stop at the URL.
  Only an image-only body or a link that now 404s leaves nothing to read: then ask for the text, or
  say the class is a candidate ([symptom routing](../SKILL.md)).

Recorded boundary: the `github` tool projections above are what the recorded calls returned;
credentials, rate limits and `max_chars` defaults are environment state and are not evidence
about a repository.
