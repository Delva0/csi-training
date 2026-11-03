# P/D proxy decode-recompute retries and lost token history

When a PD-disaggregated deployment enables the decode-side recompute scheduler, the
`examples/disaggregated_prefill_v1` load-balancing proxy can retry a request after
recomputation. Below: when that path is reachable, what the retry code actually does, and
where a token-history fix exists. Line numbers are for the refs named; re-anchor with
`git grep -n <symbol> <ref> -- <path>`.

## When the path is reachable

- `--additional-config '{"recompute_scheduler_enable": true}'`, default `false`
  (`vllm_ascend/ascend_config.py:142` at `d17a99cd`).
- `platform.py:677-684` at that ref requires `kv_transfer_config.kv_role == "kv_consumer"`:
  a PD-disaggregated **decode** node. P nodes and PD-mixed mode raise
  "recompute_scheduler_enable can only be enabled on PD-disaggregated D nodes".
- `ascend_config.py:152-161` requires the flag for
  `finegrained_tp_config.oproj_tensor_parallel_size` / `embedding_tensor_parallel_size`
  (their cross-DP collectives need uniform token counts).
- `is_pd_decode_recompute_scheduler_enabled()` (`vllm_ascend/utils.py:1122`) gates the
  attention-side behaviour. Only a request whose decode node emits
  `stop_reason="recomputed"` enters the proxy retry, so a report without that marker (or
  without the scheduler flag) is a different defect.

## Marker emission (vllm-ascend, commit `d17a99cd` on the v0.23.0 release line)

`vllm_ascend/core/recompute_scheduler.py`:

- `RecomputeReqInfo` dataclass at line 83 carries `request_id`, `output_token_ids:
  ConstantList`, `client_index` — the scheduler does retain the recomputed token history.
- The API-visible output discards it: lines 868-879 append
  `EngineCoreOutput(request_id=..., finish_reason=FinishReason.STOP, new_token_ids=[],
  stop_reason="recomputed")` after logging
  `[RecomputeScheduler] Recompute triggered for request <id>`. Grep the decode worker log
  for that message to prove recomputation happened rather than guessing.

## What the proxy does with the retry

`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` at `d17a99cd`:

- `handle_completions_impl` (line 961) captures `origin_prompt = messages[0]["content"]`
  (line 977) and `origin_max_tokens = req_data.get("max_tokens", 16)` (line 980).
- Streaming reads *bytes*, not SSE events: `aiter_bytes()` at line 849 inside
  `stream_service_response_with_retry` (line 835), then one `json.loads(chunk_str)` per
  chunk at line 1024. A chunk may split or coalesce events, so the terminal marker can be
  missed and the truncated stream forwarded as if complete.
- Accumulation keeps text only: `content = delta.get("content") or message.get("content")
  or choice.get("text") or ""`, then `generated_token += content` (lines 1037-1038).
  `delta.reasoning`, tool calls, refusal, logprobs, token ids and extra choices are lost.
- On `stop_reason == "recomputed"` (line 1047) a chat request is retried by mutating the
  first client message — `messages[0]["content"] = origin_prompt + generated_token`
  (line 1051) — and the budget is
  `req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count`
  (line 1054), with `completion_tokens` incremented per chunk for streaming (line 1042).
  Only `max_tokens` is read and rewritten; a request that set `max_completion_tokens` keeps
  its original budget, which the issue reports as making the adjusted retry budget
  ineffective (that precedence was not re-verified here).

Name these consequences rather than "garbled output": generated text is re-rendered into
the *role of `messages[0]`* (system/user as the client sent it), reasoning is dropped, and
the retry is a fresh generation from a different prompt. A streaming client therefore sees
reasoning `R1+R2` and content `C1+C2` produced from different histories; a non-streaming
client receives reasoning from one attempt with content assembled from two. The role that
gets corrupted is determined by the first message of the client request, not by the
rendered chat template (a default system prompt injected by the template never appears in
`req_data["messages"]`).

## Where a fix exists (verified reachability)

- `c120a6e6d` "[BugFix][v0.26.0rc][P/D] Preserve exact token history across recompute
  retries (#14389)" ports #13524 to `releases/v0.26.0rc`. It adds
  `vllm_ascend/patch/recompute_proxy.py` and
  `vllm_ascend/patch/platform/patch_recompute_chat.py`, transports prompt/output token ids
  in a private `_vllm_ascend_recompute_token_ids` field, parses streaming by SSE framing,
  and keeps one client-visible stream.
- `git tag --contains c120a6e6d` → `v0.26.0rc1`, `v0.26.0rc2` only;
  `git branch -a --contains` → `releases/v0.26.0rc` and `rfc/0.26.0rc1-bugfix`. Counting the fix
  files (`git ls-tree -r --name-only <ref> | grep -c 'recompute_proxy.py\|patch_recompute_chat.py'`,
  i.e. the patch module, its unit test and the platform patch) gives 3 hits at `v0.26.0rc1` and 0
  at `v0.23.0rc1`,
  `v0.23.0`, `v0.23.0.post1`, `origin/releases/v0.23.0`, `v0.24.0rc1`, `v0.25.1rc1`, `v0.27.0rc1`,
  `v0.27.1rc1`, `v0.28.0rc1`, `v0.29.0rc1`, `v0.30.0` and `main` — re-scan only a newly cut line.
  The commit message names `#13524` as a v0.23-targeted original being ported, and that original
  never landed in any 0.23 tag, so no released 0.23.x build carries the fix either. "Fixed
  upstream" or "fixed on main" is therefore wrong for this class: the only upgrade path is the
  v0.26.0rc line.
- `main` still rebuilds the retry from parsed text (`messages[0]["content"] = ...` around
  `:1220-1231`) and still streams through `aiter_bytes()`, but it *does* preserve a list-valued
  first message (multimodal content) by appending `{"type": "text", "text": generated_token}`
  when `isinstance(origin_prompt, list)`. That branch is `c94405212` "[BugFix][Proxy] Preserve
  multimodal content on recompute" (#14241), contained in `v0.26.0rc1`/`rc2`; the same lines exist
  in `main`'s file even though `c94405212` is not an ancestor of `main`, so it arrived there by a
  separate landing that was not traced. Read it as content-shape handling only — the retry still
  moves generated text into the first client message, so "main handles list content" is not a
  token-history fix.
- The fix commit also adds `tests/ut/test_recompute_proxy.py` (316 lines) and touches
  `.github/workflows/scripts/test_config.yaml`; point a "the fix exists" claim at that test rather
  than at the issue thread.
- Issue #13466 is the detailed report; it was open and labelled `triaged` with no comments when
  re-read after the fix landed, and the fix commit references it. Related reports the issue names
  (#3448 origin, #5900 re-tokenization precision, #7450 closed-unmerged token-id attempt, #8695
  non-stream text joining, #8852/#9943 cleanup) were not re-verified here.

## Boundary

Code and commit reachability were verified in the clone; no proxy or NPU run was performed,
so the user-visible corruption follows from the code path (plus the issue author's protocol
reproduction), not from a reproduced trace. Treat "recomputation caused this report's
corruption" as needing request-level evidence (the `[RecomputeScheduler]` log line for that
request id) before it is asserted.
