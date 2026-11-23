# "The load-balance proxy only answers /v1/chat/completions": endpoint coverage

## When this page applies

A PD deployment is up and `/v1/chat/completions` works through the example proxy,
while other vLLM endpoints are unreachable — `/v1/messages` (Anthropic),
`/v1/models`, `/metrics`, `/health`, `/version`, `/tokenize`,
`/v1/messages/count_tokens`, `/start_profile`. The report is usually a
feature/code-change request ("需要对所有 vllm 的接口都做兼容", "大EP对外服务api不兼容
anthropic接口和metrics接口"), sometimes with a pointer to
`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`.

Recorded #10290: vllm-ascend reported 0.18.0, snapshot v0.20.2rc1 (`367b8e62`) +
vllm v0.20.2 (`bc150f50`), GLM-5.1 大EP serve. The gap is still there at the
snapshot, so it is a live request, not a fixed-release question.

## Step 0 — one direct curl splits "proxy gap" from "their vLLM has no such route"

Before reading any proxy source, curl a **decoder's own port** (`:8000`), bypassing the
proxy: `curl -s -X POST http://<decoder>:8000/v1/messages -H 'content-type:
application/json' -d '{"model":"<served>","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}'`,
plus `curl -s http://<decoder>:8000/metrics | head`. A 200/SSE means the engine serves
them and the gap is the proxy alone — the usual case; a 404 means the reporter's vLLM
predates the package, and the rung is an upgrade, not a route patch.

At vllm v0.20.2 (`bc150f50`) that half is already source-verified: the Anthropic router
is attached **unconditionally** wherever the generation routers are
(`vllm/entrypoints/openai/generate/api_router.py:36-42`,
`register_anthropic_api_router(app)`, no flag and no `try/except ImportError`), and it
carries `vllm/entrypoints/anthropic/api_router.py:48` `POST /v1/messages` and `:94`
`POST /v1/messages/count_tokens`. So say "the decoder is routed at the pin, only the
example proxy is not" with those two paths, and do not go looking in
`vllm/entrypoints/openai/api_server.py` — it mentions neither (`include_router` has no
hit there either), which reads as "the route does not exist" and costs turns: recorded
#10290's replay spent three greps finding the mount site before its answer.

## Step 1 — read the route table, then the header: two lines of evidence

At v0.20.2rc1 the script registers exactly five routes
(`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:892-920`):
`POST /v1/completions`, `POST /v1/chat/completions`, `GET /healthcheck`,
`POST /instances/add`, `POST /instances/remove`. The file's own header states it —
":14 `Supports OpenAI-compatible /v1/completions and /v1/chat/completions
endpoints`" — which is quotable as the example's declared scope.

Grep with `rg -n '@app\.(get|post)'`, not `app.get(`/`@app.get("` variants and
not a route-name search: `_handle_completions` is a *helper* that every generation
route funnels into (`:710`), so a hit on its name says nothing about coverage.

## Step 2 — ownership: the gap is inherited, so the fix is a feature, not a regression

Upstream vLLM's `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`
(v0.20.2) registers the same two POST routes plus `/healthcheck`; the vllm-ascend
file is an adaptation of it. Say that in the first message: the engine is not at
fault, no server flag changes it, and the user's workaround exists today (a front
proxy that path-splits generation to `:9000` and everything else to any decoder).

## Step 3 — the change is three tiers, and only the middle one is mechanical

1. **Generation endpoints that must split P/D** — `/v1/messages` is feasible
   because the Anthropic request model already carries the carrier:
   `vllm/entrypoints/anthropic/protocol.py:116` `kv_transfer_params` (response
   model `:198`), and `serving.py:450` (non-stream) / `:338` (stream) forward it to
   the engine. So the route is `_handle_completions("/messages", request)`, and
   Anthropic's required `max_tokens` satisfies the proxy's
   `req_data.get("max_tokens", 16)`. **Verify per endpoint rather than assuming**:
   `rg -n 'kv_transfer_params' vllm/entrypoints/**/*protocol*.py` — a request model
   without the field belongs to tier 2, not tier 1.
2. **Two traps make it more than a decorator** (both silent, both post-smoke):
   - *The recompute-retry marker is erased.* The proxy retries P/D recompute by
     reading the OpenAI-shaped `choices[0].stop_reason == "recomputed"`
     (`:793`), which vllm-ascend emits itself
     (`vllm_ascend/core/recompute_scheduler.py:837`). The Anthropic layer maps only
     `stop` / `length` / `tool_calls` (`serving.py:93-97`) and streams through
     `self.stop_reason_map.get(finish_reason or "stop")` (`:617`), so `recomputed`
     becomes `None`: the proxy never retries and the client receives a truncated
     answer as `end_turn` with no error anywhere. Report this as a design question
     for the maintainers, not as part of the route patch.
   - *The retry body rewrite assumes string content.* `:731-737` builds
     `origin_prompt` from `messages[0].get("content", "")` and `:795-800` writes
     `messages[0]["content"] = origin_prompt + generated_token`. Anthropic
     content may be a block list (`[{"type":"text","text":…}]`), so the rewrite
     raises exactly when a recompute fires — invisible to a smoke test.
3. **Stateless passthrough** covers the rest: one catch-all
   (`@app.api_route("/{path:path}", methods=[…])`) forwarding method, query,
   headers (`anthropic-version`, `x-api-key`) and body. The in-repo template is
   `examples/disaggregated_encoder/disagg_epd_proxy.py` — `/v1/models` `:561`,
   `/health` `:570`, `/start_profile` `:676`, `/stop_profile` `:686` — so cite it
   instead of inventing the shape. Two cases are not passthrough:
   - `/metrics` and `/start_profile` / `/stop_profile` need **fan-out and
     aggregation** (one decoder's `/metrics` hides the prefiller's); the EPD
     proxy's `_profile_cmd` (`:646-660`) shows the fan-out.
   - Pooling/embedding/score/rerank/audio endpoints do not exist on a generation
     P/D instance: they need their own target list, otherwise the passthrough
     returns 404s.
4. **Docs**: the tutorials recommend this script as the only entry point, and the
   header comment `:14` repeats the two-endpoint claim. A route patch without those
   lines leaves the same report to be filed again.

## Limits

- Line numbers are the two commits above; re-grep the `@app.` block at your ref,
  and state that the reporter's own release (0.18.0 in #10290) is outside the
  snapshot so its line numbers were never verified.
- Whether the header rewrite strips `host`/`content-length` (or keeps
  `x-api-key`/`anthropic-version`) decides whether an `--api-key` serve answers
  401 after the new route is added — check
  `send_request_to_service` / `stream_service_response_with_retry`
  (`:576-655`) before claiming the patch is complete.
- `/v1/responses` and other newer OpenAI-shaped generation routes were not swept
  in the recorded case: apply the tier-1 test (`kv_transfer_params` on the request
  model) per endpoint instead of copying the `/v1/messages` conclusion.
