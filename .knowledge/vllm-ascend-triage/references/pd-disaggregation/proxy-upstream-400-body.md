# The example proxy's upstream `400 Bad Request`: which leg, and the body that decides it

## When this page applies

A PD deployment serves through
`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`, and the
failure is raised **inside the proxy**:

```
httpx.HTTPStatusError: Client error '400 Bad Request' for url 'http://<node>:<port>/v1/…'
  File "…/load_balance_proxy_server_example.py", line 832, in send_request_to_service
```

with `Attempt 1/2/3 failed …` above it, the node's own log showing only the
uvicorn status line (「就报错 400 Bad Request，其他啥也不显示」), and often the
sentence that a direct `curl` at that node "works". Recorded #15527
(vllm-ascend v0.23.0 `5cb98caa` + vllm v0.23.0 `0fc695f`, DeepSeek-V4-flash-w8a8,
2×800I A3, DP16/TP1 decode, `--max-model-len` 40960 **or** 128000 fails while 1M
works). This is a *which-component-rejected-what* question, not a transport one:
the KV connector, the fabric and the rank table are all downstream of the
frontend that answered.

The sibling direction of the same code path — an invalid `max_tokens` that
**nothing** rejects and a held P-side KV — is
`invalid-max-tokens-and-p-refill.md`. Read this page when a 400 exists.

## Step 0 — the traceback's line numbers are the pin, and they matched

At v0.23.0 `5cb98caa` the shipped file has `handle_completions_impl` `:961` with
the `assign_instances` call at `:969`, `assign_instances` `:896` with
`response = await send_request_to_service(` at `:913`, and
`send_request_to_service` `:809` with `response.raise_for_status()` `:823` and
`raise last_exc` `:832` — the four numbers in the recorded traceback, exactly. So
the reporter ran the shipped script, and a locally edited proxy is not part of
this family. One `rg -n` on those four symbols confirms or refutes it for a new
ref before any ranking.

## Step 1 — read which leg rejected: the log line says it, the port does not

The proxy has exactly two upstream call shapes, and their warning text differs:

| leg | caller | body actually sent | warning text |
|---|---|---|---|
| **decode** | `:1003` `stream_service_response_with_retry(decoder_client, api, req_data, …)` | the client's own JSON, **plus** `kv_transfer_params` copied from the prefill response (`:925`); not rewritten | `Attempt %s failed for streaming %s: %s` (`:855`, `:865`), then `Error during streaming from decoder %s:%s: …; releasing prefiller KV` (`:1076`) |
| **prefill** | `:913` `send_request_to_service` — always `ServerRole.PREFILL` | `build_prefill_request` (`:790-807`): `kv_transfer_params{do_remote_decode: true, …}`, `stream=false`, **`max_tokens=1`**, `min_tokens=1`, `max_completion_tokens=1` when present, `stream_options` popped | `Attempt %s failed for %s: %s` (`:826`) — the bare path, **no word "streaming"** |

Both default to `max_retries=3`, so the same body is rejected three times with
no visible delay between the attempts — that is the `Attempt 1/2/3` block, not
three different requests.

Ports and hosts do **not** identify the leg: `--prefiller-ports 8078 8079 8080
8081` and `--decoder-ports 8078 … 8093` overlap at 8078, and both lists name the
same two nodes. Decide from the call site: the `for streaming …` line or a
traceback naming `assign_instances`/`send_request_to_service` is the **prefill**
leg; `Error during streaming from decoder` / a bare `Attempt … for streaming` is
the **decode** leg. `releasing prefiller KV` in that message is the proxy's own
counter (`:1076-1080`), not the engine freeing anything.

## Step 2 — the proxy discards the body, and the body *is* the diagnosis

`response.raise_for_status()` (`:823`, `:847`) raises httpx's generic text and
neither handler logs `exc.response.text`; vLLM's JSON body — the only place the
rejection reason exists — is thrown away. The node's engine log is empty because
the request never reached it (frontend validation), so the access line is all the
reporter can see. Two ways to recover the reason, both cheap:

1. **Replay the identical body at the engine** (Step 3's bodies), `curl -sS -i`
   so the status line and body print.
2. **Patch the two handlers** to log the body before re-raising: in
   `send_request_to_service`'s `except` (`:825-832`) and in
   `stream_service_response_with_retry`'s (`:853-869`), add
   `logger.error("upstream body: %s", getattr(exc, "response", None) and exc.response.text)`.

## Step 3 — the four frontend rejections at this pin, all `VLLMValidationError` → 400

`create_error_response`'s default is `status_code=HTTPStatus.BAD_REQUEST`
(`vllm/entrypoints/openai/engine/serving.py:167-171`, and `:178-182` for the
streaming twin), so every entry below
reaches the proxy as a bare 400.

| body text | site | condition |
|---|---|---|
| `max_tokens=N cannot be greater than max_model_len=… Please request fewer output tokens.` | `vllm/renderers/params.py:209-220` (`TokenizeParams.__post_init__`) | `max_output_tokens > max_total_tokens` |
| `This model's maximum context length is {max_total_tokens} tokens. However, you requested {max_output_tokens} output tokens and your prompt contains {len(text)} characters (more than {max_input_chars} characters …)` | `params.py:324-343` `_text_len_check` | a **character** bound applied *before* tokenization: `len(text) > max_input_tokens * tokenizer.max_chars_per_token` |
| `… and your prompt contains {qualifier}{token_count} input tokens, for a total of {qualifier}{total} tokens.` | `params.py:415-439` `_token_len_check` | `len(tokens) > max_input_tokens` |
| `Input length ({input_length}) exceeds model's maximum context length ({max_model_len}).` | `vllm/entrypoints/serve/utils/api_utils.py:183-187` (`get_max_tokens`) | `max_model_len < input_length` |

`max_input_tokens = max_total_tokens - max_output_tokens` (`params.py:188-193`),
`max_total_tokens = model_config.max_model_len`, and `max_output_tokens` is the
**raw request value** — `build_tok_params` (`vllm/entrypoints/openai/chat_completion/protocol.py:504-524`,
`.../completion/protocol.py:202-211`) passes `self.max_tokens or 0`, *not* the
`get_max_tokens` clamp. So an oversized client `max_tokens` reaches the first
three checks unclamped; `get_max_tokens`' own raise is the fifth, separate gate.

### The prefill leg's arithmetic decides the case

`build_prefill_request` forces `max_tokens = 1`, so on that leg
`max_input_tokens = max_model_len - 1`: **a 400 there cannot be caused by the
client's output length** — the prompt alone (tokens *or* characters) is over the
configured context, or the payload is otherwise invalid. Recorded #15527 had
**both** legs 400, and the reporter's own rows (40960 fails, 128000 fails, 1M
works, same body) bound the prompt above 128000 tokens. That arithmetic — one
division by `max_chars_per_token`, one `/tokenize` call — is the answer; the
batching, DP/TP and MTP-parity knobs cannot move a request's length.

## Step 4 — the direct-`curl` control is only a control with the identical body

"单独 curl D 节点可正常返回" proves that the D engine can answer *something*, which
no hypothesis needs. Valid only when the body is the client's, byte-identical
(same `messages`/`prompt`, same `max_tokens`, same `stream`, same
`/v1/completions` vs `/v1/chat/completions`), because a shorter prompt or a
smaller `max_tokens` is a different length class — exactly the variable under
test. Make the control true locally instead:

- the client body unchanged, `curl -sS -i http://<D>:8078/v1/chat/completions`;
- the prefill-shaped body (add `"stream": false, "max_tokens": 1, "min_tokens": 1,
  "kv_transfer_params": {"do_remote_decode": true}`) at a **P** port;
- the reporter's own 1M row already says the divergence tracks the context
  budget, not the transport — quote it back as the discriminating row.

The decode comparison is still not like-for-like without `kv_transfer_params`:
with `do_remote_prefill` set, a frontend rejection is a handled path — vLLM
notifies the connector to unpin the remote-prefill blocks
(`vllm/entrypoints/openai/engine/serving.py:413-445`
`_with_kv_transfer_rejection_cleanup`, "i.e. the request never reached the
engine"). That function is the pin's own statement that the D side can reject
such a request before the engine runs.

## Step 5 — what cannot produce this 400 (route away)

- `--max-num-batched-tokens` (the recorded pair was P `8192` / D `240`),
  `--max-num-seqs`, DP/TP geometry, `--workers`, the host/port lists: scheduling
  and routing; no frontend check reads them.
- P `--speculative-config {"num_speculative_tokens": 1}` vs D `…: 3`,
  `--enforce-eager` on one side, DSA `enable_dsa_cp`, `recompute_scheduler_enable`,
  `FULL_DECODE_ONLY` on one side: KV-geometry parity — wrong tokens, connector
  asserts or a dead EngineCore (`recipe-contract-diff.md`,
  `../spec-decode-mtp/index.md`), never a shorter context budget.
- Auth: a wrong `OPENAI_API_KEY` is 401, and `auth_headers` (`:783-786`) signs
  **both** legs identically.

## Step 6 — the retry path can manufacture the 400 (name it second, not first)

With `recompute_scheduler_enable: true` on the D node, the proxy's
`stop_reason == "recomputed"` branch (`:1047-1066`) rewrites
`messages[0]["content"] = origin_prompt + generated_token` (or
`req_data["prompt"]`), sets `req_data["max_tokens"] = origin_max_tokens -
completion_tokens + retry_count`, and re-sends through `reassign_instances`. The
prompt grows by the decoded text and `max_tokens` can go to zero or negative — so
a 400 that appears only on a *later* request, with a longer prompt than the
client sent, is this branch; `proxy-example-endpoint-coverage.md` records the
`recomputed` marker being erased on `/v1/messages`. Distinguish it by whether the
failing body differs from the client's.

## Answer shape

First sentence: the 400 is the P/D node's **frontend** rejecting a request the
proxy built, the reason is in a response body the proxy discards, and here is the
one reproduction. Then: which leg (from the log line, not the port), the
`max_tokens=1` arithmetic that makes the prefill leg a prompt-length question,
the two ways to print the body, and the parity knobs listed only to be ruled out.
Ask for three things: the response body (patched proxy or identical-body curl),
the prompt's token count from the node's own `/tokenize` (the proxy does not route
`/tokenize` — `proxy-example-endpoint-coverage.md`), and confirmation that the
direct curl used the identical body.

## Pins

Proxy lines: vllm-ascend v0.23.0 `5cb98caadeff42b5b62b996e34bb2aaa29d20fd`.
Frontend checks: vllm v0.23.0 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`. Both
move between releases — re-grep `def build_prefill_request`, the two
`Attempt %s failed` strings and `_text_len_check` before quoting. The recorded
reporter's follow-up is not in the snapshot, so the mechanism (prompt length vs
another validation) stayed open; what is source-verified is the leg arithmetic,
the four 400 sites and the discarded body.
