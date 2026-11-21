# PD: an invalid `max_tokens` never errors and the prefiller's KV stays held

Symptom family: `max_tokens = 0`, or `prompt_tokens + max_tokens > max_model_len`,
is sent to a P/D deployment and **nothing rejects it** — the OpenAI client gets no
400, the decode node logs no error, the request is handed back / re-dispatched to
a prefiller and that prefiller's KV blocks stay allocated until a timeout
releases them (recorded #13539, vllm-ascend v0.23.0rc1 `f4a08bd` + vllm v0.23.0
`0fc695f`, reporter's wording: 「测试case不会报错，decode节点不会报错，而是打回请求给
Prefill节点，导致请求占用p节点的kv不释放，等待超时才释放」).

This is a *where-does-the-validation-live* question, not a connector fault. Do not
start in `mooncake_connector.py` or the transport pages: the KV is held because no
component ever decided the request was invalid, so the P-side done-signal timeout
fires exactly as designed (`llmdatadist-done-signal-and-ttft.md` explains why that
line is bookkeeping).

## Where validation actually lives, per leg (source-verified)

1. **Plain OpenAI serve** — `SamplingParams` rejects `max_tokens < 1`, and the
   renderer layer (`vllm/renderers/params.py`) carries the prompt+output bound via
   `TokenizeParams(max_total_tokens=model_config.max_model_len,
   max_output_tokens=...)`. That is the check the reporter expected.
2. **Prefill leg** — the example proxy overwrites the bound before it is ever
   checked: `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:801`
   `payload["max_tokens"] = 1`. Whatever the client asked for, the P node sees 1,
   so no P-side frontend can detect the oversized request.
3. **Native disaggregation API** (vllm v0.23.0) —
   `vllm/entrypoints/serve/disagg/protocol.py:150-154` `build_tok_params` returns
   `TokenizeParams(max_total_tokens=None, max_output_tokens=0)`: no context-length
   check and no prompt+output bound at all on `/inference/v1/generate`.
4. **The only surviving bound clamps instead of rejecting** —
   `vllm/entrypoints/serve/utils/api_utils.py:170-207` `get_max_tokens` raises only
   when `max_model_len < input_length`, then returns
   `min(model_max_tokens, fallback, override, platform_max_tokens)`. An oversized
   `max_tokens` is therefore silently truncated to the context that is left, which
   is compatible with the reporter's "校验失效".

## The bounce back to the prefiller

The proxy's decode loop keeps `origin_max_tokens = req_data.get("max_tokens", 16)`
and re-dispatches on a decoder failure (`:980` onwards, `retry`/`mark_waiting_retry`
`:441`, `release_prefill_kv` `:403`, `_finish_instance` `:885`). Its
`release_prefill_kv` only decrements the proxy's own load counters, so a
re-dispatched request leaves the first prefiller's KV allocated until the P-side
expiry. Check the decode node's log for the proxy's own
`Error during streaming from decoder …` / reassign lines and for the P-side
`Force freed expired request … (480 seconds)`-style expiry: both being present
confirms "invalid request never rejected → re-prefill → held KV" rather than a
leak in the connector.

## Answer shape

Say which leg could have caught it (none, on the native disagg API at this pin),
give the four `path:line` facts, and rank the three fix sites: validate in the
proxy before it splits the request (it holds the original `max_tokens` and the
tokenized length), restore the bound in `build_tok_params`, or make
`get_max_tokens` reject instead of clamp. Ask for the deployment's actual entry
path first — the example proxy (OpenAI JSON) and `/inference/v1/generate`
(token arrays) take different validators, and the answer differs between them.
Verified at vllm v0.23.0 `0fc695f` / vllm-ascend v0.23.0rc1 `f4a08bd`; the
reporter's only follow-up was "ok", so no fix is confirmed.
