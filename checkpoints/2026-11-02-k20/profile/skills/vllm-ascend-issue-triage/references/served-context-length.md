# Served context length and "request too long" rejections

Reports of the form "`max_tokens` above N returns 400, below N works" mix three different
things: the length vLLM actually serves, the length the launcher script thinks it serves,
and errors raised in front of the API server. Separate them before reading any model code.

## The served length is the resolved `max_model_len`

`Using max model len N` in the API-server log comes from
`vllm/config/model.py:1545 ModelConfig.get_and_verify_max_len` (verified at vLLM
`v0.14.0rc1`), and `_get_and_verify_max_len` (`:1898-2020`) resolves it as:

1. the architecture config's derived length (`max_position_embeddings` and friends, scaled by
   RoPE `factor`, LongRoPE using `original_max_position_embeddings`);
2. lowered to the tokenizer config's `model_max_length` when that file has one;
3. `--max-model-len` (or `-1` = auto-fit to memory) overrides it; a manual value above the
   derived length is refused unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN` is set.

So a launcher variable such as a logged `MAX_SEQ_LEN: 16384` is *not* a vLLM setting. It
takes effect only if the generated `vllm serve` line passes `--max-model-len`. The pasted
command in issue #7535 (0.14.0rc1, DP=2 × TP=8, MiniMax-M2.5) does not, and its log prints
`Using max model len 196608` — the model's own limit, not the script's 16384. Two-node
DP does not multiply the context length; do not offer that as the origin of a 32k ceiling
without a config or proxy that enforces it.

## `finish_reason` decides whether this class applies at all

An early stop is a length report only when `finish_reason` is `length`. The recorded `#12278` body
(`vllm-ascend:v0.18.0-a3`, Qwen3.5-397B-A17B-w4a8-mtp) carried `completion_tokens: 97`,
`finish_reason: "stop"`, `stop_reason: 248044` and a `content` that broke mid-phrase, while the
reporter's prose said "only specifying a large `max_tokens` avoids it". A numeric `stop_reason` is
the token id the sampler stopped on, which raising `max_tokens` cannot change, so that prose
describes a different run or temperature. Diagnose from the pasted body (`finish_reason`, `stop_reason`,
`completion_tokens`) and leave the reporter's `max_tokens` story out unless a `length` finish is
shown: a premature stop points at the sampler/logits path, not at the context window
(`pass_config.enable_sp` without FlashComm1 is one recorded trigger in
[env-vars-and-additional-config.md](env-vars-and-additional-config.md); its host-side mechanism is open).

## Error shape tells you who rejected the request

At vLLM `v0.14.0rc1` an oversized request fails as `VLLMValidationError` from
`vllm/entrypoints/openai/serving_engine.py:1062-1081`, with
`"This model's maximum context length is <N> tokens. However, your request has ..."` or
`"'max_tokens' or 'max_completion_tokens' is too large: ... <max_tokens> > <max_model_len>"`,
returned as `{"error": {"message", "type", "param", "code"}}`
(`ErrorResponse` at `vllm/entrypoints/openai/protocol.py:131`, `create_error_response`
at `serving_engine.py:745`). A 400 whose body is
`{"msg": "Token indices sequence length is longer than the specified maximum sequence length for this model!"}`
is neither that envelope nor vLLM wording — it is HuggingFace tokenizer phrasing — and a
`server: envoy` response header names a proxy in the path. Treat such a body as a
gateway/launcher rejection (candidate) and say which layer must be read to confirm: the API
server's own log line for that request id, or the proxy/launcher config.

## A request that cannot be admitted is not rejected at all

The 400s above are the API layer refusing an over-long request. A request that is within
`max_model_len` but whose predicted blocks exceed the pool produces no error: upstream
`vllm/v1/core/kv_cache_manager.py` `allocate_slots` returns `None` (full-sequence gate at `:354-368`,
per-step at `:385-398`, verified at vLLM `v0.23.0`), the scheduler leaves it waiting, and the client sees
only its gateway timeout. When a report shows a hang rather than a rejection, read the KV-cache sizing
path (blocks per request, pool size, hybrid block sizes) instead of the length validation, and tell the
reporter which metric or engine-log line would show the request stuck in `waiting`. The hybrid-pool case
with numbers is in [c8-dsa-deployment-flags.md](c8-dsa-deployment-flags.md) (GLM-5-Next `index_kpool`).

## Open question kept from issue #7535

Reported: `max_tokens: 32769` → 400 while 32k works, `max_tokens` is the only varying
parameter, streaming makes no difference; the maintainer thread does not answer the 32k
ceiling. The only concrete item in that thread is a separate failure from the same server
log — `ValueError: /data/ is not a multimodal model`, raised through
`_parse_chat_message_content_part` → `mm_parser.parse_image` →
`mm_processor.validate_num_items` — followed by a maintainer remark that
`--mm_processor_cache_type="shm"` was set on a model vLLM does not treat as multimodal.
That traceback must not be merged into the `max_tokens` question; the pasted request was
plain text, so the log line probably belongs to another request.

Next verification, in order: the real `vllm serve` argv (does it carry `--max-model-len`?),
the API-server log for the failing request id, then any Envoy/launcher limit. Nothing above
establishes where 32k comes from.
