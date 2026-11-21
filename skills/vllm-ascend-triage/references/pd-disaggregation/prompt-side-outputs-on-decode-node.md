# The decode node has no prompt forward: `prompt_logprobs`, `echo` and `prompt_text` across PD

Signature family — a P/D deployment whose client sends `prompt_logprobs` (or
`echo: true`) and the **decoder** dies with `OverflowError: out of range integral
type conversion attempted`; the exception surfaces in host Python inside the
tokenizer, long after the NPU work finished, so no device lane applies. Recorded
#13363: vllm-ascend v0.23.0rc1 (`f4a08bd`), vllm v0.23.0 (`0fc695f`),
`MooncakeConnectorV1` kv_producer/kv_consumer plus the example proxy.

The rule that decides the answer before any proxy read: in PD the decoder receives
K/V and never runs the prompt forward, so **every output that is a function of the
prompt forward is physically unavailable on the decode side** — prompt logprobs, the
prompt text `echo` returns, prompt-side embeddings. The deliverable is therefore a
split, not a data repair: strip the field for the decoder, or harvest it on the
prefiller and merge it into the response.

## Step 1 — the crash is upstream's, and the source confirms it

At vllm v0.23.0 (`0fc695f`), for a request the external KV connector has already
advanced:

| Anchor | What it does |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py:5441` | `start_idx = request.num_computed_tokens` — on the decoder this is `prompt_len` |
| `:5442-5443` | `start_tok = start_idx + 1`, `num_remaining_tokens = num_prompt_tokens - start_tok` → **-1** |
| `vllm/v1/outputs.py:94-113` | `LogprobsTensors.empty_cpu(…)` allocates `torch.empty(…, dtype=torch.int32)` — **uninitialised** |
| `:5449-5455` | the else branch takes `num_logits = num_remaining_tokens` and still stores the tensor in `prompt_logprobs_dict` |
| `:5456-5459` | `if num_logits <= 0: continue` → no `copy_` ever runs, the empty tensor ships |
| `vllm/v1/engine/logprobs.py:147` → `vllm/tokenizers/detokenizer_utils.py:83,98` | `convert_ids_list_to_tokens` → `tokenizer.decode([<stale negative int32>])` → the `OverflowError` |

Two traps: the helper lives at `vllm/tokenizers/detokenizer_utils.py` (**not**
`vllm/v1/tokenizers/…` — the recorded run quoted that nonexistent path, and
`:83,98` are the `def` and the `tokenizer.decode` loop), and the `empty_cpu` int32
field is uninitialised memory, so the id that ships is arbitrary — that the error is
an `OverflowError` only tells you it fell outside the tokenizer's unsigned range.

## Step 2 — `echo` is the same request on both endpoints

`prompt_logprobs = self.prompt_logprobs` then `if … is None and self.echo:` assign the
endpoint's own logprobs field — `vllm/entrypoints/openai/chat_completion/protocol.py:577-579`
(`top_logprobs`) and `vllm/entrypoints/openai/completion/protocol.py:275-277`
(`logprobs`). So a mitigation that strips only `prompt_logprobs` still crashes on
`echo: true`; strip both, and say so. Streaming already refuses the field
(`chat_completion/protocol.py:698`, `completion/protocol.py:414` — "`prompt_logprobs`
are not available when `stream=True`"), which is why these reports are non-streaming.

## Step 3 — what the prefiller can actually hand back

- `prompt_logprobs`: yes — the prefiller runs the prompt forward. Cheapest first
  check, before writing any code: send the same body **directly to the prefiller
  port** (no `kv_transfer_params`, `stream: false`) and read
  `choices[0].prompt_logprobs`. Populated and sane ⇒ the proxy-side harvest below is
  the whole fix; empty or garbage there too ⇒ the producer side carries the same
  upstream defect and no downstream mitigation can work.
- `prompt_text`: **chat schema only** at this pin (`chat_completion/protocol.py:125`,
  gated by the request's `return_prompt_text`, `:378`); `completion/protocol.py` has
  no `prompt_text` field at all, so a harvest plan that returns it from
  `/v1/completions` is wrong. One grep decides it.

## Step 4 — the example proxy at v0.23.0rc1 has zero handling

`rg -n 'prompt_logprobs|echo' examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
returns nothing, so the companion change is **to be authored, not found** — say that
instead of hunting a PR in the snapshot. The three sites:

- `build_prefill_request` (`:790`) copies the client body and overrides only
  `kv_transfer_params` / `stream` / `max_tokens` / `min_tokens` (`:800-805`), called
  from `send_request_to_service` (`:817`) — so `prompt_logprobs` *does* reach the
  prefiller: that is the producer the harvest needs.
- `assign_instances` (`:896`) reads **only** `kv_transfer_params` out of the prefill
  response (`:925-927`) and drops everything else, prompt logprobs included.
- the decoder request at `:1003` re-sends `req_data` unchanged, `prompt_logprobs`
  included — the field it must never see. `req_data` is also the retry path, so a
  strip has to be idempotent.

The layerwise variant `load_balance_proxy_layerwise_server_example.py` (640 lines) has
no `build_prefill_request`/`assign_instances`, so it cannot be patched by copy-paste;
sweep it separately rather than claiming it is covered.

## Scope

Pins for every `path:line`: vllm-ascend **v0.23.0rc1** (`f4a08bd`) and vllm
**v0.23.0** (`0fc695f`); the proxy is an example and changes without a version note.
The mechanism is upstream's and version-scoped: if the linked upstream fix is in the
reporter's release, the decode side no longer ships an uninitialised tensor and the
mitigation is optional — state which side and which commit you verified, and label the
harvest as a workaround for the versions you did verify. The generalisation is not
"prompt_logprobs is broken" but "the decode side cannot compute prompt-side outputs",
which also covers `echo` and any future prompt-derived field.
