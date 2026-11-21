# Thinking/reasoning output routing (verified in vllm v0.19.1)

Source: `vllm/reasoning/qwen3_reasoning_parser.py` and
`vllm/entrypoints/openai/chat_completion/serving.py` at vllm commit `b1388b1f`.

## The mechanism

`Qwen3ReasoningParser.extract_reasoning` (documented in its own docstring):

- strips a leading `<think>` if the model emitted one (Qwen3.5+ templates put it in the
  *prompt*, so only `</think>` appears in the output);
- if `</think>` is absent and `thinking_enabled` is False → `(None, model_output)`
  (everything is content);
- if `</think>` is absent and `thinking_enabled` is True (the default) → `(model_output, None)`
  — "output was truncated", everything is reasoning and **content is `None`**;
- otherwise splits at `</think>` and returns `content or None`.

In `serving.py`: with a reasoning parser, `content` is whatever the parser returned and
`reasoning` is only populated when the request asks for it (`include_reasoning`); without a
parser, `content` is the full text (thinking tags included).

## First gate: did the end marker survive detokenization? (#9361)

Before any template or model theory, `content: null` + the whole CoT *and* the answer
inside `message.reasoning` on a Kimi-K2.x serve (`--reasoning-parser kimi_k2`) has a
container-level explanation that costs one request to test. Verified at vllm
`v0.19.1` (`b1388b1f`):

| claim | anchor |
|---|---|
| the parser is fed `output.text`, the detokenized **generated** text only | `vllm/entrypoints/openai/chat_completion/serving.py:1374-1382` |
| a chat request decodes with `skip_special_tokens` **default `True`**, forwarded into `SamplingParams` and honoured by the decode stream | `chat_completion/protocol.py:200,510`; `vllm/v1/engine/detokenizer.py:175,182` |
| an id registered as *special* in the checkpoint tokenizer therefore never appears in `output.text` | consequence of the two rows above |
| the Kimi parser splits only on `</think>` or `<\|tool_calls_section_begin\|>`, and returns `(model_output, None)` when neither is present | `vllm/reasoning/kimi_k2_reasoning_parser.py:51-53,62-66,176-180` |
| with no `--reasoning-parser`, `content = output.text` (tags included) | `serving.py:1380-1382` |
| `reasoning_tokens: 0` alongside a populated `reasoning` is the same branch, not a second bug | `extract_content_ids` returns `[]` with no end token (`:146-147`) |

`skip_special_tokens` is a **chat body field** at this ref, so the reporter can test it
without a restart. One request decides:

```bash
curl …/v1/chat/completions -d '{"model":"kimi","messages":[…],"max_tokens":1000,
  "skip_special_tokens":false,"return_token_ids":true}'
```

`content` non-empty ⇒ the marker was emitted and stripped by detokenization (workaround:
keep the flag on the request, or drop the parser and split client-side). Still `null`
**and** no id equal to `tokenizer.convert_tokens_to_ids("</think>")` ⇒ the model never
emitted it (template/checkpoint, next section).

Two traps around that check. (a) A raw `/v1/completions` control is *not* a substitute:
it bypasses the chat template, so it tests a different prompt — #9361's control typed
`<|im_middle|><think>` by hand and opened with a garbage token, and its lack of
`</think>` said nothing about the chat path. Flip the flag on the same body instead.
(b) Ask for `completion_tokens` arithmetic first: when the completion is far longer
than the returned `content` can account for, nothing was lost — the bulk is in
`reasoning`, which is a routing decision (the #13002 rule below).

Coverage at vllm-ascend `v0.19.1rc1` for this model family is a gap worth naming and
not a cause: the support matrix lists only Kimi-K2-Thinking
(`docs/source/user_guide/support_matrix/supported_models.md:29`), the single Kimi e2e
test is disabled (`tests/e2e/multicard/4-cards/test_kimi_k2.py:28`,
`skip("CANN8.5 failed, capture stream failed")`), and the Kimi-K2.5 tutorial never
passes `--reasoning-parser`. So there is no validated launch to diff a K2.6 command
against — say that instead of implying the family is broken.

## Why this changes benchmark scores

Served with `--reasoning-parser qwen3` and thinking enabled, any request that hits
`max_tokens` while still inside the thinking block returns `message.content = null`. A
harness that only reads `message.content` records "no answer" for exactly those samples —
it looks like a hidden accuracy loss of roughly (lost samples)/(total samples), concentrated
in the subsets with the longest CoT. A harness that strips think tags from the raw text
(ais-bench's `extract_non_reasoning_content`) or reads `reasoning` too does not lose them.
That asymmetry alone can produce a multi-point tool-vs-tool gap on the same server.

## The parser *name* space: `--reasoning-parser` is not `--tool-call-parser`

Recorded follow-up on #8022. The reporter asked "qwen3_coder可以吗?" and got:

```text
KeyError: "Reasoning parser 'qwen3_coder' not found. Available parsers: deepseek_r1,
deepseek_v3, ernie45, glm45, granite, holo2, hunyuan_a13b, kimi_k2, minimax_m2,
minimax_m2_append_think, mistral, nemotron_v3, olmo3, openai_gptoss, qwen3, seed_oss,
step3, step3p5"
```

`qwen3_coder` is a **tool-call** parser (`vllm/tool_parsers/__init__.py:125-128` →
`Qwen3CoderToolParser`), not a reasoning parser: the two flags read two separate
registries (`vllm/reasoning/__init__.py:22-95` versus `vllm/tool_parsers/__init__.py`,
both at vllm v0.18.0 `bcf2be9`), so a name copied from the `--tool-call-parser` line of
the same serve command produces exactly this KeyError. The correct answer for a Coder
(or any non-thinking) checkpoint is **no `--reasoning-parser` at all**: it emits no
`</think>`, and `qwen3` with thinking enabled returns `(model_output, None)` — the
whole answer is routed to `reasoning` and `content` stays empty
(`vllm/reasoning/qwen3_reasoning_parser.py:81-87`). Keep
`--tool-call-parser qwen3_coder` (with `--enable-auto-tool-choice`) for tools and drop
the reasoning flag.

Two readings of the same mistake, both at boot and both upstream config, never the
model or the device:

| message | site |
|---|---|
| `KeyError: Reasoning parser '<x>' not found. Available parsers: a, b, …` | `vllm/reasoning/abs_reasoning_parsers.py:180-199` (`get_reasoning_parser`, reached from the serving / structured-output construction); names sorted by `list_registered` (`:200-204`) |
| `KeyError: invalid reasoning parser: <x> (chose from { a,b,… })` | `vllm/entrypoints/openai/api_server.py:503-511`, the arg pre-check |

The printed list is also a **version fingerprint**: the 18 v0.18.0 keys include the
aliases `glm45` and `holo2` (both `DeepSeekV3ReasoningWithThinkingParser`) and the
names `openai_gptoss` / `seed_oss`, which differ from their files
(`gptoss_reasoning_parser.py`, `seedoss_reasoning_parser.py`); `minimax_m2` and
`minimax_m2_append_think` are two classes in one file. Grep
`_REASONING_PARSERS_TO_REGISTER` at the claimed ref instead of trusting the spellings,
and never propose a name you have not found there or in the reporter's own list.

## DeepSeek / `deepseek_v3`: the delegation is per request, and `thinking` *is* its key

Verified at vllm v0.18.0 (`bcf2be9`); recorded #9521
(DeepSeek-V3.1-Terminus **w4a8 + MTP + QuaRot**, vllm-ascend v0.18.0rc1,
`stream: true`, `chat_template_kwargs: {"thinking": true}`, short user prompt,
`content` empty on *some* requests — "概率性出现没有content").

| claim | anchor at vllm v0.18.0 |
|---|---|
| the parser reads `thinking` **or** `enable_thinking` (each default `False`) and delegates: truthy → `DeepSeekR1ReasoningParser`, falsy → `IdentityReasoningParser` (everything stays in `content`) | `vllm/reasoning/deepseek_v3_reasoning_parser.py:32-41` |
| the `*WithThinkingParser` aliases instead **force** `thinking=True` when neither key is present | `:78-90` (the registered `glm45` / `holo2` names) |
| the parser is rebuilt **per request** from that request's kwargs merged over `--default-chat-template-kwargs` | `vllm/entrypoints/openai/chat_completion/serving.py:229-236`; merge `defaults \| request` at `vllm/entrypoints/openai/engine/serving.py:815-824` |
| the R1 branch splits on `<think>` / `</think>` | `vllm/reasoning/deepseek_r1_reasoning_parser.py:19-25` |
| in streaming every delta without the end token is emitted as `reasoning` — "reasoning content continues" | `deepseek_r1_reasoning_parser.py:63-65` |

Three consequences, all of which change what to investigate:

1. **A body-level `thinking` toggles routing request by request**, so "sometimes
   empty, sometimes fine" on one server can be two different bodies rather than a
   race, a sampler or a kernel. Diff the *requests* before the responses look
   random.
2. **For a `deepseek_v3` serve, `chat_template_kwargs: {"thinking": true}` is the
   right spelling, not the AisBench-ism the kwarg trap below warns about.** Treating
   it as wrong routes the case into a model/quantization hunt while the mechanism
   is one `content: null` + populated `reasoning`.
3. **The release-diff branch is closed here.** The reporter later reported the same
   empty `content` on `0.18.0` **and** `0.19.1rc1`
   (`issue/9521#issuecomment-4541092252`). That rules out "a newer rc fixed it" and
   leaves two live branches — routing (the caller reads only `content`) versus a
   genuinely short completion — which the same observation does **not** separate.
   The deciding artefact stays one response printed whole: `content`, `reasoning`,
   `finish_reason`, `completion_tokens`, `return_token_ids: true`. On this
   deployment class the competing explanations (MTP block-verify, w4a8-QuaRot
   degeneration, `--max-model-len` truncation) all predict a *small*
   `completion_tokens`, so that number, not the version, is the discriminator.

## The inverse shape: "thinking is inside `content` and `reasoning_tokens` is 0"

Re-verified at vllm v0.22.1 (`0decac0d96c4`) + vllm-ascend v0.22.1rc1 (`5f6faa0`),
for #11715. When *no* reasoning output appears at all — the CoT text sits in
`content`, thinking language is mixed (English thinking, Chinese answer),
`completion_tokens_details.reasoning_tokens` is `0` — the first gate is whether
the server was started with `--reasoning-parser` at all. It is **upstream
OpenAI-frontend configuration**, invariant to MTP, image tag and the Ascend
layer, so say that before any device theory.

| claim | evidence at vllm v0.22.1 |
|---|---|
| the flag defaults to empty | `vllm/config/structured_outputs.py:35` `reasoning_parser: str = ""` |
| no parser class ⇒ no split and no counting | `vllm/entrypoints/openai/chat_completion/serving.py:98,125-126,143,253-255` build the parser only when the name is set; the split branch is guarded by `elif reasoning_parser:` / `if reasoning_parser` (`:343-362,1124-1133`) |
| `qwen3` covers Qwen3 **and** 3.5/3.6 | `vllm/reasoning/__init__.py:107-110`; the parser strips a leading `<think>`, since from Qwen3.5 the template puts it in the prompt (`qwen3_reasoning_parser.py:21-24`) |
| `reasoning_tokens` is parser-derived, not a sampler counter | `vllm/reasoning/basic_parsers.py:183` `count_reasoning_tokens`, consumed at `vllm/entrypoints/openai/responses/serving.py:861-893` |
| a *wrong* name is not this failure (it is a boot failure) | two upstream boot-time `KeyError`s list the registered names — `vllm/reasoning/abs_reasoning_parsers.py:180-199` and `vllm/entrypoints/openai/api_server.py:503-511` (v0.18.0) — so a server that came up had either the right name or none; the list also fingerprints the release ("The parser *name* space", above) |

Reporter-supplied scope (#11715, later comment): 910B + Ubuntu 22.04 + CANN 25.5.1,
two images (v0.20.2rc1, v0.18.0), MTP on and off, English and Chinese prompts, both
a 35B-A3B and a 27B w8a8 Eco-Tech checkpoint — every combination identical. That
invariance is itself the argument for a frontend config gate over any
model/device/kernel cause.

Field-name drift, checked before concluding "`reasoning_content` is missing": at
vllm v0.22.1 the chat response field is `message.reasoning`
(`chat_completion/protocol.py:64`), `include_reasoning` defaults to `True`
(`:229`), and `reasoning_content` survives only as a deprecated **input** alias
that a pre-validator renames (`:429-446`). Advice to read
`choices[0].message.reasoning_content` is right for older releases and wrong for
this one; name the field the pinned version emits.

## The thinking is in the wrong language ("回答中英混杂", "the answer came back in English")

Work them in this order; a run that starts with the model's Chinese ability or the
GDN/RoPE kernels spends its budget on the wrong layer.

1. **Is the reporter reading `reasoning`, and was it truncated?** `content: null` with
   the whole completion inside the thinking block and `finish_reason: length` is the
   parser's documented "output was truncated" branch above, not a language defect. One
   repeat with `temperature: 0` and a larger `max_tokens` decides it, and it is the
   first thing to look for when the evidence is a screenshot whose OCR shows
   `"content": null` next to `completion_tokens: N`.
2. **Is the language a model/template property?** For the Qwen3.5/Qwen3.6 family it
   is: with an empty system prompt the thinking runs in English while the answer
   follows the prompt's language. Precedent `#6848` (Qwen3.5-397B-A17B on the 0day
   image `vllm-ascend:qwen3_5-v0-a2`: thinking English, answer Chinese, a single
   `</think>`) is the same symptom reported as a defect. Note its request passed
   `chat_template_kwargs: {"thinking": true}` — the wrong key, see the trap below.
   The levers are a system-prompt language instruction or disabling thinking, and
   **disabling thinking is version-gated**: `#7094` recorded
   `chat_template_kwargs: {"thinking": false}` returning 乱码 / `completion_tokens: 1`
   on that 0day image until upstream PR 34779 (the Qwen3/Qwen3.5 reasoning-parser fix)
   landed, with `#7061` as the same defect seen as `</think>` leaking into `content`.
   Before recommending it, check the installed parser carries the Qwen3.5 handling —
   its docstring says "Starting with Qwen3.5, the chat template places `<think>` in
   the prompt" and treats a missing `</think>` as truncation
   (`vllm/reasoning/qwen3_reasoning_parser.py:16-46` at vllm v0.20.2) — otherwise send
   the system-prompt language instruction instead.
3. **Before calling it model behaviour, scan the launch for a hand-written RoPE
   override.** A reporter who extended context with
   `--hf-overrides '{"text_config": {"rope_parameters": {...}}}'` (commonly
   `rope_type: yarn`, `factor: 4.0`, `partial_rotary_factor`, `mrope_interleaved`,
   `mrope_section`) has replaced the checkpoint's rope parameters wholesale:
   `get_rope(..., rope_parameters=config.rope_parameters)` is the single consumer
   (`vllm/model_executor/models/qwen3_next.py:250-255` at vllm v0.20.2, `bc150f5`),
   and yarn's low-frequency interpolation plus mscale act at *every* position, not
   only past `original_max_position_embeddings`. The validated Qwen3.5 recipes use
   `--max-model-len 133000` and no `--hf-overrides`
   (`docs/source/tutorials/models/Qwen3.5-27B.md:105-127` at v0.20.2rc1), so
   "restart without the override, same request" is the row that separates model
   behaviour from a launch that changed the numerics. Label it a hypothesis until it
   runs, and ask for the checkpoint's own `text_config.rope_parameters` — without it
   nobody knows whether the override differs from the shipped value.
4. If the model variant is absent from
   `docs/source/user_guide/support_matrix/supported_models.md` (at v0.20.2rc1 the
   matrix lists only Qwen3.5-397B-A17B and Qwen3.5-27B, not 35B-A3B) there is no
   validated recipe to diff against: say so as a version/support gap instead of
   implying the class is broken.

## The kwarg-name trap

`thinking_enabled` is read from `chat_template_kwargs["enable_thinking"]` (default `True`).
`tools/aisbench.py` in vllm-ascend instead writes `chat_template_kwargs={"thinking": True}`,
and some Qwen3.5-era templates expose the switch as `thinking`. If the served template's
switch name and the parser's key differ, thinking control silently does nothing and the
parser mis-classifies output. Do not assume either spelling: check the model's
`chat_template.jinja` and prefer `enable_thinking` with the parser. (For a
`deepseek_v3` serve **both** keys are read and `thinking` is the switch that
selects the R1 splitter — see the DeepSeek section above before calling the
reporter's key wrong. Corroborating issue
precedent: #7061 / #7094 in `history/`, which also track the upstream parser fix PR 34779.)

The spelling is per-parser, so one client body flips the branch by model family. At
vllm **v0.23.0** (`0fc695f`), `vllm/reasoning/kimi_k2_reasoning_parser.py:40-46`
does `thinking = bool(kwargs["chat_template_kwargs"].get("thinking", True))` and
falls back to `IdentityReasoningParser` only when that is falsy — the **opposite**
key from Qwen3's `enable_thinking`. Two consequences for a "`</think>` leaked into
`content`" report on a Kimi-K2.x model (#13002's shape 1) before any corruption
theory: (a) a request that sends `enable_thinking: false` does not by itself select
the identity fallback, and (b) while the thinking branch *is* active, text arriving
before an end token is emitted as `reasoning`, so a client that concatenates only
`delta.content` / reads only `message.content` cannot distinguish a routing
decision from a model-emitted token. Ask for one response that prints both
`content` and `reasoning` (and `return_token_ids: true` for the emitted ids) — that
is the whole gate; the probe is also right when the tag appears *after* a complete
answer, because of the branch order: in `extract_reasoning_streaming` the
`is_reasoning_end(previous_token_ids)` test returns `DeltaMessage(content=delta_text)`
*before* the delta's `</think>` is looked at (`:212-214`, ahead of the end-token
split below it), so once reasoning has ended any `</think>` the model emits later
reaches `content` verbatim. Two records, both decided by one response printing both
fields: an answer body that turned out to sit in `reasoning` while `content` held
only the post-tag tail, and a tag that was a duplicate emitted after the split.
The arithmetic is the cheap pre-check — when `completion_tokens` is far larger than
the returned `content` can account for, the bulk is in `reasoning` rather than lost
(recorded #13002 shape 2: `completion_tokens: 9665` against 831 characters).

## Documentation gap that produces this signature by itself

At vllm-ascend v0.22.1rc1 only seven model pages pass `--reasoning-parser`
(`DeepSeek-V4-Flash.md`, `DeepSeek-V4-Pro.md`, `GLM4.x.md`, `GLM5.md`,
`GLM5.2.md`, `Hy3-preview.md`, `MiniMax-M2.md`); **no Qwen page does** — the
Qwen3.6 serve block has none (`docs/source/tutorials/models/Qwen3.6-35B-A3B.md:106-123`)
and `Qwen3.5-27B-Qwen3.6-27B.md` does not mention the flag either. A reporter who
followed the Qwen3.6 tutorial verbatim therefore gets exactly this signature, so
quote the doc block as the reason they hit it and treat the omission as a
separate doc bug (recheck the page's flags at whatever ref the report names; the
sibling pages' flag choices move between releases).

## Checks to request from the reporter

- Server flags: `--reasoning-parser`, `--default-chat-template-kwargs`, `--max-model-len`.
- Endpoint and body: `/v1/chat/completions` (a raw `/v1/completions` or a
  text-in/text-out bench bypasses both the chat template and the parser, so all
  text must land in `content` — and because it is then a *different* prompt, it can
  neither confirm nor refute what the chat path emitted; flip `skip_special_tokens`
  on the same chat body instead), plus whether `chat_template_kwargs.enable_thinking`
  is passed (the parser reads `enable_thinking`, default `True`,
  `qwen3_reasoning_parser.py:46`). With it `False` the template prefills
  `<think>\n\n</think>` and plain content is expected.
- The checkpoint's own `chat_template.jinja` (a re-quantised export can ship a
  template without `</think>` / `enable_thinking`, which defeats the parser too).
- Mixed thinking language: work the section above in order (routing and budget, then
  model/template behaviour, then a hand-written `--hf-overrides` rope block) before
  treating `--reasoning-parser` or the model as the answer.
- One `/v1/chat/completions` call with small vs large `max_tokens`: compare `finish_reason`,
  `message.content`, `message.reasoning`.
- Harness behaviour: does it read `content` only, `reasoning` too, or strip `<think>` tags
  from raw text? Is thinking supposed to be on for an accuracy run at all?
- Whether the reported per-subset gaps are concentrated where CoT length is longest — that is
  the fingerprint of this failure mode.

## The thinking channel never closes: the stream burns the whole budget in a loop (recorded #16116)

Verified at vllm v0.26.0 (`568afb3a`) + vllm-ascend v0.26.0rc1 (`f2f74a16`): DeepSeek-V4-Pro with
`chat_template_kwargs: {"thinking": true, "reasoning_effort": "high"}`, every SSE chunk
`delta.reasoning` and **zero** `delta.content`, `output_token == max_completion_tokens == 77817`.
Three questions, three answers — this is a request-contract and model-degeneration report, not a
device fault and not a broken parser:

1. **The budget was legal.** `chat_completion/serving.py:307-311` takes
   `request.max_completion_tokens if … is not None else request.max_tokens`, so a body carrying both
   `max_tokens: 4096` and `max_completion_tokens: 77817` ran the larger one by design
   (`get_max_tokens`, `entrypoints/serve/utils/api_utils.py:170-207`, then clamps). Never read
   `output_tokens == max_completion_tokens` as a leak.
2. **No loop brake exists.** The only repetition controls are `presence_penalty` /
   `frequency_penalty` / `repetition_penalty` (`sampling_params.py:224,228,232`) — per-token logit
   shaping, and no `no_repeat_ngram_size` anywhere in the sampling params. An exact n-gram cycle the
   model prefers survives all of them; forced termination exists only as a token count. Say so
   rather than recommending a flag.
3. **A separate thinking cap exists upstream and is dead on NPU.** `thinking_token_budget` is a chat
   field (`chat_completion/protocol.py:241`) and a sampling param (`sampling_params.py:344`),
   enforced by forcing the end-of-thinking token (`v1/sample/thinking_budget_state.py`,
   `maybe_create_thinking_budget_state_holder`, built in `v1/worker/gpu_input_batch.py:111`).
   Downstream sets `self.thinking_budget_state_holder = None` with the comment *"NPU does not use
   thinking budget, so the holder is always None"* (`vllm_ascend/worker/npu_input_batch.py:56-59`),
   so the field parses, returns 200 and does nothing. Upstream has its own gate (Model Runner V2
   warns once, `config/vllm.py:2227-2231`). One request (`"thinking_token_budget": 512`, reasoning
   does not stop near 512) is the evidence for a separate downstream issue.

**The amplifier is a prompt, not a budget.** On DeepSeek-V4 vllm-ascend maps `reasoning_effort` to
a **prompt prefix** (`vllm_ascend/patch/platform/patch_deepseek_v4_thinking.py`):
`REASONING_EFFORT_PROMPTS = {"low": "", "high": "…Absolute maximum with no shortcuts permitted…
Explicitly write out your entire deliberation process, documenting every intermediate step,
considered alternative, and rejected hypothesis…", "max": …}` (`:25-48`), prepended only when
`index == 0 and thinking_mode == "thinking"` (`:85-86`). Absent effort with thinking on gives
`"low"` under the preview mapping and `"high"` otherwise (`:120-127`); `"none"` switches
`thinking_mode` to `"chat"` (`:128-129`); `"medium"`/`"minimal"` collapse to the empty `low` prefix.
So `high` cannot bound thinking — it asks for exactly the "enumerate every rejected hypothesis and
do not stop" behaviour the captured loop shows, and the loop text (`我决定…但…不准确。` ×10) is that
shape.

Deliverable order: (1) classify, and say the empty `content` proves routing works (the inverse of
the `content`-only shape above); (2) close the underspecified field in the caller's schema (a
prose enum with no none-value, e.g. `"executeType": "执行动作：新增、更新删除"`, is the prompt-side
root cause); (3) one budget field only, re-run at 8192 and expect `finish_reason: "length"`;
(4) `reasoning_effort: "low"` or `thinking: false` on the same body separates effort-prompt
amplification from model behaviour; (5) identity run ×3 at `--max-num-seqs 1` / `temperature: 0`
(`long-output-repetition.md`'s row); (6) `thinking_token_budget` as the no-op demonstration;
(7) until a server brake exists, the only working loop brake is client/gateway-side abort on a
repeated n-gram. Trap: a `finish_reason`/`usage`/`[DONE]`-less capture is a gateway truncation
artefact when `bytes_sent` far exceeds the captured window and the window starts and ends
mid-sentence — not a server defect.
