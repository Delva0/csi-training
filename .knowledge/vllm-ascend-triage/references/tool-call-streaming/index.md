# Tool-call / function-call output integrity

## Scope

The deployment is up, the engine answers, and the complaint is about a **tool
call** rather than about prose. Entry shapes, in the order they were recorded:

- the client raises a JSON error while assembling the call: `JSON Parse error:
  Unterminated string`, `Expecting ',' delimiter`, `Unexpected end of JSON
  input`, `Unterminated string starting at: line 1 column N` — often after
  `--speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}'`
  where `num_speculative_tokens: 1` "正常" (#7580);
- the arguments are reported truncated / occasionally malformed / repeated:
  "function call 模型返回时大概率会截断输出，小概率会格式错误，重复输出" (#8122,
  Qwen3-Next-80B-A3B, `stream: true`, `--async-scheduling`);
- `stream: true` plus a tool parser returns only CoT and no `tool_calls`
  (#5683, reporter-confirmed A/B: async scheduling off → correct call).

Not this topic — leave with the linked route:

- the acceptance metric collapses (`Accepted: 0`, `Mean acceptance length: 1.00`),
  a stall with drafts still produced, or a first response with `content: ""` and
  `completion_tokens: 1` → `../spec-decode-mtp/index.md`;
- long-prompt 乱码, short-prompt corruption, a VL `!!!!` run, or an
  accuracy/score gap → `../accuracy-benchmarks/index.md`;
- the thinking text sits in `content` / `reasoning_tokens: 0` →
  `../accuracy-benchmarks/reasoning-output-routing.md`;
- nothing ever started → `../startup-failures/index.md`; a traceback →
  `../hccl-and-hardware/index.md` (lane C) or
  `../upstream-downstream-drift/index.md`.

## Rules

1. **The message the reporter pastes is the client's, so the first artefact is
   the assembled string, not the model.** Concatenate
   `choices[i].delta.tool_calls[j].function.arguments` in arrival order and ask
   what the concatenation is (`bash <skill_root>/scripts/sse-toolcall-audit.sh
   <raw-sse-or-paste>`). It has three outcomes and they have different owners:
   parses as JSON → the client's accumulation/framing; missing tail
   (unbalanced `{`/`"` at the end, `finish_reason` still `tool_calls`) → the
   stream stopped short; unbalanced in the middle → the text itself is wrong.
   Do this before reading any parser, sampler or kernel. Two recorded
   calibrations: #7580's paste reassembles to `{"filePath":"/…/K8sYamlAgent.java`
   and `json.loads` reproduces the reporter's own `Unterminated string starting
   at: line 1 column 13 (char 12)`; #8122's pasted 97-chunk stream reassembles to
   fully valid JSON, so the sample in the issue was *not* one of the failing
   requests — say so instead of inventing a mechanism for a healthy stream.
2. **A numeric `stop_reason` in a chunk is upstream vLLM's own field, not
   corruption.** `ChatCompletionResponseStreamChoice.stop_reason: int | str |
   None` is vLLM-specific (`vllm/entrypoints/openai/chat_completion/protocol.py:120`
   at vllm v0.17.0) and carries the stop *token id*; it is filled from
   `output.stop_reason` where the choice is built
   (`chat_completion/serving.py:1303`, also `:1492`, `:1701`). A pasted
   `"stop_reason":154829` beside an empty `arguments` fragment says "a stop token
   ended the request here" and nothing about the JSON error.
3. **`finish_reason: "tool_calls"` does not prove the arguments were complete.**
   It is set from `auto_tools_called` when the parser streamed any tool call
   (`chat_completion/serving.py:1290-1297`), so a call whose tail never arrived
   still ends with that finish reason. Read it as "a call was started".
4. **One request decides engine-vs-assembly: add `return_token_ids: true` to the
   same body.** The stream choice then carries vLLM's per-chunk `token_ids`
   (`protocol.py:98`; filled at `serving.py:1307-1310` only under
   `request.return_token_ids`). If the token ids also stop before the closing
   tokens, the engine emitted a short sequence; if they are complete, the delta
   assembly lost them. This is cheaper and more decisive than any source walk and
   it is what the two recorded runs lacked.
5. **A constant in the code that equals the number in the report is the branch to
   name, and it is named in the first answer, not after its kernel is read.**
   `num_speculative_tokens: 3` fails and `1` works: `vllm_ascend/sample/
   rejection_sampler.py:114` is `using_block_verify = max_spec_len >= 3` with the
   comment "When num_speculative_tokens>=3, using block verify" — a threshold on
   the reported value. **Read the line at the reporter's ref first**: from
   v0.23.0rc1 it also ANDs `rejection_sampler_config.enable_block_verify`, which
   defaults `False` (`ascend_config.py:766`), so the threshold alone may select
   nothing there. Where it does hold, state it as the branch, label it
   source-verified, give the n=2 A/B that confirms it, and put the mechanism
   (`mtp-block-verify-acceptance.md`) behind that A/B. The recorded #7580 run
   reached that branch on its tenth turn and then spent its remaining nine turns
   on the acceptance kernel, the pytorch twin and the test files, and ended with
   the answer cut inside its evidence table and no action shipped — the branch
   was already the answer when it appeared.
6. **Split the three ownership layers before proposing a fix**: upstream vLLM
   OpenAI frontend (delta assembly, parser delta state, `stop_reason`/
   `token_ids` fields), vllm-ascend sampler (`sample/rejection_sampler.py`, the
   ops it patches), and the model/tokenizer (did the model emit a stop token
   before closing the JSON). The answer names which layer the evidence puts it in
   and which one instrument decides the next step.

## Pages

- `streamed-arguments-audit.md` — the audit procedure: how to obtain the raw SSE,
  how to read the script's three outcomes, the ordered ablations
  (`return_token_ids` → `stream: false` → `num_speculative_tokens: 2` →
  `temperature: 0` → `--async-scheduling` off), what to send back, and the worked
  #7580 / #8122 rows.
- `mtp-block-verify-acceptance.md` — what the `max_spec_len >= 3` branch does
  (cumulative-product acceptance over the whole draft block, recovered token,
  bonus token), which requests it cannot touch (greedy rows), why a wrongly
  accepted draft token can end a tool call early, the `parse_output` placeholder
  filter that silently removes slots, the missing equivalence test, and the
  confidence labels.
