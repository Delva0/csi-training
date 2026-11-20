# Accuracy and output cases

score/accuracy disagreement, 乱码 and repetition (short and long prompt, hybrid
block size, PD hybrid state), reasoning routing and parser names, early EOS,
spec-decode/MTP acceptance and bookkeeping, tool-call argument integrity, pooling
and embedding outputs. Routes: `../accuracy-benchmarks/index.md`,
`../spec-decode-mtp/index.md`, `../tool-call-streaming/index.md`,
`../pooling-embedding/index.md`, `../serving-frontend-faults/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file.

**#8022** vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be9`),
Qwen3.5-27B, streaming chat: with `--reasoning-parser qwen3` (or `deepseek_r1`)
`reasoning_content` stayed null, the thinking vanished from `content`, and the first
token looked "极慢". Track: output routing (upstream OpenAI frontend). The recorded
run got the mechanism right at the pinned commit — the chat field is `reasoning`
(not `reasoning_content`), the deltas carry the CoT as `reasoning`, and the "slow
first token" is the first **content** delta arriving after the thinking — and said so
in one paragraph. What it could not close: the only later evidence is the reporter's
next question ("qwen3_coder可以吗?"), who then ran `--reasoning-parser qwen3_coder` and
hit the boot `KeyError: "Reasoning parser 'qwen3_coder' not found. Available parsers:
deepseek_r1, …, qwen3, seed_oss, step3, step3p5"` — a **tool-call** parser name
(`vllm/tool_parsers/__init__.py:125`). The lesson now sits in the name-space section of
`../accuracy-benchmarks/reasoning-output-routing.md`: name both registries in the first
answer, and say that a Coder / non-thinking checkpoint needs no reasoning parser at all.

**#9355** v0.18.0 Qwen3.6-27B bf16 on 2×910B3, whose first request after start returned
`content: ""` with `completion_tokens: 1` and `finish_reason: "stop"` while later
requests answered (the first run shipped only a half-answer; its lead mechanism was a "server-lifetime first decode" story that the
reporter's later comments leave unestablished: the follow-up changed the prompt and
`temperature` and posted to a different port, and the second comment only confirmed the
image tag). Two later runs against the page this case produced both answered in full (18
and 31 messages) but neither read the draft-side config gates or offered `min_tokens`;
the sibling `Qwen3.5-27B.md` recipe was opened only by the second, and used there as a
copied workaround rather than as the deviation table
(`../spec-decode-mtp/empty-first-response.md`); #10495 v0.20.2rc1 Qwen3.6 MoE + `qwen3_5_mtp` on 2×A2 with ~64K prompts,
acceptance collapsing to exactly 0.000 with drafts still produced and throughput
stalling, where the run ranked KV capacity/queue thrash first and the reporter then
reported MTP as the owner (MTP off → normal); #10943 v0.20.2rc1 custom
`nvidia/LocateAnything-3B` served through `TransformersMultiModalForCausalLM`,
`terminate called after throwing an instance of 'std::logic_error'` /
`basic_string::_S_construct null not valid` at EngineCore init — a native
null-string abort; the run spent the session inside the exported
CANN/torch_npu/triton C++ and shipped a one-line reply, and the reporter's later
comment established that the unmodified
checkpoint crashes identically (AutoModel loads it fine), so none of the eight
model-side changes was the cause; #3902 v0.11.0rc0-a3 Qwen3-VL-235B-A22B TP=16 on
one 910C, "the machine crashes" with no log pasted, answered through the
ACL-graph capture lane (host memory was its rank 3) — the reporter's later `top`
then showed 16 python3 ranks at 88-100 GiB RES each, 1472 of 2014 GiB host RAM
with swap 0, so host memory is the number that lane owes first.

**#7580** v0.17.0rc1 A3 大EP GLM-5 + `deepseek_mtp`, `num_speculative_tokens: 3` →
`JSON Parse error: Unterminated string` while `1` was normal: the run reached
`sample/rejection_sampler.py:114 using_block_verify = max_spec_len >= 3` on its
tenth turn, then spent its remaining nine turns inside the acceptance kernel, its
pytorch twin and the test files, and was cut inside its evidence table with no
ablation plan; the reporter's later reproduction showed why the
first artefact owed was the assembled arguments — they stop at
`{"filePath":"/…/K8sYamlAgent.java`, a lost tail, which `return_token_ids: true`
splits between engine and delta assembly. #8122 v0.14.0rc1 Qwen3-Next-80B-A3B
`stream: true` + `--async-scheduling`, "流式 function call 大概率截断、小概率格式
错误、重复输出": answered late but in full, ranked on the reporter-confirmed #5683
precedent (async scheduling on → only CoT, no `tool_calls`) plus the CI gap that
`async_scheduling` never appears in `tests/e2e/multicard/4-cards/test_qwen3_next.py`;
the same audit on its 97-chunk paste returns valid JSON, so the posted sample was
not the failing request and the answer had to say so.

**#4057** v0.11.0rc0 Qwen3-Next-80B-A3B, "问到数字相关问题时精度较差" on a 19-token
prompt with no serve command in the report: the run found the downstream fp32
`torch_chunk_gated_delta_rule` prefill replacement and the tutorial's experimental
Triton-Ascend note, then spent its output budget re-deriving the FLA chunk algebra
and shipped nothing — its 14th assistant turn was cut mid-plan, with ~49 K
characters of reasoning and zero message text. The reporter's later public comment
supplied the deploy command (TP=4, `--max-model-len 8192`, `--enforce-eager`, no
MTP/EP/quantisation), which is the narrowing the answer owed
(`../accuracy-benchmarks/qwen3-next-gdn-numeric-path.md`).

**#13853** v0.20.2 + vllm-ascend v0.20.2rc1 Qwen3.5-0.8B on one 910B, 乱码 at
`--block-size 2048` with a "正常" command at `--block-size 512 --enforce-eager` (two
knobs changed in one comparison): the run had the two-clamp story (a hybrid model's
cache block is never clamped to 128 and is only *raised* by the mamba page-size
constraint) by its 10th turn, but kept walking turns 8-22 through `patch_gdn_attn.py`
chunk metadata, `gdn_chunk_meta` and the block-table plumbing, and was cut inside a
reasoning block — 22 assistant turns, ~11 K of a 16 K-token output budget, every
message a tool call, zero message text. The confound and the matrix were the
deliverable (`../accuracy-benchmarks/hybrid-block-size-corruption.md`).

**#13719** vllm-ascend v0.23.0rc1 (`f4a08bdd`) + vllm v0.23.0 (`0fc695fc`) as the
snapshot's reference, but the report's build was a daily "0.23.0": GLM-5.2 W8A8C8 on
910C, 4-node PD disaggregation (`MooncakeConnectorV1`) with `deepseek_mtp`
(n=1 P / n=5 D), 64 concurrency MMLU-Pro, `max_tokens` 32000, "2 小时后 5 条响应存在
句子的重复", both serve commands pasted and no logs at all. Delivered in full at its
10th assistant turn: the class is decode-time repetition (short
prompt, long completion — not the prompt-length corruption pages), and the
deliverable was the tutorial diff (`GLM5.2.md` 4-node 1M PD example) plus the
`--max-num-seqs 1` identity row. The run's useful correction: at this ref
`using_block_verify` also requires `rejection_sampler_config.enable_block_verify`
(default `False`, `ascend_config.py:766`), so the n≥3 block-verify story does not
transfer from #7580 (`../accuracy-benchmarks/long-output-repetition.md`).
A later replay against that page (same `SOURCES.json` pins, no new evidence) answered
in **five** assistant turns and ~6 K completion tokens instead of ten turns: the page
was enough, and what the replay still spent turns on was re-reading the tutorial's
P/D blocks the page already records — the page's own warning about re-grepping only
what it marks version-specific is the fix. The reporter's only later activity on the
thread is a failed attachment (`Failed to upload "repeated_responses.txt"`), which is
why the page asks for the samples inline.

**#13002** v0.23.0 on A2, Kimi-K2.6 W4A8, 1P1D `MooncakeConnectorV1` + eagle3,
≥16 concurrent / ≥23K-token prompts: `</think>` leaking into `content`, repetition
collapse, 乱码 in 6-10 % of requests, both serve commands pasted. **Shipped no
answer**: 21 assistant turns, 1.06 M prompt tokens and the whole 16 K output
budget, ~8 turns reading `mooncake_connector.py` in 2600-line slices; the model's
own tutorial (`docs/source/tutorials/models/Kimi-K2.6.md`, validated P/D commands
and §9.1 table) was opened only at turn 17, though its diff — `--no-enable-prefix-caching`
on both nodes vs the reporter's `--enable-prefix-caching`, `--max-model-len 32768`
vs 131072, P `HCCL_BUFFSIZE 800` vs 256, D `--max-num-batched-tokens 32/4` vs 256,
P eagle3 `num_speculative_tokens 1` vs 3 — plus the ablation matrix was the answer
(`../accuracy-benchmarks/garbled-long-context-output.md`).

**Replay, same paste and the same body** (this version also carried both full serve
commands and the reproduction script): once that page existed, the run shipped the
complete ranked answer in **11 assistant turns / 17 tool calls** — verdict plus the
one cheap action first, the three symptom shapes separated, the five-row diff and
the ablation ladder — and opened `mooncake_connector.py` exactly once, for the
`dp_size`/`tp_size` grep its P-vs-D difference needed. What the page's traps added:
the `</think>` leak resolved to the parser-key/branch-order story rather than
corruption, 9665 `completion_tokens` against 831 characters of `content` read as a
routing measurement, and the reporter's own "≥16 concurrency" axis was dropped
because their table showed 0/32 at 16-concurrency on one instance. Its remaining
gaps were honest ones: the chat template's switch key is in the weights
(`tokenizer_config.json`), which the snapshot does not carry.

**#7197** vllm-ascend v0.16.0rc1 + vllm v0.16.0, 910B3, Qwen3-Next on a customer
dataset: `ERROR [serving.py:315] Error in preprocessing prompt inputs` with
`RuntimeError: Already borrowed` raised at `hermes_tool_parser.py:63` (the tool
parser's `__init__` encoding its special tokens) under `_preprocess_chat` — a
fast-tokenizer borrow conflict between the event-loop thread and
`AsyncMicrobatchTokenizer`'s single-worker executor, with no `vllm_ascend` frame
and no downstream tokenizer code to blame. The recorded run answered on its 14th
assistant turn with the mechanism, vLLM's single existing retry for the same
exception (`vllm/multimodal/processing/context.py:440-459`, tokenizers#537), the
serial-concurrency/retry ladder and the transformers-4.57 line-number gap; the
prior-month history has no `Already borrowed` precedent at all
(`../serving-frontend-faults/tokenizer-already-borrowed.md`).

**#7689** vllm-ascend v0.13.0rc1 (`1b5d5abf`) / vllm v0.13.0 (`72506c98`), dots.ocr
on 2×910B4 (32 G) vs 910B3 (64 G) with an identical serve command: the B4 host answers
garbled or off-task text where B3 is correct. VL corruption track; the run delivered a
complete ranked answer (first ablation `USE_OPTIMIZED_MODEL=0`, then TP=1,
`--async-scheduling` off) and paid for one useful version correction: at this pin
`vllm_ascend/models/` no longer exists, the vision tower is patched by class name
(`utils.py:697` → `ops/mm_encoder_attention.py:31`) and the head-dim pad moved from
weights to **activations** (`:101-108`), so #976's weight-padding mechanism cannot be
quoted at v0.13.0rc1 — see `../accuracy-benchmarks/vl-vision-tower-degeneracy.md`.

**#8724** vllm-ascend `v0.13.0rc3` (`bbe9bc0d`) / vllm `v0.13.0` (`72506c98`), one
A3 16-card node, DeepSeek-V4-Flash-w8a8-mtp with
`--speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}'`,
DP4×TP4 + EP, `--async-scheduling`, `vllm bench serve --temperature 0.6`
(0 or unset runs clean): three tracebacks at the same second — worker `KeyError:
'chatcmpl-bench-…'` at `patch/worker/patch_model_runner.py:154`, EngineCore
`'NoneType' object has no attribute 'sampled_token_ids'`, APIServer `Counters can
only be incremented by non-negative amounts`. One provenance (a *discarded* request
gets a `-1`-filled sampled row → count 0 → `num_accepted = -1`, and
`prev_req_id_to_index` omits exactly those requests), and the failing statement is
upstream v0.13.0's, carried verbatim by the patch, whose only added hunk
(`is_kv_consumer`) a non-PD command never runs. **Shipped no answer**: 29 turns,
every message a bare tool call, ~48.5 K characters of reasoning against 73
characters of message text, cut inside a reasoning block on the very turn it began
the patch-vs-upstream diff — the first step that should have been taken
(`../upstream-downstream-drift/patch-copy-vs-upstream.md`,
`../spec-decode-mtp/async-spec-bookkeeping-desync.md`). A later replay that
started from the classification row and that page **answered it in 8 assistant
turns (8.4 K characters of message text)**: four confirmation reads (the
`KeyError` line, the `-1` fill in `eagle_proposer.py:825`, the index-map filter,
the `vllm_version_is('0.13.0')` install gate), then the ranked answer with the
upstream identity, the temperature anchor and the honest gap (which step discards
a row). The page, not more source, is what closed it.

**#10405** vllm-ascend `v0.20.2rc1` (`367b8e6`) / vllm `v0.20.2` (`bc150f5`), 4×910B2C,
Qwen3.5-35B-A3B with `qwen3_5_mtp`, `--async-scheduling` and a 1M window set through
`--hf-overrides` yarn RoPE: "回答中英混杂". The screenshot OCR was decisive —
`"content": null`, all 500 completion tokens inside `reasoning`, `finish_reason: length`
— so the answer is routing plus budget first, model/template thinking language second
(`#6848`), a hand-written rope override third (unproven row), and the variant is absent
from the support matrix (no validated recipe to diff against). Two corrections the run
established: a missing `--mamba-cache-mode align` is auto-supplied by upstream
(`vllm/model_executor/models/config.py:349-377`), and `enable_thinking: false` carries a
version gate (`#7094`). **Delivery cost**: 24 assistant turns, most of them bare tool
calls, and 16.3 K completion tokens — the answer landed at the cap.

**#10546** vllm-ascend `v0.20.2rc1` (`367b8e6`) / vllm `v0.20.2` (`bc150f5`),
DeepSeek-V4-flash 1P1D, decode instance DP2×TP4 + EP, 10-way concurrency: the
decode worker dies with `rtMemcpyAsyncEx … reason=the model stream execute
failed`, an `aivec … timeout or trap error … subErrType:4` dump and an HCCL
`AllGather` task line (`rankSize 2`, `numBlocks 2`, `size 65536`), inside a
traceback that ends at `model_runner_v1.py:1393 counts_cpu[: counts.shape[0]]
.copy_(counts, non_blocking=True)` in `_copy_valid_sampled_token_count`
(`srclen=24` → six `int32` rows on that rank). The recorded run got the
load-bearing reading right by turn 3 — the copy is a sync point, the collective
is the fault — and then verified the DSV4 config surface
(`VLLM_ASCEND_APPLY_DSV4_PATCH` default `0` gating `patch_speculative_config`,
the `deepseek_v4`→`deepseek_mtp`/`DeepSeekV4MTPModel` remap, the nightly A3 yaml
as the only in-tree recipe, `disable_padded_drafter_batch` default `False`)
before being cut with **no answer**: 14 assistant turns, 23 tool results, 16.4 K
completion tokens, ~41.6 K characters of reasoning against ~2.6 K of message
text (every turn's text was a one-line status note). **Delivery cost**: from
turn 4 on, each turn opened with "I need to answer now" and made one more call;
turn 12 spent itself writing `/workspace/scratch/findings.md` instead of message
text; turn 14 carried 3.6 K characters of reasoning and zero text. It was also heading for a wrong
mechanism: it read `rankSize 2` as "the shared-expert DP AllGather", while
`fc3_quant_x`/`mc2` are `DP×PCP×TP` groups (`parallel_state.py:85-110`) and
cannot be 2 ranks on DP2×TP4. Both lessons are in
`../spec-decode-mtp/drafter-count-copy-device-fault.md` and in the classification
row that now routes there.

**#8590** vllm-ascend `v0.14.0rc1` (`52d4acfa`) + vllm `v0.14.1` (`d7de043d`),
single 910B-class card, TP=1, Qwen2.5-1.5B: repeated 猞猁 from the first generated
token whenever `<|im_start|>`/`<|im_end|>` are in the prompt; reporter later confirmed
"0.14.0rc1 through 0.18.0rc1, same". **Shipped no answer — cut twice** (11 steps at a
16 K cap, then 3 more at 8 K after the reporter's ping), with ~21 K characters of
reasoning in one block and every message a one-line status note. The paste already
carried both discriminators it never used in message text: the printed `Input ids` are
identical on both input routes (so token ids cannot be the mechanism), and
`Using max model len 131072` is the checkpoint's own value
(`vllm/config/model.py:1545`, `_get_and_verify_max_len` at `:1898`), which stock
Qwen2.5-1.5B does not declare. Real findings, source-read and never delivered:
no id-keyed branch in `vllm_ascend/` at that ref and no `vllm_ascend/models/` at all;
`AscendVocabParallelEmbedding._forward_origin` skips the mask at `tp_size == 1`
(`ops/vocab_parallel_embedding.py:174-184`), so an out-of-range id reaches the
embedding call instead of being zeroed; and Qwen2.5-1.5B *generation* has no e2e coverage (only the pooling
checkpoint at `tests/e2e/singlecard/pooling/test_classification.py:10`). Lane:
`../accuracy-benchmarks/short-prompt-corruption.md` step 0.

**#17797** vllm-ascend `v0.23.0` (`5cb98ca`) + vllm `v0.23.0` (`0fc695f`), image
`quay.io/ascend/vllm-ascend:v0.23.0-310p`, Ascend 310P; `Qwen/Qwen3-Embedding-0.6B`
with `--runner pooling --dtype float16`, single request at a time, prompts
of ~4-10 K characters (~3-8 K tokens): wrong embedding vectors for long inputs, cosine vs three
reference services collapsing from `>0.999` to `0.2-0.6`, deterministic per serve
session and different across restarts, gone with `--no-enable-chunked-prefill`.
Real findings (source-read at the pin, never delivered): the chunked-prefill
accumulator is inherited upstream (`PoolingStates.hidden_states_cache`,
`tokwise/methods.py:35-84`, `gpu_model_runner.py:3330-3343`) so "Ascend never
populates `pooling_states`" is refuted *at this pin*; the Ascend-specific pooling
sites are `attention_v1.py:1535-1538` (`and not causal` -> the prefix-blind
`_forward_encoder_attention`), the fixed-size pooling mask branches
(`attention_mask.py:75-77`, `_310p/attention/attention_mask.py:139-145`) and the
310P dispatch `_310p/attention/attention_v1.py:329-342`; and the mask theory is a
dead end for the chunked step because `forward_chunked_prefill_310` builds its own
`get_splitfuse_mask`. **Shipped no answer**: 18 assistant turns, all 16 K output
tokens spent inside reasoning, one short status note on each of the first 17,
cut mid-sentence while it was still ranking the causes (its own notes had said
"I must deliver now" three turns earlier). Lane: pooling + chunked prefill (`../pooling-embedding/index.md`).

**#5525** vllm-ascend `v0.11.0` (`2f1aed9`) + vllm `v0.11.0` (`b8b302c`),
`quay.io/ascend/vllm-ascend:v0.11.0`, Qwen3-Embedding-8B served with `--task embed
--max-num-seqs 256 --max-num-batched-tokens 8192`; `vllm-bench`
(`openai-embeddings`, sharegpt) answers ~1800 requests, then the engine dies and
everything after gets `400 Bad Request`. Evidence is four bench screenshots only;
the `KeyError` key is unreadable, so the mechanism is source-derived, not
reporter-confirmed. Real finding (source-read, never delivered): at v0.11.0
vllm-ascend ships **its own copy** of upstream's `InputBatch`
(`vllm_ascend/worker/npu_input_batch.py:94`) with a pooling fast path upstream
lacks (`add_request` -> `num_reqs` at `:316-324`, `remove_request` skipping
`removed_append` at `:462-465`, trim-only `condense` at `:573-577`, forced by
`assert request.sampling_params` at `:303`), which leaves a `None` hole plus a
duplicated slot under concurrency and raises at
`vllm/v1/attention/backends/utils.py:798`; by v0.23.0 the copy is gone
(`class NPUInputBatch(InputBatch)`, `__init__` only). **Shipped no answer**: 20
assistant turns, 11 of them with zero message text (bare tool calls), the 16 K
output budget entirely in reasoning, cut while it was still verifying
`add_request`'s index assignment after having named the root cause two turns
earlier. Lane: pooling batch bookkeeping
(`../pooling-embedding/batch-index-desync.md`).

**#10700** vllm-ascend v0.21.0rc1 (`80610e4`) + vllm v0.21.0 (`ad7125a`), 2× Atlas
800I-A2, GLM-5.1, DP2 × TP8 + EP, `deepseek_mtp` 3, `FULL_DECODE_ONLY`,
`enable_npugraph_ex`, `--async-scheduling` + prefix caching: the title says
"未添加 enforce_eager true 运行一段时间后崩溃", and the device dump arrives in
`worker.py execute_dummy_batch` → `_dummy_run` → `_model_forward` →
`acl_graph.__call__ → stream.synchronize()` with `fftsplus aicore/aivector …
DDR address of the MTE instruction is out of range` / `507011`, plus an inner
`rtMemcpy … 107030 the current capture mode does not support this operation`
record stamped `10:01:23` — inside the same log's `init engine (…)` window — while
the traceback is stamped `10:28:19`. So "ran fine for a while" was the report
delay, not the trigger: the record belongs to this process, which is the twin of
the #9183 lesson (`../hccl-and-hardware/error-report-timestamps.md`). Route now:
`../spec-decode-mtp/draft-graph-capture-device-fault.md`. Later reporter evidence
(post-dating the recorded session): reproduced unchanged on **v0.22.1rc1**
(record `01:19:08` → report `03:00:17`), a sibling GLM-5.2-W8A8 DP2×TP8+EP16
capture named the operator — `aclnnAddRmsNormBias` in the `deepseek_mtp` **draft**
forward under `npugraph_ex` — and both workarounds are reporter-confirmed:
`"enforce_eager": true` **inside** `--speculative-config` (only the drafter goes
eager) or MTP off, with the spec-scoped one still "慢好几倍". The recorded answer
ranked the drafter-graph leg first on the spec-scoped A/B (correct) but presented
that rung as the cheap one and kept MC2/AIV and SFA rows above the plog request;
the page now carries the measured cost and the frame's real meaning (a DP dummy
pass runs the target **and** the drafter, so the traceback alone does not
discriminate).

**#12261** vllm v0.20.2 (`bc150f5`) + vllm-ascend v0.20.2rc1 (`367b8e62`), GLM-5.1
(`GLM_5_1_MEP-1.0.0`) w4a8/w8a8 on 910C 560T, `stream: true` only: the client's
assembled `tool_calls[0].function.arguments` is `{"hvd": Pura900"}` — balanced
braces, one quote short, missing the *opening* quote of the value. Reporter pasted
the raw SSE, both `/v1/completions` captures and the request body, which is what
made it answerable. Track: tool-call output integrity, the schema-type-gate branch
(`../tool-call-streaming/arg-type-gate-unquoted-value.md`, new). Mechanism, source-
verified at that pin: `hvd` is declared `anyOf: [{"type":"string"},{"type":"null"}]`,
so `_is_string_type` (`vllm/tool_parsers/glm4_moe_tool_parser.py:127-148`, only a
top-level `type: "string"` counts) returns `False`, the partial value is streamed
raw (`:408`), and when the pair closes the quoted form arrives one character
*earlier* than the streamed length, so `_compute_args_diff` (`:424`) drops the
quote for ever; `stream: false` is clean because `extract_tool_calls` (`:223`)
always `json.dumps`es the whole dict. Ownership: upstream vLLM — both Ascend GLM
patches are excluded by their own conditions (`patch_glm47_tool_call_parser.py:39`
needs a region with no `<arg_key>`; `patch_glm_tool_call_streaming.py` needs
`finish_reason == "tool_calls"` *and* non-empty `arguments` in the same chunk).
Delivery, first run: **no answer shipped** — 15 assistant turns / 23 tool calls,
~35 K characters of reasoning against 3.5 K of message text, cut on a tool call. It
had the mechanism at turn 11 and then spent four turns trying to `import vllm`
(`ModuleNotFoundError: torch`) and `import regex` and porting the two functions into
a simulation that never ran; the fragment arithmetic needed for the proof is one
subtraction (the streamed length 14 against the completed string's 18, so the diff
is the last three characters `0"}`) and can be done on paper. A later run answered
the same issue in ~9 turns from the same artefacts; its only real friction was
`scripts/sse-toolcall-audit.sh` returning the *missing-tail* verdict for
`{"hvd": Pura900"}`, which the run had to overrule against
`arg-type-gate-unquoted-value.md` — the script now classifies balanced-braces +
odd-quote-count separately (`quote-gap`) and routes it to
`tool-schema-type-audit.py`.

**#5549** vllm-ascend v0.10.0rc1 (`4604882`) + vllm v0.10.0 (`6d8d0a2`), image
`v0.10.0rc1-310p`, Ascend 310P3: `vllm serve /model/Qwen3-Embedding-0.6B --dtype
float16 --enforce-eager --gpu-memory-utilization 0.08` dies at `Starting to load
model`, shards `0/1`, with `ValueError: There is no module or parameter named
'embed_tokens' in CustomQwen3ForCausalLM` from
`vllm/model_executor/models/utils.py:277`, init line `pooler_config=None`. Cause:
an embedding checkpoint served as a **generation** model (no `--task embed`; auto
maps a `ForCausalLM` architecture to generate, `config.py:935-985`) whose tensors
are root-level (`embed_tokens.weight`, not `model.embed_tokens.weight`), so the
decoder class's prefix-strict `AutoWeightsLoader` has no owner for the first
group; the pooling wrapper strips `model.` and delegates to the inner loader
(`adapters.py:79-86`), which is why one flag fixes it. The downstream class is a
verbatim copy of upstream's loader, so this is not drift. Route now:
`../pooling-embedding/embedding-checkpoint-as-generate.md` + one call
(`scripts/checkpoint-key-prefix.py <model dir>`). Delivery: the recorded run was
cut at the 16 K output cap after 24 assistant turns — skill, SOURCES, then model
loader, pooling adapter and two `history/` greps — having emitted only status
notes ("Let me check whether vllm-ascend at this pin supports pooling/embedding
models at all") and **no answer**: it was one turn short of the question that
settles the report (`--task embed`, `pooler_config`), which is exactly what the
new page puts in the first paragraph.

**#6598** vllm-ascend v0.14.0rc1 (`52d4acf`) + vllm v0.14.1 (`d7de043`), A2 two
nodes x 8 cards, GLM-4.7 BF16, `--data-parallel-size 2 --data-parallel-size-local 1
--tensor-parallel-size 8 --enable-expert-parallel --quantization ascend
--speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}'
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' --async-scheduling`:
the service starts, inference dies with `ERR00100 PTA call acl api failed` +
`EZ9999 … When layout is TND, queryT(16) must be equal to the last element of
actualSequenceLengthQ(4)` (`CheckFASeqlenDataInTND`), pasted only as a screenshot.
Track: FIA's own TND invariant, i.e. host-side padding bookkeeping. `queryT` is the
graph bucket (local `pad_for_cudagraph`, `model_runner_v1.py:588-591`, then the
DP-wide max at `:606`), `last(actualSequenceLengthQ)` is the real per-request
cumsum, padded only on the FULL + `uniform_decode` gate (`:856-881`); the MTP
drafter pads its own list only when `graph_pad_size > num_reqs`
(`attention/mla_v1.py:565-605`, `graph_pad_size` set only at
`spec_decode/mtp_proposer.py:284`). Route now:
`../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`, one call
(`scripts/fia-tnd-seqlen-check.sh --query-t 16 --seqlen-last 4 --max-num-seqs 16
--num-spec 3 --dp 2`), rung 1 `PIECEWISE` / `--enforce-eager`, then the tree's own
GLM-4.x tutorial diff (`docs/source/tutorials/GLM4.x.md:89-104`, and `:112` for the A2 multi-node note: `num_speculative_tokens: 1`,
pinned `cudagraph_capture_sizes`, single node, EP off, "not recommended" multi-node
on A2). Delivery: the recorded run found the pair, both padding candidates and the
tutorial diff by its 8th turn, then spent the rest of the 16 K cap re-deriving which
branch it was and shipped **no answer** — the "write both readings plus the one check" rule
now in `delivery-discipline.md`.

**#13446** vllm-ascend v0.23.0rc1 (`f4a08bd`, `vllm-ascend:v0.23.0rc1` image),
vllm v0.23.0 (`0fc695f`), 910B2 / firmware 26.0.0RC1, Qwen3-Reranker served with
`--runner pooling` and `--hf_overrides '{"architectures":
["Qwen3VLForSequenceClassification"]}'`: `TypeError: Invalid type of HuggingFace
config. Expected type: …Qwen3VLConfig, but found type: …Qwen3Config` raised in the
APIServer, `ERR99999` after it. Track: architecture-name routing, pooling lane. The
recorded run was the shape this library wants: the SKILL row landed on
`../pooling-embedding/seq-cls-arch-routing.md` on its second call, one
`scripts/seq-cls-arch-check.sh` call resolved both names to text/multimodal, two
greps confirmed the pinned tutorial pages, and it answered in Chinese with the
registry rewrite table, exact line numbers matching the reporter's own traceback
(866/883) and a version gap. Two additions came out of it: the `found type:` class
is the **fingerprint of the checkpoint that actually loaded** (the command named a
VL path while the text class was found, i.e. path/weights mismatch, not a card or
driver problem), and the quoted page's *name* dates the document — the pinned
v0.23.0rc1 tree still has `Qwen3_reranker.md` (correct), the recorded #14817 defect
is the hyphen-named `Qwen3-Reranker.md` at v0.23.0, and the reporter's
`v0.24.0rc …/Qwen3-Reranker.html#42-…` is newer than both their image and the
snapshot. One detour: `scripts/history-titles.sh -i 'reranker'` exited with
`no such history root: reranker`; the script now takes `-i` and rejects other flags
with its usage.

**#15560** vllm-ascend v0.18.0 (`e18643f`) + vllm v0.18.0 (`bcf2be9`),
Qwen3.6-27B-w8a8, TP=2, 2×910B3, offline `LLM(...)` with
`enable_sleep_mode=True` and `compilation_config={"cudagraph_mode":
"FULL_DECODE_ONLY"}`: the first `llm.generate()` returned garbage and the second
was normal. Track: first-call corruption of a text-only launch of a VL-capable
checkpoint. Delivery: the recorded run got the lane right and produced the
token-signature reading (a constant token id repeated to `finish_reason: "length"`
= degenerate logits, which retires the numerics lanes) but its ranked lead for the
`AttributeError: 'NoneType' object has no attribute 'size'` was "the ACL-graph
replay calls the compiled hybrid forward with a None tensor", and the answer was
cut mid-table at the output cap. The reporter's later comments supplied both
corrections: the crash is the unfixed `is_multimodal_model` use in the v0.18.0
branch (#14638, fixed on main by PR #8556, backport #14618) rather than the graph
— and the artifact itself is a start-up sync problem ("start a `LLM` object and
give some time, then no error will occur"). Version scope from the reporter:
still reproducing on v0.23.0 and v0.26.0rc1 (`token_ids=[0]*100`, second call
normal), clean on nightly main at vllm 0.27.1. Page:
`../accuracy-benchmarks/first-generate-degenerate.md`.
