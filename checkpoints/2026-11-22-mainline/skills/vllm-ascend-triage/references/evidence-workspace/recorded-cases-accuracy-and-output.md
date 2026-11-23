# Accuracy and output cases

score/accuracy disagreement, 乱码 and repetition (short and long prompt, hybrid
block size, PD hybrid state), reasoning routing and parser names, early EOS,
spec-decode/MTP acceptance and bookkeeping, tool-call argument integrity, pooling
and embedding outputs. Routes: `../accuracy-benchmarks/index.md`,
`../spec-decode-mtp/index.md`, `../tool-call-streaming/index.md`,
`../pooling-embedding/index.md`, `../serving-frontend-faults/index.md`.

Inherits the scope and rules of `recorded-cases/index.md`, whose page for this
family carries one line
per case in this file.

**#9430** vllm-ascend v0.19.1rc1 (`da421afa`) pinned, reporter on a **model-specific
image** `quay.io/ascend/vllm-ascend:deepseekv4` labelled "框架版本：0.18.0": DeepSeek-V4-Pro
w4a8, 4-node 2P2D PD, `MooncakeHybridConnector` with `prefill {dp2,tp16}` /
`decode {dp8,tp4}`, `deepseek_mtp` 1 on both sides, hybrid KV manager on, prefix caching
off, `recompute_scheduler_enable` on the decode side only, `VLLM_ASCEND_ENABLE_FLASHCOMM1`
on the prefill side only, `USE_MULTI_GROUPS_KV_CACHE=1` / `USE_MULTI_BLOCK_POOL=1` on
both. Symptom: *"响应中经常出现 hilabihan 特殊字段"* — fluent, correct-looking Chinese with
an occasional inserted meaningless word; the pasted body is `prompt_tokens: 6`,
`finish_reason: "stop"` and contains the title's token **nowhere** (both the title and the
body were re-scanned for it, twice, before the shape was accepted). The recorded run
found the sharpest fact — the pinned tree carries none of the stack
(`MooncakeHybridConnector`, `deepseek_v4`, `USE_MULTI_GROUPS_KV_CACHE` all empty; tutorials
stop at DeepSeek-V3.2), i.e. the reporter's branch is unverifiable here — but used it only
as a caveat: it shipped six reporter-side ablations (eager, EP off, MTP off, recompute off,
P/D TP parity, weight md5) and **never offered the image change**. The reporter's later
comment on the issue (*"v0.20的版本问题解决了"*) closes it as a version boundary, not any of
the six. Lesson: absence of the whole feature stack from the pin is a rung-1 trigger, not a
closing caveat (`version-contract.md` §*old-tag rung*, feature-image paragraph); and a
title token that the body does not contain is the symptom's name, not a string to hunt
(`../accuracy-benchmarks/short-prompt-corruption.md` §*Grade the symptom first*).

**#5725** vllm 0.13.0 + vllm-ascend v0.13.0rc1 (`72506c9834` / `1b5d5abf865f`) pinned,
reporter on the same pair, 910B4 x1, Qwen3-Embedding-0.6B `--task embed` with
`--max-model-len 32000 --max-num-seqs 20 --tensor-parallel-size 1
--gpu-memory-utilization 0.5`: one ~1.2 K-character Chinese string embedded repeatedly
returns `cos 0.9997` against the reference on call 1, `0.5855`/`0.5815` on call 2, and
calls 3-5 bit-identical to call 2 — filed as a numeric 误差. The recorded run classified
it as a cache/reuse signature on its third turn and delivered the ranked answer at its
21st, with `--no-enable-prefix-caching` as rung 1 and the pooling attention branch
(`actual_seq_kvlen = actual_seq_lengths_q`) as the source-verified mechanism; the reporter
later confirmed that **disabling prefix caching removed the issue** (2026-03-23), after
reproducing it on v0.14.0rc1 and v0.15.0rc1. What it cost: ~18 turns of source walking
after the discrimination was already in hand (a repeat that freezes is a cache signature,
so the ablation was the answer on turn 3), one wrong rung
(`usage.prompt_tokens_details.cached_tokens`, which the embeddings response never fills —
`../pooling-embedding/prefix-cache-hit-embedding-drift.md` §*A check that looks cheap*),
and no ask for the serve command's caching flags, which would have made the first message
decisive. The page now carries the reporter-confirmed workaround, the nonce-suffix
confirmation and both pins' line numbers.

**#15939** vllm-ascend v0.26.0rc1 (`f2f74a16`) pinned, reporter on the same rc,
DeepSeek-V4-Flash-073-w4a8 on A2 1P1D (P: TP4/DP1, spec commented out; D: TP1/DP4,
`--speculative-config {"method": "dspark", "num_speculative_tokens": 7}` plus
`rejection_sampler_config {enable_block_verify, enable_entropy_verify,
posterior_threshold 0.8}`), one node running P + D + `load_balance_proxy_server_example`:
"精度失真" and "TTOT 速度过快", prefix caching on/off changing the frequency, all five
uploaded logs failed so only the launch scripts were available. The recorded run
delivered a ranked answer from the code and docs: sampler config first (documented
"reduced sampling precision", `additional_config.md:163-171`, #7807 as the block-verify
precedent) and the P/D `num_speculative_tokens` parity second
(`DeepSeek-V4-Flash.md:1076`). The reporter's later "verification" request/response pair
(made public afterwards) is `max_tokens: 128`, `content: null`, `finish_reason:
"length"`, `completion_tokens: 128`, and its `reasoning` works an unrelated
∀-formalisation prompt — so the sample shows a budget truncation and a
request↔response pairing question, and neither was in the deliverable. What it cost:
the ranking was built without first asking for one raw body with the `id` that matches
the request — `../accuracy-benchmarks/short-prompt-corruption.md` ("Grade the symptom
first").

**#14884** vllm-ascend `v0.18.0rc1` image, Qwen3-VL-30B-A3B, TP4 +
`--enable-expert-parallel`, `--enforce-eager`, `--no-enable-prefix-caching`,
`temperature: 0`, `max-num-seqs 32` / `max-num-batched-tokens 16384`, and the
determinism env block exported: "enabling `VLLM_BATCH_INVARIANT=1` results in
inconsistent output for the same request in multi-concurrency scenarios". Track: the
engine compared with itself across **batch composition** — the sibling of #7180, not a
repeat-run report. The recorded run got the verdict right (the feature's per-op
envelope, not reporter misconfiguration: attention is only replaced in the AscendC
branch, EP/MoE and FIA `_v2` are never replaced, and the tree's own tested models and
e2e files exclude VL and EP) and it corrected one instruction in this library — the
start-up INFO line cannot prove attention was patched, because it is keyed on
`HAS_TRITON or HAS_ASCENDC_BATCH_INVARIANT`. Delivery held (answer at assistant turn
8, verdict-first), but ~35 K characters of reasoning re-derived the whole coverage map
from source because no page carried it, and the first turn's two calls
(`cat -n vllm_ascend/batch_invariant.py`, the 645-line e2e test) are now summarised.
Page: `../accuracy-benchmarks/batch-invariant-coverage.md`.

**#11936** vllm-ascend `v0.21.0rc1-310p` image (`80610e4`) + vllm v0.21.0 (`ad7125a4`), one
310P3 box, CANN 9.1.0-beta.1: GLM-OCR and MinerU2.5-2509-1.2B both "可以正常拉起，但是精度有问题 /
随便跑个图片识别就能复现", while PaddleOCR-VL was "好像没问题" (MinerU2.5 appears in **both** lists of the
same paragraph). Track: unvalidated model/SoC pair, returning as an accuracy report. The paste is a full
environment dump plus the PaddleOCR-VL serve command only — no response body, no log. Later evidence on
the same issue is a commenter's reply plus the reporter's rebuttal of it: the commenter said neither model
is officially supported yet and GLM-OCR needs PP-DocLayout-V3, and the **reporter** answered that both *do*
launch and infer on the official image with no extra operations, 只是输入简单的图片识别都是输出乱码. So the
launch half is closed by the reporter and the artefact is open: the ranked answer stays pipeline stage >
unvalidated pair > engine numerics, recorded as hypothesis, with the support half a two-grep doc lookup that
is context rather than a verdict: at this ref
`docs/source/tutorials/models/` has the PaddleOCR-VL page with an Atlas-300 tab (`:51, :60, :93-115`)
and **no** GLM-OCR or MinerU page, while `hardwares/310p.md:96-172` validates text-only Qwen3-W8A8SC.
The recorded run spent 28 assistant turns and its whole 16 K output budget inside the 310P rope and
attention internals, emitting **zero** message text, after guessing the split was "the models on the
M-RoPE / partial-rotary branches" — partly refuted, since `uses_mrope` comes from the checkpoint config
(`mrope_section`), not the family (`paddleocr_vl.py:1097` defines `get_mrope_input_positions` anyway).
The page now carrying it, with the branch map and the verified dead ends (`scale_value`, the head-dim pad
path, `ApplyRotaryEmb`), is `../soc-platform-paths/310p-vl-ocr-garbage-output.md`. **The next replay of
the same paste delivered**: it took that page's pins at face value (they equal `SOURCES.json`), re-ran the
two doc lookups, and shipped the verdict, three ranked legs with one probe each, the dead-end list and the
ask list in one answer on assistant turn 5 (~5.5 K output tokens) — the procedure holds when the page is
read before `_310p/`, and its `MinerU in both lists` catch is what the ask list led with.

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
reported MTP as the owner (MTP off → normal) — a later replay built from the owning
page led with the MTP-off row, put capacity third and shipped it in six assistant turns,
so the ordering rule is the part that was missing, not the mechanism; #10943 v0.20.2rc1 custom
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
`async_scheduling` never appears in `tests/e2e/multicard/4-cards/test_qwen3_next.py`.
The audit of its 97-chunk paste returns valid JSON (268 chars, braces 1/1, 40 quotes)
**and the sample is failing anyway**: the last value is `guest_phone: "138123"`
against the request's `13812345678`, a value cut inside complete punctuation. The
recorded answer said "the posted sample is not the failing request", and a repeat
session (2026-08) repeated "11 位数字全部到齐" to the reporter — a balanced audit is
not a healthy sample (`../tool-call-streaming/streamed-arguments-audit.md` §Step 3A).

**#12658** vllm-ascend v0.21.0rc1 (`80610e44`) + vllm v0.21.0 (`ad7125a4`), Atlas
800I A2, DeepSeek-V4-Flash-w8a8-mtp TP8+EP with `--tool-call-parser deepseek_v4
--reasoning-parser deepseek_v4`, MTP n=1, `FULL_DECODE_ONLY`,
`--async-scheduling`: "Non-think, tool call empty". The report shipped the four
dumps (`bad_case.json`, `rendered.json`, `rendered_prompt.json`,
`raw_completion.json`) and they decide it in one call: `chat_template_kwargs
{"enable_thinking": false}`, `stream: false`, one tool `get_weather`;
`rendered_prompt.json` carries `## Tools` + `<｜DSML｜tool_calls>` + `get_weather`
and ends `<｜Assistant｜></think>`; `rendered.json` holds 5944 prompt ids at
`temperature 1.0`; `raw_completion.json` is 132 tokens of Chinese prose with
`finish_reason: "stop"` and **no `｜DSML｜` anywhere**, `prompt_tokens: 5944`.
Prompt rendering is fine and the sampled sequence contains no call, so no parser
can create one — the model-choice lane, and the deliverable is a rate plus the
mode/vendor-API ablations (`../tool-call-streaming/no-tool-call-in-raw-output.md`).
A later public comment from the reporter is the confirmation: 1000 requests in
non-thinking mode missed the call with "a certain probability", thinking mode
always called, and the official `dsv4-flash` API reproduced the non-thinking miss,
so w8a8 quantisation and the engine are excluded. The recorded run never opened
those files — it grepped the attachments **manifest** for the model name and the
GitHub file-id digits, read the empty result as "no attachments", and spent all
27 assistant turns and the 16 K-token budget inside `vllm/parser/abstract_parser.py`,
`patch_deepseek_v4_thinking.py` and `tokenizers/deepseek_v4_encoding.py`, shipping
no answer at all.

**#12398** vllm-ascend v0.18.0 (`e18643f8`) + vllm v0.18.0 (`bcf2be96`), GLM-5.1
W4A8, 1P1D on 800T A3: GLM's own in-band protocol text (`<tool_call>`/`<arg_key>`/
`<arg_value>`, a leading `</think>`) leaking into streamed `content`. Delivered in
full at its 16th turn: the strings are the GLM-4.x XML grammar
(`vllm/tool_parsers/glm47_moe_tool_parser.py`) and nothing consumes them because
the serve line lacks `--tool-call-parser glm47 --reasoning-parser glm45`; the
tree's own `docs/source/tutorials/models/GLM5.md` single-node w4a8 blocks
(:152-247 at that ref) omit both flags while its multi-node PD examples carry
them, so the tutorial diff is the first artefact for this class. Also verified
there: `[tool_call:Name]{…}` appears in no source file of either tree, i.e. that
notation is the reporter's client rendering, not engine output.

**#4730** image `quay.io/ascend/vllm-ascend:v0.11.0rc1` (pin v0.11.0rc3 `b6d63bbd` ↔
vllm v0.11.0 `b8b302cde434`), Qwen2.5-VL-3B, one card (`ASCEND_RT_VISIBLE_DEVICES=7`,
TP=1), `--max_model_len 8192 --mm_processor_kwargs '{"max_pixels": 2038400}'
--limit_mm_per_prompt '{"image": 3}'`, the screenshot upload failed: the same serve
structured the answer correctly on `v0.11.0rc0` and "返回的html结构都错乱" on rc1. The
recorded run classified it right and shipped a ranked ladder (truncation at ~7 800 image
tokens vs an 8192 context; `USE_OPTIMIZED_MODEL=0` for the head-dim-padded tower;
`--enforce-eager`; text-only; rc0 re-run) plus the honest limits (single-commit export, no
rc0→rc1 diff, no attachments) — but framed the answer as "a regression inside the rc0→rc1
delta" and never asked for the stack dump **inside each image**, which is the only way to
attribute a whole-tag delta. The reporter's later public comment (2025-12-05) says **rc2
and rc3 also fail**, same garbled structure, now suspected as 精度 — so the delta was not
a one-rc regression that a later tag repairs, and the tag rung was inverted for this
report. What it cost: the leading frame could not survive the reporter's next message —
`../accuracy-benchmarks/vl-vision-tower-degeneracy.md` §*Fifth pin*,
`../accuracy-benchmarks/harness-vs-engine-attribution.md` §*When the older image is the
good one*.

**#4057** v0.11.0rc0 Qwen3-Next-80B-A3B, "问到数字相关问题时精度较差" on a 19-token
prompt with no serve command in the report: the run found the downstream fp32
`torch_chunk_gated_delta_rule` prefill replacement and the tutorial's experimental
Triton-Ascend note, then spent its output budget re-deriving the FLA chunk algebra
and shipped nothing — its 14th assistant turn was cut mid-plan, with ~49 K
characters of reasoning and zero message text. The reporter's later public comment
supplied the deploy command (TP=4, `--max-model-len 8192`, `--enforce-eager`, no
MTP/EP/quantisation), which is the narrowing the answer owed
(`../accuracy-benchmarks/qwen3-next-gdn-numeric-path.md`).
A later replay at the same pins answered in **7 assistant turns / 5.3 K completion
tokens** (library grep → the entry → the page + the screenshot's `.ocr.txt` → one pin
confirmation → the message), so the page was sufficient; what it still got wrong was
delivery. It quoted the reporter's TP=4/`--enforce-eager` line as a mitigation and then
asked for "your actual `vllm serve` command line" in the same message, and its opening
sentence asserted the GDN path and "no config knob fixes this version" while its own
table labelled cause 1 *mechanism-verified, not established as a defect*. Both edits are
in the page's *Reporter-confirmed narrowing* section.

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

**#14085** vllm-ascend v0.23.0rc1 (`f4a08bd`) + vllm v0.23.0 (`0fc695f`): GLM-5.2-w4a8c8
across 8×910B (A2) PD disaggregation — `MooncakeConnectorV1`, both C8 sparse flags `true`,
`--decode-context-parallel-size 8` + `--cp-kv-cache-interleave-size 128`, P `--enforce-eager`
and D `FULL_DECODE_ONLY` with MTP 3, `--max-model-len 200000`; `opencode` against the proxy
reports "偶现乱码，或重复说话" with no log at all. The run shipped a complete ranked answer
inside ~10 assistant turns and its cap landed only in the closing scope paragraph: the C8
lane first (the A2 8-node block of the same tutorial sets both flags `false`, `:1059-1060`,
`:1142-1143`, while the reporter carried the 1M block's lattice verbatim, `:1365`/`:1436`),
then the C8 write path, then MTP + graph, with the capture-size `298` lane closed by the
upstream round-up. Two lessons are now on the pages: the reporter's later comment gave the
evidence the report lacked — **one fixed prompt, 29 repeats in 193 runs (~15 %)** — so a
two- or three-run clean check certifies nothing and the count belongs in the first answer's
ask (`../accuracy-benchmarks/long-output-repetition.md`); and a per-block recipe lattice
match beats a flag-by-flag delta (`../pd-disaggregation/recipe-contract-diff.md`).

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
A later replay (same pins, one library grep for the exception text on its 2nd
assistant turn) shipped the verdict and ladder in **6 assistant turns / 6.4 K completion
tokens**. Its one paid call was the page's ast trio, which corrected the v0.16.0 parser
list from three to four (`llama_tool_parser.py:47`); its `rg` of `vllm/renderers/base.py`
exited 2, because that file exists only at v0.17.0 — at v0.16.0 `get_async_tokenizer()`
is in `vllm/renderers/protocol.py:60`.

**#7689** vllm-ascend v0.13.0rc1 (`1b5d5abf`) / vllm v0.13.0 (`72506c98`), dots.ocr
on 2×910B4 (32 G) vs 910B3 (64 G) with an identical serve command: the B4 host answers
garbled or off-task text where B3 is correct. VL corruption track; the run delivered a
complete ranked answer (first ablation `USE_OPTIMIZED_MODEL=0`, then TP=1,
`--async-scheduling` off) and paid for one useful version correction: at this pin
`vllm_ascend/models/` no longer exists, the vision tower is patched by class name
(`utils.py:697` → `ops/mm_encoder_attention.py:31`) and the head-dim pad moved from
weights to **activations** (`:101-108`), so #976's weight-padding mechanism cannot be
quoted at v0.13.0rc1 — see `../accuracy-benchmarks/vl-vision-tower-degeneracy.md`.

A later recorded session on the same report **shipped no answer**: ten verification
rounds and the whole output budget spent re-confirming what the page and
`../accuracy-benchmarks/hybrid-block-size-corruption.md` already state (no 910B3/B4
branch anywhere in `vllm_ascend/` except the `core/scheduler_dynamic_batch.py:151`
comment; `--block-size 16` is a no-op while prefix caching or chunked prefill forces
128 at `utils.py:1068-1076` — pin-verified here; `mm_encoder_tp_mode` defaults to
`weights` at vllm `config/multimodal.py:111`; `rg -i dots docs tests .github` is
empty), cut on an empty final turn. Deliver the three levers and the saved request
JSON's `prompt_tokens` comparison in the first message, then spend the rest on the
one thing the paste cannot answer.

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
a row). The page, not more source, is what closed it. A third replay answered it in
the same shape in six assistant turns and 7.4 K characters of message text, and is the
one that found the page's temperature claim under-determined: the stack's only
branch is greedy vs stochastic, so the title's `(0,1)` window is not a source
boundary. Both pages now carry the two control facts that settle it — `vllm bench
serve` fills `temperature: 0.0` when the flag is absent
(`vllm/benchmarks/serve.py:1412-1413` at vllm v0.13.0) while a raw client's omitted
field arrives as the generation config's value or `1.0`
(`vllm/entrypoints/openai/protocol.py:390,406-408`) — which makes
`--temperature 1.0` the run that tests the reading.

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

**#10892** vllm-ascend `v0.21.0rc1` (`80610e4`) / vllm `v0.21.0` (`ad7125a`), 300I Duo
(310P) ×4, Qwen3.6-35B-A3B-w8a8, `presence_penalty: 1.5`: "过度思考" — the serve had no
`--reasoning-parser`, so 2548 completion tokens of English CoT, one `</think>` and a JSON
tail all arrived in `message.content` while `message.reasoning` was `null`. **Answered in
8 assistant turns / 9.8 K completion tokens** with a ranked list that was correct on the
routing mechanics (parser gate, `enable_thinking` default, `thinking_token_budget` a
no-op on NPU at `npu_input_batch.py:56-58`, the support-matrix gap and the 310P preview
note) and **wrong on the lane**: it opened with "这不是 300I Duo（310P）的硬件/量化精度问题"
and ranked `presence_penalty` only fourth. The reporter's later comments put the
complaint where it belonged — "content 中英混杂，带着思考的内容一直思考，不能给正确的结果"
— and the maintainer's resolution was a **precision defect in the MoE gating top-k
softmax op**, fixed downstream (PR to reference the issue); the reporter also confirmed
"输出很快". Calibration: the run used "text is coherent, single `</think>`,
`finish_reason: stop`" to retire the corruption *and* numerics lanes. That evidence only
retires truncation/repetition/乱码 — plausibility is not correctness. The free check it
skipped is one line of the paste: the JSON tail invented a vacuous "pass" and a page
number that the prompt's own rules do not support, so the answer was wrong, and no
parser flag can move logits (`../accuracy-benchmarks/moe-gating-topk-precision.md`).

**#15161** (same batch, sibling forward) vllm-ascend `0.23.0` on 310P, Qwen3.6-35B-A3B
w8a8, aisbench with `presence_penalty=1.5`, "一到测试阶段就会崩溃", **no log pasted**:
**shipped no answer** — 32 assistant turns, every message a bare tool call, zero message
text, cut at the 16 384-token cap. Its own reads had already refuted its leading
hypothesis (`AttributeError: '_OpNamespace' '_C' object has no attribute
'apply_repetition_penalties_'`, precedent #2533/#2544): at vllm v0.23.0 (`0fc695f`)
`_custom_ops.py:387-411` dispatches `apply_repetition_penalties` to a **torch** path when
`not logits.is_cuda`, and vllm-ascend v0.23.0 (`5cb98caa`) `Dockerfile.310p:63` uninstalls
`triton`/`triton-ascend` (`Dockerfile:70-71`, `Dockerfile.a3:72-73` reinstall
`triton-ascend==3.2.2`), so the 310P sampler inherits `apply_penalties` and falls back to
that torch path. Two lessons: a precedent
op-name is not the finding until the dispatch gate in the pinned version is read, and a
run that reaches that point with 900 characters of message text would have had something
to hand over.

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

**#6533** vllm-ascend `v0.14.0rc1` (`52d4acfa`) + vllm `v0.14.1` (`d7de043d`), one
**910B4** (reporter-confirmed 2026-02-05, answering triage's "B4 or B4-1?" — so the
32 GB card, not the 64 GB `910B4-1`), `Qwen3-VL-8B-Instruct`: a **text-only** chat request
(`prompt_tokens: 12`, no image, no `temperature`, `max_tokens: 100`) comes back degenerate
from about the fifth token (`你好，，，我是一个AI，我叫Qwen，…我，我，我`) to
`finish_reason: "length"`, with no serve command and no log in the report. The recorded run
classified it correctly at turn 3 and had an answer available by turn 10 (the pin's own
`tests/e2e/models/configs/Qwen3-VL-8B-Instruct.yaml`: `hardware: "Atlas A2 Series"`,
`mmmu_val 0.55`, wired at `nightly_test_a2.yaml:120`; the two one-request controls), then
spent five calls trying to reopen the page's #8590 precedent — `history-body.sh 8590
2>/dev/null` (the miss is printed to stderr), then `/repos/…/issues/8590` (404 in this dated
replay) — and its last four turns on `AscendMRotaryEmbedding.forward_oot` → upstream
`triton_mrope` argument parity. **Shipped no message text at all**: 45 turns, 1.26 M prompt
tokens, output-capped. Two harness corrections came from it: `history-body.sh` now prints a
missing id on stdout as well, and a shared-kernel hypothesis is checked against the pin's
wired accuracy row first — `../accuracy-benchmarks/short-prompt-corruption.md` Step 0b.

**#8590** vllm-ascend `v0.14.0rc1` (`52d4acfa`) + vllm `v0.14.1` (`d7de043d`),
single 910B-class card, TP=1, Qwen2.5-1.5B: repeated 猞猁 from the first generated
token whenever `<|im_start|>`/`<|im_end|>` are in the prompt; reporter later confirmed
"0.14.0rc1 through 0.18.0rc1, same". **Shipped no answer — cut twice** (11 steps at a
16 K cap, then 3 more at 8 K after the reporter's ping), with ~21 K characters of
reasoning in one block and every message a one-line status note. The paste already
carried both discriminators it never used in message text: the printed `Input ids` are
identical on both input routes (so token ids cannot be the mechanism), and
`Using max model len 131072` (`vllm/config/model.py:1545`,
`_get_and_verify_max_len` at `:1898`) is longer than the 32768 stock Qwen2.5-1.5B
declares. That second one is a lead, not a verdict: the line returns the user's
`--max-model-len` verbatim when the command carries one (`:175` default `None`,
returned at `:2026`; above the derived length it raises unless
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `:1999-2026`), and this body has **no serve
command at all**, so the flag is not excluded — "your checkpoint is not stock" is
only the first of two branches. Real findings, source-read and never delivered:
no id-keyed branch in `vllm_ascend/` at that ref and no `vllm_ascend/models/` at all;
`AscendVocabParallelEmbedding._forward_origin` skips the mask at `tp_size == 1`
(`ops/vocab_parallel_embedding.py:174-184`), so an out-of-range id reaches the
embedding call instead of being zeroed; and Qwen2.5-1.5B *generation* has no e2e coverage (only the pooling
checkpoint at `tests/e2e/singlecard/pooling/test_classification.py:10`). Lane:
`../accuracy-benchmarks/short-prompt-corruption.md` step 0. A **third** run on the
same pin and body shipped an answer at last — assistant turn 16, 15 tool-only
messages in front of it, 978 891 prompt tokens, 6.7 K characters of answer cut inside
its ask-back section, and it opened with the "not a stock checkpoint" branch above as
if it were settled; the page's step 0 now makes that branch conditional on the serve
command, which the run never asked for.

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
reporter-confirmed. The real finding (source-read, never delivered) is the pooling fast path on
vllm-ascend's own `InputBatch` copy
(`vllm_ascend/worker/npu_input_batch.py:94`), spelled out in
`../pooling-embedding/batch-index-desync.md`; gone by v0.13.0rc1 (thin
`NPUInputBatch` shell).
**Shipped no answer**: 20
assistant turns, 11 of them with zero message text (bare tool calls), the 16 K
output budget entirely in reasoning, cut while it was still verifying
`add_request`'s index assignment after having named the root cause two turns
earlier. A later replay at the same pins delivered the whole answer from that
page in seven assistant turns, so nobody owes the slot arithmetic again: it
grepped the four `.ocr.txt` for the symbol fragments (`Error|assert|Traceback|
raise|Exception`), read two windows around the hits, then confirmed the frames
at the pin in one call. Six of those seven turns still carried no message text —
the pattern that cost the recorded session — safe only because this replay
finished. Lane: pooling batch bookkeeping
(`../pooling-embedding/batch-index-desync.md`).

**#5687** vllm-ascend `v0.11.0rc3`, Qwen3-Embedding-8B, `--max-num-seqs 256` on a
`ray_distributed_executor`: `KeyError: None` at `model_runner_v1.py:1260
_prepare_inputs` -> `tokens = [scheduler_output.num_scheduled_tokens[i] for i in
req_ids]`, gone at `--max-num-seqs 8`; the ask is "要怎么找到报None值到底出现在哪里",
and the reporter's own probes were in `reorder_batch_to_split_decodes_and_prefills`
and around `update_states`/condense — both legitimately empty. The delivered
answer named the `InputBatch` slot bookkeeping (the #5525 mechanism, stated as a
precedent because 0.11.0rc3 is not in the snapshot), said the first consumer in
this version is `_prepare_inputs` rather than the reorder helper, and gave a
one-launch probe at the listcomp. **Reporter-confirmed afterwards**: `print(input_batch)`
showed `[None,None,xxx,297]` in the second batch becoming `[None,298,299,xx]` as
requests were added, "初步排查是因为condense侧的删除逻辑有问题" — i.e. the ranked
diagnosis held, and the decisive evidence came from dumping the batch state, not
from any consumer frame (`../pooling-embedding/batch-index-desync.md` §instrument
the producer). Lane: pooling batch bookkeeping.

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
discriminate). Its 2026-07-01 comment is also the reporter's own `问题分析报告`
(ACL-graph compile defect first, `VLLM_ASCEND_ENABLE_ACL_GRAPH=0` as the fix): grade
each lever before quoting it — `../dispatch/controls-and-ablation.md`, *A ranked
analysis the reporter pasted*.

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

**#14656** vllm-ascend v0.23.0 (`5cb98ca`) / v0.23.0rc1 images (`v0.23.0-a3`,
`v0.23.0rc1-a3`) + vllm v0.23.0 (`0fc695f`), GLM-5.2-w4a8c8 on 910C 560T, PD 4P/8D
with `MooncakeConnectorV1` + `AscendStoreConnector` (memcache), `--tool-call-parser
glm47`, `--reasoning-parser glm45`, MTP `deepseek_mtp` n=1 (P) / 3 (D), `--max-model-len
524400`, no `--async-scheduling` on either side: in `stream: true` the tool-call
`arguments` lose their leading `[`/`"`, a fragment repeats, the escapes are wrong —
"多出现于 ascii ⬅➡ 中文字符交界处", `stream: false` clean, both 1M and shorter
configs, w4a8c8 and w8a8. Track: tool-call output integrity, **streamed text
delta** (`../tool-call-streaming/detokenizer-text-delta.md`, new). What the run
verified at the pin: the #12261 gate is **fixed** here (`_is_string_type` uses
`extract_types_from_schema`, `glm4_moe_tool_parser.py:130-136`; complete string
values render via `json.dumps` `:348`, partial via `:396`), the length-prefix diff
survives (`_compute_args_diff` `:409-417`), the frontend `remaining_args`
reconciliation is gone from vllm v0.23.0 so `patch_glm_tool_call_streaming.py:140`
is inert, block/entropy verify default `False` (`ascend_config.py:797-798`), and
`vllm_ascend/` contains no detokenizer. It never grepped the server log for
`resetting decode stream` — the one line that decides the fast `DecodeStream`
path — and shipped **no answer** (see `delivery-discipline.md`). The reporter's
later statements supply the missing evidence: both serve commands (above; the
tutorial's `--async-scheduling` is absent, and `--speculative-config` carries
`"enforce_eager": true`), the client-side workaround ("手动添加 jsonl 格式校验与
补充，可以缓解"), and that a **later parser refactor** showed no recurrence. Treat the
client-side validate-and-repair as the actionable answer for these versions and
keep both parser and text-delta owners open.

**#7682** vllm-ascend v0.17.0rc1 (`e20f0b1a`) image `v0.17.0rc1-a3-glm5-v2`,
GLM-5-w4a8 on one A3 16-card node, `--tool-call-parser glm47 --reasoning-parser
glm45`, MTP `deepseek_mtp` n=3, `--async-scheduling`, `FULL_DECODE_ONLY`, chunked
prefill: the non-streaming `/v1/chat/completions` returns the complete
`{"todos": "- [x] 分析项目结构…"}`, the streaming one returns
`"arguments":"\"}"` — two characters — after 28 `delta: {}` chunks. Track:
tool-call output integrity, the **suffix-only / head-lost** variant of the
bookkeeping family (`../tool-call-streaming/suffix-only-arguments.md`, new):
`{` 0 / `}` 1, one quote, `json.loads` → `Unterminated string starting at: line 1
column 1 (char 0)`, and the assembly is a strict suffix of the non-streaming
arguments. The reporter restated the same A/B in a follow-up minutes after filing
(2026-03-26T08:29Z, `#issuecomment-4132650891`), so the paste is a genuine failing
request — as is #8122's, whose cut sits inside a value rather than at the head.
Coverage gap at this pin: the case view exports vllm **v0.13.0**
(`72506c9`), whose tool/reasoning registries hold `glm45` only — a deployment
running `--tool-call-parser glm47` cannot be serving that tree, so the parser code
under the export is not the code that ran. Delivery: the recorded run called the
shape on turn 3 ("not the model … only its last two characters"), then spent ~8
turns on the `glm47`-absent registry and the parser/sampler walk (`serving_chat`
`remaining_call`, `rejection_sampler.py:114`, `patch/` sweep) and ended
`status: output_truncated` at 16 384 output tokens with **no answer text ever
sent** — the last block was reasoning about which file to open next. Two fixes
landed: `scripts/sse-toolcall-audit.sh` had labelled this paste `quote-gap`
(braces "balanced" although 0 `{` vs 1 `}`) and pushed it to
`tool-schema-type-audit.py`; it now reports `head-lost` with the suffix
arithmetic, and the doc route is rule 1's new outcome plus the new page.

**#5065** vllm-ascend v0.12.0rc1 (`42ceaf08`) + vllm v0.12.0 (`4fd9d6a8`), A2
910B2C ×8 (`910B2C`, TP=8) with an `A+X` box, Qwen3-Next-80B-A3B-Instruct W8A8 via
`--quantization ascend`, `VLLM_ASCEND_ENABLE_NZ=1`, `FULL_DECODE_ONLY`, chunked
prefill on: `who are you?` comes back as an answer to a different question and a
Chinese prompt as an English CAI essay, `finish_reason: stop`, `prompt_tokens`
12/24. Track: accuracy/corruption on a hybrid GDN model
(`../accuracy-benchmarks/qwen3-next-gdn-numeric-path.md`, new
`v0.12.0rc1` section). The run's citable finds at this pin: the release's own
Known Issues carry "DeepSeek 3/3.1 and Qwen3 doesn't work with FULL_DECODE_ONLY
graph mode" (`release_notes.md:42`) and the `VLLM_ASCEND_ENABLE_NZ` default note
(`:46`), the Upgrade Note requires CANN 8.3.RC2 while the reporter has 8.2.RC1
(`:39`), the W8A8 e2e case is annotated `# TODO: will conduct accuracy
verification after the subsequent version becomes stable`
(`tests/e2e/multicard/test_qwen3_next.py:109`) and the batch-chunked-prefill case
`# TODO: Fix the accuary of batch chunked prefill` (`:64`), and the GDN path now
binds to `vllm_ascend/ops/triton/fla/*` (`patch/worker/patch_triton.py:2-16`)
instead of the v0.11 fp32 torch reference. Delivered a ranked answer on turn 15
(NZ, graph mode, unverified W8A8, CANN/wheel pair) and correctly refused to treat
the paste's own corrupted JSON keys as detokenizer evidence.

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

**#2071** vllm-ascend v0.9.2rc1 (`b5b7e0e`) + vllm v0.9.2 (`a5dd03c`), Qwen3-32B
(dense, team `--quantization ascend`, `--compilation_config` capture sizes), served
fine through ~40 s of `vllm bench serve --dataset-name random` and then every rank
died in the same second: `AssertionError` at `attention_v1.py:413`
`assert attn_metadata.attn_mask is not None`, reached through
`unified_ascend_attention_with_output` from `<eval_with_key>.2`. Cause: ngram spec
decode has no branch in the dense backend — `SpecDecoding` is declared at
`attention_v1.py:117` and consumed by nobody there, the runner hands it to every
non-`use_eagle` method (`model_runner_v1.py:977-981`), `_make_attention_mask`
returns `None` for it (:712-714), and only `mla_v1` reads `runner.spec_attn_mask`.
Delivered: verdict first with the "remove `--speculative-config`" rung, the
`--enforce-eager`-is-not-the-fix statement, the first-draft-hit timing, and the
tree's own `pytest.skip("Not current support for the test.")` at
`test_v1_spec_decode.py:72` as the project's statement of non-support. Cost 22 tool
calls over 13 turns / 871 K prompt tokens: the decisive facts arrived by re-reading
`model_runner_v1.py` in three overlapping windows, where one
`rg -n 'SpecDecoding|spec_attn_mask' vllm_ascend/attention/` returns the whole
consumer table. Page `../spec-decode-mtp/dense-attn-state-gap.md`. Open: whether
`deepseek_mtp` on a dense model fails identically (same branch, no run), and whether
a later ref implements the state (no later ref in this snapshot).

**#1791** vllm-ascend `v0.9.2rc1` (`b5b7e0e`) + vllm `v0.9.2`, Qwen3-30B-A3B,
`lm_eval --tasks gsm8k --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn`, TP2 +
DP2 + `--enable-expert-parallel` + `additional_config.expert_tensor_parallel_size: 2`:
`exact_match` 0.1653 flexible / 0.1547 strict, i.e. the wrong-answer floor. Reporter's own
control (later comment, 2025-07-14): the same body at **TP4** scores 0.8840 / 0.8704. The
run established the decisive arithmetic — `platform.py:135-149` forces
`expert_tensor_parallel_size = 1` under EP, so both runs are EP=4 with the same
32-experts-per-rank shard and the DP replica is the only differing dimension — and the
project's own gate (`tests/e2e/long_term/accuracy/accuracy_multicard.py:62`, TP4 + EP, one
node, 0.888) with no DP>1 row, then spent its last several turns inside the DP all-reduce metadata
path (`model_runner_v1.py:547-559`) and shipped **no final answer**: 24 turns, 37 tool
calls, 70 K characters of reasoning against 2.1 K of message text, cut at the 8 192-token
segment limit. Corrective: the score comparison, the EP arithmetic and the coverage gap
already were the deliverable. Page: `../accuracy-benchmarks/index.md` (rule on
`expert_tensor_parallel_size` under EP).

**#1592** vllm-ascend v0.7.3.post1 (`c69ceac`) + vllm v0.7.3 (`ed6e907`), Kylin aarch64
+ 910B, Qwen2.5-32B: a chat request carrying `response_format: {"type":
"json_object"}` answers `500 Internal Server Error` while plain chat works. The log
prints three guided-decoding fallback warnings and dies at
`xgrammar_decoding.py:89  vocab_type = xgr.VocabType.RAW`. Track: upstream
guided-decoding **backend-selection loop** on a host where xgrammar cannot exist
(`requirements-common.txt:24` marks it `platform_machine == "x86_64"`). This one
delivered: the answer landed on the 8th assistant turn after ~5 checks — the
warning-by-warning chain with `path:line`, `--guided-decoding-backend
lm-format-enforcer` as the escape (including the verification that the per-request
`xgrammar` default is stamped only for `json_schema`, so the server flag does apply),
and why `--guided-decoding-backend outlines` cannot escape (the `json_object` rewrite
preempts `outlines_decoding.py:127`) — with `findings.md` written before the final
message. The one waste was a whole-file `cat SOURCES.json` on turn 1 (rule 7); its
replacement, `scripts/sources-summary.sh`, was used by the same run two turns later.
Page: `../serving-frontend-faults/guided-decoding-backend-aarch64.md`.

**#7180** image 0.13.0 + CANN 8.5.0, 910B4, `QwQ-32B-GRPO-INT8`: "同一条 prompt，nv 卡
vllm 推理若干次的输出 tokens 长度一致，而 npu A2 推理若干次的输出 token 长度差异大" — no
log, no request body. Track: reproducibility, not accuracy
(`../accuracy-benchmarks/repeated-run-nondeterminism.md`). The recorded run delivered in
shape (verdict, ranked causes, action ladder, `findings.md` before the final message) and
was source-accurate on the batch-invariance mechanism, but it **ranked "采样参数没有固定" as
cause 1**; the reporter's later public comment pasted the body — `temperature: 0`, `top_p: 1`,
`top_k: 1`, `repetition_penalty: 1.1`, `max_tokens: 65536`, prompt ≈8782 tokens, and three
screenshots of `time curl` runs — so the only live branch was the one it ranked second. Two
things it had and did not use: the reporter's own cross-platform control already implied a
pinned body, and `vllm/v1/sample/sampler.py` `all_greedy` → `greedy_sample` (argmax, before
any RNG) refutes the sampler at `temperature: 0` from source. It also never cited the tree's
own FAQ #16 (`docs/source/faqs.md:167-203`, four determinism env vars), leading instead with
the beta `VLLM_BATCH_INVARIANT=1` program and its gates. Page:
`../accuracy-benchmarks/repeated-run-nondeterminism.md`.

**#9521** vllm-ascend `v0.18.0rc1` + vllm `v0.18.0`, DeepSeek-V3.1-Terminus-
w4a8-mtp-QuaRot, `stream: true`, `chat_template_kwargs: {"thinking": true}`,
`temperature: 0.1`, a one-word user prompt, no `max_tokens`: "概率性出现没有content".
Track: response contract versus completion length, not a crash. The recorded run
classified it there and verified the two facts that decide the shape — `deepseek_v3`
reads `thinking`/`enable_thinking` from the **request's** kwargs and rebuilds the
parser per request (vllm v0.18.0 `deepseek_v3_reasoning_parser.py:32-41`,
`chat_completion/serving.py:229-236`) — then delivered a five-cause ranking with an
ablation ladder (routing → budget → MTP/w4a8 → detokeniser → template key) and the
ask for one response printing `content`, `reasoning`, `finish_reason`,
`completion_tokens`, `return_token_ids`. Reproducing two log greps and a DeepSeek
tutorial read cost turns the page now spares. Later reporter data (same behaviour on
`0.18.0` **and** `0.19.1rc1`, `issue/9521#issuecomment-4541092252`) closes the
"fixed in a later rc" branch and leaves routing vs a genuinely short completion open
— which is exactly what the one whole-response dump separates, so this case supports
asking for that artefact earlier, not a claim about the cause.
**#14339** vllm-ascend `v0.23.0rc1` (`f4a08bd`) + vllm v0.23.0
(`0fc695fc`), Atlas 300I DUO / 200I PRO: "[v0.23.0][Bug]: the accuracy of the MTP and
prefix cache of the 310P Qwen3.5 series models is abnormal", with an empty
`collect_env` block and a two-sentence body — no log, no request, no serve command.
Track: hybrid GDN state x spec decode x prefix caching on a SoC-conditional fallback.
The recorded run localized the right surface (align mode, `num_speculative_blocks`,
the 310P torch fallbacks, the fp16 SSM state) and its rank 1 was defensible, but it
spent ~42 K characters of reasoning over 22 tool calls with **no message text on 17
assistant turns**, then began the ranked answer on turn 20 and was cut at the output
cap inside cause 2 — the matrix, ladder and scope never shipped. The page now carries
the coverage verdict (the 310P prefix-cache test and the 310P MTP test never combine
the features) and the three launches that decide it:
`../accuracy-benchmarks/mtp-prefix-cache-310p.md`.

**#17338** vllm-ascend `v0.23.0` (`5cb98caa`) + vllm v0.23.0 (`0fc695fc`), Atlas 300I
Duo 310P3 ×4 at TP4: "Qwen3.8-27B-w8a8-310p (GDN hybrid) with `--enable-prefix-caching`
and MTP both on emits token-level corruption", with the serve command, the acceptance
numbers and the sample outputs pasted. Track: the same superposition as #14339, on a
27B W8A8 checkpoint at `FULL_DECODE_ONLY`/`[1,4]` — and the one datum the earlier
report lacked: **acceptance 70-100 % while the text is corrupt**, which bounds the fault
to the SSM state / block boundary and keeps `../spec-decode-mtp/acceptance-collapse.md`
out of the answer. The run re-verified every cited `path:line` at the release pin
(the page had been written at `v0.23.0rc1`), then delivered the coverage verdict, the
three-launch matrix and the scope in one answer. Page:
`../accuracy-benchmarks/mtp-prefix-cache-310p.md` (second recorded report).

**#16980** vllm-ascend `v0.23.0` (`5cb98caa`) + vllm v0.23.0 (`0fc695fc`), Atlas 800T
A2 (910B3×8): "[Bug]: gemma4 MTP speculative decoding has a 0 % draft acceptance rate —
draft layers on multiple KV cache groups share group[0]'s block table", `gemma-4-12B-it`
/ `-31B-it` + the matching `-assistant` draft, `num_speculative_tokens: 1`, eager,
TP1 and TP2. Track: family parity in spec decode. The run found the decisive structure
— upstream has `gemma4_mtp` plus a dedicated `Gemma4Proposer`/`Gemma4Speculator` with
per-group block tables, constant draft positions and draft→target KV sharing, while
`vllm_ascend/` has zero `use_gemma4_mtp` hits and `get_spec_decode_method` sends every
`mtp` launch to `AscendEagleProposer` — plus the downstream port template
(`AscendStep3p5MTPProposer`'s two overrides) and the coverage verdict (no `gemma4`
test or tutorial). Two gaps the page now closes: it said "downstream has no gemma4
support" where the release notes carry "experimental Gemma4 support on A2 and Ascend
950" with a named A2/A3 512-dim-head attention fallback `TODO` (three levels, not one),
and it shipped no support-plan paragraph — the same reporter's later public follow-up
asks only for that ("gemma4 mtp 910B 上的支持计划是什么样的"). Page:
`../spec-decode-mtp/family-proposer-parity.md`.

**#16116** vllm `v0.26.0` + vllm-ascend `v0.26.0rc1`, 910 8-card, CANN 9.1.0:
  "[Bug]: DeepSeek-V4-Pro thinking stream loops in delta.reasoning and never emits content", a
  well-instrumented body (376 SSE chunks all `delta.reasoning`, `output_token ==
  max_completion_tokens == 77817`, an ~812-character self-rejection cycle, the request and gateway
  metadata pasted). The recorded run answered in 21 messages — every turn carried draft text, the
  verdict moved into the message at turn 3 — and shipped the full ranked answer at the cap of its
  own budget rather than the harness's: `max_completion_tokens` outranks `max_tokens`
  (`chat_completion/serving.py:307-311`), no loop brake exists, `thinking_token_budget` 400s without
  an enabled reasoning config or off the V1 runner (`input_processor.py:102-117`) and no-ops on NPU
  once admitted (`npu_input_batch.py:56-59`), and `reasoning_effort` on DeepSeek-V4 is a prompt
  prefix (`patch_deepseek_v4_thinking.py:25-48,85-86`) that *asks* for the enumerated-rejection loop
  — but only the string the checkpoint's branch maps to it does (the replay's "`high` reaches the
  high prefix in both mappings" was wrong: under the preview mapping `high` is the empty prefix).
  Page: `../accuracy-benchmarks/reasoning-output-routing.md`, last section.

**#17391** image vllm-ascend 0.23.0, GLM-5.2-w4a8c8 / GLM-5.3-w8a8c8, 8×A2 P/D,
  `--tool-call-parser glm47` + `--reasoning-parser glm45`, MTP n=1 (P) / n=4 (D):
  streaming ends at a lead-in line ending `:` with no `tool_calls`, and the reporter's
  own harness printed `[OK] … JSON 校验通过` because `empty_arg_frags=0` — there was
  nothing to validate. The run delivered the ranked answer in 8 turns from two
  source facts: an unclosed `<tool_call>` is withheld from **both** `content` and
  `tool_calls` (`glm4_moe_tool_parser.py:238-303`, `:453` at vllm v0.23.0), so the cut
  point is the marker's fingerprint and `return_token_ids` decides it — not the
  model-choice/rate lane the "empty call" entry shape otherwise implies. Page:
  `../tool-call-streaming/index.md`, `no-tool-call-in-raw-output.md` table.

**#15925** vllm-ascend v0.26.0rc1 (`f2f74a1c`) + vllm v0.26.0 (`568afb3a`),
4 machines in a 1×2P / 1×2D PD layout on Atlas 800 A3, GLM-5.3-w8a8c8 vs GLM-5.2-w4a8c8, an
ifbench-style strict/loose instruction-following benchmark with
`reasoning_effort: "max"`: "the w8a8c8 weights score lower than the w4a8c8
weights". Track: a quantized-sibling score gap, not a kernel defect. What the
recorded run got right: the classification, the three-variable count (model ×
quant format × release label, with the env dump saying 0.26.0rc1 while the table
rows say 0.23.0), the within-model control (GLM-5.2-w8a8 73.67/76.67 against
w4a8c8's 73.67/76.5 — the row that retires "8-bit is 18 points worse"), the
loose−strict arithmetic (9.66/8.83 against 3.33/2.83, i.e. a widened spread), the
pinned recipes (`GLM5.2.md:211,261` sfa_c8 off vs `:505,579` all three on), the
release's model-restricted validated list, and the fact that `reasoning_effort`
has no downstream mapping for GLM. What it cost: the recipe diff and the lane's noise floor live in
the C8 page and were reached only around the tenth tool block, so the answer was
cut mid-sentence at the 16 384-token output cap (786 K cumulative prompt tokens,
3 975 characters delivered) and ranks 2+ plus the whole ablation ladder never
shipped — and the 38 KB accuracy index it opened first carried no route to them.
The page above now opens with the variable count and the control row and routes
to the flag authority in one line: `../accuracy-benchmarks/quantized-sibling-score-gap.md`.


**#4962** vllm-ascend 0.11.0rc2 as reported, pinned at v0.11.0rc3 (`b6d63bbd`) +
vllm v0.11.0 (`b8b302cd`), one Atlas 800 A3 node, two dies exported
(`ASCEND_RT_VISIBLE_DEVICES=14,15`, `--data-parallel-size 2`, `--async-scheduling`,
no `--enforce-eager`), Qwen2.5-Omni-7B served as `qwen-omni`: an `audio_url`
request asking for a transcript comes back with a plausible first clause, then
repeats "然后呢，嗯，" to `finish_reason: "length"` at `temperature: 0`
(`prompt_tokens: 775`, `completion_tokens: 256`). Track: corruption/degeneration on
a modality route. What the recorded run got right: the shape ("退化循环", not a
score), the doc-vs-code gap that is the actual finding — the pinned tutorial's
online section says the OpenAI-compatible server does not support audio inputs
(`docs/source/tutorials/single_npu_audio.md:122`, citing vllm#19977) while upstream
vllm v0.11.0 does parse `audio_url` (`chat_utils.py:69,77,816-904`) — the ownership
split (only `self.visual` is replaced downstream, so a pure-audio request never
touches the Ascend vision tower or `USE_OPTIMIZED_MODEL`), the `prompt_tokens`
placement argument, and the ladder (offline `LLM()` tutorial path, then
`--enforce-eager`, then `--async-scheduling`, then DP=1). What it cost: its last
two blocks went into the `audio_feature_lengths` tensor-vs-list hazard in
`model_runner_v1.py:774-801` → upstream `mrope.py:1082`; the run correctly refused
to publish it as a bug, so the turns bought no claim, and it never took the two
one-grep coverage facts that would have hardened the verdict (the audio encoder is
`transformers`' `Qwen2_5OmniAudioEncoder`, whose modeling file is absent from the
partial export; the only audio e2e at the pin is offline Qwen2-Audio,
`tests/e2e/singlecard/test_vlm.py:92-124`, `max_tokens=10`, assert-not-empty).
Page: `../accuracy-benchmarks/omni-audio-chat-degeneration.md`.

**#1798** vllm-ascend `v0.9.2rc1` (`b5b7e0e`) + vllm v0.9.2 (`a5dd03c`),
DeepSeek-V3-0324-w8a8, 8×910B-64GB × 2 nodes, `DP4 TP4` + EP, torchair graph and
the Ascend scheduler on: the pasted test script sends **no** `tools` field and the
serve line carries **no** `--enable-auto-tool-choice --tool-call-parser
deepseek_v3`, so the model's `<｜tool▁calls▁begin｜>…<｜tool▁calls▁end｜>` text comes back in
`content`; the screenshot's OCR shows 7/10 requests well-formed and 2/10 looping
on the opening marker. The recorded run read the OCR, pinned the parser gate
(`serving_chat.py:495`, `cli_args.py:311`) and ranked sampling degeneration for the
loops correctly, but its action ladder ended at "add the two flags". The reporter's
follow-up refutes that as a complete fix: with `--tool-call-parser deepseek_v3
--chat-template tool_chat_template_deepseekv3.jinja --enable-auto-tool-choice` the
failing case returns **both** `content` and `tool_calls` empty — the DeepSeek-V3
parser's silent drop, i.e. start marker seen, strict fenced-call regex unmatched,
`tools_called=True, tool_calls=[], content=None` (`deepseekv3_tool_parser.py:84-107`
at vllm v0.9.2) — so the deliverable for the second shape is the raw text vs the
regex comparison, not another flag
(`../tool-call-streaming/deepseek-parser-silent-drop.md`, new). The reporter's own
parser-free route (tool schemas in the system prompt, read `content`, verified on
H200) is recorded there with its scope.

**#6525** reported 0.14.0rc1; snapshot pins vllm-ascend v0.14.0rc1 (`52d4acf`) ↔ vllm
v0.14.1 (`d7de043`). Qwen3-VL-235B-A22B-Instruct, 双机 deployment, body = one sentence
("循环输出问题如何解决，重复惩罚参数无法解决") plus an **empty** `collect_env` block: no
log, no serve command, no request body. Track: output degeneration with no evidence
(“答案 = 排序原因 + 消融矩阵 + 缺失物料”, not a mechanism).

What the recorded run got right: the verdict that `repetition_penalty` cannot fix a
logits collapse (and the check that makes it a fact rather than an opinion:
`rg 'repetition_penalty|penalties' vllm_ascend/sample/` is empty at the pin, so the
penalty is upstream's and "it did nothing" is not a penalty bug); the discovery that the
pin ships **two** two-node recipes for this exact checkpoint — the tutorial's MP
DP2×TP8 + EP + `--async-scheduling` (`docs/source/tutorials/Qwen3-VL-235B-A22B-Instruct.md`)
and the only multi-node nightly, which is PD with an asymmetric pair
(`tests/e2e/nightly/multi_node/config/Qwen3-VL-235B-disagg-pd.yaml`: P DP2×TP8, D
DP4×TP4 + FULL_DECODE_ONLY, `--max-num-batched-tokens 8192`,
`--no-enable-prefix-caching`, MooncakeConnectorV1) — so "which topology" is the first
question; and the ViT-sharding row (`--mm-encoder-tp-mode data`, accepted because
`vllm/model_executor/models/qwen3_vl.py:1246 supports_encoder_tp_data = True` at this
ref) together with `USE_OPTIMIZED_MODEL=0` for the head-dim activation pad
(`vllm_ascend/ops/mm_encoder_attention.py:102-112`, gate at `envs.py:71`).

What it cost: the second-ranked cause was the chunked-multimodal-input chain
(`--disable-chunked-mm-input` under the recipe's `--max-num-batched-tokens 4096`) — a
real code path (`vllm/config/scheduler.py:110` default False, Ascend forces nothing,
CUDA only forces it for `is_mm_prefix_lm` at `vllm/platforms/cuda.py:236-249`) but one
the report does not select, since nothing says the request contained an image. It also
never read `../accuracy-benchmarks/long-output-repetition.md`, and it invented a
Qwen3-VL vision `head_dim` from memory before noticing the export has no such config —
both avoidable. Page: `../accuracy-benchmarks/vl-vision-tower-degeneracy.md`
(“The lever is older than this pin”, “A 双机 / multi-node VL report”).

**#1553** reported vllm-ascend 0.9.1 (`6db7dc2`), pin v0.9.1rc1 (`c30ddb8`) + vllm v0.9.1
(`b6553be`), `vllm-ascend/DeepSeek-V3-Pruning` TP4, lm_eval `--tasks mmlu
--apply_chat_template --fewshot_as_multiturn --batch_size 1`, MMLU `0.2584 ± 0.0037` with
`enable_prefix_caching: true` + `torchair_graph_config.enabled: true` +
`ascend_scheduler_config.enabled: true` (and a no-op `"enable_chunked_prefill": false`): the
recorded run answered from the pinned v0.9.1rc1 release-notes Known-Issue row
`release_notes.md:37` (#1350, "prefix cache … does not work with the Ascend Scheduler but
without chunked prefill enabled") and made "turn APC off, MMLU recovers" its rank-1 ladder
rung. The reporter's later public comment (`2025-07-01`) is the control: `enable_prefix_caching:
false` with everything else identical scored `0.2585 ± 0.0037` — the row is a functionality
statement about the cache, not this score's cause, so the remaining stack (graph capture, the
scheduler fork, the `-Pruning` checkpoint, the harness) owned it and the answer's first rung
was already refuted. Same pin and same three-flag cell as the `decode_hs_or_q_c` contract, so
the doc/test anchors were cheap to reach — and the right move was to rank them, not to quote
the known issue. Page: `../feature-enable-recipes/torchair-mla-graph-and-chunked-prefill.md`
§*chance-level*.

**#12060** vllm-ascend v0.20.2rc1 (`367b8e62da79`) + vllm v0.20.2 (`bc150f502991`),
official image `quay.io/ascend/vllm-ascend:v0.20.2rc1`, 4 cards, Qwen2.5-7B,
PP2+TP2, no log at all: "pp+chunked prefill 场景下首token不对", with both controls
already stated (off chunked prefill fine, off PP fine). The intersection is real and
untested at the pin — the PP e2e prompts are 5-6 tokens
(`tests/e2e/multicard/4-cards/test_pipeline_parallel.py:35-38`) and the only
multicard chunked e2e is context-parallel — but the recorded run answered it as a
mechanism hunt (upstream `scheduler.py:1079-1099` token-id path, the downstream
`sync_and_slice_intermediate_tensors` fork) and shipped a truncated verdict with
four source hypotheses, 43 K characters of reasoning against 2 K of text. The
reporter's later public comment (2026-07-17) is the resolution: *"tried the latest
version, it works fine"* — the rung the answer owed first was the release lane.
Page: `../pipeline-parallel/pp-chunked-prefill-numeric.md`.

**#5643** vllm-ascend v0.11.0 (`2f1aed98ccdb`) + vllm v0.11.0, RL rollout (1 训 2 推,
reduced layers, weight re-send) with static EPLB (`additional_config.expert_map_path`)
and a garbled-OCR screenshot; "静态负载均衡是否有精度问题", 主线推理正常. Verified at
that pin in this review: `vllm_ascend/ops/expert_load_balancer.py` reshapes the file's
`layers_num`/`ranks_num` and never compares them with the deployed `num_moe_layers`/
EP size, the off-rank `log2phy` fallback picks a copy with unseeded `random.choice`,
`check_expert_map_tensor` only compares the file across ranks, and `expert_map_path`
appears in `tests/ut/` + the docs only — the two nightly `*_eplb` files exercise
**dynamic** EPLB (`dynamic_eplb: true` + `init_redundancy_expert: 16`, gsm8k gate
baseline 95 / threshold 5, `test_deepseek_r1_w8a8_eplb.py:87`,
`test_qwen3_235b_a22b_w8a8_eplb.py:83-85`) and no `.github/workflows/` file names eplb —
so the static-map path has no accuracy regression coverage while the dynamic one has a
baseline. The recorded run delivered the ranked answer (map-file
shape vs deployment, weight re-send vs physical slots, unseeded fallback, support
surface) with a 4-rung ladder and the screenshot gap stated; no reporter
confirmation. The map row is picked by `moe_instance_id`, a construction counter
(`common_fused_moe.py:148-149`), so a reduced-layer export shifts every layer's entry,
and a replay of this report pinned the gates that decide whether the map is in play at
all — the readable-path guard (`:181-183`, silent fallback `:204-211`), the `Eplb
supports only w8a8_dynamic quantization.` refusal (`:227-231`), the failed apply's one
warning (`:197-199`) and the expert-count line (`:214-219`) — with the ablation ladder
they imply: `../accuracy-benchmarks/eplb-static-map-deployment-shape.md` (new).
Route: an RL/weight-reload accuracy report is a *deployment-shape* audit first, and the
OCR of a screenshot is not evidence when it is garbled.

**#2934** vllm-ascend v0.9.1 (`0740d10211fc`) + vllm v0.9.1 (`b6553be1bc75`), Qwen3-8B
offline `LLM`, `SamplingParams(temperature=0.8, top_p=0.95, logprobs=20, n=2)`, body a
screenshot only: "输出中存在较多 logprob=-inf 的情况，gpu上不会出现类似的问题，对采样得到的效果
影响比较大". The attachment OCR is the evidence and the recorded run decoded it correctly:
the `-inf` entries are the vocabulary's lowest ids in ascending order (`!`, `"`, `#`,
`$`, `%`, `&`, … — ids 0,1,2,…), filling exactly the slots the finite entries left, which
is `torch.topk` over a row that was already masked. It then verified the two facts that
matter (upstream `raw_logprobs` is computed before the filters, `sampler.py:37-39`; the
downstream in-place `apply_top_k_top_p` is gated by `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`,
default off) — and spent the remaining budget on how a mask could reach `raw_logprobs`,
ending at the 16 384-token cap with an **empty** final message and ~0.9 K characters of
text against ~44 K of reasoning. The reporter's later public comment
(2026-06-29) restates the cross-platform control and the rank observation and closes
"我再试试吧，时间蛮久了" — i.e. the session's owed deliverable was the two-launch ladder
(env echo, `top_p=1.0`), which costs the reporter one run.
Page: `../accuracy-benchmarks/topk-logprob-negative-inf.md` (new).

**#8350** vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be9`), Qwen3.5-122B-A10B
TP8+EP on 8×910B2, `--enable-prefix-caching --block-size 128`, **no `--reasoning-parser`**:
"think 混乱且不可关闭，关闭就返回空list". Reporter-confirmed afterwards on the issue
(2026-04-21): `--reasoning-parser qwen3` for thinking on, `--default-chat-template-kwargs
'{"enable_thinking": false}'` for thinking off (served defaults merge under request kwargs,
`entrypoints/openai/engine/serving.py:815-824`; parser key `enable_thinking`,
`qwen3_reasoning_parser.py:39-42`), and at this pin the response field is `message.reasoning`
— a client reading `reasoning_content` shows an empty list. The recorded run had both flags
by turn 3 but then took the reporter's `--block-size 128` / prefix-caching flags as a
hybrid-KV-cache confound (turns 36-50 on `mamba_cache_mode`, `refresh_block_size`,
`release_notes.md`) and shipped **no answer**: 16 assistant turns, ~44 K characters of
reasoning, 185 characters of message text. The guard: one serve command and no
works-vs-fails control means there is no confound to grade — `hybrid-block-size-corruption.md`
applies only once the reporter has two commands differing by a scheduling knob.
Page: `../accuracy-benchmarks/reasoning-output-routing.md` (#8350 section).

**#5760** vllm-ascend v0.11.0 (`2f1aed98ccdb`, image `v0.11.0-openeuler`) + vllm v0.13.0
(`72506c9`), Qwen3-VL-8B-Instruct: `max_tokens: 1024` prints the answer, `max_tokens: 4096`
"waits until completion and receives empty" on the terminal, while `> response.json` has
content. Client-side display/transport: the reporter's own file redirect is the control that
proves the 200 OK body was complete, and the deliverable is `curl -w 'code=%{http_code}
bytes=%{size_download}'` plus `"stream": true` with `-N`. The run answered inside the gates
(verdict + ladder on its 3rd assistant turn, one correction after) and that correction is the
durable fact: the paste's `skip this step for we receive the data from remote disaggregate
prefill node` looks like PD and is not — it is guarded by `not has_kv_transfer_group()`
(`vllm_ascend/worker/model_runner_v1.py:1929-1936`), i.e. "no tokens scheduled this step".
Route: `../observability-and-logs/index.md` rule 4.

**#7191** vllm-ascend `0.11.0` (`2f1aed9`) + vllm `0.11.0` (`b8b302c`), CANN 8.3.RC1,
`torch_npu` 2.7.1, TP4, `enforce_eager=True`: offline `LLM(sys.argv[1],
enforce_eager=True, tensor_parallel_size=4)` on DeepSeek-R1-Distill-Qwen-7B/32B
returns `'\n\n,\n\n ( (\n ( or'` with a `SamplingParams` the body writes as
`temperature=0.6`, "from v0.7.3 to v0.11.0". **The reporter resolved it
themselves (public comment, 2026-03-27): changing the temperature to 0.6 —
"anything number >0.1" — fixed it on the 7B model**, so the effective value in
the failing runs was ≤ 0.1 and the body's `0.6` is not the run that produced the
sample. The recorded run (completed, 37 messages) ranked chat templating,
first-`generate()` state, `--dtype` and TP sharding, and offered
`SamplingParams(temperature=0, top_p=1, max_tokens=128)` as its cheapest check —
the reproducing configuration — so its ladder pointed away from the fix. Two
harness rules came from it: the sampler value is read before any engine mechanism
on a reasoning-distill checkpoint, and a pasted `SamplingParams` is a claim about
the run rather than the run —
`../accuracy-benchmarks/short-prompt-corruption.md` §*Grade the symptom first*.

**#562** vllm-ascend v0.7.3rc2 (`00459aea`) + vllm v0.7.3 (`ed6e9075`), torch-npu
`2.5.1.dev20250320`, CANN 8.0.0, 8×910B4: Qwen2.5-VL-7B after a LLaMA-Factory SFT
answers 全是感叹号, with the body's own two controls — "使用llamafactory推理是正常的，在v100上
也是正常的" — and a screenshot as the only "log". Track: VL degeneration, but the
controls put it in the numerics lane. **The reporter closed it themselves (public
comment, 2025-04-23): 「已解决，将精度改为 bfloat16 就可以了」**. Two later public facts from
the same thread: the post-SFT weight was asked for on 2025-04-21 and **declined** ("Sorry,
this model weight is not open to the public"), so the reporter's two controls can never be
re-run here and "the weight" does not belong on a send-back list; and on 2025-04-22 an
intermediate comment sent them to `INF_NAN_MODE_ENABLE=0` (MindIE experience), the reporter
answered "Okay, thanks. I'll try it", and a maintainer corrected the advice the same day —
`INF_NAN_MODE_ENABLE shouldn't be set to 0 on the current device … please export
INF_NAN_MODE_FORCE_DISABLE=1` — on a `2.5.1.dev20250320` build, i.e. the guard is a **build**
fact, not a version-number one. The recorded run (38
messages — 16 assistant turns — cut at the output cap) verified the right structural fact — at that pin
`Qwen2_5_VLForConditionalGeneration` is upstream's, so the padding family and
`USE_OPTIMIZED_MODEL` cannot apply — and then ranked `--tensor-parallel-size 1` and
the checkpoint/SFT row above the dtype axis, which the reporter's clean controls
refute, planning the whole answer inside its final reasoning block and shipping no
text. Three harness rules came from it: the pinned tutorial's own `--dtype bfloat16`
(`docs/source/tutorials/single_npu_multimodal.md:147`) is the axis to diff first, bf16
is available on every 910B SKU (`IsBF16Supported()`), and the `INF_NAN_MODE_*` name is
build-scoped, not version-scoped (the 2024-era 2.5.1 tree reads only
`INF_NAN_MODE_ENABLE=0`; guarded builds — 2.7.1 verified, 2.8.x reported, and this
reporter's 2.5.1.dev build per the maintainer — abort on that with a message naming
`INF_NAN_MODE_FORCE_DISABLE=1`) —
`../accuracy-benchmarks/precision-dtype-degeneracy.md` (new).

**#7365** vllm-ascend v0.17.0rc1 (`e20f0b1a0d2f`) + vllm v0.17.0 (`b31e9326a7d9`), one
910B3, `Qwen/Qwen3-8B` target with `{"model": "Qwen/Qwen3-0.6B", "method": "draft_model",
"num_speculative_tokens": 5}`: the first request raises
`RuntimeError: The expanded size of the tensor (1024) must match the existing size (4096)
at non-singleton dimension 1. Target sizes: [5, 1024]` at `eagle_proposer.py:1008`. The
recorded run did the mechanism well — 24 tool calls, the draft-sized `hidden_states`
buffer (`eagle.py:79-82,161-163`), the `not needs_extra_input_slots` branch, the guarded
sibling copy, the `parallel_drafting` route, the e2e coverage gap — and shipped a full
ranked answer, but it read the *owner* wrong: it concluded the defect is not
Ascend-specific and told the reporter a GPU run of the same snippet must fail identically.
Upstream `DraftModelProposer` passes `pass_hidden_states_to_model=False` into the init that
computes `needs_extra_input_slots` (`vllm/v1/spec_decode/draft_model.py:19-30` →
`eagle.py:88-93`), so upstream takes the guarded branch; the fabric comes from the fork's
base class (`SpecDecodeBaseProposer(EagleProposer)` receiving the flag `True` from
`eagle.py:1691-1696` and the flag reassigned at `eagle_proposer.py:95`). The lesson is
one grep at the top of the session: *which* `__init__` computes the derived flag —
`../spec-decode-mtp/draft-model-serial-hidden-size-crash.md` (new).

**#6424** vllm-ascend v0.11.0 (`2f1aed98ccdb`) / vllm v0.11.0 (`b8b302cde434`) pinned,
reporter on v0.11.0 (bge-m3, 910B, CANN 8.3.RC2, K8s, `/dev/shm` already cleared by
the reporter): `POST /v1/embeddings` hangs to timeout while the server logs
`(APIServer pid=1) INFO: 10.16.x.x:0 - "POST /v1/embeddings HTTP/1.1" 200 OK` and a
`(Worker_TP0 pid=411)` `torch.save with "_use_new_zipfile_serialization = False" is
not recommended for npu tensor` warning. The recorded run found the one grep that
closes the warning (the same text in this release's own embedding tutorial,
`docs/source/tutorials/single_npu_qwen3_embedding.md:96`, printed in a successful run)
and still spent the rest of the budget on the ZMQ/tensor serialization path
(`vllm/v1/serial_utils.py`, `torch/_tensor.py::_reduce_ex_internal`,
`outputs.py:230`'s `.tolist()`) — 8 assistant turns, 12 tool calls, ~55 K characters
of reasoning, one 363-character message, no verdict, cut at the cap. The
discrimination it never used: the body is materialised in the API server
(`serving_embedding.py:134` → `outputs.py:225-230`) *before* the 200 is written, so a
completed response is a transport statement and one in-pod loopback `curl` decides it.
Page (new): `../pooling-embedding/client-hang-after-200-ok.md`.

**#2318** vllm-ascend v0.9.1rc2 (`b9f715d6ce30`) + vllm v0.9.1 (`b6553be1bc75`) — the
pin *is* the reporter's version — Qwen3-32B on 4 devices, a directory they named
`/path/to/my/qwen32b-w8a8pertoken`, `vllm serve --quantization ascend --tensor-parallel-size 4`
with an `ascend_scheduler_config` block: W8A8 per-token weights that answer garbage in
vLLM while the same weights "works in Transformers". The reporter's later comment
(2025-08-13) supplies the resolution the session never reached: the model had been
quantized with **m4 (an improved version of SmoothQuant)**, and that conversion recipe
is what produced the garbage. The recorded run never asked that question — it read
`quantization/{quant_config,quantizer,w8a8_dynamic}.py`, the downstream
`models/qwen3.py` decoder path, the fused `AddRMSNormW8A8Quant`, the quantizer dispatch
table, the pin's W8A8 tests and the docs across ~1.0 M cumulative prompt tokens, and was
cut by the output cap with no message text. The corrective is the ordering, not a new
mechanism: on weights the reporter converted, provenance comes before any engine read,
because the engine consumes numbers the converter chose and can neither validate nor
explain them — `../accuracy-benchmarks/quantized-checkpoint-conversion.md` rule 2 (and
its published-checkpoint split as the one-command discriminator).

**#5835** vllm-ascend **v0.11.0** reported, the snapshot pinning only v0.13.0rc1
(`1b5d5abf865f`) + vllm v0.13.0 (`72506c98349d`); Qwen3-235B, env dump literally `none`,
no serve command, no log: `response_format: {"type": "json_schema"}` answered as
doubled/interleaved characters (`"{\"{\"eexxxtttrrraaacctceded__eentntnitiiteesss\"…`),
`finish_reason: "stop"`, `completion_tokens: 207`. The recorded run found both the
mechanism hazard (the Ascend runner applies the xgrammar mask through a host float32
roundtrip, so the bitmask row is whatever `input_batch.req_id_to_index` plus the
spec-decode offsets say) and the two corroborating precedents (#4323 v0.11.x
Qwen3-32B + `eagle3` → `tctctctctc…`; #5144 v0.11.0 fp16 → `!!!!` on the first request
of a pair), yet shipped **74 characters** of message text across 15 assistant turns and
27 tool calls, cut while checking `vllm/entrypoints/openai/processor.py` — a path the
aarch64 guided-decoding page verified at vllm v0.7.3, where the whole-tree answer at
this pin is that `guided_decoding_backend` is gone and the backend is server-resolved.
Page (new): `../accuracy-benchmarks/structured-output-garble.md`; the delivery half is
`delivery-discipline.md` (*Verifying what a page already verified*).

**#10068** evalscope + MMLU-Pro scoring 5-6 points below ais-bench on
vllm-ascend v0.19.1rc1 (`da421afa`) + vllm v0.19.1 (`b1388b1f`), env dump matching the
pin exactly; the only evidence was three OCR'd screenshots of the two harness reports and
a vendor blog table. The recorded run shipped the ranked harness-vs-engine answer on its
sixth turn (reasoning-output routing → decode budget → sampling → thinking switch →
aggregation), correctly refused to pair the OCR'd per-subset labels with the numeric
block, and closed the coverage question arithmetically (evalscope's `Num` column sums to
12 032, the full MMLU-Pro test set). What it cost was a reasoning block spent on the
unpairable OCR table and an `rg -rn 'mmlu'` that ran `--replace`; the fixes are the
§*Reading the OCR'd score table* section of
`../accuracy-benchmarks/harness-vs-engine-attribution.md` (pair only where an
independent line fixes the mapping; quote ais-bench's `naive` **and** `weighted` columns
against evalscope's single `OVERALL`; ask for `summary_*.csv`/`reports/*.json`).

**#10413** vllm-ascend `v0.20.2rc1` (`367b8e62da79`) + vllm `v0.20.2`
(`bc150f502991`), A2 16×NPU per machine, 4P1D Mooncake deployment, 128K,
DeepSeek-V4-Flash-w8a8-mtp, `VLLM_ASCEND_APPLY_DSV4_PATCH=1`, ais-bench
`gpqa_gen_0_shot_str` with `chat_template_kwargs={"thinking": False}`,
`temperature=1.0`, `max_out_len=65536`, `extract_non_reasoning_content`: GPQA
63-57 against the official 71. Track: score-below-official, harness side. The
recorded run answered in one message after 16 turns and ~615 K prompt tokens, and
its rank 1 was right (the body runs DSV4 in non-thinking "chat" mode; upstream and
the downstream patch both default `thinking`/`enable_thinking` to False, patch
opt-in via `envs.py:116`), with the `prompt_tokens` probe, the project's own
`tools/aisbench.py:154-156,163` values and a PD/quant ladder behind it. Two things
only the reporter's later comments settle, both now in
`../accuracy-benchmarks/harness-vs-engine-attribution.md` §*thinking checkpoint*:
the effort value is a second, dataset-calibrated variable (MMLU-Pro closed at
thinking **max**, 85.91 vs 86.2; GPQA still short at max and matched at `<think>
HIGH`), and the hand-created `hccn_conf` the reporter asked about is a connectivity
configuration, not an accuracy lever.

**#4616** vllm-ascend `v0.11.0rc2` (`a2e4c3fe`) + vllm `v0.11.0` (`b8b302cd`),
910B 32G, official image, `Qwen3-Reranker-4B` served with
`--hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],
"classifier_from_token": ["no","yes"], "is_original_qwen3_reranker": true}'` and no
`--task`/`--runner`, `/rerank` with 6 Chinese documents: HTTP 200, scores
0.807-0.889, the exact match fourth behind an unrelated document, filed as
精度问题. Track: pooling/rerank scoring. The recorded run spent its whole 16 384
token output budget inside `adapters.py`/`pooler.py`/`registry.py` and shipped no
message text, although the decisive arithmetic was in the response body it had
pasted (`usage.total_tokens: 88` over 6 pairs ≈ 15 tokens per pair). Now
`../pooling-embedding/rerank-untemplated-score.md`: `get_score_prompt`'s three
branches at v0.11.0 and the fact that only `jina_vl.py` implements
`SupportsScoreTemplate` mean a Qwen3-family `/rerank` call at that pin scores a
raw `query + document` pair, and the release's own route is the embedding endpoint
with a client-built `Instruct:` prefix.

**#8844** reported vllm `0.18.0` / vllm-ascend `0.18.0` (neither is in the snapshot;
the case pins vllm-ascend v0.13.0rc3 `bbe9bc0d` + vllm v0.13.0 `72506c98`), 4-node PD:
GLM-5/5.1 `GLM-new-w8a8`, `MooncakeConnectorV1` with `use_ascend_direct`, DP2 × TP16 +
`--enable-expert-parallel`, `--speculative-config '{"method":"deepseek_mtp",
"num_speculative_tokens":3}'`, P with `--enforce-eager` (4096 max batched tokens) and D
with `FULL_DECODE_ONLY` + `--async-scheduling` (512), `--gpu-memory-utilization` 0.95 /
0.92, `--max-num-seqs` 64 / 128, `VLLM_ASCEND_ENABLE_FLASHCOMM1` on the P node only,
`multistream_overlap_shared_expert` written twice in one JSON object, and
`"recompute_scheduler_enable": true` on **both** nodes. The symptom is a score, not a
crash: GPQA-diamond 67.17 in the title, 67.8 % in the body, "lower than the result
reported in the paper", with two reporter hypotheses — a large TP (≤4 is fine) and a
loss that appears "when the output length is long, concurrency is high, and GPU memory
utilization remains saturated for an extended period". The reporter's later comment on
the issue closed it: *"This issue arises from recalculations after the gpu memory usage
is fully utilized. Disabling recalculations eliminates this problem."* — i.e. the
decode-side recompute path, and its off-switch is the `recompute_scheduler_enable` their
own paste sets to true. The recorded run shipped a complete six-cause ranked answer
(measurement terms first, then truncation/`--reasoning-parser glm45` routing, then
MTP + `--async-scheduling` bookkeeping, then the P/D recipe asymmetry, then KV
saturation as "a correctness trigger, not a fix", then TP16/DP2), correctly reading the
duplicate `multistream_overlap_shared_expert` key and the `layer_sharding` /
FLASHCOMM1 asymmetry — and never opened the recompute flag, although one of its own
greps had printed `recompute_scheduler_enable` among the reporter's keys. What that cost
is the whole session's answer: the flag-vs-state control was one restart with a key the
reporter had already typed. The durable rule is in
`../accuracy-benchmarks/pd-recompute-after-kv-saturation.md` (new): when the reporter's
hypothesis names a *state* and the `--additional-config` carries a flag gated on that
state, the flag-vs-state arm is rung 1.
