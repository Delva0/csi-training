# Accuracy and score discrepancies

## Scope

Reports of the shape "score with tool A is N points below tool B / the vendor number /
another image", and reports of accuracy regressions after a code change. Covers MMLU-Pro,
gsm8k, ceval and friends as seen through evalscope, ais-bench, lm_eval and vendor blogs.
Also covers the corruption end of the same track — a prompt that comes back as 乱码 /
garbled / repeated / empty text rather than as a wrong answer — and the shortest sibling
symptom, output that is fluent but stops after tens of tokens ("被截断") — because both are
decided by the same first question (is anything being compared, or is the engine wrong per
se), then by `finish_reason`. A report that pastes only a response body — thinking text in
`content`, `reasoning`/`reasoning_content` missing, `reasoning_tokens: 0` — is also this
topic: the answer is the serve flag's presence, not a kernel.
One shape here compares the engine **with itself** rather than against a reference —
the same prompt or body run several times, with lengths or contents differing
("nv 卡多次一致，npu 多次不一致", "`temperature=0` 结果不确定"). That is
reproducibility, not accuracy: `repeated-run-nondeterminism.md` owns it, and the
request body plus the *pattern* of which runs differ decide it. Its sibling compares
the engine with itself across **batch composition** instead of across repetitions —
the same body differing only with concurrency while `VLLM_BATCH_INVARIANT=1` is set:
`batch-invariant-coverage.md` owns that one, and the answer is the feature's per-op
envelope rather than the flag failing to apply.

Seven entry shapes leave this topic immediately and are not accuracy questions at all:

- Wrong or alternating answers in a **PD/EPD** deployment on a hybrid
  linear-attention model (GDN/Mamba: Qwen3.5/Qwen3.6, DeepSeek-V4, GLM) — the
  suspect is the state the decode node received, not the numerics. Go to
  `../pd-disaggregation/hybrid-state-transfer.md` (connector matrix plus the
  `mamba_cache_mode` gate) and bring back its ablation rows as the deliverable.
- A number printed while the checkpoint is being *converted* (per-layer
  `MSE loss`, an epoch table, a bad quantized artifact) is converter output and
  precedes any engine comparison — go to `quantized-checkpoint-conversion.md`.
- Speculative decoding / MTP owns the symptom — `Accepted: 0` with `Drafted: N > 0`
  and `Mean acceptance length: 1.00`, a stall with drafts still being produced, or a
  first response with `content: ""` and `completion_tokens: 1` — go to
  `../spec-decode-mtp/index.md`; the ordered ablation (MTP off first) lives there.
- A **tool call** the client cannot parse (`JSON Parse error: Unterminated
  string`, `Expecting ',' delimiter`) or that arrives truncated / 格式错误 /
  重复输出: that is protocol assembly against the model's own stop, not the text
  corruption on this page — go to `../tool-call-streaming/index.md` and run its
  audit on the raw SSE first.
- A **vision** request answered as a long run of one token (`!!!!!!!!`, 感叹号,
  `，，，`) with "and it is very slow", **or** a VL/OCR answer that is merely
  *wrong* — one coarse element, boxes without text, unreadable OCR — only when the
  serve is multi-card while the single-card run of the same request is correct
  (two screenshots, no log), **or** the deployment is 1P1D/EPD and a single-instance
  serve of the same body, image and flags answers it, **or** a vision request whose
  answer is fluent but stops after a word or two (`"The"`, `completion_tokens: 2`,
  `finish_reason: stop`, no `length` — #6909) — that is not a score question and not
  the text-corruption profile: the downstream vision tower (or the split itself) owns
  it, and one flag (`--mm-encoder-tp-mode`, v0.17.0rc1) or one env var decides it. Go
  to `vl-vision-tower-degeneracy.md` **first** (§*PD/EPD split* for the PD row),
  before the early-EOS page: an image in the request plus an absurdly short answer is
  graded by `usage.prompt_tokens` against the model tutorial's own sample response
  (that page's grading bullets), and that comparison — not the `stop_reason` lane — is
  what names the subject. The same page owns the **mis-structured** variant — 结构化
  内容错乱, the returned HTML/JSON layout garbled, while the *older* image tag of the same
  serve structured it correctly (#4730) — where the tag rung is inverted and the delta
  must be enumerated inside each image before any code claim (§*Fifth pin*).
- A **VL or OCR** checkpoint on **310P / Atlas 300I** answers garbage while a text
  model on the same box is fine: the support verdict and the layout → recognise
  *pipeline* come first, because the VLM alone is one stage of a document-parsing
  product and its Atlas-300 support is a per-model doc statement, not a matrix
  column. Go to `../soc-platform-paths/310p-vl-ocr-garbage-output.md`; the rope
  branch there is one probe (`scripts/310p-rope-branch.py`), not a walk.
- The comparison is between **embedding vectors** (cosine, `/v1/embeddings`,
  `--runner pooling`, `--task embed|score|classify`) rather than between answers:
  no text is generated, so `finish_reason` and extraction play no part and the
  engine can be wrong per span rather than per token. Go to
  `../pooling-embedding/index.md`; a crash under concurrency on the same serve is
  `../pooling-embedding/batch-index-desync.md`, and a report whose control is **the
  same string against itself** (call 1 right, call 2 wrong, call 3+ identical to call 2)
  is the prefix-cache form — `../pooling-embedding/prefix-cache-hit-embedding-drift.md`,
  reporter-confirmed `--no-enable-prefix-caching` (#5725).

## Rules

- **Only `temperature>0` corrupts and greedy is byte-identical** — `<pad>` / token id
  0, one rare id, repeats spliced into otherwise exact text, the rate rising with
  concurrency, clean under `--enforce-eager` / `--no-async-scheduling` — is the
  sampler's side-stream `q`, not a kernel and not a model: rank the allocator
  *lifetime* (`record_stream`) above the "missing stream sync" story such reports
  usually tell, because `wait_stream` already orders the fill before the read. Take
  `VLLM_BATCH_INVARIANT=1` on the same canary as the one-env-var probe →
  `sampler-side-stream-corruption.md`. A **greedy** body that differs is the sibling
  page, `repeated-run-nondeterminism.md`.
- A corruption report whose "clean" and failing commands differ by a **scheduling**
  knob — `--block-size`, `--enable-prefix-caching`, `--mamba-cache-mode`,
  `--enforce-eager` — on a **hybrid linear-attention** model (Qwen3.5/Qwen3.6 GDN and
  siblings) is a KV-cache-geometry question, not a kernel-numerics one: on those
  models the cache block is never clamped to 128 and is only ever *raised* to match
  the mamba page size. Separate the knobs (the recorded #13853 "clean" command
  changed two) and go to `hybrid-block-size-corruption.md`. A short *digit-heavy*
  prompt on the same model family (Qwen3-Next) with **no** knob difference is the
  other hybrid shape: `qwen3-next-gdn-numeric-path.md`.
- **A repeated newline/punctuation loop from a reasoning distill at low temperature
  is the sampler value, not a code path.** DeepSeek-R1-Distill-Qwen-7B/32B offline
  `LLM.generate` returns `'\n\n,\n\n ( (\n ( or'` at temperature ≤ 0.1 and
  normal text at `temperature=0.6` — reporter-confirmed #7191, *"anything number
  >0.1"*, after the same script had "failed from v0.7.3 to v0.11.0". So here the
  greedy row is the reproducing condition rather than the control that the
  `sampler-side-stream-corruption` / `repeated-run-nondeterminism` rules below
  impose, and the first ladder rung *raises* the temperature. Read the pasted
  `SamplingParams` as a claim about the run, not as the run: #7191's body names
  0.6 while its sample is the low-temperature one →
  `short-prompt-corruption.md` §*Grade the symptom first*.
- **The whole answer is one repeated character or token** (`!!!!!!!!`, `，，，`, a
  single word) on a text serve: grade it at the id level before any theory, because
  a constant character is a constant token id and a constant-tie logits vector
  retires weights, quantization, tokenizer, sampling and prompt content at once.
  The onset then splits the lane — *always* is a config × capability gap (recorded
  #1754: `--pipeline-parallel-size` on an ACLGraph serve made all four prompts return
  `Output: '!!!!!!!!!!!!!!!!'`), *after hours of healthy serving, every new
  conversation too, restart restores* is persistent process/device state (recorded
  #3547). One `"temperature": 0, "logprobs": true, "top_logprobs": 5` body on a
  trivial prompt, captured **while still broken**, decides it →
  `constant-token-degeneracy.md` first, then the VL/sampler/first-call pages the
  scope list there excludes.
- A **first-call-only** degenerate answer — a constant token id repeated to
  `finish_reason: "length"`, an immediate `<|im_end|>` after `<think>`, or an
  `AttributeError: 'NoneType' object has no attribute 'size'` raised through
  `compilation/acl_graph.py` or traced at `torch/_dynamo/utils.py::call_size` — on
  an offline `LLM` at TP≥2 with call 2 correct, **or** on a `vllm serve` whose
  first request kills the worker while `--enforce-eager` runs clean (#7947, no call
  2 exists there) — is not a numerics question: a constant-tie logits vector retires weights,
  quantization, tokenizer, block size and prefix caching at once, and the two real
  owners are the text-only-multimodal `is_multimodal_model` drift and the
  first-request state that `enable_sleep_mode=True` exposes.
  Go to `first-generate-degenerate.md` before any kernel read.
- **A kernel the whole path shares is refuted by the pin's own wired accuracy row.** Before
  ranking rope/mrope, attention, an embedding or the sampler as the corrupting site, read
  `tests/e2e/models/configs/<model>.yaml` (the value *and* its `hardware:` line) and confirm
  a `.github/workflows/*.yaml` (or `misc/model_list.json`) row actually runs it. At
  `52d4acfa` the `Qwen3-VL-8B-Instruct.yaml` row is `hardware: "Atlas A2 Series"`,
  `mmmu_val acc 0.55`, wired at `nightly_test_a2.yaml:120`, so a text-only Qwen3-VL
  degeneration cannot be a shared-kernel corruption (#6533): the hypothesis must name the
  regime difference — text-only positions vs image, TP, dtype, batch, card SKU — or be
  dropped. `short-prompt-corruption.md` Step 0b.
- A **quantized** serve (`--quantization ascend`, a `…-W8A8`/W4A8 checkpoint) on a
  **triton-ascend-backed** family — Qwen3-Next and its GDN siblings, whose
  linear-attention kernels are triton-ascend
  (`../image-dependency-faults/triton-ascend-pair.md`) — answering as one punctuation
  token repeated to the cap (`!!!!!!!!`, `，，，`), `finish_reason: length`, nothing
  crashing: resolve the installed **triton-ascend build** before any numerics, NZ or
  sampler row. The maintainer attributed exactly this shape to the wheel (the `1107`
  nightly yielding NaN in quantized operators) and the reporter cleared it by changing
  builds (#5606) — and the wheel the pinned Qwen3-Next tutorial installs *is* that
  `1107` build. First ask: `pip show triton-ascend`, which the `collect_env` block does
  not reliably carry. A bf16 run of the same prompt does **not** control for it (the
  NaN is scoped to the quantization scenario); take it after the build row,
  `qwen3-next-gdn-numeric-path.md` §*triton-ascend build*.
- A **tiny** prompt (a handful of tokens) answered as *one token repeated* to
  `finish_reason: length` at `temperature: 0` on a DeepSeek-V2-Lite serve (MLA,
  `q_lora_rank: null`) is the model-scoped shape: the recorded #894 pairing bump
  (vllm `f32fcd94` + vllm-ascend `f24375f`, incl. #905) did **not** fix it
  (reporter-confirmed), so the nightly-vs-image stack question is answered — go to
  `deepseek-v2-lite-mla-degeneracy.md`, take its checkpoint/prompt row first (the
  validated row is `-Chat`, not the base model) and keep the two V2-Lite-only
  branches (`q_proj` when `q_lora_rank is None`, the `global_num_experts == 256`
  MoE gate) as labelled suspects rather than causes.
- An accuracy report on **310P** naming **MTP *and* prefix caching together** — the
  one-line form ("Qwen3.5 series + MTP + prefixcache 精度异常", empty `collect_env`) or the
  full paste with acceptance numbers pasted beside the corrupt text (token-level breakage
  at **normal** acceptance, clean with prefix caching off):
  on a hybrid GDN model both features move the same per-block SSM state, and 310P is the
  one lane whose state copy/postprocess is a torch/numpy fallback with an fp16 SSM state.
  The deliverable is the coverage verdict — two 310P CI tests, each feature alone, never
  the combination — plus the launches and the hit/miss split, `mtp-prefix-cache-310p.md`.
- A **chance-level** aggregate — MMLU 4-way `0.2584 ± 0.0037` with every one of the 57
  subsets between 0.17 and 0.34 — on a serve whose only distinguishing flags are an
  `additional_config` feature stack (torchair graph + `ascend_scheduler_config` + prefix
  caching) is decided by *which flag you can remove and keep the score*, and it needs one
  response body before any source read: a real engine defect and a deliberately size-reduced
  checkpoint export (`…-Pruning` weights) look identical in the table. Recorded #1553
  (vllm-ascend v0.9.1rc1, DeepSeek-V3-Pruning TP4, lm_eval `--apply_chat_template
  --fewshot_as_multiturn`) matches the pinned release-notes Known-Issue row for exactly that
  flag cell (#1350, "prefix cache … does not work with the Ascend Scheduler but without
  chunked prefill", `release_notes.md:37`) — and the reporter's later APC-off re-run scored
  `0.2585 ± 0.0037` against `0.2584`, so the row (a *functionality* statement about the
  cache) was not the cause. When the pinned known issue and the symptom do not match
  classes, say so and use the ablation order in
  `../feature-enable-recipes/torchair-mla-graph-and-chunked-prefill.md` §*chance-level*:
  graph mode off first, then scheduler fork off (chunked prefill on, APC on), then a
  checkpoint/harness baseline with all three off.
- First question: **did both numbers come from the same server, weights and serve flags?**
  If yes, sampling, decoding budget, output routing, extraction and aggregation are the
  attributable knobs — but **"same server" does not mean identical logits on Ascend**: on a
  non-batch-invariant path the same prompt can produce different logits run to run
  (a single-sample comparison then differs by luck), so before drawing a score conclusion
  from one run per side, check the repeat-run pattern
  (`repeated-run-nondeterminism.md`). If no, nothing is attributable until the serve flags
  are matched; list them as candidates instead of choosing a culprit. When the two sides are
  two **images**, identically-spelled commands are not matched flags — the newer image's
  default-on graph-fusion stack (`npugraph_ex` under `FULL`/`FULL_DECODE_ONLY`) is part of
  its run, so the noise floor and that one ablation come first
  (`harness-vs-engine-attribution.md`, "Two images").
- Compare the *pattern*, not only the mean. Bidirectional or subset-specific gaps (some
  subsets up, others down several points) are characteristic of harness-level effects
  (truncation, answer extraction, thinking output routing). A uniform drop accompanied by
  garbled/repeated/empty output is the shape of an engine or quantization defect. Say which
  pattern the report actually shows, and reject the other explicitly.
- Recompute aggregates from the reported numbers before comparing them: micro (sample
  weighted) vs macro (subset mean) can differ by a point or more on MMLU-Pro, and the two
  averages are often printed side by side. Show the arithmetic.
- A reference from **another platform** — an H100/H200 golden scored against the fixed
  token sequence with `prompt_logprobs`, deviation deterministic, **no signed bias**,
  concentrated where the golden's own confidence is low, unaffected by
  `--enforce-eager`/graph mode and by the torch_npu precision switches — is neither a
  harness question nor repeat-run noise. The answer is the pin-scoped switch inventory,
  the topology match and the bucket arithmetic: `cross-platform-logprob-parity.md`.
  Before quoting the FAQ determinism recipe, check each variable is still live at that
  ref: the two `ATB_*` vars bind only to the ATB calls the release still makes
  (`torch_npu.atb.npu_paged_cache_load`, the one-shot `_warm_up_atb` matmul) and the
  served matmuls are aclnn/AscendC — and the grep key for that is `torch_npu.atb`, not
  `import atb`, which returns nothing and supports a claim that is false.
- A score comparison **across two quantized formats of the same model family**
  (`w8a8c8` vs `w4a8c8`, `w4a8` vs `w8a8`) is not a quant-width test: the two formats
  ship *different* documented `--additional-config` sparse-C8 recipes and usually a
  different node count, and this lane's own noise floor is 2-3 points run to run
  (`release_notes.md:270`, #14378). Take the reporter's same-model row as the control,
  then ablate the recipe — `quantized-sibling-score-gap.md`.
- A corruption report whose request carries **audio** — `audio_url`, a `file://` media
  path, a data URL — is a modality-pipeline question before it is a text-MoE one, and
  `usage.prompt_tokens` is what says the audio got in. At v0.11.0rc3 the online audio
  route is accepted by upstream but has no e2e and the tutorial says it is unsupported
  — `omni-audio-chat-degeneration.md` (its Rule 4 ladder, not the MoE matrix).
- No numeric claim from OCR screenshots without a cross-check (a log line, caption or
  reproducible sum). Screenshot tables lose row/label alignment; annotate the caveat.
- Prefer `finish_reason`, `content`, `reasoning` and dump-file statistics over speculation:
  the discriminating evidence is per-sample, not aggregate.
- Establish the phase before the component: a loss printed *before* the server starts is
  the quantizer's output, so attribute it to the converter (msit/msmodelslim) rather than
  to the engine, and say which of the two documented gates the reporter has not shown.

## Pages

- `harness-vs-engine-attribution.md` — knob-by-knob checklist (thinking, max tokens,
  sampling, extraction, aggregation), where the project's own baselines live, and the
  decisive experiment to run.
- `reasoning-output-routing.md` — output-routing in both directions, and the place to
  answer a pure response-contract question that pastes no score at all: (a) verified
  upstream vLLM v0.19.1 behaviour that turns truncated thinking into `content: null`,
  plus the `enable_thinking` vs `thinking` kwarg trap (which is per-parser: Kimi-K2's
  reads `thinking`, Qwen3's reads `enable_thinking` — vllm v0.23.0
  `kimi_k2_reasoning_parser.py:40-46`); (b) re-verified at vllm v0.22.1
  that "thinking inside `content` + no `reasoning` field + `reasoning_tokens: 0`" is a
  missing `--reasoning-parser` (default `""`), that the chat field is now `reasoning`
  with `reasoning_content` only a deprecated input alias, and that the Qwen3.6 tutorial
  omits the flag while seven sibling pages pass it. (c) A "thinking is in the wrong
  language / 回答中英混杂 / the answer is in English" report: first **is the text after
  `</think>` right at all?** If it is wrong, or the reasoning keeps re-deciding itself,
  routing is display only and the lane is numerics on a MoE serve
  (`moe-gating-topk-precision.md`, #10892 — the maintainer's resolution there was a
  gating top-k softmax precision fix, while the run had answered with this page's rows).
  If it is right: routing plus `finish_reason` budget, then model/template thinking
  language (`#6848`), then a hand-written `--hf-overrides` rope block (`#10405`,
  hypothesis row), then the version gate before recommending `enable_thinking: false`
  (`#7094` / `#7061`, upstream PR 34779).
  (d) A **boot-time parser `KeyError`**: `Reasoning parser '<name>' not found. Available
  parsers: …` or `invalid reasoning parser: <name> (chose from {…})`, typically right
  after a name copied from the reporter's `--tool-call-parser` line (`qwen3_coder`) —
  `--reasoning-parser` and `--tool-call-parser` read two separate registries, so the
  flag is the whole gate, the printed list fingerprints the release, and a Coder /
  non-thinking checkpoint needs no reasoning parser at all (#8022; the name-space
  section of the page). (e) A `deepseek_v3` serve: that parser reads
  `thinking`/`enable_thinking` from **each request's** kwargs and delegates to the
  R1 splitter or to the identity parser (vllm v0.18.0
  `deepseek_v3_reasoning_parser.py:32-41`, `chat_completion/serving.py:229-236`), so
  the reporter's `chat_template_kwargs: {"thinking": true}` is the parser's own key
  and "概率性没有content" is first a **request-body diff**, not a model defect;
  #9521 reproduced it on 0.18.0 and 0.19.1rc1, which closes the release-diff branch
  and leaves `completion_tokens` (small ⇒ genuinely short completion) as the
  discriminator against routing. (f) A **thinking stream that never
  closes** — every chunk `delta.reasoning`, zero `delta.content`, `output_token ==
  max_completion_tokens` — at vllm v0.26.0: `max_completion_tokens` outranks `max_tokens`
  by design, no loop brake exists, `thinking_token_budget` is accepted and **ignored** on NPU
  (`npu_input_batch.py:56-59`), and DeepSeek-V4's `reasoning_effort` is a prompt prefix that
  amplifies deliberation rather than capping it; the ordered ladder is the #16116 section.
  (g) **"think 混乱" plus "关闭思考就返回空 list" on a Qwen3.5 serve with no
  `--reasoning-parser`** (#8350): the answer is two serve flags, reporter-confirmed —
  `--reasoning-parser qwen3` to split, `--default-chat-template-kwargs
  '{"enable_thinking": false}'` to switch thinking off for every caller (served defaults
  merge under request kwargs, vllm v0.18.0 `engine/serving.py:815-824`; the parser reads
  `enable_thinking`, `qwen3_reasoning_parser.py:39-42`), plus the field the pin emits
  (`message.reasoning` at v0.18.0 — a client reading `reasoning_content` shows nothing).
  One serve command and no works-vs-fails control means the hybrid `--block-size` /
  prefix-caching page does **not** apply yet; the page's #8350 section lists what stays open.
- `quantized-sibling-score-gap.md` — "w8a8c8 scores N points below w4a8c8" on a
  GLM-5.x/DSA-sparse family model in a multi-machine PD deploy: the variable count in
  the reporter's own table and the within-model control row, the two formats'
  non-interchangeable C8 recipes (`GLM5.2.md:211,261` vs `:505,579`) and the 4-node e2e
  split, the lane's documented 2-3 point run-to-run floor plus the DCP-symmetry rule
  (#14378/#14322/#14011/#14329), the release's model-restricted validated list,
  strict/loose arithmetic as a truncation hypothesis, and where `reasoning_effort` is
  actually handled at the pin (upstream template kwarg; vllm-ascend only for DeepSeek-V4).
- `omni-audio-chat-degeneration.md` — an Omni/audio request over
  `/v1/chat/completions` that loops a phrase to `finish_reason: length` or returns
  garbage: the stage/owner table (audio encoder = `transformers` and absent from the
  partial export; only `self.visual` is downstream), the pinned doc's "OpenAI server
  doesn't support audio inputs" (`single_npu_audio.md:122`) against upstream's
  `audio_url` parsing, the pin's offline-only 10-token audio e2e, `prompt_tokens` as
  the placement test, and the offline-vs-online / eager / async / DP ladder.
- `moe-gating-topk-precision.md` — fluent but **wrong** answers on a MoE (A3B) serve:
  mixed-language CoT that keeps re-deciding, a final JSON that contradicts the prompt's
  own rule, normal tokens/s. A parser/`enable_thinking` change cannot move logits, so a
  correctness gate decides the lane; then the 310P selector's fused
  `torch_npu.npu_moe_gating_top_k_softmax` (`_310p/fused_moe/experts_selector.py:61`),
  the `torch.allclose` its e2e test computes and discards
  (`test_gating_top_k_softmax.py:36-37`), the fp32 probe
  (`scripts/moe-gating-topk-probe.py`), and #10892's reporter-confirmed resolution.
- `quantized-checkpoint-conversion.md` — per-layer loss tables and bad artifacts from
  msmodelslim conversion: the two gates (engine version plus the msit branch, which
  differs between doc pages), the published-checkpoint split test, layer-localization by
  shape invariance, and the ownership boundary (Ascend/msit, not vllm-ascend).
- `garbled-long-context-output.md` — long-input 乱码/garbage reports: grade the symptom,
  hand the reporter a feature-ablation matrix (DCP, sparse c8, dsa_cp, KV offload,
  length sweep), the `dcp_size == tp_size` / interleave constraints that make rows
  un-runnable, the tested envelope at the pinned release, and the #12905 / #14320
  precedents. When the report pastes **both** serve commands for a model the tree
  documents, its first section is the move: diff them against
  `docs/source/tutorials/models/<model>.md` (validated P and D commands, §9.1
  `Recommended Configurations`) and make that diff the opening answer — the worked
  Kimi-K2.6 v0.23.0rc1 table is there, with #13002 as the cost of skipping it —
  its five deviations are ranked rows, not proven causes.
- `constant-token-degeneracy.md` — the text-only sibling of the VL `!!!!` page:
  one repeated character/token as the entire answer, always or after hours of
  healthy traffic until restart. The id-level reading (`completion_tokens` vs the
  character count; `logprobs` ties or `-inf`), the persistence control the reporter
  already supplied ("新会话也坏 + 重启即好" retires prompt, sampling, parsers and
  gateway in one sentence), the always-on precedents with their owning statement
  (#1754's PP × graph known-issue line, #374's 100%-reproducible upstream-clean run),
  and the #3547 ladder — including why that serve's `ascend_scheduler_config`
  `enable_chunked_prefill: true` means `schedule()` delegates to upstream, so "the
  fork's chunked scheduling is buggy" is not available there.
- `first-generate-degenerate.md` — the first offline `LLM.generate()` is garbage
  (constant token id / immediate stop) or dies while call 2 is correct, on a
  hybrid GDN checkpoint of a VL-capable architecture at TP≥2: the response-field
  table that retires the numerics lanes, the source-verified `is_multimodal_model`
  vs `supports_mm_inputs` drift in `_dummy_run`
  (`model_runner_v1.py:2265,2431` at v0.18.0 vs upstream `gpu_model_runner.py:5152`,
  `_prepare_mm_inputs`, `_dummy_mm_kwargs`) with #14638/#8556/#14618 as the
  maintainer statement, the sleep-mode matrix and the reporter's "give the `LLM`
  some time" sync finding, the `expandable_segments` confound inside the
  `enable_sleep_mode=False` row (`platform.py:519-541`), and the version scope
  (0.18.0/0.23.0/0.26.0rc1 broken, nightly main clean).
- `vl-vision-tower-degeneracy.md` — VL image/video requests that degenerate into one
  repeated token, with the co-reported "very slow": the two-implementation gate
  (`USE_OPTIMIZED_MODEL`, v0.9.2rc1 `b5b7e0e` `models/__init__.py:23-33`), the head-dim
  padding branch (`MIN_PAD_SIZE`/`MAX_PAD_SIZE` 64/128, `enable_pad`, weight-name-keyed
  `load_weights`), the six-row ablation matrix, the #976 / #131 / #445 / #562 precedent
  family, why "`!` is token id 0" and a recalled `head_dim` must stay labelled, and a
  **third pin** (v0.17.0rc1) where the tower is weight-sharded by default: the
  `--mm-encoder-tp-mode data` row, its `supports_encoder_tp_data` gate, the
  `AscendMMEncoderAttention` consumer, and the two-screenshot reading of the
  TP=1-vs-TP=2 shape (#7680), and the **PD/EPD split** variant: the pinned 1P1D recipe
  (`pd_disaggregation_mooncake_single_node.md`) is Qwen2.5-VL-7B, so the entry point is
  the single-instance control, `usage.prompt_tokens` on a good and a bad run, and the
  prefix-caching flag the recipe disables on both sides — not the connector, and a
  **fourth pin** (v0.11.0rc0, Qwen3-VL-30B-A3B with a **video**): Qwen3-VL is registered
  unconditionally there (`USE_OPTIMIZED_MODEL` is not a row for it) and the *image tag* is
  rung 1 — #3806's identical body on `quay.io/ascend/vllm-ascend:latest` answered correctly
  (reporter-confirmed), while the pinned rc0 predates any fix.
- `early-eos-short-output.md` — "only tens of tokens / 输出被截断, worked before I added a
  flag": `finish_reason` decides the lane (length = budget, stop = early stop), an
  integer `stop_reason` names the *declared* stop id that ended it (and not why the
  model emitted it), then audit every reader of the toggled key, then the shipped
  support matrix — and check the checkpoint's `generation_config.json`
  `max_new_tokens`, which upstream renames to `max_tokens` and which makes "only a
  large `max_tokens` helps" literal. Worked
  cases #12253 (v0.17.0rc1) and #12278 (v0.22.1rc1), `pass_config.enable_sp` on a
  non-VL MoE w4a8 model: the key is honored, not dropped; the matrix excludes that
  class (at v0.22.1rc1 because non-VL **and** quantized — the VL + MoE cell changed,
  and so did `sp_threshold` → `sp_min_token_num`); upstream disables the flag itself
  at TP=1; FC1 is a complementary feature, not a rename; the key also selects the
  MoE all2all backend, so "just ignore it" is not safe; plus the ablation rows.
- `hybrid-block-size-corruption.md` — 乱码 on a hybrid GDN/Mamba model that follows
  `--block-size` (or prefix caching / `--enforce-eager`): the two-knob confound in the
  reporter's own command pair, the verified clamps at v0.20.2rc1 (`refresh_block_size`
  hybrid early-return, the upward page-size override in `patch_mamba_config`, the 16:1
  physical→logical block split), the five-row matrix whose prefix-caching row decides
  align-mode state block vs attention-side split, and the stop rule that cost #13853
  its session. Recorded case #13853 (Qwen3.5-0.8B, 2048 vs 512 + eager).
- `short-prompt-corruption.md` — the same symptom on a few-thousand-token prompt, **or the
  tens-of-tokens form on a dense model — including a text-only request to a VL-capable
  checkpoint where `prompt_tokens` already excludes the vision tower (Step 0b, #6533)**:
  read the
  pasted body's `max_tokens`/`completion_tokens` and whether its text answers the *sent*
  prompt before ranking anything (a sample cut at a 128-token `max_tokens` measures
  nothing, and a `reasoning` that works a different question is a request↔response
  pairing question — #15939); then test the
  trigger claim in the paste first (identical ids on both input routes refute an id-driven
  mechanism, and `Using max model len N` is the checkpoint's own number, so a mismatch
  points at the local `config.json` before any kernel — #8590), then the
  prompt shape (a repeated body at temperature 0 can collapse on its own), then the
  MoE/EP/quantisation ablation rows (EP off, FLASHCOMM1, prefix caching, `--enforce-eager`,
  `max-model-len`, both nodes' weights and image), and the gateway/model-echo caveat.
  `usage.prompt_tokens`, not the size of the pasted request, decides which of the two
  corruption pages you are in — and a **short** prompt with a tens-of-thousands-token
  completion that repeats sentences is neither: see `long-output-repetition.md`.
- `deepseek-v2-lite-mla-degeneracy.md` — DeepSeek-V2-Lite (MLA, `q_lora_rank: null`)
  at TP>1 answering a 4-token prompt with 50 copies of one token, nothing crashing:
  the reporter-confirmed refutation of the stack-pairing hypothesis (#905 bundle did
  not fix it), the checkpoint/prompt row that comes before any kernel read (validated
  row is `-Chat`; the tutorial's own sample output is already malformed prose),
  TP=1 / CPU-reference / torch_npu rows, the pin's forced `enforce_eager`
  (`platform.py:124`) that removes the graph row, and the two V2-Lite-only code
  surfaces with one grep each (#894, #374, #406).
- `long-output-repetition.md` — a few responses in a long accuracy run come back with
  sentences repeated ("5 条响应的句子重复", "偶现重复说话"), fluent and on-topic, on a
  documented model with PD + MTP + prefix caching and a large `max_tokens`. It starts
  with the reporter's own trial count when there is one: a rate (recorded #14085:
  29 repeats in 193 runs of **one** prompt) sizes the identity rerun as `N ≈ 3/p` and
  makes a two- or three-run clean check worthless, so ask for the count in the first
  answer. Then the identity row (`--max-num-seqs 1`, `temperature: 0`, same request
  `N` times), the tutorial P/D diff for the decode side, the three traps in reading
  that diff (`--enable-prefix-caching` is default-on at vllm v0.23.0; a Mooncake
  `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` hit is a log line, not a generation bound; the
  block-verify gate is off by default there), and the recorded #13719 diff table. A body
  that also says `repetition_penalty` 没有效果 under `--async-scheduling` gets its own
  one-grep check there: on an image older than v0.19.1rc1 the ids the processor reads are
  CPU placeholders, so the penalty was never applied (#7769).
- `batch-invariant-coverage.md` — the same request returns different text only when
  concurrency changes, with `VLLM_BATCH_INVARIANT=1` exported and a normally serving
  engine (recorded #14884: Qwen3-VL-30B-A3B, TP4 + `--enable-expert-parallel`, greedy,
  `--enforce-eager`, `--no-enable-prefix-caching`). Why the start-up log cannot say
  whether attention was patched (the INFO line is keyed on `HAS_TRITON or
  HAS_ASCENDC_BATCH_INVARIANT`), the two coverage branches at v0.18.0
  (`batch_invariant.py:96-124`), the ops no branch replaces (EP dispatch/combine,
  `npu_grouped_matmul`, FIA `_v2`, the VL vision tower), the project's own
  tested-model and e2e envelope that excludes VL and EP
  (`test_batch_invariant.py:160` has `enable_expert_parallel` commented out), the
  three causes a reviewer will otherwise pick (HCCL timing, sampler, custom-op
  fusion — all refuted) and the one-axis-per-launch matrix.
- `cross-platform-logprob-parity.md` — per-token logprobs (or logit deltas) differ from
  an H100/H200 golden on Ascend, deterministically and without signed bias: the
  pin-scoped reading of the FAQ determinism recipe and why its two `ATB_*` vars cannot
  move served numerics, the fp32 `log_softmax` parity that retires the logprob-math lane
  (`vllm/v1/sample/sampler.py:291-292` vs `vllm_ascend/sample/sampler.py:144,157`), the
  ranked causes with one check each (TP/topology first, prefill chunking, default-on
  fusion with `VLLM_BATCH_INVARIANT=1` as its only kill switch, VL preprocessing), and
  `scripts/logprob-bucket-fit.py` for the flat-`mean|Δ log p|/(1-p)` test that decides
  "one mechanism" against "this bucket needs a `top_logprobs` dump".
- `topk-logprob-negative-inf.md` — the *entries themselves* come back `-inf` on ordinary
  prompts while the same script on a GPU is finite (`logprobs=20`, Qwen3-8B, v0.9.1): why
  `-inf` is a mask/extreme question rather than the parity page's numeric floor (fp32
  `log_softmax` cannot turn a finite logit into `-inf`, and the bucket statistic is
  undefined), the id-pattern reading that splits "ascending lowest ids ⇒ a masked row" from
  "fixed high ids ⇒ the logits", the rank semantics that answer the reporter's "the `-inf`
  ranks differ, so it is not masking" (top-k ranks are list positions,
  `vllm/v1/engine/logprobs.py:182-192`), the v0.9.1 sampler facts (`raw_logprobs` before the
  filters, `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`-gated in-place `apply_top_k_top_p` vs
  upstream's scatter), and the two-launch ladder (`VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` echo,
  then `top_p=1.0`).
- `sampler-side-stream-corruption.md` — `temperature>0` output carrying `<pad>` /
  token id 0 / repeated or garbage tokens while greedy is byte-identical and the rate
  tracks concurrency: `random_sample()`'s `q` on `global_stream()`
  (`sample/sampler.py:18-41` at v0.20.2rc1 `367b8e6`), why `wait_stream` **does**
  order the fill before `div_` so the uncovered invariant is `q`'s cross-stream
  lifetime (precedent #13673; no `record_stream` anywhere in `vllm_ascend/`), what the
  `enable_async_exponential` row does and does not prove (it removes `random_sample()`
  from the path; zero cost belongs to pre-generating `q`, not to a host sync inside
  the sampler), the `VLLM_BATCH_INVARIANT=1` probe, the `test_sampler.py` coverage
  gap (no temperature>0 case on the default path), and the RFC #5304 → PR #6089
  attribution.
- `repeated-run-nondeterminism.md` — the engine compared with itself: the same prompt
  or body run 3–5×, lengths or contents differing, typically NPU "差异大" against a
  stable NV-GPU baseline at `temperature: 0`. The reporter's own control refutes any
  platform-independent mechanism, so "没有 seed / 采样没钉死" cannot lead when the
  control is stable — recorded #7180 ranked it first and the reporter's next comment
  pasted a greedy body (`temperature: 0, top_p: 1, top_k: 1, repetition_penalty: 1.1,
  max_tokens: 65536`, prompt ≈8782 tokens). Then the pattern of which runs differ
  (cold vs cache-warm prefill, all-runs, `finish_reason: length`), the documented
  determinism recipe at `docs/source/faqs.md:167-203` (FAQ #16, four env vars) before
  the beta `VLLM_BATCH_INVARIANT=1` program with its 910B/custom-op/coverage gates, and
  the amplification rule for long-CoT checkpoints (one flipped greedy token forks
  `max_tokens: 65536`). Precedents #9373, #5725, #221.
- `qwen3-next-gdn-numeric-path.md` — the hybrid linear-attention (Gated-DeltaNet) shape:
  a 19-token to few-hundred-token prompt on Qwen3-Next-80B-A3B answered as a digit run or a
  phrase loop while ordinary chat is fine, **and** the quantized `!` run whose owner is
  the installed triton-ascend wheel build (§*triton-ascend build*, #5606). What
  vllm-ascend replaces on that path at
  v0.11.0rc0 (the prefill `chunk_gated_delta_rule` swapped for an fp32 pure-torch
  re-implementation with a hard-coded x2 head expansion, while decode keeps the patched
  Triton kernel), the reporter-confirmed serve command that already rules out graph mode,
  MTP, EP and quantisation, the ablation matrix that separates model, platform and kernel
  ownership, and the #3689 / #3571 / #4057 precedents with what remains unestablished.
- `mtp-prefix-cache-310p.md` — 310P (Atlas 300I) Qwen3.5-class hybrid model with
  MTP **and** prefix caching on: the verified chain `--enable-prefix-caching` →
  `mamba_cache_mode="align"` (`vllm/model_executor/models/config.py:349-379`) → the
  310P-only config patch (`patch_mamba_config_310.py:84-85`) → `num_speculative_blocks`
  plus `accept_token_bias` in `preprocess_mamba`/the postprocess, where
  `_can_launch_triton_batch_memcpy() = not is_310p()` swaps the Triton copy for
  torch/numpy fallbacks (`patch_mamba_utils.py:22-23,108-183`), and the GDN op forces an
  fp16 SSM state (`_310p/ops/fla/gdn_310.py:134-151`). Coverage verdict from the two 310P
  PR-lane tests that never combine the features, the launch matrix (incl. the cache-hit vs
  cache-miss split), the stop rule that cost #14339 its answer (42 K characters of
  reasoning, no text on 17 turns), the re-check at the release pin `5cb98caa`, and the
  second recorded report #17338 (27B W8A8, TP4, `FULL_DECODE_ONLY`) where the corruption
  arrives at 70-100 % acceptance.
