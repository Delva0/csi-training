# The first `LLM.generate()` is degenerate, the second is correct

Scope: an **offline** `LLM(...)` run (the same engine a serve's first request
reaches) on a hybrid GDN checkpoint of a *multimodal-capable* architecture
(Qwen3.6/Qwen3.5 — the stack shows `vllm/model_executor/models/qwen3_5.py` →
`qwen3_next.py`), `tensor_parallel_size ≥ 2`, where call 1 is garbage or dies and
call 2 is correct. Two independent owners are in this shape and one field of the
response decides which: a **crash/degenerate first call with a text-only run on a
VL-capable checkpoint** (Step 2, `is_multimodal_model` drift) and a
**first-request state/sync problem that `enable_sleep_mode=True` exposes** (Step 3).
Inherit the rules in `index.md`; the empty-answer sibling (one blank token,
`completion_tokens: 1`) is `../spec-decode-mtp/empty-first-response.md`, and the
repeated-single-token page for *image* requests is
`vl-vision-tower-degeneracy.md`.

## Step 1 — read the response, not the flags

| Observation | Reading |
|---|---|
| A **constant token id** repeated to `finish_reason: "length"` — e.g. `text='!!!!…'`, `token_ids=[0]*100` (v0.23.0 and v0.26.0rc1, same script) | Greedy argmax over an all-equal/degenerate logits (or embedding) vector: id 0 wins the tie. This **retires** the weights, w8a8/NZ, tokenizer, attention-numerics, block-size and prefix-cache lanes — none of them produce a constant-tie vector |
| `text='\n\n<think>'`, `token_ids=[271, 248068, 248046]`, `finish_reason: "stop"` on call 1 (v0.18.0, graph mode `NONE`) | The same family seen earlier: degenerate logits that emit `<|im_end|>` immediately. Not a second bug |
| `AttributeError: 'NoneType' object has no attribute 'size'` at `torch/_dynamo/utils.py::call_size`, stack `model_runner_v1._model_forward` → `compilation/acl_graph.py` → `qwen3_5.py:773` → `qwen3_next.py:1358` | **Step 2**, not an ACL-graph defect: the graph wrapper is only the compiled call chain |
| call 1 and call 2 return identical `token_ids` | Not this page: no first-request defect |

`'!' == token id 0` is *recalled* for the Qwen tokenizers, not verified in this
snapshot (the `huggingface/transformers` export carries no Qwen tokenizer file);
give the reporter `AutoTokenizer.from_pretrained(...).convert_tokens_to_ids('!')`
instead of asserting it.

## Step 2 — the crash is the text-only-multimodal drift, not ACL graph

Verified at vllm-ascend v0.18.0 `e18643f` + vllm v0.18.0 `bcf2be9`:

- Upstream `vllm/v1/worker/gpu_model_runner.py:460` computes
  `self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)`,
  and `vllm/multimodal/registry.py:103-111` returns **False** when no modality is
  enabled — its own docstring says "effectively running in text-only mode".
- The model-level flag is a *different* question:
  `vllm/config/model.py:1424` `is_multimodal_model = multimodal_config is not None`,
  and `:600-621` builds a `MultiModalConfig` for every multimodal-capable
  architecture — `language_model_only` is stored **inside** it (`:299`, `:367`),
  it does not make the config `None`. So on a text-only launch of a VL-capable
  checkpoint `is_multimodal_model` is **True** while `supports_mm_inputs` is **False**.
- The downstream fork's dummy/capture path branches on the model-level flag:
  `vllm_ascend/worker/model_runner_v1.py:2265` (`_dummy_run`) at `:2431` —
  `if self.is_multimodal_model and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:`
  → `input_ids = None; inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]`
  (`self.is_multimodal_model` is set at `:280`). Upstream's equivalent branch
  (`gpu_model_runner.py:5152`) gates on `supports_mm_inputs`, calls
  `_prepare_mm_inputs()` (`:2959-2968`, which keeps `input_ids` when
  `model.requires_raw_input_tokens`) **and** passes `**_dummy_mm_kwargs(num_reqs)`.
  The fork has neither the gate nor the mm kwargs.
- Result: the compiled `Qwen3_5`/`Qwen3Next` forward is called with
  `input_ids=None` and no `pixel_values`/mm kwargs; dynamo's guard evaluates
  `x.size(i)` on `None` → the `AttributeError` above, on the first request because
  that is when the size case is captured/dispatched.

Known issue: **#14638** "v0.18.0 branch has an unfixed incorrect use of
`is_multimodal_model`" carries this exact stack; its body states the fix is on
main (PR **#8556**) and **not merged into the v0.18.0 branch**, with backport PR
**#14618** opened. The reporter confirmed on 2026-09-07 that with the
`is_multimodal_model` problem fixed the `AttributeError` no longer appears. Both
are issue-text claims — the snapshot carries no PR data — so label the fix
attribution and keep the line numbers above as the verified half.

Checks, one call each:

1. `rg -n 'is_multimodal_model|supports_mm_inputs' vllm_ascend/worker/model_runner_v1.py`
   in the image/report's tree — a `_dummy_run` still branching on
   `is_multimodal_model` predates the fix.
2. `python -c` print of `model_config.is_multimodal_model` and
   `MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config)` on the reporter's
   checkpoint: `True`/`False` is the drift, `True`/`True` means real images are
   enabled and the mismatch has to be found elsewhere.
3. Grading rows: dropping `language_model_only=True` / `--language-model-only`, and
   `--enforce-eager`. The first changes which branch is *right*; the second only
   avoids capture, so a still-degenerate answer moves the case to Step 3.

## Step 3 — the degenerate output is first-request state, and sleep mode exposes it

Recorded matrix (reporter-run, 2×910B3, Qwen3.6-27B-w8a8, TP=2, v0.18.0, same
script both times):

| `cudagraph_mode` | `enable_sleep_mode` | call 1 | call 2 |
|---|---|---|---|
| `FULL_DECODE_ONLY` | `True` | EngineCore crash during ACL-graph replay (Step 2 signature) | not reached |
| `NONE` | `True` | wrong: `'\n\n<think>'`, 3-token stop | normal |
| `NONE` | `False` | normal | normal, identical ids |
| `FULL_DECODE_ONLY` (nightly main, vllm 0.27.1) | `True` | normal | normal, identical ids |

So graphs **amplify** (wrong answer → crash) while sleep mode is the condition the
clean row removes. The reporter's own later diagnosis is the useful half: *"This
seems some sync problem, if you start a `LLM` object and give some time, then no
error will occur."* That is a startup sync / async-H2D race on first-request
metadata, not numerics — and it is cheaper to test than any source walk:

1. `llm = LLM(...)`; `time.sleep(N)`; then the real calls — or one throwaway
   `llm.generate(PROMPT, SamplingParams(max_tokens=1))` first. Call 1 then matching
   call 2 makes the first-request state the owner **and** the warm-up the mitigation.
2. `llm.sleep(level=1); llm.wake_up()` once after construction — separates "the
   sleep-mode pool/tag path is stale at first use" from a plain timing race.
3. TP=1 on the same script (the reporter reports it clean) — narrows it to the
   per-rank first-request path.

**Separate the allocator confound before blaming the pool.** At v0.18.0,
`vllm_ascend/platform.py:519-541` injects `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
only when `enable_sleep_mode` is **False** ("NOTE: We should not set this
environment variable in RL (sleep mode) scenarios"), so the `sleep=False` row
changes two variables at once. The discriminant row is `sleep=True` **with**
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` exported: the code appends only
when the key is absent, so an explicit value is honored. Clean there ⇒ the
allocator config owns the artifact, not the sleep-mode pool.

`platform.py:105` `is_sleep_mode_available()` returns `True` on the Ascend
platform, so the finding is about the first-request state under that mode, not
about an unsupported flag.

## Step 4 — version scope and what to hand back

- Reproduces: v0.18.0, v0.23.0, v0.26.0rc1 (reporter-run, same script, firmware
  25.3.rc1). Clean: nightly main at vllm 0.27.1 (reporter-run). No released tag in
  this snapshot can be named as fixed — say "post-0.26.0rc1, not yet in a release
  you can install", not "fixed".
- Candidate fixes named by the reporter (#12255 GDN/metadata, #12071 async H2D
  metadata race) are **unverified** here; present them as the maintainers' list to
  bisect, not as the cause.
- Deliverable: the workaround (`enable_sleep_mode=False`, or a warm-up request when
  sleep mode is required), the backport status of #14638 on the reporter's branch,
  and a regression assertion of `run1.token_ids == run2.token_ids` — not
  "non-empty", because the failure is a well-formed 100-token answer.
- "`vllm serve` works" is not a mechanism: the chat template yields a different,
  longer prompt. Ask for both response bodies before ranking anything on it.

## Stop rule

With a constant-token-id first answer, do **not** open the GDN kernels,
`refresh_block_size`/`patch_mamba_config` geometry, prefix caching or
`mamba_cache_mode` align. The tokens are degenerate, so the question is which
buffer was read and when — `index.md`'s "same server, weights and flags" rule
already says the logits, not the kernel, are the evidence.
