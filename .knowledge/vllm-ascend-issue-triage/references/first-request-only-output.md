# First-request-only output: empty, id-0 fill, or crash on the first decode

Use when the report says the **first** request / first `llm.generate()` is empty, truncated,
garbage or kills the engine, and **later** requests are normal. That "then it works" half is the
signature: it separates this class from steady-state quality reports
([sampling-and-output-quality.md](sampling-and-output-quality.md)) and from the graph-mode support
boundaries ([full-decode-only-graph-mode.md](full-decode-only-graph-mode.md)). On the crash branch
also exclude the plugin/upstream method-contract seams that surface at request 1 in a hybrid-Mamba
P/D run ([pd-disaggregation.md](pd-disaggregation.md)): there the wrong-typed return recurs, while
this class needs call 2 to be normal. Recorded reports:
vllm-ascend#9355 (serve, bf16, 0.18.0), #15560 (offline `LLM`, w8a8, 0.18.0; the reporter
re-tested 0.23.0 and 0.26.0rc1), and #14638 (the crash branch).

## Read the first-call shape before reading code

Three outcomes, three different owners:

| first-call shape in the report | read it as | go to |
|---|---|---|
| `content: ""`, `completion_tokens: 1`, `finish_reason: stop` | the first sampled token was an EOS/stop token, not a dropped string | branch 3 / open candidate (#9355) |
| short junk run, or a full run of token id **0** (detokenizes to `!`/punctuation) | placeholder/garbage ids written by the first decode | branches 2 and 3 |
| `EngineCore crashed during ACL Graph replay` on the first request | a separate warm-up mismatch, not the graph replay itself | branch 1 |

Ask for `token_ids` and `finish_reason` of call 1 and call 2 rather than the text: id 0 renders as
punctuation, and an immediate EOS renders as "empty". A reporter's "changing the prompt/temperature
fixed it" is confounded whenever the retry was simply the second call (#9355's follow-up is exactly
that) — the discriminating variable is call order, not the prompt.

## Branch 1 — `--language-model-only` warm-up mismatch (verified root cause of the first-call crash)

Symptom tail (this exact shape identifies the branch; the frames above it are ordinary):

```text
vllm_ascend/worker/model_runner_v1.py  _model_forward -> run_model()
  vllm_ascend/compilation/acl_graph.py  __call__ -> self.runnable(*args, **kwargs)
  vllm/model_executor/models/qwen3_5.py forward  -> self.language_model.model(...)
  vllm/model_executor/models/qwen3_next.py forward
  torch/_dynamo/utils.py  call_size -> return x.size(i)
AttributeError: 'NoneType' object has no attribute 'size'
```

Mechanism, stated in PR #8556 (merged to `main` 2026-04-27): with `--language-model-only`,
`is_multimodal_model` stays `True` (the architecture still is) while `supports_mm_inputs` becomes
`False`. `_dummy_run` chose the warm-up input path from `is_multimodal_model` (`input_ids = None`,
`inputs_embeds = tensor`) while `_preprocess` chose the token-ids path, so torch.compile/dynamo hit
`None.size()` **on the first inference request**.

- `v0.18.0`: `vllm_ascend/worker/model_runner_v1.py:2431`
  `if self.is_multimodal_model and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:`
  (the branch is `_dummy_run`; note the Python precedence — the `or` binds outside the `and`).
- `main` (2026-11-01): `vllm_ascend/worker/v1/model_runner.py:4506-4513` uses
  `self.supports_mm_inputs` (plus `not requires_raw_input_tokens`).
- Backport to `releases/v0.18.0` is **PR #14618, open** at the snapshot, so a 0.18.0 report keeps
  the crash. Issue #14638 is the same defect ("错误使用 is_multimodal_model", fixed on main by
  #8556, not in the 0.18.0 branch).
- Workaround to offer: drop `language_model_only` / `--language-model-only`; otherwise the one-line
  condition swap. Any multimodal-architecture checkpoint (Qwen3.5/3.6, Qwen3-VL, ...) loaded
  language-only is in scope.

Do **not** fold this into the acl_graph story: the graph only appears in the stack because the
compile/graph wrapper is what calls the mismatched forward. The reporter of #15560 naming #14638 as
a prerequisite is right about this frame and says nothing about the wrong-token half.

## Branch 2 — `enable_sleep_mode` (thread-verified trigger; mechanism still open)

A contributor control matrix inside #15560 (real `Qwen3.6-27B-w8a8`, 2×910B3, image v0.18.0,
`temperature=0`, `max_tokens=100`, `seed=42`, same prompt; no explicit `sleep()`/`wake_up()` call):

| vllm-ascend | ACL graph | `enable_sleep_mode` | first `generate()` | second |
|---|---|---|---|---|
| 0.18.0 | `FULL_DECODE_ONLY` | True | EngineCore crash during ACL Graph replay (branch 1's `None.size()`) | not reached |
| 0.18.0 | `NONE` | True | wrong early stop after 3 tokens (`'\n\n<think>'`, ids `[271, 248068, 248046]`) | normal 100 tokens |
| 0.18.0 | `NONE` | False | normal | normal, identical ids |
| nightly main | `FULL_DECODE_ONLY` | True | normal | normal, identical ids |

Reporter's own retest on v0.23.0 and v0.26.0rc1: first call = 100× token id **0**, second normal —
the wrong-output form outlives 0.18.0; the crash cell above was not reported on those versions.

- "`enable_sleep_mode` is an important trigger for the bad first-request state, `FULL_DECODE_ONLY`
  amplifies it" is the contributor's conclusion (contributor comment, not a maintainer fix).
- "Fixed on main at `f6983134` (vllm 0.27.1)" is a single-revision observation, **not bisected**;
  #12255 (GDN/metadata) and #12071 (async H2D metadata race) are named candidates only.
- Workarounds that are claimable: disable `enable_sleep_mode` (thread-verified), discard the first
  call as warm-up (reporter), serve online instead of offline `LLM` (reporter), `--tensor-parallel-size 1`
  (reporter) — the last two are configuration observations, not root causes.
- In-tree surface at `v0.18.0`: `vllm_ascend/device_allocator/camem.py` (CANN-mem pluggable
  allocator; `:66` "Failed to import vllm_ascend_C: ... Sleep mode will be disabled",
  `:102-109 use_memory_pool_with_allocator`, `:226 use_memory_pool(tag)`),
  `vllm_ascend/platform.py:105 is_sleep_mode_available` / `:522`, and the two pools in
  `vllm_ascend/worker/worker.py` (`:430-433` tag `weights`, `:517-519` tag `kv_cache`, with
  `assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."`),
  plus `csrc/camem_allocator.cpp`. Nothing there yet explains "first request only"; treat the
  allocator/pool path as the place to instrument, not as the answer.

## Branch 3 — diff the serve line against the tag's own tutorial, then the drafter's graph gate

At `v0.18.0` the Qwen3.5-27B tutorial (`docs/source/tutorials/models/Qwen3.5-27B.md`, the model the
`qwen3.5` label and the `Qwen3.6-27B` title both resolve to) prescribes:

```
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
--no-enable-prefix-caching --async-scheduling --quantization ascend --max-model-len 133000
```

The recorded #9355 line differs in: bf16 weights (doc: w8a8-mtp), `--max-model-len 262144`,
`method: qwen3_next_mtp`, `num_speculative_tokens: 2`, explicit `cudagraph_capture_sizes`,
`disable_padded_drafter_batch: false`, no `enforce_eager`, no `--async-scheduling`. Name the delta,
do not rank it — except for the one gate that is source-verified:

`vllm_ascend/spec_decode/eagle_proposer.py:136-142` at `v0.18.0`:

```python
self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager
if self.method == "mtp":
    self.use_cuda_graph = (self.use_cuda_graph
        and not self.use_async_scheduling
        and not self.speculative_config.disable_padded_drafter_batch)
```

So for MTP the **drafter** is ACL-graph captured only when none of `enforce_eager`,
`--async-scheduling`, `disable_padded_drafter_batch` stop it. The tutorial's `enforce_eager: true`
pins the drafter eager while the target still runs `FULL_DECODE_ONLY`; the #9355 line leaves the
drafter graph-captured — a drafter that has just been captured succeeding on later requests fits a
first-request-only wrong token. Candidate, cheap to test: re-run the first call with
`enforce_eager: true` inside `--speculative_config` (or `--async-scheduling`, or
`disable_padded_drafter_batch: true`), then with `--enforce-eager`, then with spec decode off.

## Anti-detours (each verified; these are the wrong turns the recorded runs were taking)

1. **The MTP method *name* is not the defect.** At upstream `v0.18.0`
   (`vllm/config/speculative.py:33-48`, `:354-358`) every MTP method — `qwen3_next_mtp`,
   `qwen3_5_mtp`, ... — is deprecated and normalized: `self.method = "mtp"` with a warning, and the
   architecture comes from the target's `hf_config_override`. A serve line with `qwen3_next_mtp`
   therefore reaches the same code as the doc's `qwen3_5_mtp`; say so and move on, and read the
   logged spec config rather than the pasted flag.
2. **`Replaying aclgraph` is `logger.info_once`** (`vllm_ascend/compilation/acl_graph.py:196` at
   `v0.18.0`): once per worker process by construction. Its appearance during the first request
   does not show that later requests skip graphs, and its absence is not evidence graphs are off.
3. **`speculative_config.enforce_eager` is a drafter switch, not the target's graph mode.** Upstream
   `vllm/config/compilation.py:60-67`: `FULL_DECODE_ONLY = (FULL, NONE)`, and `mixed_mode()` returns
   `NONE`, so `EagleProposer.initialize_cudagraph_keys` (`v1/spec_decode/eagle.py:362-377`) picks
   `NONE` anyway. In the plugin the only route is `eagle_proposer.py:136` above.
4. **The two log lines pin the report exactly**: `model_runner_v1.py:1213` "Running batch with
   cudagraph_mode ..." and `acl_graph.py:196` "Replaying aclgraph" both resolve to `v0.18.0` — no
   need to reconcile a `dev<N>+g<sha>` build string for that report. The reporter's batch line
   `num_tokens=3, num_reqs=1, uniform=True` is `1 + num_speculative_tokens(2)`, i.e. a uniform MTP
   decode step.

## Evidence boundaries

- Branch 1 is the only verified root cause here: code at both refs, PR #8556 merged, backport
  #14618 open, issue #14638 (Chinese) with the identical traceback.
- Branch 2 rests on one contributor comment and one reporter retest inside #15560; the allocator
  mechanism is unexplained, the main fix is not bisected.
- #9355 has no maintainer root cause (open, labels `bug`/`triaged`/`qwen3.5`/`multimodal-understanding`;
  replies are "please take a look" and "which image?"). Its first-token-EOS is a candidate shared
  with #15560's wrong-token cell, not an established one.
- Next checks, in order: `language_model_only` present? → branch 1. `enable_sleep_mode` present? →
  branch 2. Else the drafter-gate matrix above. A worker-side `plog` or per-call logits excerpt
  separates "wrong token" from "wrong sampler input".
