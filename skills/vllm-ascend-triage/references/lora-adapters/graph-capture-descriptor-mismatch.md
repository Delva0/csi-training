# A LoRA serve that dies on the first request with "graph capturing at an inappropriate time"

## Symptom

Engine up and serving, then one request kills the workers:

```
(EngineCore_DP0 pid=480) ERROR … [patch_core.py:70] RuntimeError: Worker failed with error
'CUDA graph capturing detected at an inappropriate time. This operation is
currently disabled.', please check the stack trace above for the root cause
```

Recorded #6279 (v0.13.0rc2, Qwen3-32B with **multi-LoRA** and graph mode on
910B): the request that killed it selected the **base** model; the report does not
say whether adapter requests were tried, so the asymmetry below is the first
thing to establish rather than something the paste proves. Neither `patch_core.py:70` nor the `Worker failed with
error` text is the cause — the first is vllm-ascend's re-raise, the second is
upstream `v1/executor/multiproc_executor.py` relaying the worker's `str(e)`.

## Owner: upstream's capture guard, armed only during startup capture

`vllm/compilation/monitor.py:48-57` (vllm v0.13.0, `72506c9`) raises that
`RuntimeError` from `validate_cudagraph_capturing_enabled()` whenever the
module-level flag is `False`. Exactly one place toggles it:
`vllm/v1/worker/gpu_model_runner.py:4520` sets `True` immediately before the
startup capture, `:4577` sets `False` right after it, with the comment "any
unexpected cudagraph capturing will be detected and raise an error after here".

So the message never means "the hardware refused a graph"; it means **a graph
capture was attempted at request time**. Two wrappers call the guard:
upstream `vllm/compilation/cuda_graph.py:242` and downstream
`vllm_ascend/compilation/acl_graph.py:131`. Both capture **lazily** — for a
`BatchDescriptor` not yet in their `concrete_*_entries` dict they create an
entry and capture it (`acl_graph.py:110-136`). A request therefore reaches the
guard only when the dispatcher advertised a key that startup capture never
built.

## The key: `BatchDescriptor` and `has_lora`

`vllm/forward_context.py:28-48`: the dispatch key is
`(num_tokens, num_reqs, uniform, has_lora)`; `relax_for_mixed_batch_cudagraphs()`
keeps `has_lora`, so PIECEWISE does not hide it.

- Keys advertised: `vllm/v1/cudagraph_dispatcher.py:90-141`
  `initialize_cudagraph_keys()` builds `product(cudagraph_capture_sizes,
  lora_cases)` where `lora_cases = [True, False]` for a LoRA config with
  `cudagraph_specialize_lora` (default `True`, `vllm/config/compilation.py:518`),
  `[True]` when LoRA without specialization, `[False]` otherwise.
- Cases captured: `gpu_model_runner.capture_model` builds the *same* list
  (`:4526-4538`) and `_capture_cudagraphs` loops
  `for num_tokens, activate_lora in compilation_cases` (`:4618`), passing
  `activate_lora=activate_lora` into `_dummy_run` (`:4649`, `:4658`), which
  forwards it as `force_has_lora=` into `_determine_batch_execution_and_padding`
  (`:4081`, `:2794-2798`). The case is what makes the descriptor real.

**The drift (source-verified at vllm-ascend v0.13.0rc2 `a17059a` + vllm
v0.13.0 `72506c9`):** the downstream fork's `_dummy_run` ignores that flag.
`vllm_ascend/worker/model_runner_v1.py:2107` accepts `activate_lora` in the
signature and never reads it; `:2124` computes
`has_lora = True if self.lora_config and self.compilation_config.cudagraph_specialize_lora else False`
and dispatches with it (`:2126`, and again `:2182`). Its `capture_model`
(`:3041-3052`) delegates to upstream `GPUModelRunner.capture_model(self)`, but
`self._dummy_run` resolves over the MRO to the fork, so **both** capture
iterations write the `has_lora=True` descriptor and no `has_lora=False` graph
exists. At runtime the fork computes the descriptor from the live batch —
`has_lora = len(self.input_batch.lora_id_to_lora_request) > 0`
(`:607`, `:1537`; also `spec_decode/eagle_proposer.py:466`,
`spec_decode/mtp_proposer.py:279`). A base-model request therefore gets
`False`, the dispatcher matches the advertised FULL (or relaxed PIECEWISE) key,
the wrapper has no entry, it captures — and the guard raises. Adapter requests
hit `True`, which was captured, and work. Confidence: reporter-confirmed
symptom; mechanism source-verified at those two commits, not in a rerun.

One call reproduces the whole audit:
`bash <skill_root>/scripts/lora-graph-descriptor-audit.sh <vllm-ascend-root> <vllm-root>`
— it labels each `has_lora` producer `LIVE-BATCH` vs `CONFIG-CONST`, reports
whether `activate_lora` is ever consumed, and prints the guard and the
`lora_cases` on both sides. Its part 4 also settles the ownership question when
the image's vllm predates `monitor.py`: then the guard does not exist in that
tree and the pasted line comes from a newer vllm than the one being blamed.

## Action ladder

1. **`--enforce-eager`** (or `--compilation-config '{"cudagraph_mode": "NONE"}'`)
   on the same traffic: if the crash disappears, capture is the trigger and the
   rest of the ladder is about keeping graphs, not about the adapter.
2. **Send only adapter requests** (`model: <lora-name>`) on the graph-mode
   server, then only base-model requests. The asymmetry is the signature; it
   also tells the reporter which half of their traffic is at risk.
3. **`--compilation-config '{"cudagraph_specialize_lora": false}'`** is the
   workaround that keeps a graph for adapter traffic at this ref: keys become
   `has_lora=True` only, capture writes `has_lora=True`, and a base-model batch
   now falls through `dispatch` to `CUDAGraphMode.NONE`
   (`cudagraph_dispatcher.py:182`) instead of lazily capturing. Cost:
   base-model requests run eager. Check the direction first — a fork that
   produces the case the keys *lack* crashes on the other request type.
4. File it as a defect with the traceback, the `Capturing ACL graphs …` startup
   case list and the failing request's model name; the fix is one line in
   `_dummy_run` (use the per-case flag), so do not propose a patch the reporter
   must carry.

## Version scope and open items

Verified at vllm-ascend v0.13.0rc2 + vllm v0.13.0. Before quoting on another
ref, re-grep `activate_lora|force_has_lora` in `model_runner_v1.py` and
`cudagraph_specialize_lora` in upstream `config/compilation.py` — either may
have moved, and the same guard now protects any descriptor field, so a
non-LoRA variant (a batch shape or uniform/non-uniform switch that startup never
captured) is the same mechanism with a different missing field. Not verified
here: whether #6279's traceback shows the `Capturing a aclgraph on (FULL,
BatchDescriptor(…))` debug line, which would name the missing descriptor
directly (`acl_graph.py:126-131`, needs `VLLM_LOGGING_LEVEL=DEBUG`).
