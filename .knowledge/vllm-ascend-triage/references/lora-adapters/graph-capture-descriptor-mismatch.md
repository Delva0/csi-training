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

Reported again as #7150 (ms-swift `swift rollout`, Qwen2.5-7B-Instruct, TP2,
`{"cudagraph_mode": "FULL_DECODE_ONLY"}`, 910B2, CANN 8.3.RC2) — and there **no**
request carries an adapter at all: the command passes `--vllm_enable_lora true` but
no `--adapters`, so every rollout request is a base-model request. So the asymmetry
to establish is a client question first: which side of the traffic sends
`model: <adapter>` / a `lora_request`? A launcher can enable the capability and then
never use it.

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

**Re-verified unchanged at vllm-ascend v0.14.0rc1 (`52d4acf`) + vllm v0.14.0
(`b17039b`), the #7150 ref:** fork `model_runner_v1.py:1935` (`activate_lora` in
`_dummy_run`'s signature, never read), `:1952` (config-const `has_lora`), `:1954`
and `:2010` (dispatch), `:1394` (live-batch `has_lora` at runtime), `:2828`
(`capture_model` delegates to `GPUModelRunner.capture_model`); upstream
`force_has_lora` at `gpu_model_runner.py:2963`/`:2985-2989`, passed as
`activate_lora=` from `_capture_cudagraphs` (`:4887` loop, `:4918,4927`), guard
armed `:4789` … disarmed `:4846`, wrapper guard `acl_graph.py:126` (debug line
`%s,%s` at `:124`). So the defect spans at least v0.13.0rc2 → v0.14.0rc1.

**Grade the paste's own decode step before spending a DEBUG round trip.** The other
three descriptor fields come from the scheduled batch, so the dumped
`SchedulerOutput` names them: `total_num_scheduled_tokens` / `num_scheduled_tokens` give
`num_tokens`, `scheduled_new_reqs` + `scheduled_cached_reqs` give `num_reqs`, and one
live request on a decode step is `uniform=True`. Compare that with the
`cudagraph_capture_sizes` list the same paste prints in its `compilation_config`
(default `[1, 2, 4]` + steps of 8 and 16 — `vllm/config/vllm.py:1148-1158` at v0.14.0
`b17039b`; newer pins add a `performance_mode="interactivity"` branch that is
contiguous `1..N`, v0.23.0 `:1704-1706`, which still covers small counts) and with
`cudagraph_specialize_lora`. A token count inside the list then leaves `has_lora` as the
only field startup could have missed, and the DEBUG line above becomes confirmation
instead of triage: recorded #7150's paste carried `total_num_scheduled_tokens=1` with one
cached request against a list starting at `1`, so the size-gap variant was excluded from
the report alone. A token count *outside* the list (a capture-size gap under
`--async-scheduling`, or a non-uniform mixed batch) sends the same guard to the
size/shape variant instead — state which one the paste is in before asking for logs, so
the DEBUG ask is not spent on a field the paste already settled.

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
3. **`"cudagraph_specialize_lora": false` does _not_ keep a graph here** — it moves
   the failure from the request to engine start, because both sides of the ledger
   read the *same* flag. `initialize_cudagraph_keys` advertises `lora_cases=[True]`
   when specialization is off, while the fork's `_dummy_run` computes
   `lora_config and cudagraph_specialize_lora` = **False**; `dispatch` then finds no
   key for `has_lora=False`, returns `NONE`
   (`cudagraph_dispatcher.py:182-183`), and the capture call's own mode
   (`cudagraph_runtime_mode=FULL`/`PIECEWISE`, passed at upstream
   `gpu_model_runner.py:4928`) trips the fork's
   `ValueError: Aclgraph runtime mode mismatch at dummy_run. Expected NONE, but got
   FULL.` (`model_runner_v1.py:2026`). Source-derived at v0.14.0rc1, not
   reporter-tested — re-check `rg -n 'Aclgraph runtime mode mismatch at dummy_run'`
   before quoting it at another ref. Check the direction too: a fork that produces
   the case the keys *lack* crashes on the other request type.
4. **What does keep a graph**, cheapest first: (a) if the traffic carries no
   adapter, **do not enable LoRA** — `lora_config is None` ⇒ `lora_cases=[False]`
   and both sides compute `has_lora=False`, so capture and replay agree (#7150's
   own command allows this); (b) send every request through an adapter; (c) the
   one-line fix (`_dummy_run` should take the per-case flag, i.e. upstream's
   `force_has_lora`) — one line, so do not propose a patch the reporter must carry.
5. File it as a defect with the traceback, the `Capturing ACL graphs …` startup
   case list and the failing request's model name.

## Version scope and open items

Verified at vllm-ascend v0.13.0rc2 + vllm v0.13.0, and again at vllm-ascend
v0.14.0rc1 + vllm v0.14.0 (line numbers in both sections above). Before quoting on another
ref, re-grep `activate_lora|force_has_lora` in `model_runner_v1.py` and
`cudagraph_specialize_lora` in upstream `config/compilation.py` — either may
have moved, and the same guard now protects any descriptor field, so a
non-LoRA variant (a batch shape or uniform/non-uniform switch that startup never
captured) is the same mechanism with a different missing field. Not verified
here: whether #6279's traceback shows the `Capturing a aclgraph on (FULL,
BatchDescriptor(…))` debug line, which would name the missing descriptor
directly. That line is **not** gated on a separate flag:
`CUDAGraphOptions.debug_log_enable` defaults `True`
(`vllm/compilation/cuda_graph.py:134`) and is forced per partition, so
`VLLM_LOGGING_LEVEL=DEBUG` is all it takes — one line per capture (first subgraph),
carrying the runtime descriptor, e.g. `has_lora=False`. Neither #6279 nor #7150
pastes a DEBUG log, so this stays the check to request rather than a
reporter-confirmed fact.

Precedent search wants the guard's sentence plus the descriptor field, not the mode
name: `-b 'capturing detected at an inappropriate time|cudagraph_specialize_lora'`
returns 6 ids with #6279 among them, while
`validate_cudagraph_capturing_enabled|FULL_DECODE_ONLY` returns 35 ids that are
mostly launch lines carrying `FULL_DECODE_ONLY`.
