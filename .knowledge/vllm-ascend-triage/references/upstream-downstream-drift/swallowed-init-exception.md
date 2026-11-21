# A missing `__init__` attribute is a swallowed first exception

## Scope

`AttributeError: '<AscendRunner…>' object has no attribute '<x>'` where a grep of the
pinned upstream `__init__` shows `<x>` assigned **unconditionally** (even if only to
`None`), and the downstream subclass calls `super().__init__()` inside a
`with <wrapper>():` block.

Then the attribute is not the diagnosis: upstream's constructor raised, vllm-ascend's
`@contextmanager` **caught the exception and returned**, and `__init__` continued
half-built. Recorded case: vllm-ascend #5571, image `v0.13.0rc1`, a multimodal
deployment (`AttributeError: 'NPUModelRunner' object has no attribute 'mm_budget'` at
upstream `profile_run`). The two readings reporters arrive at — "vllm-ascend dropped
support for multimodal" and "upstream renamed the attribute" — are both wrong, and each
sends the next session into a different, innocent source tree.

## The two calls that settle it

1. **Grep the upstream assignment, not the missing name.** At vllm v0.13.0
   (`72506c9`) `self.mm_budget` is assigned at
   `vllm/v1/worker/gpu_model_runner.py:565-573` —
   `MultiModalBudget(...) if self.supports_mm_inputs else None` — so a **completed**
   `GPUModelRunner.__init__` always binds it; the reads are `:612` (`reset_mm_cache`),
   `:1115` (`_dummy_mm_kwargs`), `:3946` (`_get_mm_dummy_batch`) and `:4434`
   (`profile_run`, the frame in the report). An unconditional assignment means the
   `AttributeError` cannot be a rename, a moved initialiser or "the Ascend runner is
   not a multimodal runner".
   The same message also separates the neighbouring failures: had the value merely
   been `None`, `:4434-4435` raises `AssertionError` from `assert mm_budget is not
   None`, **not** `AttributeError`. Read which of the two the reporter got.
2. **Read the `super().__init__` call site and its wrapper.** At vllm-ascend
   v0.13.0rc1 (`1b5d5ab`) `NPUModelRunner.__init__`
   (`vllm_ascend/worker/model_runner_v1.py:191-193`) calls
   `super().__init__(vllm_config, device)` inside `with _torch_cuda_wrapper():`, and
   `_torch_cuda_wrapper` (`:3453-3488`) is a `@contextmanager` whose `try:` at
   `:3467` encloses the **`yield` at `:3475`**, with `except Exception:` at `:3476`
   and a `finally:` at `:3482`. A generator context manager that catches the
   exception thrown into it and then *returns* suppresses that exception:
   `contextlib.__exit__` sees `StopIteration` instead of the original value and
   returns `True`. Verified with 6 lines, no NPU and no vllm needed:

   ```python
   from contextlib import contextmanager
   @contextmanager
   def w():
       try:
           yield
       except Exception:
           print("swallowed")
   with w():
       raise ValueError("boom")      # prints "swallowed", then execution continues
   ```

   So **every** `Exception` raised by upstream `GPUModelRunner.__init__` is dropped,
   and `NPUModelRunner.__init__` keeps assigning its own attributes afterwards. The
   first thing that reads an attribute the constructor had not reached yet raises the
   attribute error the reporter pasted.

   Two side effects of the same helper, which are **not** the diagnosis but do
   constrain later theories: the `finally:` runs on the success path too and leaves
   `torch.cuda.Event`/`Stream`/`default_stream`/`current_stream` bound to the
   placeholder classes (so a post-init CUDA-stream use is a silent no-op), and
   `:3485` (`torch.cuda.Stream = torch.cuda.Stream`) is dead text — do not read this
   helper as "restore what was there".

## Where the raise happened (and why the paste cannot say)

`self.mm_budget = (MultiModalBudget(...) …)` is a single statement, so an exception
inside `MultiModalBudget.__init__` leaves the attribute unbound. The reporter's run got
past weight load and KV-cache init into `profile_run`, which the statements before
`:565` are what set up — so the raise is at that statement (its `MultiModalBudget(...)`
call) far more often than earlier in the constructor. `MultiModalBudget`
(`vllm/v1/worker/utils.py:25`) builds the model's HF **processor** through
`mm_registry.get_mm_limits_per_prompt` / `get_max_tokens_per_item_by_modality`
(`vllm/v1/worker/utils.py:44-55` → `vllm/multimodal/registry.py:145-169`,
`create_processor` + `MultiModalProfiler.get_mm_max_tokens`) and then
`compute_mm_encoder_budget`. Candidate raises are therefore processor/transformers
version mismatches, a modality whose max-token table cannot be built, or an
`mm_processor_kwargs`/limit value the processor rejects — **model-specific, and
recoverable only from the exception the wrapper destroyed.** Say that; do not invent
the inner error, and do not blame the model's architecture.

## What to tell the reporter (ranked, cheapest first)

1. **Recover the first exception** by neutralising the swallow in their image —
   insert `raise` as the first statement of the `except Exception:` branch at
   `vllm_ascend/worker/model_runner_v1.py:3476` (or narrow the `try` to the
   `torch.cuda.*` assignments only), restart the same serve command, and read the
   traceback that appears before the attribute error. This is the only step that
   identifies the cause; everything below is a workaround.
2. **No-edit smoke test:** `--skip-mm-profiling` (upstream flag at vllm v0.13.0,
   `vllm/engine/arg_utils.py:475,969`, field `vllm/config/multimodal.py:131`,
   consumed at `gpu_model_runner.py:4426-4432`) makes `profile_run` skip the
   `mm_budget` read, so startup proceeds. It does **not** fix anything: the budget was
   never built, so the first multimodal *request* is the real test, and peak encoder
   memory is then the reporter's to estimate (the field's own docstring says so).
   Never present this as the fix.
3. **Confirm the pair before ranking upstream:** `pip show vllm` /
   `git -C /vllm-workspace/vllm rev-parse HEAD`. A paste whose line matches
   `gpu_model_runner.py:4434` is the v0.13.0 tree, which is what the tag claims — the
   `mm_budget` init is older than the release, so a "newer upstream moved it" theory
   is already refuted by step 1.
4. **File it against the wrapper, not the model.** The defect is "`_torch_cuda_wrapper`
   converts any constructor failure into a missing-attribute error", so the fix is a
   `raise` after the placeholder assignments (or `try/except` around the assignments
   only). Ask the maintainers for that plus the real inner error from step 1.

## Discriminators against the neighbouring pages

| Shape | Owner / page |
|---|---|
| Attribute assigned in upstream `__init__`; downstream calls `super().__init__()` in a swallowing wrapper | this page — the message is a *masked first exception* |
| Attribute is a property/method and the message says `Did you mean: '<method>'` | `stale-module-binding.md` |
| Attribute belongs to the *caller's* signature (`attention_type`, a new keyword argument) | `patched-forward-api-drift.md` |
| Class replaced by name in an OOT registry, so upstream code runs on the Ascend class | `oot-class-registration-drift.md` |
| Value is only wrong (`None` where an int is expected) or a capability flag is off | not a drift: read the gate/flag, the message is `AssertionError`/`TypeError` |

## Scope and recheck conditions

- Verified at vllm-ascend `v0.13.0rc1` (`1b5d5ab`) × vllm v0.13.0 (`72506c9`). The
  wrapper is an implementation detail: re-grep per version
  (`rg -n 'def _torch_cuda_wrapper|with _torch_cuda_wrapper' vllm_ascend/`).
- The **v2** runner does not have this hazard at the same ref: its
  `torch_cuda_wrapper` (`vllm_ascend/worker/v2/utils.py:7-33`) uses `try/finally`
  with no `except`, and `VLLM_USE_V2_MODEL_RUNNER` defaults to `0`
  (`vllm/envs.py:246,1565-1566`), so the legacy `NPUModelRunner` is the default path.
  If the reporter set that env var, this masking cannot have happened — re-derive.
- Unrecoverable from the report: which statement inside `MultiModalBudget.__init__`
  raised, and the model's modality set. Ask for the re-run of step 1 plus the
  `vllm serve` line; the fix is model/processor-specific.
- Delivery: step 1 already supports a verdict sentence ("the attribute is assigned
  unconditionally upstream, so your constructor was interrupted — the real error was
  swallowed"). Emit it with the tool call that starts step 2. The recorded #5571 run
  verified exactly these two facts by turn 6, then spent six more turns re-checking
  them against the rename hypothesis, emitted 226 characters of message text in
  total, and was cut with **no answer**.
