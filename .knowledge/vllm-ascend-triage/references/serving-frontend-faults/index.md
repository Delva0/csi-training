# Faults owned by the serving frontend or the image's Python stack

## Scope

The report carries a Python traceback or error line that surfaces in the OpenAI
server process but whose frames are **only** `vllm/**` and the installed Python
packages: no `vllm_ascend/**` frame, no `npu_*` / `aclnn` / `EZ…` / `HCCL` text,
no plog. #7197 is the recorded case: `RuntimeError: Already borrowed` from a
transformers fast tokenizer while the chat preprocessor builds a tool parser.

The deliverable is an ownership verdict plus the cheapest client-side A/B. The
NPU stack cannot own a raise that never enters it — say that in the first
sentence, then give the mechanism at the pinned upstream files.

Not this topic, leave immediately:

- A Python `AttributeError`/`TypeError` from an Ascend replacement of an upstream
  class, or raised inside an upstream file *on a class vllm-ascend replaces* →
  `../upstream-downstream-drift/index.md`.
- The symptom is log text (a flag that prints nothing, a missing stage, a `404`
  route) rather than an exception → `../observability-and-logs/index.md`.
- The process aborts without a Python frame, or dies in a bundled native module
  (`corrupted size vs. prev_size`, `std::logic_error`) →
  `../image-dependency-faults/index.md`.
- The engine never becomes ready, or a worker raised under a `WorkerProc` prefix
  → `../startup-failures/index.md`.

## Rules

- **Count the frames per package before any Ascend reasoning.** Zero
  `vllm_ascend` frames plus a `vllm/**` or `site-packages/**` raise is an upstream
  or dependency fault; the model, the SoC and the dataset then enter only through
  the code path that was reached (a chat template, a parser flag, a request
  field), never as suspects. One `grep -rn "<symbol>" vllm_ascend/` that returns
  nothing is the positive form of that claim and belongs in the answer.
- **Ask the pinned vLLM tree whether it already knows the message**:
  `grep -rn "<exact message>" -g '*.py' vllm/`. A single hit that catches and
  retries the exception is the cheapest ownership proof there is — it shows the
  condition is expected upstream and names the one path that was protected. Do
  not search the `vllm_ascend` tree first.
- **Do not request plog, `npu-smi`, tiling or per-rank logs for this class.** The
  artefacts that decide it are the full serve command and its parser flags, the
  request body, the client concurrency, and whether *every* request fails or only
  some (sporadic = a race, every-request = read the message again).
- **An error response is not a crash.** Upstream chat preprocessing catches
  `(ValueError, TypeError, RuntimeError, jinja2.TemplateError)` and returns an
  error to the client (`vllm/entrypoints/openai/chat_completion/serving.py:313-316`
  at vllm v0.16.0), so the engine stays healthy while requests fail. Separate the
  two in the answer; a reporter who says "the service failed" may only mean the
  requests did.
- **One `history` grep, then stop.** These races are usually absent from the
  prior-month corpus (no `Already borrowed` hit at all across the recorded
  history); say "no precedent" and cite the upstream issue/comment instead of
  hunting for a matching title.

## Pages

- `tokenizer-already-borrowed.md` — the worked #7197 case: `RuntimeError: Already
  borrowed` at `Hermes2ProToolParser.__init__` inside `_preprocess_chat`, the two
  threads that share one HF fast tokenizer in the API-server process, the single
  retry upstream already ships for this exception, the serial-request / retry
  ladder, and the transformers-version evidence gap.
