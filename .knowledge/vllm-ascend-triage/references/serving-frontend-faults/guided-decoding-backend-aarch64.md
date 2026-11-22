# A JSON-output request 500s on aarch64: the guided-decoding fallback loop

## Signature

`POST /v1/chat/completions` that carries `response_format` returns
`500 Internal Server Error` (`EngineCore encountered an issue. See stack trace
(above) for the root cause.`), while plain chat on the same server answers
normally and the engine never restarts. The log prints **three** fallback
warnings, then a traceback whose frames are all upstream:

```text
WARNING … [__init__.py:48] xgrammar is only supported on x86 CPUs. Falling back to use outlines instead.
WARNING … [__init__.py:84] xgrammar module cannot be imported successfully. Falling back to use outlines instead.
WARNING … [__init__.py:91] outlines does not support json object. Falling back to use xgrammar instead.
…
  File "vllm/model_executor/guided_decoding/__init__.py", line 117, in …
  File "vllm/model_executor/guided_decoding/xgrammar_decoding.py", line 40, in from_guided_params
  File "vllm/model_executor/guided_decoding/xgrammar_decoding.py", line 89, in get_tokenizer_data
    vocab_type = xgr.VocabType.RAW
```

There is **no** `vllm_ascend/**` frame and no NPU text: this is upstream's
guided-decoding dispatch, so the NPU stack, the SoC and the model are not
suspects. Recorded shape: `#1592`, vllm-ascend v0.7.3.post1 image on a Kylin
aarch64 host, 910B, `Qwen2.5-32B`. The reporter's own summary — "正常测试不会报错,
只有要求 JSON 输出的 prompt 才崩" — is the discriminator, not a coincidence.

## Mechanism (vllm v0.7.3 @ `ed6e907`)

`maybe_backend_fallback` (`vllm/model_executor/guided_decoding/__init__.py:22-94`)
is a sequence of **independent** `if` blocks, so a later block can undo an earlier
one:

1. `:47-50` — `current_platform.get_cpu_architecture() is not CpuArchEnum.X86`
   rewrites `xgrammar` → `outlines`, with warning `:48`.
   `get_cpu_architecture` maps `platform.machine()` (`platforms/interface.py:282-292`),
   and vllm-ascend does not override it (`rg -n get_cpu_architecture vllm_ascend/`
   is empty at `c69ceac`), so aarch64/arm64 hits it.
2. `:81-86` — `elif not xgr_installed:` warns (`:84`) and sets `outlines` again.
   `xgr_installed` is false because the requirement is arch-marked:
   `requirements-common.txt:24  xgrammar == 0.1.11; platform_machine == "x86_64"`
   — on aarch64 upstream never installs it.
3. `:89-93` — the *last* block is `if backend == "outlines" and json_object is not
   None: … backend = "xgrammar"`, with warning `:91`. This is what closes the loop:
   the only request shape that reaches it is `response_format:
   {"type": "json_object"}` (or `guided_json_object`), because `json`/`regex`/
   `grammar` each have their own earlier branch.
4. Dispatch (`:114-119`) then imports the xgrammar backend unconditionally.
   `xgrammar_decoding.py:15-21` catches the `ImportError` with a bare `pass`, so the
   name `xgr` is left **unbound** and `:89` raises `NameError: name 'xgr' is not
   defined` — inside the logits-processor construction, before any sampling.

## Decisive checks

1. **The request body.** A prompt that merely *asks* for JSON in words does not
   crash; only the `response_format` field does. Ask for both bodies (the reporter
   already showed the working one).
2. `python -c "import xgrammar"` in the *serving* process's environment —
   `ModuleNotFoundError` at this pin is the report restated.
3. `platform.machine()`. If it says `x86_64`, this page is the wrong story: the arch
   gate never fired, so read which of `:53-79` fired instead.

## Escape, cheapest first

1. `--guided-decoding-backend lm-format-enforcer` at serve time. LMFE is only left
   for `grammar` or an unsupported *json-schema* feature
   (`__init__.py:26-40`), and bare `json_object` maps to `JsonSchemaParser(None)`
   (`lm_format_enforcer_decoding.py:44-46`). A per-request default does not defeat
   it: `entrypoints/openai/protocol.py:455-465` stamps
   `guided_decoding_backend="xgrammar"` only for `response_format.type ==
   "json_schema"`; for `json_object` the field stays `None` and the engine-level
   default applies (`guided_decoding.backend = guided_decoding.backend or
   default_guided_backend` in `engine/async_llm_engine.py`).
2. `--guided-decoding-backend outlines` does **not** help, although
   `outlines_decoding.py:127` has its own `json_object` branch: the `:89-93`
   rewrite preempts it for every request at this pin.
3. Making xgrammar importable is version-bound. The x86_64 marker is upstream's
   statement about the version *it* pins; the snapshot's `mlc-ai/xgrammar` tree
   declares cibuildwheel `aarch64` archs, so newer xgrammar does publish aarch64
   wheels — but do not promise it for the pinned `0.1.11` without checking that
   release's assets, and note that upgrading it past vllm's pin is unsupported.

## Scope

Verified at vllm v0.7.3 (`ed6e907`) + vllm-ascend v0.7.3.post1 (`c69ceac`). The
rewrite is version-specific: later vLLM releases raise a clear error instead of
looping, and the reporter's v0.8.5rc1 / v0.9.0rc2 runs were clean — re-grep
`maybe_backend_fallback` at the reported ref before reusing the chain, and answer
inside the pinned version. Its **anchors are also version-bound**: at vllm v0.13.0
`guided_decoding_backend` no longer exists anywhere in `vllm/`
(`rg -ln guided_decoding_backend vllm/` is empty), the launch flag is
`--structured-outputs-config '{"backend": …}'`, `_backend` is resolved in
`v1/engine/input_processor.py:258-352`, and a request that selects a backend is
rejected — so `entrypoints/openai/protocol.py:455-465` / `openai/processor.py` are
v0.7.3 paths and a miss on them is not evidence about a newer tree
(`../accuracy-benchmarks/structured-output-garble.md`). Two delivery notes: the
ownership sentence (upstream frontend, proven by the empty `rg` in `vllm_ascend/`)
belongs in the first paragraph, and the V1 feature matrix (history `#414`) marking
Guided Decoding "🟢 Functional" is a *doc* gap for aarch64 + `json_object` at this
pin, worth one line rather than a section.
