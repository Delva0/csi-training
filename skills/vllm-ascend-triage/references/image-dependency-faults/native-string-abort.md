# Native null-string abort at engine init (`std::logic_error`)

Symptom, verbatim (`#8745` v0.13.0rc3 + HDK 25.5.2, DeepSeek-V4-Flash; `#10943`
v0.20.2rc1 + CANN 8.2.RC1, `nvidia/LocateAnything-3B` through
`TransformersMultiModalForCausalLM`):

```text
terminate called after throwing an instance of 'std::logic_error'
  what():  basic_string::_S_construct null not valid
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

No Python traceback, no per-rank worker prefix, no NPU/collective text, no plog
error, no device dump. The last application lines are engine-start only:
`Platform plugin ascend is activated` → `Registered model loader …
ModelNetLoaderElastic` (#8745), or `Initializing a V1 LLM engine (v…) with
config: …` (`vllm/v1/engine/core.py:108-113` at the pinned vllm v0.20.2,
`bc150f5`, #10943). `Failed core proc(s): {}` is a generic parent-side message —
the empty set is not evidence about which rank died.

## What the text proves, and what it does not

- Proves: libstdc++ threw from `basic_string::_S_construct` (via
  `std::__throw_logic_error`) because native code built a `std::string` from a
  **null `char*`** — a null `getenv`, a null SoC/device-name or path pointer, in
  whichever shared object owns that frame. `terminate` rather than a Python
  `RuntimeError` means the throw happened where nothing catches it: a static
  initialiser, a destructor, an `atexit` path, or a native worker thread (pybind11
  translates a caught `std::exception` into a Python exception, so an escaped
  `std::logic_error` is by construction outside a bound-call frame).
- Does **not** prove anything about the checkpoint, its architecture, its
  processor or its config. Same two lines, two unrelated models, two different
  images: #8745 (DeepSeek-V4-Flash, whose support vllm-ascend does not question)
  and #10943 (a custom VL checkpoint). In #10943 the reporter later confirmed that
  the **unmodified** upstream `nvidia/LocateAnything-3B` crashes identically, that
  `transformers.AutoModel.from_pretrained(trust_remote_code=True)` loads it fine,
  and that none of eight model-side changes moved the crash point (arch
  registration, LFS `tokenizer.json`, optional `decord`/`lmdb`/`cv2` imports,
  `_get_num_multimodal_tokens()`, nulled `rope_scaling`/`template`/
  `transformers_version`, `_attn_implementation` `magi`→`sdpa`, `_name_or_path`,
  V0 vs V1). Do not rank "the architecture is unregistered" or "the model config
  is malformed" for this text: registration and config already succeeded — that
  is why the config printed at all.
- Phase: the abort sits between that config log (`vllm/v1/engine/core.py:108-113`)
  and the model load, i.e. inside worker creation `executor_class(vllm_config)`,
  which at v0.20.2rc1 reaches `vllm_ascend/worker/worker.py:108`
  (`_register_atb_extensions()`), `:112` (`check_ascend_device_type()`), `:256`
  `_init_device` (incl. `:267` `import torch_npu._inductor`), `:306`
  `init_device_properties_triton()`. Any of those is a candidate region; the text
  cannot pick one.

## Steps

1. **Decide whether the model is involved at all — one run, same host, same
   image.** Serve a stock text checkpoint with the same flags:
   `vllm serve Qwen/Qwen2.5-0.5B-Instruct --enforce-eager --max-model-len 4096`.
   Identical abort → image/host native fault; this is the cheapest discriminator
   and it decides the whole tree. It serves → the difference is in the
   engine-init *inputs* the custom model supplies (`max_model_len`, dtype,
   multimodal limits, the `trust_remote_code` import), and that difference is what
   the next artefacts must isolate.
2. **Probe the native layer without vLLM** (three one-liners on the reporter's
   host; any bare abort with the same text answers the question):
   - `python -c "import torch, torch_npu; print(torch.npu.is_available(),
     torch.npu.device_count()); print(torch.npu.get_device_name(0));
     print(torch_npu.npu.get_soc_version())"` — the same call vllm-ascend makes in
     `vllm_ascend/platform.py:215-216` and `vllm_ascend/utils.py:848-861`, both of
     which run during worker init.
   - `python -c "import torch, torch_npu, triton;
     print(triton.runtime.driver.active.utils.get_device_properties(0))"` — this is
     `init_device_properties_triton` (`vllm_ascend/ops/triton/triton_utils.py:46-52`,
     called at `worker.py:306`) with vLLM removed.
   - `python -c "import torch_npu.op_plugin.atb._atb_ops"` — the import whose
     registration `worker.py:108` triggers.
3. **Name the shared object** (the decisive artefact; the paste cannot contain
   it):
   `gdb --batch -ex 'break std::__throw_logic_error' -ex run -ex 'bt 25' -ex
   'thread apply all bt 8' --args vllm serve <model> <the same flags>`.
   The innermost non-libstdc++ frame names the owner (`libtorch_npu.so`,
   `libtriton.so`, `libascend*.so`, ATB, the vllm-ascend extension). Keep the full
   `thread apply all bt`: if the throwing thread is not the main one, that is the
   finding.
   Cheap localiser when `gdb` is unavailable: re-run with
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` (`vllm/envs.py:129,1103-1104`, consumed at
   `vllm/v1/engine/llm_engine.py:148`) so the engine core runs in the foreground
   process, and read which step's log line is last — ordering evidence, not the
   owner's name.
4. **Version correspondence**: image tag ↔ vllm-ascend ref ↔ torch_npu ↔ CANN/HDK
   (`../startup-failures/environment-correspondence.md`). The two known instances
   sit on different pairs (v0.13.0rc3/HDK 25.5.2 and v0.20.2rc1/CANN 8.2.RC1), so
   a partially installed or mismatched toolkit stays a live candidate, and a
   later image that works is a valid answer for the reporter.

## Do not spend the session on these

- **Grepping the exported `cann/*`, `Ascend/pytorch`, `triton-ascend` trees for
  `std::string(getenv(`, `_S_construct` or "null string" sites.** The throw is in
  libstdc++, not in repo text, and those exports are truncated content dumps
  (`SOURCES.json`: `Ascend/llvm-project` 1115/146342 files, `cann/hccl`,
  `cann/ge`, `cann/opbase` `unavailable`). The recorded #10943 run spent six
  tool calls there, found one unrelated `std::getenv("HOME")` guard, and shipped
  no answer inside the 16384-token cap.
- **Re-testing model-side config fixes.** They are already listed above and all
  eight were tried without moving the crash point.
- **Reading the custom model's modeling/processing files.** They explain nothing
  about a throw inside a bundled `.so`; the AutoModel load the reporter already ran
  is the proof they are fine.

## Answer shape for this class

- Direct answer: this is a native library aborting on a null string during engine
  start, not a bad checkpoint and not an unsupported architecture — the model never
  reaches the device. Say that the two model-side questions the reporter will ask
  ("is my config wrong", "is the arch registered") are already answered no.
- Cheapest first: the stock-model run, the three probes, then `gdb`/`catch throw`
  output to name the `.so`; a working later image if one is known for that
  hardware.
- What to send back: the stock-model result, the three probe outputs, the gdb
  backtrace, `pip show vllm vllm-ascend torch_npu`, `cat
  /usr/local/Ascend/ascend-toolkit/latest/version.cfg`, `npu-smi info`, and the
  child's stderr from plugin activation to the abort.
- Scope: image tag and ref; label the null-pointer mechanism as source-verified
  libstdc++ behaviour and the owning `.so` as open until the backtrace arrives.
