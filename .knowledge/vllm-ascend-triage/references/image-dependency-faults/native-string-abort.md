# Native null-string abort at engine init (`std::logic_error`)

Symptom, verbatim (`#8745` v0.13.0rc3 + HDK 25.5.2, DeepSeek-V4-Flash; `#10943`
v0.20.2rc1 + CANN 8.2.RC1, `nvidia/LocateAnything-3B` through
`TransformersMultiModalForCausalLM`; `#11300` v0.22.1rc1 `5f6faa0` + vllm v0.22.1, a
DeepSeek-V4-Flash-w8a8-mtp tutorial deployment with `--data-parallel-size` > 1):

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

**Release-scoped correction**: the "no Python traceback / no per-rank worker prefix"
shape above is true for #8745 and #10943 but **not** at v0.22.1rc1, where #11300's
paste carries `(EngineCore_DP1 pid=1036)` prefixes and a full stack —
`multiproc_executor.py:200 _init_executor` → `:747 wait_for_ready` →
`Exception: WorkerProc initialization failed due to an exception in a background
process. See stack trace for root cause.`, then the child's
`multiprocessing/process.py:314 _bootstrap` → `vllm_ascend/patch/platform/patch_balance_schedule.py:650`.
Every one of those frames is the *reporting* path: `wait_for_ready` re-raises the
child's death and the balance patch only re-dispatches `run_engine_core`. The root
cause is still the two `terminate`/`what():` lines the child printed above them, so
do not chase `wait_for_ready`, the balance patch or `--model-loader-extra-config`
(`Registered model loader …RForkModelLoader` is printed unconditionally by
`vllm_ascend/__init__.py:56-63` and is merely the last line before the child dies).

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

0. **Before any probe: was this command the one for these cards?** The model pages
   tab their deployment section per SoC and the tabs are **not** interchangeable.
   At v0.22.1rc1 `docs/source/tutorials/models/DeepSeek-V4-Flash.md` §5.1, the A2
   tab (`:146-191`) is `--data-parallel-size 1 --tensor-parallel-size 8
   --max-model-len 133120 --max-num-batched-tokens 8192 --max-num-seqs 32
   --no-enable-prefix-caching` plus `"enable_dsa_cp": true` (`:189`), while the A3
   tab (`:194-238`) is dp4/tp4, `--max-model-len 1048576`,
   `--max-num-batched-tokens 10240`, `--api-server-count 1`, `--max-num-seqs 64` and
   **no** `enable_dsa_cp`; the image tags differ too (`:51` `…|<ver>|-a3` versus
   `:94` the unsuffixed A2 tag). #11300 is the worked instance: the reporter's image
   was `vllm-ascend:v0.22.1rc1-a3`, the command they pasted is **the A2 block**
   (dp1/tp8, `--max-model-len 133120`, `"enable_dsa_cp": true`), the log's
   `EngineCore_DP0`/`DP1`/`DP2` lines imply dp ≥ 3, and the thread closed with the
   reporter's own "已解决，A3机器跑了A2的命令" — the A2 block on the A3 machine. So
   this paste disagrees with itself on two axes at once (command vs log dp, command
   vs image tag), which is exactly what one grep settles. That is why the tab check
   is step 0 and not a footnote: it costs one command
   (`uv run --no-project python scripts/recipe-tab-diff.py --page <page>
   --section 5.1 --command <the reporter's script>`), it is decided entirely in the
   pinned tree, and it is the only hypothesis in this family the reporter can
   settle in one line. Two guard rails, both from the same thread:
   - **The log's process names are the real readout of what ran.** `EngineCore_DP<N>`
     exists only when `data_parallel_size > 1` (vllm v0.22.1 `0decac0`,
     `vllm/v1/engine/utils.py:136,151`; `vllm/v1/engine/core.py:1104-1110`), so a
     paste carrying `EngineCore_DP1`/`DP2` beside `--data-parallel-size 1` is two
     different launches. #11300 pasted exactly that pair, and the DP-prefixed shape
     matches the **A3** dp4 tab — never assume the pasted command produced the
     pasted log.
   - **The tab is an eliminator, not a closure.** The same reporter pasted the
     identical two lines again 15 minutes later labelled "A3命令报错". If the
     matching tab still aborts, the recipe lane is closed with that one datum and
     steps 1-5 below are live; say so, and leave the null-pointer mechanism open.
   - **When the reporter's own script IS the tab's block, the lane is closed.**
     #8745's reporter later posted the command they actually ran: the env head
     `USE_MULTI_BLOCK_POOL=1 OMP_PROC_BIND=false OMP_NUM_THREADS=10
     PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ACL_OP_INIT_MODE=1
     TRITON_ALL_BLOCKS_PARALLEL=1` plus `vllm serve <…>/DeepSeek-V4-Flash-w8a8-mtp
     --max_model_len 65536 --max-num-batched-tokens 8192 --max-num-seqs 16
     --data-parallel-size 1 --tensor-parallel-size 8 --enable-expert-parallel
     --quantization ascend --block-size 128 --async-scheduling --additional-config
     '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": true}'
     --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp"}'
     --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, described by
     them as "A2单机" on the unsuffixed `v0.13.0rc3` image. That is the **A2 tab
     verbatim** (`docs/source/tutorials/DeepSeek-V4.md:136-167` at v0.13.0rc3;
     their edits are the model path, `--port`, `--served-model-name` and an added
     `--chat-template`). They posted it as the script they referenced, with no
     resolution in the thread, so the command behind the abort is the matching-tab
     block: the tab lane is closed for #8745 and an "export the A3 variables"
     answer is wrong there. Say which comment and date you are citing.
     `scripts/recipe-tab-diff.py --page <page> --command <script>` prints that
     verdict from the pinned tree alone (worked: 4 tabs, closest A2, distance 3).
   - **Env exports do not identify the tab at this ref.** At v0.13.0rc3 both tabs
     export `OMP_PROC_BIND`, `OMP_NUM_THREADS`, `PYTORCH_NPU_ALLOC_CONF`,
     `ACL_OP_INIT_MODE` and `USE_MULTI_BLOCK_POOL=1` (A2 `:142-147`, A3 `:176-184`);
     the A2↔A3 env delta is only `ASCEND_A3_ENABLE=1`, `HCCL_BUFFSIZE=1024`,
     `VLLM_ASCEND_ENABLE_FUSED_MC2=1`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, beside
     `--data-parallel-size 2` vs `1`, `--port 8005` vs `8006` and the
     `--served-model-name`. So "they are missing the other tab's exports" is not a
     hypothesis an env list can support; the identifying flags are dp + the image
     suffix.
   - **Read a tab block whole.** A line window cut into it (`sed -n '145,208p'`)
     starts inside the block and hides its export head — that is how the recorded
     #8745 answer came to attribute `USE_MULTI_BLOCK_POOL`/`OMP_*` to the A3 tab
     only and to offer them as the fix. Anchor on the `{tab-item}` marker, or
     `rg -n '^export|vllm serve' <page>`, or run the script above.
1. **Decide whether the model is involved at all — one run, same host, same
   image.** Serve a stock text checkpoint with the same flags:
   `vllm serve Qwen/Qwen2.5-0.5B-Instruct --enforce-eager --max-model-len 4096`.
   Identical abort → image/host native fault; this is the cheapest discriminator
   and it decides the whole tree. It serves → the difference is in the
   engine-init *inputs* the custom model supplies (`max_model_len`, dtype,
   multimodal limits, the `trust_remote_code` import), and that difference is what
   the next artefacts must isolate.
2. **Bisect the recipe's own env exports — `USE_MULTI_BLOCK_POOL` first.** At
   vllm v0.13.0 `AsyncEngineArgs.add_cli_args` calls `load_general_plugins()`
   (`vllm/engine/arg_utils.py:1998`), which prints every `Registered model loader`
   line (`vllm/model_executor/model_loader/__init__.py:109`), and *then*
   `current_platform.pre_register_and_update(parser)` (`:2014`) →
   `vllm_ascend/platform.py:124-129` `adapt_patch(is_global_patch=True)` →
   `vllm_ascend/patch/platform/__init__.py:28-34`, which imports six extra modules
   whenever `envs.USE_MULTI_BLOCK_POOL` is on (`patch_kv_cache_coordinator`,
   `patch_kv_cache_utils`, `patch_model_config`, `patch_speculative_config`,
   `patch_transformers`, `patch_vllm_config`; the same flag switches
   `distributed/__init__.py:25` to `mooncake_state_connector` and
   `ops/fused_moe/moe_mlp.py:223` to the fused grouped-matmul/swiglu path). On
   this abort shape the last printed line is followed by exactly that import set,
   so unsetting one export at a time (`USE_MULTI_BLOCK_POOL=0` first) costs one
   restart, needs no model change and no `gdb`, and moves six modules in or out of
   the window. Pair it with `VLLM_ENABLE_V1_MULTIPROCESSING=0` (step 4) to
   also learn whether the throw is in the engine-core child or the parent. The
   mechanism stays open: those imports are Python, so a native static initialiser
   reached through one of them is a hypothesis, not a finding.
3. **Probe the native layer without vLLM** (three one-liners on the reporter's
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
4. **Name the shared object** (the decisive artefact; the paste cannot contain
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
5. **Version correspondence**: image tag ↔ vllm-ascend ref ↔ torch_npu ↔ CANN/HDK
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
  no answer.
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
- Cheapest first: the tab/image-tag match against the reporter's card generation
  (step 0 — quote the dp/tp/`max-model-len`/`enable_dsa_cp` diff, and the
  `EngineCore_DP<N>` prefix as the check that the pasted command is the one that
  ran), then the stock-model run, the one-env bisect of the recipe (step 2), the
  three probes, then `gdb`/`catch throw` output to name the `.so`; a working later
  image if one is known for that hardware.
- If the reporter has already pasted a launch script, score it against the pinned
  tabs in the first message and say which tab it is and whether that closes the
  lane (A2 on A2 hardware closes it); do not answer with an export list from the
  other tab.
- What to send back: the stock-model result, the three probe outputs, the gdb
  backtrace, `pip show vllm vllm-ascend torch_npu`, `cat
  /usr/local/Ascend/ascend-toolkit/latest/version.cfg`, `npu-smi info`, and the
  child's stderr from plugin activation to the abort.
- Scope: image tag and ref; label the null-pointer mechanism as source-verified
  libstdc++ behaviour and the owning `.so` as open until the backtrace arrives.
  Where a reporter has already reported a resolution ("跑了 A2 的命令"), cite it as
  their claim with its date, note whether a later paste contradicts it, and do not
  upgrade it to "this abort always means the wrong tab" — #11300's own follow-up
  did.
