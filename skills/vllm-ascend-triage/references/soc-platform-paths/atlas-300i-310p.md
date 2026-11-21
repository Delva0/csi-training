# Atlas 300I / 310P: what the platform changes, and what the docs admit

Verified at vllm-ascend **v0.11.0rc3** (`b6d63bbd`) with vllm **v0.11.0**
(`b8b302cd`); the `-310P` image is built from `Dockerfile.310p.openEuler`
(`FROM quay.io/ascend/cann:8.3.rc2-310p-openeuler24.03-py3.11:18`,
`SOC_VERSION=ASCEND310P3:51`, `VLLM_TAG=v0.11.0:37`). Re-grep per version: the
gate list and line numbers move, the shape does not.

For "there is no 310P artefact for my machine" (arm64 host, an arm64 tag that pulls
amd64, an aarch64 wheel) the owner is the image pipeline, not this page:
`../image-dependency-faults/image-arch-and-tag-availability.md`.

## The documented status (quote it, then move on)

- `docs/source/faqs.md:18`: "[Experimental] Atlas 300I inference series (Atlas
  300I Duo). Currently for 310I Duo, the **stable version is vllm-ascend
  v0.10.0rc1**."
- `docs/source/tutorials/single_node_300i.md:3-7`: 300I is "currently
  experimental", "only supports eager mode and the float16 data type", known
  issues `#3316` / `#2795`, "You can use v0.10.0rc1 version first."
- **The same file contradicts itself**, and this is the trap in #4812: its
  Qwen2.5-VL-3B tab (`:126-131`) ships exactly the failing command
  (`vllm serve … --tensor-parallel-size 1 --enforce-eager --dtype float16
  --compilation-config '{"custom_ops":["none","+rms_norm","+rotary_embedding"]}'`)
  while the image line at `:17` still says `v0.10.0rc1-310p`. When a reporter
  pastes a command that is verbatim from a tutorial, say so — it removes "user
  error" from the ranking and it is a documentation finding in its own right.
- Environment the tutorial requires and reporters routinely omit (`:43-49`):
  `VLLM_USE_MODELSCOPE=True` and
  `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`. That value is the project's
  300I fragmentation workaround, not general tuning, and from v0.18.0 on it also
  switches off vllm-ascend's own `expandable_segments:True` default
  (`vllm_ascend/platform.py:518-528` at v0.18.0; no such injection exists at
  v0.11.0rc3) — state both effects
  (`../runtime-memory/hbm-growth-and-oom.md`, step 3).

A VL or OCR checkpoint that **serves and answers garbage** on this SoC is the
other shape of the same platform question — support is a per-model doc statement
and document parsing is a pipeline: `310p-vl-ocr-garbage-output.md`.

## An image that carries triton-ascend cannot import the worker at all

Verified at vllm-ascend **v0.14.0rc1** (`52d4acfa` — the release whose notes say
"310P support is back now", `docs/source/user_guide/release_notes.md:9`, PR #5776)
with vllm **v0.14.1** (`d7de043d`), image `…:v0.14.0rc1-310p`. A different pin
from the rest of this page, because the 310P lane was re-opened there; re-grep
for the next 310P release.

Shape (#6309): `EngineCore failed to start` in the same second the worker class is
resolved — before any config check and before any weight load:
`resolve_obj_by_qualname` → `vllm_ascend/_310p/worker_310p.py:21` →
`worker/model_runner_v1.py:96` → `patch/worker/__init__.py:20-21`
(`if HAS_TRITON: import … patch_triton`) → `patch_triton.py:4` →
`ops/triton/fla/chunk.py:15` → upstream
`vllm/model_executor/layers/fla/ops/utils.py:141`, a **module-level**
`device = "cuda" if … else get_available_device()` →
`triton.runtime.driver.active.get_current_target()` →
`triton/backends/ascend/driver.py:188,91` `get_arch()`, ending
`SystemError: <built-in function get_arch> returned NULL without setting an exception`.

Two facts make this a one-call answer:

- The owner is the **triton-ascend build in the image**, not the model, not the
  310P capability table: nothing model-specific has run yet, and `get_arch()`
  returns NULL because that backend cannot map the SoC.
- Upstream `get_available_device()` catches only `(RuntimeError, AttributeError)`
  (`utils.py:118-123`), so `SystemError` escapes at import time — while the
  listener on the vllm-ascend side is `HAS_TRITON`-gated and 310P is eager-only
  (no Triton kernels), so removing triton removes the whole chain:
  `pip uninstall -y triton triton-ascend`, then
  `python -c "from vllm.triton_utils import HAS_TRITON; print(HAS_TRITON)"` must
  print `False`. The same frame can be probed in one line without vLLM:
  `python -c "import triton; print(triton.runtime.driver.active.get_current_target())"`.

Labels: source-verified from the reporter's own traceback plus this pin. The
`pip uninstall` action is **not** reporter-confirmed, and whether the shipped
`-310p` image carries that triton build is a registry fact this snapshot cannot
check — ask for `pip list | grep -i triton` before asserting it. Do not answer
this shape with the version survey the reporter asked for: at this pin
`v0.14.0rc1-310p` is the image that reintroduced 310P, so "which tag supports
310P" is the wrong question.

## The 310P gates that change behaviour

| Gate | Code (v0.11.0rc3) | Consequence to check |
|---|---|---|
| SoC detection | `vllm_ascend/utils.py:66-71` (`:65-69` at v0.11.0rc1) — `is_310p()` reads `_build_info.__soc_version__` | every branch below keys off the build stamp, not the device name. At **v0.11.0rc1** there is no runtime guard at all — `check_ascend_device_type()` (the assert in `device-type-build-vs-runtime.md`) does not exist in that tree, so a generic unsuffixed image on a 310P host silently keeps `is_310p() == False` and takes the A2 path; the tag suffix is the only signal |
| Custom ops not activated | `vllm_ascend/platform.py:287-288` — `# Activate custom ops for v1, except on 310P` / `if not is_310p(): compilation_config.custom_ops = ["all"]` | on 310P the reporter's `--compilation-config custom_ops` list is the only one in play; on other SoCs vllm-ascend overwrites it. A "custom op choice made no difference" observation is expected here |
| Global NZ format | `vllm_ascend/worker/model_runner_v1.py:146-154` — `torch.npu.config.allow_internal_format = True`; `set_compile_mode(jit_compile=False)`; `ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ` on 310P (ND otherwise) | every `_convert_torch_format` call site becomes a real conversion on 310P and a no-op elsewhere |
| Weight NZ conversion after load | `model_runner_v1.py:2685-2694` converts every `MergedColumnParallelLinear` / `QKVParallelLinear` / `RowParallelLinear` weight | load-time failures on 310P can be format casts, not op bugs |
| Cache NZ conversion | `model_runner_v1.py:2716-2721` (`_convert_torch_format`) called from the KV-init paths (`:2826`, `:3076` for `k_cache`) | see `../runtime-memory/kv-init-and-format-cast-alloc.md` — this frame is an allocation site |
| dtype gate on NZ | `vllm_ascend/utils.py:74-84` — `is_enable_nz()` returns **False for float16/bfloat16**, else `VLLM_ASCEND_ENABLE_NZ` (default 1, `vllm_ascend/envs.py:168-169`) and False for `qwen3_next` | at this pin an fp16 cache is returned unchanged, so a traceback whose deepest frame really executed `torch_npu.npu_format_cast` implies a non-fp16/bf16 cache (int8 / quantised) **or** a tree whose gate differs from this pin. Say which you verified |
| Attention mask NZ | `vllm_ascend/attention/attention_v1.py:253-262` (`nd_to_nz_2d` / `nd_to_nz_spec` + `npu_format_cast`) and further `is_310p()` branches at `:72,347,403,531` | mask-shape/format errors on 310P belong to this branch, not to the model's attention code |
| Separate 310P patch module | added later than this pin: at v0.18.0 `patch/platform/__init__.py:21-26` imports `patch_mamba_config_310` when `is_310p()` (the file does not exist at v0.11.0rc3) | a 310P-only patch can drift independently of the A2 one, and the gate list itself moves between releases — re-run the grep |

Two of these gates combine into the 310P LoRA answer: there is no graph mode
to amortise small launches (300I is eager-only) **and** `PunicaWrapperNPU` binds
the PyTorch `vllm.lora.ops.torch_ops` callables whenever the device type is
310P — at v0.9.2rc1 unconditionally, at v0.13.0rc2 by that explicit predicate
even though the AscendC `vllm_ascend/lora/lora_ops.py` exists in the same tree
(`vllm_ascend/lora/punica_npu.py:27-39`). So "LoRA is 4× slower here" is
structural on this SoC, and the lever is merging the adapter or moving the
release, not a flag — `../lora-adapters/decode-overhead-vs-base-model.md`.

## At v0.12.0rc1: the gate is renamed, the tutorial path moved, and "when will 310 get graph mode" has no answer

Verified at vllm-ascend **v0.12.0rc1** (`42ceaf0`) with vllm **v0.12.0**
(`4fd9d6a8`) while triaging #5075 (300I Pro, reporter on vllm-ascend 0.10.0rc1,
Qwen2.5-VL-3B, `--enforce-eager`, "310系列什么时候支持图模式和量化", 8 s vs T4 3 s
end-to-end on an ID-card OCR pipeline):

- **`is_310p()` does not exist at this ref.** `rg -n "is_310p" vllm_ascend/` returns
  nothing — a version fact, not a missing gate. The predicate is spelled
  `get_ascend_device_type() == AscendDeviceType._310P` (`utils.py:677` `_310P = 2`,
  `get_ascend_device_type()` at `:710`; call sites `platform.py:297`,
  `worker/model_runner_v1.py:138,2342`, `quantization/w8a8.py:51,163,320,349,366,386`,
  `lora/punica_npu.py:9`). Grep `_310P` before concluding "this tree has no 310P path".
- **The tutorial path moved, and the reporter's command is copied from whichever one
  their tag ships.** `docs/source/tutorials/single_node_300i.md` (≤v0.11, quoted
  above) → `docs/source/tutorials/310p.md` (v0.12.0rc1: eager-only note at `:5`,
  Qwen2.5-VL-3B-Instruct tab at `:120-131` whose serve line is
  `--tensor-parallel-size 1 --enforce-eager --dtype float16 --compilation-config
  '{"custom_ops":["none","+rms_norm","+rotary_embedding"]}'` verbatim) →
  `docs/source/tutorials/hardwares/310p.md` (v0.22+). A paste that matches a tab
  byte-for-byte is the doc's own recipe, so "user error" leaves the ranking; the old
  path still exists only as `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/single_node_300i.po`,
  which dates a paste that cites it.
- **Quantization already has a 310P path at this ref.** `--quantization ascend` plus
  the 310P branch of `npu_quant_matmul` (`quantization/w8a8.py:163`, transpose
  decision at `:51`, `fused_experts_310p` at `:320`), so "310 量化什么时候支持" is
  answered "already — with a 310-targeted checkpoint", not "wait"
  (`310p-quant-checkpoint-soc-match.md`). Graph mode is **not** supported: `310p.md:5`
  says the 310I series "only supports eager mode and the float16 data type".
- **Establish the coverage verdict from test *paths*, then from the workflow.** At
  this pin the 310P suite is one file, `tests/e2e/310p/test_offline_inference_310p.py`,
  and every case in it is `enforce_eager=True` + fp16 (Qwen3-0.6B,
  Qwen2.5-7B-Instruct, and a Qwen2.5-VL-3B case — the reporter's own model):
  content-grepping `310p|_310P` inside `tests/` misses it because the SoC is only in
  the **directory name**, so use `rg --files tests/ | rg -i 310`. The wiring is
  `.github/workflows/vllm_ascend_test_310p.yaml` (label-gated, not per-PR):
  `:112-115` runs that file on `linux-aarch64-310p-1` and
  `tests/e2e/310p/test_offline_inference_parallel_310p.py` on `-4` — **a file that
  does not exist at this commit** (`git ls-tree -r --name-only HEAD | rg 310p` lists
  only the one test), so the 4-card leg cannot run and the only real 310P coverage at
  this ref is single-card eager.
- **"什么时候支持" is answered from the project's own staging, not from a guess.** The
  RFC that owns this platform is #1309 `[RFC]: Support Altlas 300I series`:
  `Stage1: E2E workflow work` (310P CI runner, e2e, wheels — several unchecked) and
  **`Stage2: Performance improvement` with an empty body**. That empty heading is the
  citable statement that graph mode/quantization work on 300I has no committed scope
  or date; the first *documented* 310P `FULL_DECODE_ONLY` is the v0.22+
  `hardwares/310p.md` block below, and which release between v0.12 and v0.22 landed it
  is unanswerable here (`one_commit_git` per repo — say so, do not guess). Roadmap
  issues #448 (Q2 2025) and #1168 (Q3 2025) carry a "Graph mode" line with no 310P
  date; find them with `scripts/history-titles.sh -i '300I|310P'`.
- **Do not import the v0.22 mask trap backwards.** `attention/attention_v1.py` has
  **no** `_310P` branch at v0.12.0rc1 (`rg -c _310P` = 0), so the
  `O(max_model_len²)` fp16 mask allocation quoted for v0.22.1rc1 below cannot be
  attributed to a v0.12 paste.

## At v0.22.1rc1: the `_310p/` subpackage, the doc's validated table, two traps

Verified at vllm-ascend **v0.22.1rc1** (`5f6faa0`) + vllm **v0.22.1**
(`0decac0d`) while triaging #12015 (`v0.22.1rc1-310p-openeuler` image); five
distinctive log strings of that paste appear in the pinned tree with line numbers
within ±1 of the log's own `[file:line]` tags (`platform.py:60-61` breakable
cudagraph, `:526` eager-mode default, `:759` allocator injection, `:1036-1037`
GPU-only parameter, `ascend_config.py:772` Dynamic EPLB), so the pin is citable
for that release even though the image is not byte-verified.

- **The 310P code is now a subpackage.** `vllm_ascend/_310p/` holds
  `worker_310p.py`, `model_runner_310p.py`,
  `attention/{attention_v1,attention_mask,metadata_builder}.py`,
  `kv_block_zeroer.py`, `quantization/modelslim_config.py`, `ops/…`. The
  substitutions are `platform.py:651-652` (`worker_cls =
  "vllm_ascend._310p.worker_310p.NPUWorker310"`), `:807-810`
  (`AscendAttentionBackend310`) and `:197-201`
  (`AscendModelSlimConfig310`, which also prints the
  `quantization method 'ascend' … will be overwritten` warning). The v0.11-style
  `worker/model_runner_v1.py` line numbers in the gate table above no longer
  exist at this ref — re-grep inside `_310p/`.
- **The doc's validated table is narrow.** `docs/source/tutorials/hardwares/310p.md:95-176`
  validates only `Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310-vllm` (TP1, TP1, TP4),
  every row carrying `--dtype float16 --quantization ascend --load_format
  sharded_state --no-enable-prefix-caching --max_model_len 16384` (20480 for 32B)
  `--gpu_memory_utilization 0.90 --max_num_seqs`. `:86-87` warns that the
  attention mask is `O(max_model_len^2)` and asks for an explicit, conservative
  `--max-model-len`. A paste serving anything else — a VL checkpoint, an fp16
  dense model, prefix caching left on (the doc passes
  `--no-enable-prefix-caching`; #12015's own config echo says
  `enable_prefix_caching=True`) — is outside the validated set: say so, and make
  "run the doc's row on a small model" rung 1.
- **Trap: the fusion flag in those very doc commands is inert on 310P.** All three
  rows pass `--additional-config '{"ascend_compilation_config":
  {"fuse_norm_quant": false}}'`, but
  `vllm_ascend/compilation/graph_fusion_pass_manager.py:54` and `:69` read that
  key with `and not is_310p()`, so the pass is never added here. The
  `Enabled custom fusions: norm_quant, act_quant` line is **upstream**
  (`vllm/config/compilation.py:313`) and reports
  `compilation_config.pass_config`, which the upstream *optimization level* fills
  (`vllm/config/vllm.py:99-107` `enable_norm_fusion`, the O2 table at `:188-…`,
  default level `:353`, applied at `:1110`) — not the `additional_config` key. Under
  `--enforce-eager` the mode is `NONE` (`platform.py:525-528`) and no pass runs at
  all. So "the flag I set is not what the log lists" is a two-registry artifact,
  not a smoking gun: do not walk it (#12015 spent ~8 calls on it and ran out of
  output before answering).
- **Trap: the 310P mask allocation is real, but it is sized by `max_model_len`.**
  `_310p/attention/metadata_builder.py:80` constructs
  `AttentionMaskBuilder310(self.device, max_model_len)` and
  `_310p/attention/attention_mask.py:68` allocates
  `torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)`
  (class default `max_seqlen = 16384`, `:38`), and `gen_causal_additive_mask`
  (`:66-70`) holds a `bool` `tril`, its `~tril` complement and the fp16 mask at
  once — **≈4·N² bytes per rank**, so `--max-model-len 100000` is ~40 GB peak on a
  43 GiB die before any weight (60 K → ~14 GB, 32768 → ~4.3 GB). That is what the
  doc's `:86-87` warning is about, and it is an argument for an explicit
  `--max-model-len` — never a licence to read a **CPU** `torch.zeros` top frame in
  a segfault dump as this allocation
  (`../hccl-and-hardware/worker-segfault-no-python-frames.md`,
  `## The startup variant`).

## At v0.22+/v0.23: a quantized model that loads and then dies in an op

When the weights load and the failure is an operator-level dtype sentence
(`Bias dtype should be INT32, actual dtype is DT_FLOAT16`) rather than a Python
traceback, the owning variable is the **checkpoint conversion**, not this page's
gate table: 310P needs its own `-310`/`-310P` artifact, and the `_310p/`
`W8A8_DYNAMIC` scheme passes the model's fp16 bias into `npu_quant_matmul` while
the static/SC schemes pass the checkpoint's int32 `quant_bias` —
`310p-quant-checkpoint-soc-match.md` (v0.23.0 line set, plus the doc trap that
the 27B tutorial's 300I tab points at the generic `-w8a8` repo).

## The `TransData` / `EZ3002` rejection at `profile_run` (the bf16 form)

Verified at vllm-ascend **v0.9.1rc1** (`c30ddb8`) with vllm **v0.9.1** (`b6553be`),
recorded for #1383 (310P3, `Qwen/Qwen2.5-0.5B-Instruct`, no `--dtype` so the
checkpoint's bf16 is used, V0 engine, `profile_run` at
`determine_num_available_blocks`). The block is long; only four lines decide it:

```
EZ3002: Optype [TransData] of Ops kernel [AIcoreEngine] is unsupported. Reason:
  [tbe-custom]:op type TransData is not found in this op store.
  [Dynamic shape check]: The format and dtype is not precisely equivalent to
  format and dtype in op information library
Optype [TransData] of Ops kernel [aicpu_ascend_kernel] is unsupported. Reason:
  Transdata op, groups should be greater than 1, but now is 1.
build op model failed, result = 500002
```

Read it as a **dtype/format rejection**, not as a missing kernel: the op and the
engine were found and the *input combination* was refused. The missing-kernel shape
is different text — `EZ9999 … Parse dynamic kernel config fail` /
`AclOpKernelInit failed opType` / `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE failed` →
`kernel-package-vs-card.md`. The `aicpu … groups should be greater than 1` line is the
CPU fallback's own complaint about the same node: a red herring, not a second cause.

Why bf16 is cause 1 at this pin (all citable):

| Fact | Where |
|---|---|
| The checkpoint's dtype reaches the device: the engine line prints `dtype=torch.bfloat16` and the tree has no 310P dtype gate | the report's own `Initializing a V0 LLM engine …` line |
| On 310P the rope **always** takes the ATB fallback: the custom-op branch is `custom_rotary_embedding_enabled(...) and not is_310p()`, else `torch_npu._npu_rotary_embedding` — where the TransData is inserted | `vllm_ascend/ops/rotary_embedding.py:52`, fallback `:69-76` |
| So the tutorial's `custom_ops: ["+rms_norm","+rotary_embedding"]` **cannot** move rope off ATB here — "add the flag" is not an action, and the doc/code disagreement is a finding | same file, plus `docs/source/tutorials/single_node_300i.md:175` |
| The same tag's tutorial states the constraint outright | `single_node_300i.md:175`: `dtype="float16", # IMPORTANT cause some ATB ops cannot support bf16 on 300I series` (repeated `:224`, `:274`) |
| Precedent on the same SoC | #752 (310P3, CANN 8.0.0, Qwen3): `RopeOperation_0 Atlas inference products can not support bf16` / `setup check fail` |

Ladder: `--dtype float16` first (with `--enforce-eager`, the tutorial's other
mandatory flag); then the image/kernel line — `Dockerfile.310p:18` bases on
`cann:8.1.rc1-310p-…` while the generic `Dockerfile:18` bases on `…-910b-…`, so a
generic image plus a missing `Ascend-cann-kernels-310p` is cause 2; then
`ASCEND_LAUNCH_BLOCKING=1` + `ASCEND_GLOBAL_LOG_LEVEL=1` for the node's real
dtype/format. State the doc contradiction as it stands at the pin:
`docs/source/faqs.md:18` still says 300I/310P "might be supported on 2025.Q2" while
`single_node_300i.md` ships a working fp16 recipe — experimental channel, not a
status answer.

## Ranked actions for a 310P startup report

1. Confirm the SoC and the real versions: `python -c "from vllm_ascend import
   _build_info; print(_build_info.__soc_version__)"`, `pip show vllm vllm-ascend`,
   `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`,
   `cat /usr/local/Ascend/driver/version.info`, `npu-smi info`. The image pins
   CANN `8.3.rc2-310p`; a host driver older than that pairing is a real candidate.
2. Apply the 300I environment (`PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`),
   `--enforce-eager`, `--dtype float16` — the tutorial treats all three as
   mandatory, and the reporter usually has only the last two. Write an explicit
   value: `unset` re-enables vllm-ascend's `expandable_segments:True` injection, and
   on 310P (v0.23.0rc1 `-310p` image, CANN 9.1.0-beta.1, torch_npu 2.10.0.post2)
   that value crashed every rank in `torch.npu.empty_cache()` with `unmapHandles …
   507899` while `expandable_segments:False` deployed (reporter-confirmed #12534 —
   `../runtime-memory/allocator-expandable-segments-unmap-crash.md`). Also quote the
   graph budget when TP > 1: `FULL_DECODE_ONLY` is supported but "the number of
   capturable graphs is limited and depends on the model depth … Qwen3-32B can
   capture and replay 2 graphs", with no limit at TP=1
   (`docs/source/tutorials/hardwares/310p.md:20-30` at v0.23.0rc1) — so a
   "2 cards fine, 4 cards stall" report takes `--enforce-eager` and
   `cudagraph_capture_sizes: [1]` before any HCCL theory. The same stall across a
   **different checkpoint** retires every model lane (`#12985`: Qwen3.5-35B-A3B,
   then Qwen3-VL-32B, reporter asking "是硬件的问题么" — the answer is these
   documented platform limits, not a broken card). The cheapest discriminator is
   already in the report: the reporter's own *working* command, usually a
   screenshot whose OCR carries it (`../evidence-workspace/index.md`). The recorded
   pair differed on exactly three axes — `--enforce-eager` + 60 K
   `--max-model-len` + `gpu_memory_utilization 0.4` (worked) vs
   `FULL_DECODE_ONLY` + `enable_npugraph_ex` + 100 K + 0.6 at TP 8 (stalled behind
   the 60 s `shm_broadcast` warning) — so diff that pair instead of theorising from
   the pasted traceback. Co-tenancy (`mindie_llm_backend … --npu_device_id 2`
   beside `Worker_TP0..3` in that screenshot's `ps`, while `npu-smi` util% reads
   idle) is visible only there, never in `npu-smi`.
3. Shrink the budget (`--max-model-len`, `--gpu-memory-utilization`) before
   blaming the platform: 310P cards are small and the documented KV/profile
   arithmetic in `../startup-failures/max-model-len-budget.md` applies unchanged.
4. Then the documented-status sentence: v0.11.0rc2-310P is experimental, the
   project itself points at v0.10.0rc1 for 300I, and `#3316` / `#2795` are the
   open known issues the tutorial cites. Say explicitly that this is the
   *documented* state and not a diagnosis of the reporter's host.
5. Only then propose code ownership, and name the gate you think is responsible
   from the table above.

## Limits

- The snapshot carries **one commit** per repo (`mode: one_commit_git`), so
  "which commit introduced this between v0.10.0rc1 and v0.11.0rc2" is
  unanswerable here; say so instead of guessing. `SOURCES.json` also marks the
  reported tag `v0.11.0.rc2-310P-openeuler` as `no_named_release` — the pinned
  reference is v0.11.0rc3.
- Title/command mismatches (7B in the title, 3B in the command) are common in
  this family; ask which one failed before reasoning about memory.
