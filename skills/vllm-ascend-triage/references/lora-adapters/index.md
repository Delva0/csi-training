# LoRA adapters: graph-mode capture crashes, decode overhead, and ViT adapters on VL models

## Scope

The deployment was started with `--enable-lora` / `--lora-modules` and the
complaint is one of six shapes:

- **It dies on a request**, after a clean startup, with an ACL/CUDA
  graph-capture error (`CUDA graph capturing detected at an inappropriate time.
  This operation is currently disabled.`, wrapped as `Worker failed with error
  '…'`), often when the request selects the **base** model rather than an
  adapter. This is a `BatchDescriptor` bookkeeping mismatch, not a device,
  HCCL or CANN fault → `graph-capture-descriptor-mismatch.md`.
- **It is several times slower than the same model served without adapters**
  (recorded #1812: 9 tok/s → 2 tok/s on Qwen3-8B + LoRA, Atlas 300I Duo), with
  no error and no OOM → `decode-overhead-vs-base-model.md`.
- **It never starts: the worker dies in the memory-profiling forward** with
  `Could not guard on data-dependent expression Ne(u0, 1)` (or another
  data-dependent expression) and a `from user code:` block whose last frames are
  `punica_npu.py` → `vllm/lora/ops/torch_ops/lora_ops.py` `bgmv_shrink` →
  `torch.einsum`, under `VLLM_USE_V1=1` / `PIECEWISE` compile. That is the same
  PyTorch punica binding as the slow-decode shape, seen from the compile side:
  the adapter math is traceable, so the first compiled forward cannot guard on
  it → `pytorch-punica-compile-guard.md`.
- **It never starts: the worker dies in the memory-profiling forward** with an
  `AttributeError` naming a **vllm-ascend class's own attribute** — e.g.
  `'AscendRMSNorm' object has no attribute 'next_need_quant_fusion_linear'` out of
  `vllm_ascend/ops/layernorm.py` — on a quantized checkpoint
  (`--quantization ascend`), one line after `Using PunicaWrapperNPU.`, while the
  same checkpoint without `--enable-lora` serves. The name is a *property* whose
  getter raised; the real missing name is `quant_method` on the LoRA wrapper
  upstream swapped in for `qkv_proj`/`gate_up_proj`, which the norm+quant fusion
  dereferences unguarded → `fusion-reader-vs-lora-wrapper.md`.
- **The model is a VL model and the adapter covers the vision tower**
  ("LoRA（with ViT）", an RL rollout whose `target_modules` name `visual.*`), and
  the complaint is an OOM in multimodal prefill ("请求显存明显超出预估"), answers
  worse than the training run, or one source line the reporter says blows up.
  ViT adapters are **filtered out by upstream** at these refs, so they are never
  the crash and the quoted tensor shape is not the allocator's victim →
  `multimodal-vit-lora-ignored.md`.
- **The base model is a MoE** (`Qwen3-*-A3B/A22B`, DeepSeek, Mixtral …) and the report is
  "<model> cannot use `enable-lora`": the run dies before the first token, and the exception
  names *which* of five gates failed — the rank default, the adapter's key set, the warm-up
  dummy in `profile_run`, a `None` weight half in `set_lora`, or a fused-MoE LoRA layer that
  does not exist / is CUDA-only at the pin. Each gate has its own sentence, so read the map
  in `moe-base-enable-lora-gates.md` before choosing one; on these models the answer is a
  ranked gate list plus the `--enable-lora`-without-`--lora-modules` ablation, never a
  single flag.

Evidence that decides these: the two serve commands (with and without
`--enable-lora`), **which phase** the failure sits in (engine init / profile run
vs request time vs capture), the SoC and whether graph capture is possible on it, the
adapter `r` / `target_modules` / dtype, and for the crash the **worker**
traceback above the `Worker failed with error` wrapper. For the VL shape, the
`target_modules` list and the server's `… will be ignored` warning decide it,
and the memory question needs the allocator line plus `npu-smi info`, not a
shape.

Not this topic, leave immediately:

- `kv_lora_rank` / `qk_rope_head_dim` in an MLA/DeepSeek config — KV-compression
  geometry, nothing to do with adapters →
  `../startup-failures/connector-config-attributes.md`.
- The adapter never loads (`AdapterNotFound`, path/permission, wrong
  `base_model_name`) — upstream's worker manager, before any graph question.
- Same adapter, different **scores** → `../accuracy-benchmarks/index.md`.
- Spec-decode or mixed-batch symptoms on a LoRA serve →
  `../spec-decode-mtp/index.md`, `../mixed-batch-splits/index.md`.

## Rules

1. **Name the punica backend before anything else; it is version- and
   SoC-dependent.** `vllm_ascend/lora/punica_npu.py` binds the bgmv/sgmv
   callables in `__init__` (the module path moves: `vllm_ascend/lora/punica_wrapper/punica_npu.py`
   at v0.9.1rc1 `c30ddb83`, `vllm_ascend/lora/punica_npu.py` at the later refs in
   this library — `ls` or grep both, the audit script probes both). At
   v0.9.2rc1 (`b5b7e0e`) the binding is
   unconditional `from vllm.lora.ops.torch_ops import …` — pure PyTorch, so each
   apply is a gather of the adapter weight stack + cast + `einsum` + strided
   write-back, per slice, per layer, per step. At v0.13.0rc2 (`a17059a`) the
   AscendC module `vllm_ascend/lora/lora_ops.py` is used **unless**
   `get_ascend_device_type() == AscendDeviceType._310P` **or**
   `lora_config.max_lora_rank >= 128` (`:27-39`). One call:
   `bash <skill_root>/scripts/lora-graph-descriptor-audit.sh <vllm-ascend-root> [vllm-root]`
   prints the binding and its predicate. Consequence to carry into advice: on
   310P the PyTorch backend is the *selected* backend at every ref in this
   library, and raising `--max-lora-rank` past 128 silently demotes an A2/A3
   serve to it. The **same binding reads both ways**: where it is transitive
   PyTorch, compile cannot tolerate it (engine-init dynamo guard,
   `pytorch-punica-compile-guard.md`) and a graph cannot amortise it (slow
   decode) — so "the SoC has ACL graph" does not by itself make an adapter
   cheap; ask whether the *LoRA path* is graph-visible.
2. **A LoRA × graph-mode crash — capture or compile — is a defect at refs whose
   own docs claim the pair.** v0.9.1rc1 `c30ddb83` claims it twice — `suppoted_features.md:11`
   (LoRA 🟢 Functional on both engines) and `release_notes.md:62` ("supported for
   V1 Engine now") — while `tests/conftest.py:87` runs every e2e LoRA case
   eager, so nothing exercises LoRA under compile. v0.13.0rc2 ships `docs/source/user_guide/feature_guide/lora.md:8,16`
   ("You can run LoRA with ACLGraph mode now… enables the ACLGraph mode by
   default"). So the answer is an ablation ladder plus a bug report, never
   "LoRA is not supported with graphs". At v0.9.2rc1 (`b5b7e0e`) the same page's
   only tip is the opposite — "if you fail to run … with LoRA, … disable graph
   mode and try again" (`:7-8`) — so quote the page at the reported ref rather
   than carrying a newer claim backwards.
3. **The logged frame is a re-raise, never the owner.** `patch_core.py`
   `logger.exception("EngineCore encountered a fatal error.")` + `raise e`, and
   `Worker failed with error '<text>'` from upstream
   `v1/executor/multiproc_executor.py`, both only relay. The raising module for
   a capture error is upstream `vllm/compilation/monitor.py`; for the slow shape
   there is no raise at all and the owner is the call count in
   `punica_npu.py`.
4. **"LoRA costs 4-5×" is never an answer, and the ratio is not SoC-specific.**
   Convert the two reported rates to ms/step (`1000 / tok_s`), price the
   adapter's extra FLOPs (order 1 % for a rank-16 adapter on a dense 8B), and put
   the remainder on dispatch — then say which backend and whether a graph can
   amortise it on that SoC. Recorded ratios: 9 → 2 tok/s on eager-only 310P
   (#1812) **and** 10-12 → 2-3 tok/s on graph-capable 2×910B3 TP2 (#1629,
   reporter-confirmed base-only baseline), so a graph-capable SoC does not
   short-circuit the question.
5. **On a VL model, ViT adapters are dropped by upstream before any Ascend code
   runs.** `vllm/lora/models.py:613-624` filters everything under the model's
   `tower_model` / `connector` prefixes and only warns (`:489-497`), so rank a
   vision-tower mechanism for a crash or an OOM on a VL serve only after that
   grep comes back empty — and read `multimodal-vit-lora-ignored.md` for the
   `[L, 1, H]` shape anchor before treating a quoted tensor shape as the failing
   allocation.
6. **`--enable-lora` moves the *module graph*, not only the math, so a fusion path
   that indexes `model_instance` breaks on a launch with no other change.**
   Upstream replaces the linear modules **in place**
   (`vllm/lora/models.py:474-502` `replace_submodule` → `from_layer`), so a
   `*WithLoRA` wrapper takes the place of the `LinearBase` the fusion cached and
   exposes no `quant_method`. Before ranking a quant/fusion crash on a LoRA serve,
   run `rg -n 'base_layer' vllm_ascend/` — no hits outside `tests/` means no
   downstream site is LoRA-aware, so the bug is the plugin's, and the report's
   three-command matrix (quant+LoRA fails / fp16+LoRA works / quant alone works) is
   the whole attribution → `fusion-reader-vs-lora-wrapper.md`.
7. **On a MoE base, "cannot use `--enable-lora`" is five gates, and the paste names which.**
   Do not answer it with `--max-lora-rank` alone, however well the body's
   `adapter_config.json` supports that rung (`"r": 32` against `max_lora_rank: int = 16`,
   `vllm/config/lora.py:33`): the same recorded report (`#3377`, whose log arrived later)
   showed `AttributeError: 'NoneType' object has no attribute 'shape'` at
   `vllm/lora/layers/base_linear.py:125` from the **warm-up dummy** block of
   `profile_run`, i.e. before the adapter was ever loaded. Two facts widen most such
   reports: `vllm/lora/models.py:59-67` warns `For MoE models, vLLM currently does not
   support fused MoE LoRA inference …` on **every** MoE `--enable-lora` start (scope, not
   cause), and the MoE × LoRA intersection is dead on both sides — no
   `vllm/lora/layers/fused_moe.py` at v0.11, and at vllm `v0.13.0` that file exists but
   `select_gemm_impl` reaches a CUDA-only import, so `NameError: name 'TritonExperts' is
   not defined` (`#6114`). One run splits adapter from pin: the same command with
   `--enable-lora` and **no** `--lora-modules` → `moe-base-enable-lora-gates.md`.
