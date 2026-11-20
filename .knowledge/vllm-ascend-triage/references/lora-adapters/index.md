# LoRA adapters: graph-mode capture crashes and per-step decode overhead

## Scope

The deployment was started with `--enable-lora` / `--lora-modules` and the
complaint is one of two shapes:

- **It dies on a request**, after a clean startup, with an ACL/CUDA
  graph-capture error (`CUDA graph capturing detected at an inappropriate time.
  This operation is currently disabled.`, wrapped as `Worker failed with error
  '…'`), often when the request selects the **base** model rather than an
  adapter. This is a `BatchDescriptor` bookkeeping mismatch, not a device,
  HCCL or CANN fault → `graph-capture-descriptor-mismatch.md`.
- **It is several times slower than the same model served without adapters**
  (recorded #1812: 9 tok/s → 2 tok/s on Qwen3-8B + LoRA, Atlas 300I Duo), with
  no error and no OOM → `decode-overhead-vs-base-model.md`.

Evidence that decides both: the two serve commands (with and without
`--enable-lora`), the SoC and whether graph capture is possible on it, the
adapter `r` / `target_modules` / dtype, and for the crash the **worker**
traceback above the `Worker failed with error` wrapper.

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
   callables in `__init__`. At v0.9.2rc1 (`b5b7e0e`) the binding is
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
   serve to it.
2. **A LoRA × ACLGraph crash is a defect at refs whose own docs claim the
   pair.** v0.13.0rc2 ships `docs/source/user_guide/feature_guide/lora.md:8,16`
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
4. **"LoRA costs 4-5×" is never an answer.** Convert the two reported rates to
   ms/step (`1000 / tok_s`), price the adapter's extra FLOPs (order 1 % for a
   rank-16 adapter on a dense 8B), and put the remainder on dispatch — then say
   which backend and whether a graph can amortise it on that SoC.
