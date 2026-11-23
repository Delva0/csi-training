# "`<MoE model>` cannot use `enable-lora`": the load-path gates and the sentence each one raises

Verified at vllm `v0.11.0rc3` (`8ce5d319`) × vllm-ascend `v0.11.0rc0` (`00ba071`) for the
rank gate and the `AttributeError` shape (recorded #3377: Qwen3-30B-A3B, TP4, `vllm serve`,
report carried the command plus `adapter_config.json` and **no** server log), and at vllm
`v0.13.0` (`72506c9`) × vllm-ascend `v0.13.0rc1` (`1b5d5ab`) for the `NameError` shape
(recorded #6114: Qwen3-235B-A22B, offline `LLM(enable_lora=True)`, traceback only as a
screenshot whose `.ocr.txt` kept the exception text).

Use this page when a MoE checkpoint is served or run with `--enable-lora` /
`enable_lora=True` and the run fails **before the first token**, however little of the log
survived. The gates below are reached in this order and each raises a **different class of
exception**, so "which gate" is decided by one sentence of the paste — never by the model,
the card, TP, CANN or the adapter's dtype.

## 1. The gates, in the order a start reaches them

| # | Gate | Site (pins above) | The reporter's own sentence |
|---|---|---|---|
| 1 | adapter rank vs the engine default | `vllm/config/lora.py:33` `max_lora_rank: int = 16`; `vllm/lora/peft_helper.py:113-127`; called from `vllm/lora/worker_manager.py:105` before any weight is loaded | `ValueError: LoRA rank 32 is greater than max_lora_rank 16.` (`peft_helper.py:119-122`); legal values are `(8, 16, 32, 64, 128, 256, 320, 512)` (`config/lora.py:104`) |
| 2 | adapter key set vs the modules this model expects | `vllm/lora/models.py:227-238` (`check_unexpected_modules`), matched on the **last path component** only | `ValueError: While loading <dir>, expected target modules in [...] but received [...]. Please verify that the loaded LoRA module is correct` |
| 3 | the warm-up **dummy** adapter inside `profile_run` | `vllm/v1/worker/lora_model_runner_mixin.py:32` `LORA_WARMUP_RANK = 8`, `:86-160` | no message of its own — this is where gate 4 fires |
| 4 | a weight half reaching `set_lora` | `vllm/lora/layers/base_linear.py:102-129`, dereference at `:124-126` | `AttributeError: 'NoneType' object has no attribute 'shape'` |
| 5 | a fused-MoE LoRA layer exists at all | 0.11: **absent** (`vllm/lora/layers/` has no `fused_moe.py`); the module-level warning at `vllm/lora/models.py:59-67`, emitted once per manager creation at `:379` | the `For MoE models, vLLM currently does not support fused MoE LoRA inference …` WARNING |
| 5b | the same gate at ≥0.13, where the layer exists but is CUDA-only | `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:31-37` (import) and `:137` (use); `vllm/platforms/interface.py:174-176`; `vllm/lora/layers/fused_moe.py:44,119,129` | `NameError: name 'TritonExperts' is not defined` |

Gates 1 and 2 are **reporter-side facts** — the adapter's `adapter_config.json` and its key
list decide them without a log. Gates 3-5 are pin-side. A body that carries only the
`adapter_config.json` therefore still settles gate 1: `"r": 32` on a pin whose default is
16 means `--max-lora-rank 32` is required *whatever else is wrong*, and the recorded #3377
report is exactly that shape (`"r": 32`, `"target_modules": ["k_proj","q_proj","v_proj","o_proj"]`,
`"use_dora": false`, `"modules_to_save": null`). Quote the flag as one rung; do not let it
become the whole answer, because the log that later arrived showed gate 4 instead.

## 2. Gate 4, the `NoneType` shape: what the frame proves and what it excludes

```text
determine_available_memory -> profile_run -> _dummy_run
  with self.maybe_dummy_run_with_lora(self.lora_config, …)      # lora_model_runner_mixin.py:153-157
  -> maybe_select_dummy_loras -> _set_active_loras              # :119-148
  -> worker_manager.set_active_adapters -> _apply_adapters -> add_adapter
  -> activate_adapter -> module.set_lora(index, module_lora.lora_a, …)
  -> vllm/lora/layers/base_linear.py:125  0, :lora_a.shape[0], :lora_a.shape[1]].copy_(
AttributeError: 'NoneType' object has no attribute 'shape'
```

* **The block is the warm-up one.** `maybe_dummy_run_with_lora` enters
  `maybe_setup_dummy_loras` (which registers `LoRARequest(lora_name="warmup_<id>",
  lora_path="/not/a/real/path")` at rank 8) and then `maybe_select_dummy_loras`, which
  activates ids `1..max_loras`. `--lora-modules` is a **frontend** registration
  (`vllm/entrypoints/openai/serving_models.py:53-79`) and reaches the worker as an
  `add_lora` RPC, so on a plain `vllm serve … --enable-lora --lora-modules …` start the
  objects present at `profile_run` are the dummies (`worker_manager.py:145-156` skips
  creating a dummy only for an id already registered).
  → **One run decides owner: the same command with `--enable-lora` and no
  `--lora-modules`.** It still crashes ⇒ the pin cannot start LoRA on this model and the
  adapter is exonerated (deliver the defect + merge-offline workaround). It starts ⇒ the
  weights came from the adapter and gate 2/4 is a key-set question.
* **It is not a partial packed adapter.** `MergedColumnParallelLinearWithLoRA.set_lora`
  guards every slice (`vllm/lora/layers/column_parallel_linear.py:260-284`,
  `if (lora_a_i := lora_a[i]) is not None`), and `PackedLoRALayerWeights` keeps `None`
  holes deliberately (`vllm/lora/lora_weights.py:152-177`). Only a **single-slice** wrapper
  (`RowParallelLinearWithLoRA` for `o_proj`/`down_proj`, `ReplicatedLinearWithLoRA`) reaches
  the unguarded statement — `RowParallelLinearWithLoRA` (`o_proj`/`down_proj`),
  `ColumnParallelLinearWithLoRA` (a single un-packed projection) or
  `ReplicatedLinearWithLoRA` — and the traceback never names which, so ask for the module
  or a debugger stop on `set_lora` rather than guessing one.
* **It is not the rank gate.** Gate 1 raises `ValueError` with its own sentence from
  `peft_helper.validate_legal`, which runs before any `set_lora`; an `AttributeError` in
  `set_lora` therefore says the rank check either passed or was never reached.
* **It is not `lora_b`.** `LoRALayerWeights.optimize()` runs first and would raise
  `TypeError` on a `None` `lora_b` whenever scaling ≠ 1 (`lora_weights.py:41-47`); a
  `None` **`lora_a`** is what survives to `:125`.
* Instrument, not a guess: `VLLM_LOGGING_LEVEL=DEBUG` prints
  `Activating LoRA. int id: <id>, slot index: <i>` (`vllm/lora/models.py`,
  `activate_adapter`) — an id ≤ `--max-loras` is a warm-up dummy, i.e. the pin's defect.

## 3. Gate 5b, the `NameError` shape at ≥0.13 (MoE + LoRA has no non-CUDA path)

| Step | Site (vllm `v0.13.0` `72506c9`) |
|---|---|
| wrap the MoE module in place | `vllm/lora/model_manager.py:254-292` (`_create_lora_modules` → `from_layer`, `:289`) |
| dispatch to the fused-MoE LoRA class | `vllm/lora/utils.py:87` `from_layer` → `vllm/lora/layers/fused_moe.py:44` `FusedMoEWithLoRA` |
| inject: ask the quant method for a gemm impl | `fused_moe.py:57` `_inject_lora_into_fused_moe` → `:119` → `:129` `self.base_layer.quant_method.select_gemm_impl(…)` |
| crash | `unquantized_fused_moe_method.py:137` `return TritonExperts(self.moe_quant_config)` |
| why the name is unbound | `TritonExperts` is imported **inside** `if current_platform.is_cuda_alike():` (`:31-37`), and `is_cuda_alike()` is `self._enum in (CUDA, ROCM)` (`vllm/platforms/interface.py:174-176`) — NPU matches neither |

Downstream does not intercept it, which makes the report a defect rather than a config
error: `AscendFusedMoE(FusedMoE)` (`vllm_ascend/ops/fused_moe/fused_moe.py:138`) is still
an upstream subclass, so `isinstance(module, FusedMoE)` wraps it, and
`AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod)` (`:55`) overrides
`process_weights_after_loading` / `apply` but **not** `select_gemm_impl` (0 hits in the
file). One grep confirms: `rg -n 'select_gemm_impl' vllm_ascend/`.

**Adding the import is not a fix** — `TritonExperts` is the CUDA Triton modular-kernel
expert class (`vllm/model_executor/layers/fused_moe/fused_moe.py:2032`) and the injection
also installs a `FusedMoEModularKernel` over `punica_wrapper.moe_lora_align_block_size`,
i.e. the whole LoRA-MoE path is CUDA-shaped. Say that, and put the ladder on offline merge
/ dense base / upstream report. Sibling at the same pin: `assert not self.base_layer.use_ep`
(`fused_moe.py:49-51`) means `--enable-expert-parallel` + LoRA raises
`AssertionError: EP support for Fused MoE LoRA is not implemented yet.`

## 4. The MoE warning is a scope statement, not the cause

`is_moe_model()` (`vllm/lora/models.py:59-67`) warns once whenever the model holds any
`FusedMoE`, from `LoRAModelManager.__init__` — so it prints on **every** `--enable-lora`
start of a MoE model, one line after `Using PunicaWrapperNPU.`, whether or not the adapter
carries expert weights, and it prints on a start that then succeeds. Read it as "this pin
has no fused-MoE LoRA inference", quote it for that, and do not report it as the adapter's
fault or as the crash.

## 5. Answer shape

1. **Verdict first, per gate:** which gate the paste shows (one sentence each for the
   `ValueError` / `AttributeError` / `NameError` forms), and — when no log was pasted —
   that the rank gate is settled from `adapter_config.json` alone while the crash needs the
   traceback. Do not offer one flag as "the" fix for a body that also names a crash.
2. **Ranked rungs, cheapest first:** `--max-lora-rank <r>` with the adapter's own `r`
   (only if `r > 16`); the `--enable-lora`-without-`--lora-modules` ablation; the adapter
   key dump (`adapter_model.safetensors` keys grouped per module, A/B pairing); offline
   merge of the adapter into the base weights (no LoRA at serve time); a dense base for the
   same evaluation.
3. **Name the acceptance check**, never "the engine starts": one real completion through
   the adapter compared with the same prompt on the base model. A dropped or ignored
   adapter half passes startup and returns wrong numbers.
4. **Scope:** both pins above are rc exports; "fixed later" is unanswerable from them — give
   the recheck (`rg -n 'select_gemm_impl' vllm_ascend/`, and whether
   `vllm/lora/layers/fused_moe.py` gained a non-CUDA gemm impl). Claim coverage, not
   absence: at `v0.11.0rc0` the only LoRA e2e cases are Llama dense
   (`tests/e2e/singlecard/test_ilama_lora.py`, `tests/e2e/multicard/test_ilama_lora_tp2.py`),
   so no MoE × LoRA case exists at either pin.
5. **Ask for:** the full per-rank traceback (not the `Worker failed with error` wrapper),
   `adapter_config.json` + the adapter's key list, `--max-loras` / `--max-lora-rank`, and
   whether `--enable-lora` alone fails.

## Recheck conditions

Gate 1 moves with `LoRAConfig`'s default (`vllm/config/lora.py`); gate 2's matcher, gate 4's
line numbers and gate 5's module layout all moved between `0.11.0rc3` (`vllm/lora/models.py`,
`vllm/lora/layers/base_linear.py`) and `0.13.0` (`vllm/lora/model_manager.py`, the same
`layers/` package). Re-read the file names before citing a line: a `vllm/lora/models.py`
grep at 0.13 returns the class but not `_create_lora_modules`.
