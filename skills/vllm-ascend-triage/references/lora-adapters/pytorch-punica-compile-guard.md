# A LoRA serve dies at engine init inside the profile run: `Could not guard on data-dependent expression`

## Symptom this page answers

`vllm serve … --enable-lora --lora-modules …` never reaches "Application startup
complete". The worker's traceback ends

```text
RuntimeError: Worker failed with error 'Could not guard on data-dependent expression
Ne(u0, 1) (unhinted: Ne(u0, 1)). (Size-like symbols: u0)  …'
```

and the block above it is a **dynamo** report whose `from user code:` list ends in
the LoRA math, not in a model or device op:

```text
vllm/lora/layers.py:581 forward → :422 apply
  → vllm_ascend/lora/punica_wrapper/punica_npu.py:296 add_lora_linear
  → :175 add_shrink → :148 _apply_shrink → :36 _shrink_prefill
  → vllm/lora/ops/torch_ops/lora_ops.py:64 sgmv_shrink → :78 bgmv_shrink
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
```

Recorded #1629 (vllm-ascend v0.9.1rc1 `c30ddb83` + vllm v0.9.1 `b6553be1`,
2×910B3 TP2, DeepSeek-R1-Distill-Qwen-32B = dense 64-layer Qwen2, CANN 8.1.RC1,
torch-npu 2.5.1.post1.dev20250528). The same report's **other** half — the same
image serving the same adapter at 2-3 tok/s when it does start — is the same
cause and is priced at `decode-overhead-vs-base-model.md`; #1629 is its
graph-capable-SoC data point (base-only serve 10-12 tok/s → LoRA serve 2-3 tok/s,
reporter-confirmed).

## Phase, in one line

`worker_v1.py:144 determine_available_memory` → `model_runner_v1.py:1815
profile_run` → `_dummy_run` (`:1774`) → `qwen2.py:477 forward` →
`vllm/compilation/decorators.py:239 self.compiled_callable(...)` → `torch/_dynamo/**`.
That is the **memory-profiling dummy forward**, i.e. the first compiled forward,
after the weights loaded and before capture. So this is not request-time (that is
`graph-capture-descriptor-mismatch.md`) and not a capture-stream sync (that is
`../startup-failures/captured-stream-host-sync.md`); it is **compile tracing**.
Ordering in the paste: the config-time `PIECEWISE compilation enabled on NPU …` /
`Calculated maximum supported batch sizes for ACL graph: 9` lines, then
`Starting to load model …`, the shard bar, `Loading model weights took 31.2558 GB`,
then the traceback three seconds later — no `Memory profiling takes …`, no
capture line, no request served.

## Owner: the LoRA op *binding*, not the graph, the model or the version pairing

Anchors verified at those pins (`../…` paths are the other pages that quote the
same lines):

| Fact | Anchor |
|---|---|
| The ref binds the punica wrapper to **pure PyTorch** ops unconditionally — nothing hides them from dynamo | `vllm_ascend/lora/punica_wrapper/punica_npu.py:6` `from vllm.lora.ops.torch_ops import (bgmv_expand, …, sgmv_shrink)`; the module lives under `lora/punica_wrapper/` at this ref, while later refs use `vllm_ascend/lora/punica_npu.py` — probe both spellings |
| That wrapper is what the platform hands back | `vllm_ascend/platform.py:236` `get_punica_wrapper` → `…punica_npu.PunicaWrapperNPU` |
| The traced body is a gather + cast + `einsum` + strided write-back, per slice; the gathered extent is the token count, which is symbolic in the profiling forward — hence `Ne(u0, 1)` | `vllm/lora/ops/torch_ops/lora_ops.py:68-80` (`selected_loras = lora_b_weights[lora_indices_tensor].to(…)`, `torch.einsum("bi, boi -> bo", …)`, `output_tensor[:, :outputs.shape[1]] = scaling * outputs[:]`) |
| V1 *does* turn on custom ops, but only its own: `custom_ops = ["all"]` covers the plugin's ops, never upstream's `torch_ops` helpers, so these stay traceable | `vllm_ascend/platform.py:209-211` (`if envs.VLLM_USE_V1: if not is_310p(): compilation_config.custom_ops = ["all"]`) |
| Upstream GPU reaches an **opaque** implementation — Triton custom ops, which dynamo does not trace into | `vllm/lora/punica_wrapper/punica_gpu.py` → `vllm.lora.ops.triton_ops`; `vllm/platforms/cuda.py:331-333` `get_punica_wrapper` → `punica_gpu.PunicaWrapperGPU` |
| No AscendC LoRA kernel exists in this tree at all (so "use the fast backend" is not a launch flag here) | `ls vllm_ascend/lora/` → `__init__.py`, `punica_wrapper/`; `ls csrc/kernels` has no `bgmv*` |
| CI never exercises this pair: the shared e2e runner defaults to eager, so every LoRA e2e case runs without compile | `tests/conftest.py:87` `enforce_eager: Optional[bool] = True` (passed through at `:99`); `tests/e2e/singlecard/test_ilama_lora.py:50`, `tests/e2e/multicard/test_ilama_lora_tp2.py:12` set only `max_loras=4` |

A `fullgraph` compile turns what would be a graph break into a hard error:
`vllm/compilation/wrapper.py:51` `fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE`,
whose default is `"1"` (`vllm/envs.py:311-313`). The message's own remedy
("add `guard_size_oblivious` to framework code") is not reachable from a serve
command, and `torch._dynamo.config.suppress_errors = True` would convert the
crash into a silent eager fallback **inside** a compiled region — offer it only
with an output check, never as the fix.

## The pinned docs claim the pair, so this is a defect, not "V1 lacks LoRA"

`docs/source/user_guide/suppoted_features.md:11` marks LoRA 🟢 Functional on both
engines with the V1 multi-LoRA PR, and `docs/source/user_guide/release_notes.md:62`
says "LoRA、Multi-LoRA And Dynamic Serving is supported for V1 Engine now"
(#893); `vllm_ascend/worker/model_runner_v1.py:65,131,959,1842` imports and calls
upstream's `LoRAModelRunnerMixin` (`load_lora_model`, `set_active_loras`,
`maybe_dummy_run_with_lora`). So the answer is an ablation ladder plus a bug
report, exactly as `index.md` rule 2 requires. Do **not** let the reporter's
"VLLM_USE_V1=1 部署模型服务，服务无法正常启动" phrasing become "V1 does not
support LoRA": the same V1 engine is the only engine here that *tries* to give
them a graph.

## Ladder, cheapest first

1. **Base-only serve on the same host** (`vllm serve <model>` with no
   `--enable-lora`). This decides the lane before anything else: #1629's
   reporter measured **10-12 tok/s** there versus 2-3 tok/s with the adapter, so
   the LoRA path — not the card, not TP2, not `--max-model-len` — owns the cost.
   If base-only is *also* a few tok/s, you are in
   `../performance-throughput/index.md`, not here.
2. `VLLM_USE_V1=1 … --enforce-eager` on the **same** command. Compile off ⇒ no
   dynamo trace ⇒ predicted to start (mark it a prediction), and it then
   reproduces the slow decode, which unifies both halves of the report. Confirm
   from the paste's own line `Compilation disabled, using eager mode by default`
   (`platform.py:164` at this ref — grep the sentence, the line moves) rather
   than from "it started".
3. Unset `VLLM_USE_V1` (V0). At this ref V0 does not compile the forward, so V0
   is *why* a slow run exists beside this crash — that inference needs no log
   from the reporter, but label the run's engine as inferred, not read.
4. Merge the adapter offline (`peft` `merge_and_unload`, matching dtype) and
   serve without `--enable-lora`: the only route to base-model speed while the
   backend is PyTorch. Cost: no dynamic adapter swapping.
5. File the defect with the two commands and the phase: the repair is an AscendC
   `bgmv/sgmv` (what later refs added, `vllm_ascend/lora/lora_ops.py`) or
   `NO_COMPILATION` when `lora_config` is set. The second is a guard, not a fix:
   its acceptance is **the same request returning the same output as the V0
   run**, plus the eager-baseline speed.

Do not offer, at a ref with this binding: a rank change (`--max-lora-rank`), a
`--max-loras` change, or "use the fast LoRA backend" — none of them selects a
different backend here, and the rank→backend predicate arrives only in much
later refs (`decode-overhead-vs-base-model.md` step 1 has the version table).

## Traps

- **The guard sentence has no model in it.** `u0` is the profiling forward's
  token-count symbol and `selected_loras` is the gathered adapter stack; a
  reporter reading `Ne(u0, 1)` as "batch size 1 is unsupported" is reading the
  symptom. The routing token is the deepest **user** frame:
  `vllm/lora/ops/torch_ops/lora_ops.py` reached from `punica_npu.py`.
- **One attached log, two runs.** #1629's `errorlog.txt` covers only the failing
  run; the 2.4 → 1.6 tok/s figures live in screenshots whose OCR keeps
  `Avg generation throughput: N tokens/s` and the `lora_request: LoRARequest(…)`
  line. Screenshots are the metrics, the log is the crash — ask for both serve
  commands before asserting a flag. Watch the OCR spelling of `LoRARequest`
  (`LoRRARequest`, `nothunk-lora`) and the `2@@`/`@` digit corruption in
  `max_tokens=`; quote the number, not the glyph.
- **Same-binding sibling refs.** v0.9.2rc1 `b5b7e0e` ships the same
  unconditional `torch_ops` import (`docs/source/user_guide/feature_guide/lora.md:10-11`
  there only says "if you fail to run … disable graph mode and try again"), so
  this crash shape and the 2-3 tok/s shape should be expected there too; re-grep
  the binding before carrying the verdict forward.
- **Do not price the step from the crash.** `2.4 tok/s` = 417 ms/step and
  `10-12 tok/s` = 83-100 ms/step is the arithmetic that belongs in the answer;
  the number of `add_lora_linear` calls per step comes from the model's fused
  linears at that pin, not from a generic "every layer × every target" count.
