# `ub overflow, requires N bits while M bits available` — a Triton kernel's tile is too big

## Scope

The engine starts, the weights load, and the **first request** dies inside a
Triton-Ascend kernel compile with:

```text
[ERROR] … [core.py:1136] …
[ConvertLinalgRToBinary] encounters error:
loc(".../vllm_ascend/ops/triton/rope.py":24:0) …  Failed to run BiShengHIR pipeline
loc("/tmp/tmpXXXX/kernel.ttadapter.mlir":1:1): error: Failed to run BiShengHIR pipeline
loc("/tmp/tmpXXXX/kernel.ttadapter.mlir":2:3): error: ub overflow, requires 15589248 bits
  while 1572864 bits available! (possible reason: tiling basic block is too large or block
  number is more than what user expect due to multi-buffer feature is enabled and some ops
  need extra local buffer.)
```

Deepest frames: `vllm_ascend/ops/rotary_embedding.py:168` `query, key =
rope_forward_triton(` → `vllm_ascend/ops/triton/rope.py:285` `_triton_rope[(n_row,)]()` →
`triton/runtime/jit.py` → `triton/compiler/compiler.py`. Pins for the `path:line` below:
**vllm-ascend v0.20.2rc1 (`367b8e62da79`)**, vllm v0.20.2 (`bc150f502991`); precedence
spans v0.17.0rc1 → v0.20.2rc1 and the anchors move per release — re-grep them.

This is **not** the wheel/compiler pair lane and **not** an AscendC/aclnn device fault:
nothing ran on the device, the pair compiled every other kernel on that image, and the
detail line names a *budget* rather than IR that could not be lowered. The owner is the
kernel's own `tl.constexpr` tile sizes multiplied by the model's shape.

## Rules

1. **Read the two bit numbers; ignore everything else on the line.** The `available`
   half is a per-SoC constant, not a release fact: `1572864 bits` (192 KiB, the
   vector-core unified buffer) is printed identically by a v0.17.0rc1 report and by a
   v0.20.2rc1 report. The whole finding is `required ÷ available`:
   **1.17×** (`1835776` bits, #8037, stock hybrid model at v0.17.0rc1) is one tile step
   over the budget, so the same model on an older/newer tile constant can go either way;
   **9.9×** (`15589248` bits, #18354, custom model at v0.20.2rc1) is a structurally
   oversized tile, not a marginal one. Quote the ratio and the two numbers; they are
   arithmetic the reporter can check on their own paste.
2. **Do not reconstruct the per-buffer byte accounting.** The compiler's figure already
   folds in `--enable-auto-multi-buffer=True` (it is on the `bishengir-compile` command
   line by default), the `.to(tl.float32)` promotions inside the kernel and the
   lowering's temporaries; hand-summing
   `tile elements × dtype size × N buffers` lands near, not on, the printed number.
   What the number *does* support is the direction and the size of the fix: the demand
   is roughly linear in the tile constant, so divide it by ≥ the ratio (rule 6).
3. **The detail line decides the lane — before any reinstall advice.** Under the same
   `[ConvertLinalgRToBinary]` banner: `operation's operand is unlinked` /
   `bishengir-compile … returned non-zero exit status 1` with IR diagnostics is the
   wheel × compiler pair (`../image-dependency-faults/triton-ascend-pair.md`);
   `libkernel.so: failed to map segment from shared object` is the mapping-count lane
   (`../runtime-memory/dlopen-map-count-exhaustion.md`); `ub overflow, requires …` is
   **this page**, and `pip install`/`.run` advice costs the reporter a cycle and changes
   nothing.
4. **Take the tile constants from the pinned file, and use the traceback's line number —
   not the OCR'd call name — to say which variant ran.** In `vllm_ascend/ops/triton/rope.py`
   at v0.20.2rc1:
   - `_triton_rope` tiles q and k heads in `BLOCK_SIZE_HEAD` chunks (`:103`, `:142`); its
     own docstring says the tiling exists "to avoid UB overflow" (`:50`).
   - `rope_forward_triton` (`:256`) picks `BLOCK_SIZE_HEAD = 64` (neox) / `32`
     (interleaved) with a `# TODO: use a more robust method` (`:273-277`), sets
     `pad_rope_dim = triton.next_power_of_2(rope_dim)` (`:284`) and launches at `:285`
     (`:316-317` for the explicit `cos`/`sin` branch). Demand ∝
     `BLOCK_SIZE_HEAD × pad_rope_dim`.
   - `rope_forward_triton_siso` (`:349`) has **no head tiling**: `BLOCK_SIZE =
     pad_n_head` (`:363-365`) and the kernel loads a `[pad_n_h, pad_rope_dim//2]` tile in
     one go (`:232-249`). Its demand is ∝ `pad_n_head × pad_rope_dim`, so it is the
     variant that overflows on a many-head model.
   So `_triton_rope[(n_row,)]` at `:285` (the #18354 traceback) is the *tiled* variant and
   the overrun is `pad_rope_dim`-driven; an un-tiled `_triton_rope_siso` line would mean the head
   count is the axis. The shape inputs are `(head_size, rotary_dim, is_neox_style,
   n_q_head, n_kv_head)` — all from `config.json`, never from the weights.
5. **The validated envelope is the test file's parameter list, not the model zoo.**
   `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py:14-31` covers
   `(head_size, rotary_dim)` ∈ {(64, 32), (128, 128)} only, `(num_q_heads, num_k_heads)`
   ∈ {(64, 1), (96, 8)}, bf16/fp16, tokens {1, 4, 8, 16, 1024} — and both neox styles, so
   non-neox at (128,128) is *exercised*. A custom model with `rotary_dim` outside
   {32, 128}, `head_dim` outside {64, 128}, a partial-rotary factor (head_dim 128 with
   rotary_dim 64), or >128 q heads is **unvalidated**, and "this is a supported
   configuration" cannot be argued from the presence of a model in the tree.
6. **Levers at that pin, cheapest first.** (a) Shrink the tile constant in the installed
   file — `BLOCK_SIZE_HEAD` 64 → 16/8 (it must stay a power of two for `tl.arange`) is
   numerically exact: the tiling is a scheduling detail, and the check that it stayed
   exact is the same request's output against the non-Triton path. (b) Remove the compile
   step: the same function's `else` branch (`rotary_embedding.py:189`, `:205`) and the
   310P impl (`vllm_ascend/_310p/ops/rotary_embedding.py:121,141,157` — "`npu_apply_rotary_pos_emb`
   only supports rotary_dim 64 or 128") call `torch_npu._npu_rotary_embedding` /
   `torch_npu.npu_apply_rotary_pos_emb`, which have no Triton compile and therefore no UB
   tiling — but **no env switch forces that branch at v0.20.2**: `rope_forward_oot`
   (`:153`) selects Triton on `if HAS_TRITON:` (`:166-168`), and `HAS_TRITON`
   (`vllm/triton_utils/importing.py:12-13`) is `find_spec("triton")` plus an active-driver
   count, disabling itself only when that driver check fails. Uninstalling triton to reach
   the `else` branch also disables every other `if HAS_TRITON:` op — say so before
   suggesting it. (c) A newer release: the fix is invisible in a one-commit snapshot, so
   state the version gap instead of deriving it.
7. **A screenshot-only report can still be pinned — match the traceback's `path:line` to
   the release tree.** #18354 arrived as a photo; an offline OCR turned `_triton_rope` into
   `triton_rope` and digits into garbage, yet `rotary_embedding.py` "line 168
   `query, key = rope_forward_triton(`" and `ops/triton/rope.py` "line 285
   `_triton_rope[(n_row,)]`" match the pin byte-for-byte, which is what proves the
   reporter's `0.20.2rc1` wheel *is* this tree and lets the answer cite `path:line`. Do
   that before asking for a clean log, and re-check the other file the frames name
   (`rope.py:24:0` in the `loc(...)` header is the `@triton.jit` definition site, not a
   second fault).
8. **What to ask for, and what not to.** The two bit numbers (they are the arithmetic),
   the model's `config.json` rope block (`head_dim`, `rope_parameters` /
   `partial_rotary_factor`, `rope_scaling`, `is_neox_style`), `pip list | grep -i triton`,
   `which bishengir-compile`, the SoC from `npu-smi info`, and whether the identical model
   serves on another release. Do not ask for weights, `--enforce-eager`, graph-mode
   changes or a driver/CANN pairing: the failure is at kernel-compile time, before any
   graph, and eager runs compile the same kernel.

## Precedents

- **#8037** (v0.17.0rc1, Qwen3.5-35B-A3B, TP4, CANN 8.5.1): `requires 1835776 bits while
  1572864 bits available`, `Failed to run BiShengHIR pipeline` — the marginal-overrun
  reading, on a stock model, so reconcile the ratio with `BLOCK_SIZE_HEAD × pad_rope_dim`
  for that shape before crediting a custom checkpoint. Resolution not available in the
  snapshot (comments are).
- **#5983** (`[Bug]: bisheng-ub overflow`) is a forwarding stub with no environment and no
  log: cite it as the same signature, nothing else.
- **#18354** (v0.20.2rc1 `367b8e62da79`, A3, custom 讯飞星火 model, TP1, bf16, eager):
  `requires 15589248 bits` = 9.9×, OCR-only paste, `rope.py:285` = the tiled variant.

## Answer shape

Verdict first: `real, host-side: the Triton <kernel>'s compile-time tile exceeds the
vector-core UB (required N bits vs 1572864 available, R×), so it is the kernel's tile
constants × your rope shape — not the wheel/compiler pair and not the NPU`. Then the
ratio, the pin-matched `path:line` for the tile constant, the model fields that set
`pad_rope_dim`/head count, the tile-shrink or non-Triton-path lever with its exactness
check, and the version gap. Label the compiler's internal accounting as not derivable
from the tree, so the ladder does not depend on it.
