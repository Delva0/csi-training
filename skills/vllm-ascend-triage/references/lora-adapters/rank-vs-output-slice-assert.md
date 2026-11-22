# `hidden in should be smaller than hidden out`: `--max-lora-rank` against the narrowest LoRA slice

Source-verified at vllm-ascend **v0.23.0** (`5cb98caa`) + vllm **v0.21.0** (`ad7125a4`),
from the trees only, for recorded #14728 (Qwen3.5-4B, SWIFT PEFT adapter `"r": 64`
targeting `in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|q_proj|k_proj|v_proj|o_proj|out_proj|
gate_proj|up_proj|down_proj`,
`--enable-lora --max-lora-rank 64`, image `v0.23.0rc1`, one 910B4). That session ended
`output_truncated` with no answer while it held this mechanism by its third turn — see
*Cost control* below. No reporter confirmation and **no in-tree fix**: state it as
source-verified at those commits and keep the fix attributed to `csrc/`.

Second driver, same vllm-ascend pin but vllm **v0.23.0** (`0fc695fc`): recorded
**#16586** (Qwen3.5-27B, 910B3 ×8, adapter `r=8`, `--max-lora-rank 8`,
`--tensor-parallel-size 4`, `--enforce-eager`, GDN `in_proj_ba` `[48, 48]`), where the
reporter arrived with the mechanism *and* a patch. Both hold, with one attribution
correction that matters: the `TP × rank` inflation is **not** in `punica_npu.py` (no
`all_gather` in `vllm_ascend/lora/` at all) but in upstream `_mcp_apply`. On a serve
line carrying `--tensor-parallel-size` > 1 read the TP section below before ranking
anything: the flag is small, the operand is not.

## The report

- The engine never becomes ready: the container starts, `vllm serve … --enable-lora
  --max-lora-rank 64` dies during init, and the only error text is
  `RuntimeError: hidden in should be smaller than hidden out`. `--enforce-eager`
  changes nothing (reporter-confirmed) — which is the first tell, because nothing in
  this path touches graph capture.
- The body carries the adapter's `target_modules` and `"r"`, and usually not the
  model's `config.json`. Those two are enough: the failing value is a *width*, and
  both widths are computable from them plus the serve line.
- **At `--tensor-parallel-size > 1` the same message arrives on a serve no flag
  should have broken**: `--max-lora-rank 8`, equal to the adapter's own `r`, against a
  48-wide packed part (12 per card at TP=4) still aborts, because the operand width
  is `TP × max_lora_rank` = 32. A TP>1 report quoting a *small* rank is therefore not
  "the flag was too big" — ask for the TP size before answering, and never send
  "lower `--max-lora-rank`" to a serve that already runs it at `r`.

The message names neither operand — a fair complaint, and the pin's own
`docs/source/user_guide/feature_guide/lora.md` documents no rank/width limit at all
(its only caveat at `:30` is MoE/EP/quantisation), so the reporter's ask for an
actionable error is part of the answer, not a side note.

## The arithmetic (this is the whole diagnosis)

`rg -n 'hidden in should be' <vllm-ascend-root>` returns exactly four hits in
`csrc/torch_binding.cpp`:

| line | op | check |
|-|-|-|
| `:435`, `:519` | `bgmv_shrink`, `sgmv_shrink` | `TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out")` — module *input* width vs rank |
| `:476`, `:561` | `bgmv_expand`, `sgmv_expand` | `TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out")` — the reported one |

So the failing op is the **expand** of a **sliced** (packed) target:

1. `x.size(1)` is the shrink buffer's width, and that is the engine's
   `--max-lora-rank`, **not** the adapter's `r`:
   `vllm_ascend/lora/punica_npu.py:315` `r = lora_b_stacked[0].size(-1)` sizes the
   buffer, and the stack's last dim is `lora_config.max_lora_rank`
   (`vllm/lora/layers/base_linear.py:151-172` `create_lora_weights`). **This holds at
   `--tensor-parallel-size 1` only** — at TP>1 upstream multiplies it by TP (next
   section).
2. `slice_size` is the per-part width of the packed module:
   `vllm/lora/layers/column_parallel_linear.py:196-202`
   `output_slices = tuple(divide(output_size, tp_size) …)`, one slice per
   `output_sizes` entry, passed down as `y_slice_size`
   (`punica_npu.py:106-145` `_expand_slice_prefill`/`_expand_slice_decode`).
3. Those entries are the model layer's own split. For Qwen3.5's GDN,
   `in_proj_ba` is `[num_v_heads] * 2` and `in_proj_qkvz` is
   `[key_dim, key_dim, value_dim, value_dim]`
   (`vllm/model_executor/layers/mamba/gdn_linear_attn.py:435-455` `create_ba_proj`,
   `:409-433` `create_qkvz_proj`, `gqa_interleaved_layout=False` on Qwen3.5), and
   `linear_num_value_heads` defaults to **32**
   (`vllm/transformers_utils/configs/qwen3_5.py:64`; the 4B checkpoint may set less).

**Verdict sentence: at `--tensor-parallel-size 1` the assert fires iff
`--max-lora-rank` exceeds the smallest output slice among the adapter's target
modules; at `TP > 1` the predicate is `TP × --max-lora-rank <= smallest per-rank
slice`, because the operand is the all-gathered shrink buffer.** For the recorded
report `64 <= 32` is false on `in_proj_b`/`in_proj_a`, and every other target in that
regex (`q_proj` 2048+, `o_proj`, `down_proj`, …) is wide enough — which is why the
same adapter would work with those two modules removed.

The project's own control is in the tree: the Qwen3.5 dense LoRA e2e runs
**`max_lora_rank=8`** on `Qwen/Qwen3.5-4B`
(`tests/e2e/pull_request/one_card/lora/test_qwen35_densemodel_lora.py:82`), i.e.
`8 <= 32` passes and CI never reaches the guard. Say so instead of "LoRA on Qwen3.5
is broken" — the delta is one flag value against one config field.

## At `--tensor-parallel-size > 1` the left side is `TP × max_lora_rank`

Nothing in `vllm_ascend/lora/` gathers anything (`rg -n 'all_gather'
vllm_ascend/lora/` → 0 hits at v0.23.0), so a report that blames `punica_npu.py` for
the multiplication is half right: the gather is **upstream**, one frame above every
LoRA apply on a column-parallel layer.

- `_mcp_apply` (`vllm/lora/layers/column_parallel_linear.py:24-80`, used by the
  column-parallel, merged-column, QKV and variable-slice wrappers) sizes the shrink
  buffer `[n_slices, tokens, local_lora_rank]` (`:43-44`), fills it, and then does
  **`buffers = tensor_model_parallel_all_gather(buffers)`** (`:64`).
  `tensor_model_parallel_all_gather` defaults to `dim=-1`
  (`vllm/distributed/communication_op.py:17-21`), so the width becomes
  `TP × local_lora_rank`. `slice_lora_a` returns `lora_a` unchanged (`:102-103`), so
  the gathered chunks are identical replicas and the reporter's "mathematically
  redundant" argument is right.
- **At TP=1 the gather returns its input untouched** — `all_gather` bypasses at
  `world_size == 1` (`vllm/distributed/parallel_state.py:636-639`) — which is exactly
  why the TP=1 arithmetic above, the pin's one-card e2e and every CI LoRA row hold,
  and why the same image fails on a TP=4 serve.
- `punica_npu.py:214-255` (`add_expand`) forwards `x[slice_idx]` (`:247`) unchanged to
  `_apply_expand` → the AscendC binding sees width `TP × rank`.
- Upstream GPU never trips: the triton expand takes `K = lora_b_weights[0].shape[-1]`
  (`vllm/lora/ops/triton_ops/lora_expand_op.py:206`) and strides the gathered rows, so
  the extra columns are ignored. One read gets you that file — it defines
  `_lora_expand` / `_lora_expand_kernel`, so grep `lora_expand` there, not
  `def lora_expand` (#16586 lost two turns to that name and to `lora_op.py`, which
  does not exist at v0.23.0).

**The binding's contract is stricter than its `TORCH_CHECK`.** The check compares the
width against the *slice* (`csrc/torch_binding.cpp:476,561`), but the same width is
passed to the kernel as the **rank**: `int lora_rank = x.size(1);` (`:489`, `:576`),
and the kernel builds its weight stride and reduction from it —
`singleLoRAWeightLen_ = maxLoRARank_ * outputHiddenDim_` (`csrc/kernels/bgmv_expand.cpp:67`,
`sgmv_expand.cpp:67`), `numOutputElementsPerInputTile_ = BLOCK_REDUCE_NUM_REPEATS *
(NUM_ELEMENTS_PER_REPEAT / maxLoRARank_)` (`:87`) and a rank switch that only computes
for `{8,16,32,64}` (`:252-276`). So `width != rank` is wrong math or a weight read
outside the adapter's block **even when `width <= slice_size` passes**. Source-derived,
not reporter-confirmed, and worth one line in the answer: on a TP>1 serve whose
targets are all wide (only `q_proj|k_proj|v_proj|o_proj`, say) the same mismatch can
start cleanly and still give adapter answers that differ from TP=1. The discriminator
is one **TP=1 vs TP=4** run of the same adapter, base-vs-adapter compared — not
`--dtype`, not `--enforce-eager`.

**No flag rung exists at TP>1 for a column-parallel target** (row-parallel wrappers do
not gather, so `o_proj`/`down_proj` adapters are unaffected). The `--max-lora-rank 128`
demotion in *Rungs, cheapest first* below does **not** rescue it: that path is
`torch.einsum("bi,boi->bo", inputs, selected_loras)`
(`vllm/lora/ops/torch_ops/lora_ops.py:123`), which needs the input width to equal the
rank, so the gathered buffer raises a size mismatch instead of a `TORCH_CHECK` — and
310P, always on that path, is in the same position. Rungs, cheapest first:

1. **`--tensor-parallel-size 1`** as the diagnostic: it confirms the multiplier in one
   run, and is a legitimate short-term serve for a single-card-fits model.
2. **Narrow inside `add_expand` before the op** — `x_i = x[slice_idx][..., :w.shape[-1]]`
   **and** `.contiguous()`. The copy is required, not cosmetic: the binding hands
   `x.data_ptr()` to the kernel with no stride arguments
   (`csrc/torch_binding.cpp:482,567` → `:489,576`), so a strided narrow is read as
   contiguous (silent garbage), which is the reported decode hang's mechanism. Make it
   an assert rather than a silent `if` (`assert x_i.shape[-1] % w.shape[-1] == 0`) so a
   future change to the gather fails loudly; it costs nothing at TP=1, where the widths
   are equal and no copy happens.
3. **Drop the packed/column targets or `merge_and_unload`** when `r` itself exceeds the
   per-rank slice (the TP=1 rung 2 arithmetic, with the slice divided by TP).
4. **File the structural repair** where it belongs: narrow before the gather (upstream),
   or let the AscendC binding take the row stride instead of a raw pointer, and print
   both operands in the `TORCH_CHECK`.

**Recheck condition:** any pin whose `punica_npu.add_expand` narrows to
`lora_b_stacked[i].shape[-1]`, or whose upstream stops gathering the shrink buffer,
closes this. Two greps decide it: `rg -n 'all_gather|shape\[-1\]'
vllm_ascend/lora/punica_npu.py` and `rg -n 'tensor_model_parallel_all_gather'
<vllm-root>/vllm/lora/layers/column_parallel_linear.py`.

## Why it fails at init, and why `--enforce-eager` cannot help

The LoRA math runs during the memory-profiling warm-up, before any request:
`vllm/v1/worker/gpu_worker.py:388-392` wraps `profile_run()` in `memory_profiling`,
and the runner's `_dummy_run` executes under
`maybe_dummy_run_with_lora(...)` (`gpu_model_runner.py:5552` →
`lora_model_runner_mixin.py:237-266` `maybe_setup_dummy_loras`), which installs
zero-initialised dummy adapters (`vllm/lora/model_manager.py:485` `create_dummy_lora`)
through **every** wrapped module. Since the buffer width is `max_lora_rank` — times
`TP` once the gather has run — whatever the dummy adapter's own rank is, a single
narrow target aborts startup on a serve that would otherwise work. `--enforce-eager` and `--max-loras` are inert here: the
check is host-side, inside the op binding, before any device op is launched.

## Rungs, cheapest first

1. **`--max-lora-rank` down to the adapter's real rank** (it must still satisfy
   `max_lora_rank >= r`, `vllm/lora/peft_helper.py:120-123`; the default is 16,
   `vllm/config/lora.py:34` at v0.21.0, `:33` at the earlier refs). A serve that
   merely over-provisioned the flag (rank-8
   adapter, `--max-lora-rank 64`) is fixed by this alone (`8 <= 32`).
2. **When `r` itself exceeds the narrowest slice, no flag value works.** Drop those
   targets: retrain/export the adapter without `in_proj_a|in_proj_b`, or
   `merge_and_unload` into the base weights (upstream GPU has no such limit — see
   *Upstream/downstream* — so the merged serve is the only exact local path).
3. **`--max-lora-rank 128` demotes the serve to the PyTorch punica**
   (`punica_npu.py:23-43`: `310P` **or** `max_lora_rank >= 128` selects
   `vllm.lora.ops.torch_ops`), where `bgmv_expand_slice` is
   `einsum(...)` + `output[:, off:off+size] += …`
   (`vllm/lora/ops/torch_ops/lora_ops.py:110-128`) and rank ≤ slice is legal. **Not a
   rung at TP>1** — that `einsum` needs the width to equal the rank, so the gathered
   buffer raises a size mismatch (TP section below). Cost:
   that path is dynamo-visible — expect `Could not guard on data-dependent
   expression …` unless eager (`pytorch-punica-compile-guard.md`), and it is slower
   per decode step (`decode-overhead-vs-base-model.md`). Present it as a diagnostic
   that confirms the mechanism, not as the recommended serve.
4. **File it as a `csrc/` change, with the two numbers.** The guard is a *tiling*
   guard, not a typo: at rank 64 one input tile computes
   `BLOCK_REDUCE_NUM_REPEATS * (NUM_ELEMENTS_PER_REPEAT / maxLoRARank_) = 128`
   output elements (`csrc/kernels/bgmv_expand.cpp:87`, with
   `W_IN_TILE_NUM_ELEMENTS = 8192` at `:48` and `SUPPORTED_RANKS = {8,16,32,64}` at
   `:31`), so a 32-wide slice has no lane. Deleting the `TORCH_CHECK` would compute
   past the slice; the repair is a narrow-slice/tail path in
   `csrc/kernels/{bgmv,sgmv}_expand.cpp` or a padded slice in the binding.

## Upstream/downstream

Upstream vLLM has no rank-vs-width constraint: the triton expand op asserts nothing
about `r` against the slice — it silently reduces over
`K = lora_b_weights[0].shape[-1]` (`vllm/lora/ops/triton_ops/lora_expand_op.py:206`)
and strides past any wider operand — and the torch fallback adds a well-formed
`[M, size]` GEMM (`vllm/lora/ops/torch_ops/lora_ops.py:110-128`). Both tolerate a
*superset* operand where the AscendC op requires an exact rank, which is the whole
drift: upstream's tolerance is what lets `_mcp_apply` all-gather the shrink buffer
unconditionally (`column_parallel_linear.py:64`), and the downstream binding inherits
an operand its kernel cannot use. The refusal is **downstream and AscendC-only**, and
it is version-scoped to the refs whose `punica_npu.py` binds
`vllm_ascend.lora.lora_ops` (re-grep the predicate; 310P always takes the torch
path, and so does any `--max-lora-rank >= 128`). What would invalidate the claim:
a `bgmv/sgmv_expand` binding at the pin without the two `TORCH_CHECK`s, a
`create_ba_proj` whose `output_sizes` are no longer one entry per checkpoint part, a
`punica_npu.add_expand` that narrows to the weight's last dim, or an upstream
`_mcp_apply` that stopped gathering.

## Not the cause (do not rank these)

- **Graph capture.** `--enforce-eager` leaves the crash unchanged; the check runs in
  the Python binding, not in a captured graph.
- **The adapter files or the LoRA loading gate.** A rank gate raises its own
  `ValueError` (`LoRA rank 64 is greater than max_lora_rank 32`), and a broken
  adapter fails earlier or silently (`identical-output-adapter-dropped.md`).
- **`VLLM_ASCEND_LORA_MATMUL_FASTPATH`**, which several third-party training recipes
  export: no consumer at v0.23.0 (`rg` over the tree returns 0 hits, `punica_npu.py`
  included). One grep, then say it is inert at the pin.
- **dtype, `--max-loras`, version skew.** The message is a width comparison; none of
  these changes a width.

## Cost control (the recorded #14728 failure)

Two greps and two arithmetic facts end this case: the `TORCH_CHECK` line and the
binding/`output_slices` pair, then `max_lora_rank` vs `num_v_heads/tp`. #14728 found
the `csrc` line by its third turn and then spent the remaining budget on
`bgmv_expand.cpp`'s 368 lines of AscendC tiling, upstream `vllm/lora/layers/`, the
Qwen3.5 config defaults, release notes and a 31-hit title search — and shipped
nothing. The kernel's tiling arithmetic only matters for rung 4's wording; it cannot
change the verdict, and the message the reporter needed was the verdict.

#16586 is the second instance: the gather at `column_parallel_linear.py:64` and the
binding's `lora_rank = x.size(1)` were both in hand by its eighth turn, yet it shipped
its answer on its 22nd with 15 847 of the 16 384 output tokens spent — two of those
turns on a guessed triton path (`lora_op.py`) and the `def lora_expand` name that does
not exist. With the TP section above, the answer is: the predicate with `TP ×`
restored, the upstream-narrows/downstream-binds drift, one TP=1 ablation, and the
`[..., :rank].contiguous()` patch — no kernel walk, no upstream `lora/` survey.

## Ask for (only what changes the answer)

`adapter_config.json` (`r` and the full `target_modules` regex), the serve line's
`--max-lora-rank` **and `--tensor-parallel-size`**, and the model's `config.json`
(`linear_num_value_heads`, `linear_num_key_heads`, `linear_value_head_dim`). If they
can paste it, the frame *between* `Using PunicaWrapperNPU.` and the `RuntimeError`
names the module, which confirms which slice; without it the arithmetic above already
names it. For a TP>1 serve add the one run that decides which mechanism is live: the
same adapter at `--tensor-parallel-size 1`, comparing adapter output against the base
model — equal answers mean the width contract, not the dtype gate
(`identical-output-adapter-dropped.md`).
