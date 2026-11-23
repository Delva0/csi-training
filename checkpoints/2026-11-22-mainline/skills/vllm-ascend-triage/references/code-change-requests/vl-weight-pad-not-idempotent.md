# VL vision qkv pad re-applied on a live weight reload

Scope: an RL/GRPO training loop (verl, ms-swift, trl) over a **Qwen2.5-VL /
Qwen2-VL** family checkpoint where **step 1 runs** and a later step dies with

```text
RuntimeError: shape '[-1, 3, 80, 1280]' is invalid for input size 1966080
```

(or the same shape with the reporter's own head dim / hidden size). PyTorch's own
wording is `invalid for input of size N` — reporters' titles often drop the `of`, so grep
`invalid for input` plus the shape, not the whole sentence. The paste names
no `vllm_ascend` frame because the raise is inside `Tensor.reshape`, and the reporter
usually offers no launch script. This is **not** a token-count, pixel or
`image_grid_thw` question, and not a model-constant hunt: it is the downstream
vision-tower weight **pad** being applied a second time, because the RL rollout
re-loads weights every step and `load_weights` rewrites `param.data` in place. The
deliverable is a change plan (the pad must be idempotent), so this belongs to this
topic even though the report carries a traceback.

Not this page: the same tower producing **wrong or degenerate answers** with no
traceback (`../accuracy-benchmarks/vl-vision-tower-degeneracy.md` — same pad path,
accuracy surface); a load that dies on the **first** load (`utils.py` shard assert,
`../startup-failures/checkpoint-file-not-safetensors.md`); an RL weight sync that
dies inside a **collective** (`../hccl-and-hardware/weight-sync-hccl-cascade.md`);
a reload whose destination was **emptied** by `dispose_layer`
(`live-weight-reload-vs-disposal.md`, `AssertionError: param_data.shape == loaded_weight.shape`).

## Step 1 — one division decides the family

The reshape's own last two dims are the tower's `origin_hidden_size_per_attention_head`
and its `hidden_size`, so `rows = M // hidden` and the tensor is the qkv weight.

`uv run --no-project python -c "M=1966080;H=1280;print(M//H, M/(3*80*H))"`
→ `1536 6.4`. Two readings, and only one survives:

| `M / (3 · origin_head_dim · hidden)` | Reading |
|---|---|
| integer (`4·` at TP=4, `8·` at TP=2 …) | the **unpadded** weight: first load, 3·hidden/tp rows |
| `1.6 × heads_per_partition` — 25.6 / 12.8 / **6.4** / 3.2 / 1.6 at TP 1/2/4/8/16 | the **already padded** weight: `3 · (16/tp) · 128` rows, i.e. head_dim raised 80 → 128 |

`1.6 = 128/80` is the pad ratio, so a factor that is exactly `1.6 × heads_per_partition`
with **no rounding** is the padded stack and nothing else. At v0.7.3 on a
1280/16/80 tower (Qwen2.5-VL **3B and 7B share this ViT**) the padded element counts
are TP1 `7864320`, TP2 `3932160`, TP4 `1966080`, TP8 `983040` — so the reporter's
`1966080` fixes TP=4, and the number is a *consistency check*, not the cause: the same
bug fires at every TP ≥ 2 with a different number in the message. Do not build a
TP-theory on the literal; read TP from their launch script.

## Step 2 — owner, verified at v0.7.3 (`779eebb`)

`vllm_ascend/models/qwen2_5_vl.py` (the same code is in `qwen2_vl.py`, so Qwen2-VL is
affected too), all at that commit:

| Line | Fact |
|---|---|
| `:45-46` | `MIN_PAD_SIZE = 64`, `MAX_PAD_SIZE = 128` |
| `:70-71` / `:168`, `:191-197` | head_dim in the **open** interval (64, 128) → head_dim := 128, `enable_pad = True` on the transformer |
| `:239-247` | `pad_qkv_weight` is exactly the failing reshape: `data.reshape(-1, 3, self.origin_hidden_size_per_attention_head, self.hidden_size)` |
| `:224`, `:259` | sibling `pad_qkv_bias` (`(-1, 3, 80)`) and `pad_proj_weight` (`(1280, -1, 40)`) — which one raises first depends on the shard order, so the paste's shape may be one of three and the family is the same |
| `:267-299` | `load_weights` runs the standard loader, then `param.data = self.pad_qkv_weight(param.data)` for **every** `attn.qkv.weight` on **every** call; `enable_pad` is never re-checked against the parameter's current shape |
| vllm v0.7.3 `ed6e907`, `vllm/model_executor/parameter.py:103-109` | the column-shard loader takes `shard_size = self.data.shape[output_dim]`, i.e. the **padded** row count, so the second write's offsets no longer line up with the checkpoint tensor and the out-of-range ranks raise before the pad |

Consequence: nothing prevents a second pad, and the pad can never succeed on a padded
parameter (`1.6 × heads_per_partition` rows of `hidden` never divide by
`3 · 80 · hidden`).

## Step 3 — why "step 2": the rollout re-loads weights

A plain `vllm serve` / one-shot `LLM()` loads the model **once** and cannot reproduce
this, which is the cheapest control to state in the answer: **the same tag, weights and
cards are fine for inference; only a per-step weight sync reaches the second
`load_weights`.** RL trainers call it through
`<engine>.model_executor.driver_worker.worker.model_runner.model.load_weights(...)`
or `collective_rpc("update_weights")` after each actor update (verl's
`fsdp_vllm` sharding manager is the recorded path), so:
step 1 = first load (works), step 2 = first sync (dies), and every run
"works for N-1 steps and fails on step N" with the same N.

The confirming evidence is in the **per-rank** log, and it is a two-message prediction
(one look at the other ranks' tracebacks, no source work):

- the ranks whose `tp_rank · padded_rows` still fits inside the checkpoint tensor
  (`3·hidden` rows) reach the pad and print the `reshape … invalid for input size`
  `RuntimeError` — at TP=4 that is ranks 0-1;
- the remaining ranks raise earlier, in the shard loader, as
  `IndexError: start (3072) + length (1536) exceeds dimension size (3840)` (or vLLM's
  `Attempted to load weight … into parameter …`);
- both messages in one run, on the same key, on the same step = this bug.

If instead the log shows a single consistent error on every rank, or the failure is on
step 1, this page is the wrong owner.

## Step 4 — the change plan (and what not to recommend)

1. **The pad must be idempotent, and the guard is the unpadded shape.** Pad only when
   the parameter is still unpadded — `param.data.shape[0] == 3 * hidden // tp_size` for
   qkv weight (bias: `3 * hidden // tp_size` elements; proj: `shape[1] == hidden // tp_size`),
   which is exactly what the column/row loader allocates.
2. **A guard alone is not sufficient, and say so.** Because the shard loader derives its
   offset from the *padded* destination, a second in-place write lands the wrong rows on
   every rank (rank 0 included, since the row stride itself moved) and leaves the previous
   pad's zero tail in place: the arithmetic failure turns into silent corruption of the
   vision tower. The correct fix is to re-derive the weight from the incoming shard and pad
   *that* (write the unpadded stack, then pad once at the end of each `load_weights`), or to
   keep an unpadded master copy and re-pad from it — not `if shape != padded: pass`.
3. **A release that can skip the padded tower is the workaround, if the pin has one.**
   From ~v0.9.x the registry picks `qwen2_5_vl_without_padding` under
   `USE_OPTIMIZED_MODEL` (`../startup-failures/environment-correspondence.md`), which
   removes this `load_weights` entirely. **At v0.7.3 it does not exist**:
   `vllm_ascend/models/` holds only `qwen2_5_vl.py` / `qwen2_vl.py`, and
   `models/__init__.py:9-12` registers the padded class unconditionally (`:5-7` does the
   same for Qwen2-VL) — so an upgrade, not an env var, is that rung. Re-grep the registry
   at the pin before offering it.
4. **`--enforce-eager`, `ASCEND_LAUNCH_BLOCKING`, `gc.collect()`, spawn start method and
   `num_speculative_tokens` remove no part of this path** (the pad runs in
   `load_weights`, before any graph or kernel), so none of them is a rung.
5. Report it as a defect of the pad's reload contract (a serving-only optimization that
   breaks live reload) — that framing, not "the reshape is wrong", is what a reviewer
   accepts, and it is the same class as `live-weight-reload-vs-disposal.md`.

## Step 5 — what to ask for, and what stays open

Ask for: the trainer's step-2 **per-rank** tracebacks (to check the two-message
prediction), `tensor_parallel_size` from the rollout config, the model directory's
`config.json` `vision_config` (`hidden_size`, `num_heads`, so `head_dim` is read, not
recalled), and confirmation that plain inference with the same tag does not crash.

Open here: which release first guarded the pad (the snapshot carries v0.7.3 only); whether
the reporter's later public reproduction (Qwen2.5-VL-**3B**, `tensor_model_parallel_size=2`,
so the message would read `3932160`) is the same run or a reduced one — the mechanism is
identical, the number is not, and the TP arithmetic above is what keeps the answer
independent of that ambiguity.

## Recorded case

**#809** (vllm 0.7.3 + vllm-ascend 0.7.3, verl GRPO, second step). The recorded run
derived this mechanism from the pin correctly and then went looking for a corpus
precedent it did not need (`history-titles.sh`/`/search/issues` both empty for the
issue's own corpus), was cut at the output limit and shipped **zero message text**.
The derivation is three calls — the division, one `rg -n` for the pad, one read of
`load_weights` — so the first message is the answer plus the per-rank check.
