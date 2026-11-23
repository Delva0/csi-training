# Static expert map (EPLB) and accuracy: the map is a deployment artefact

## Scope

An Ascend MoE serve that carries a **static expert map** —
`--additional-config '{"expert_map_path": …}'`, optionally with
`init_redundancy_expert` — or records one with `expert_map_record_path`, and reports an
accuracy/score loss. The sharpest shape is a training/rollout pipeline (RL "1 训 2 推",
reduced layers, per-step weight re-send, a P/D or P/2D layout) whose mainline inference
"scores fine" while the rollout path does not, or the bare question
"静态负载均衡有没有精度问题" with a screenshot and no log (`#5643`).

Not this topic: EP on = garbage / EP off = correct with **no** map
(`ep-mode-reduce-double-count.md` — the model's second all-reduce); a dead EPLB helper
subprocess that wedges the next forward (`../scheduler-admission/
background-helper-death-wedge.md`); an engine that dies **during load** with the map
(`../startup-failures/index.md`); a two-image score gap with no map in the command
(`harness-vs-engine-attribution.md`).

## Step 0 — decide whether the map is in play at all (one log grep, one config check)

Verified at vllm-ascend v0.11.0 (`2f1aed98ccdb`), `vllm_ascend/ops/common_fused_moe.py`:

| Gate | Line | Consequence |
|---|---|---|
| applied only when `expert_map_path` exists **and** is readable | `:181-183` | a wrong/unreadable path falls to the plain `determine_expert_map` branch (`:204-211`) and prints **nothing** — the serve looks normal and the map is not used |
| the file's own redundancy count replaces `init_redundancy_expert`, and `global_num_experts` is recomputed from it | `:186-189` | taken **before** the apply is attempted, so a later fallback keeps the file's counts |
| the apply is one `try/except Exception` | `:190-199` | a failed apply logs exactly one warning, `Init expert map of mtp/eagle when using sample.{e}`, then falls back to the default map (with the file's counts already adopted) |
| a successful apply on any non-`W8A8Dynamic` quant method raises | `:227-231` | `ValueError: Eplb supports only w8a8_dynamic quantization.` — so a **bf16** checkpoint that "runs with static EPLB" is running *without* the map: either the raise, or the silent fallback |
| the row index is `moe_instance_id`, a construction counter | `:148-149` | the file's row *n* is whatever MoE layer was built *n*-th — a reduced-layer export, PP splitting or a different model shifts every layer's entry |

Then read the two launch-log lines that say which of the three states you are in
(default map / file applied / file applied then refused):
`[EP Rank r/N] Expert parallelism is enabled. Local/global number of experts: L/G`
(`:214-219`; `G` must equal `num_experts + init_redundancy_expert`), and
`Init expert map of mtp/eagle …` above. No `/health`, no throughput and no crash line
distinguishes these states. Coverage at this pin, one grep apart, and the difference
matters: **dynamic** EPLB does have an accuracy gate — `tests/e2e/nightly/models/
test_deepseek_r1_w8a8_eplb.py:87` and `test_qwen3_235b_a22b_w8a8_eplb.py:83-85` pass
`{"dynamic_eplb": true, "init_redundancy_expert": 16}` and score gsm8k against
baseline 95 / threshold 5 — while the **static** replay (`expert_map_path`) appears in
`tests/ut/` and the docs only, and no `.github/workflows/` file names eplb at all. So a
green CI says nothing about a static map, and a dynamic-EPLB report has a baseline to
quote.

## Step 1 — the file against the deployment, one jq

```bash
jq '{moe_layer_count, devices: (.layer_list|length), first_device: .layer_list[0].device_count}' eplb.json
```

- `moe_layer_count` must equal the number of MoE layers the served model **actually
  builds**, not the checkpoint's `num_hidden_layers` before reduction.
- `first_device` / `layer_list[*].device_count` must equal the deployment's expert-parallel
  world size. `expert_load_balancer.py:49-61` builds an
  `(layers_num, ranks_num, num_experts)` tensor from exactly these two numbers and never
  compares them with the deployed `num_moe_layers` or EP size.
- The doc's contract, which the reporter has to match:
  `expert_map_record_path` records, `expert_map_path` replays, and
  `init_redundancy_expert` **must equal the TP size**
  (`docs/source/user_guide/feature_guide/eplb_swift_balancer.md:39-52,72`; its own pitfalls
  list `:97-98`). A map recorded on the unreduced model, on another topo, or with a
  different redundancy is the shape this report describes.
- `get_global_redundant_expert_num` reads `expert_map_tensor[0][0]` (`:98-102`) — it
  assumes every layer carries the same redundancy, so a hand-edited or partly recorded
  file is mis-read rather than rejected.
- In a P/D deployment, one file per instance is one chance to diverge:
  `check_expert_map_tensor` (`:104-119`) only `dist.all_gather_object`s inside **one**
  dist world, so prefill and decode instances are never cross-checked.

## Step 2 — the ladder, cheapest first

1. **Map ablation**: the same serve command with `expert_map_path` removed, EP still on,
   same requests. Accuracy returns ⇒ the map lane (below); it does not ⇒ stop ranking
   maps and take `ep-mode-reduce-double-count.md` (EP arithmetic) or
   `harness-vs-engine-attribution.md` (measurement). One restart, and it splits the whole
   report.
2. **Zero-redundancy map**: record with `init_redundancy_expert: 0` and replay it. The
   off-rank `log2phy` slots are filled by an **unseeded** `random.choice`
   (`expert_load_balancer.py:83`), so equal-by-construction copies are never guaranteed;
   recovery on a zero-redundancy map localises the fault to the redundancy slots.
3. **Count consistency**: `w8a8_dynamic.py:209` asserts
   `…[1] == global_num_experts - global_redundant_expert_num`
   (`Number of global experts mismatch (excluding redundancy)`), so a map whose
   per-layer redundancy differs from the launch's `init_redundancy_expert` is a loud
   failure at the routing site, while the same mismatch on another quant path is silent.
4. **Per-instance identity** in P/D: `sha256sum` of every instance's map plus identical
   `--tensor-parallel-size`/`--enable-expert-parallel`. Mismatched maps put experts on
   ranks the instance does not have — weights partially loaded, fluent output, no crash.
5. **Weight re-send vs physical slots** (hypothesis, one check): in an RL/rollout loop
   that re-sends weights every step, ask whether the re-send order is the *physical*
   (map-permuted) slot order — the engine's own expert-weight update is
   `vllm_ascend/eplb/adaptor/vllm_adaptor.py` and
   `eplb/core/eplb_device_transfer_loader.py` (`rg -n 'update_expert_weight'
   vllm_ascend/eplb/`), while the trainer side is outside this snapshot. A re-send in
   logical order leaves stale copies behind only where redundancy exists, which is why
   step 2 separates it; do not rank this above the ablation while the map file's shape is
   unknown.

## Step 3 — what the answer owes

Verdict first: whether the map can be in play at their checkpoint's quant method
(bf16 ⇒ no), then the ablation rung and the two log lines to grep. Ask for the map file's
`jq` output, the three serve commands (P and each D), the `L/G` and
`Init expert map …` lines, and how layers were reduced — the map is data the reporter
holds, not something this snapshot can settle. Scope every claim to the pin above: the
file→tensor reshape and the w8a8 gate move between releases, so re-grep
`moe_layer_count|Eplb supports only` at theirs.

## Recorded case

`#5643` — v0.11.0 (`2f1aed98ccdb`), RL 1 训 2 推 with reduced layers and
`expert_map_path`, "静态负载均衡精度正常 / 减层后对不上", evidence one garbled-OCR
screenshot. The record and its 4-rung ladder live in
`../evidence-workspace/recorded-cases-accuracy-and-output.md` (§`#5643`); no reporter
confirmation, so the ablation above is the deliverable rather than a diagnosis.
