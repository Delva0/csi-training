# Output stops after tens of tokens ("被截断", early EOS)

Scope: the answer is fluent but short — "只输出几十个 tokens", "输出默认被截断" —
and the reporter says a large `max_tokens` makes it go away. `usage.prompt_tokens`
is small (a few hundred to a few thousand), the process is alive and there is no
traceback. This is not `garbled-long-context-output.md` (100K+ prompts) and not
`short-prompt-corruption.md` (word salad rather than early stop): here the text is
sane, it just ends too soon.

## Step 1 — one response field decides the lane

| What the response says | Lane |
|---|---|
| `finish_reason: length` and `usage.completion_tokens == max_tokens` | Budget: the effective `max_tokens` is smaller than the request thinks (`--default-max-tokens`, `generation_config.json`'s `max_new_tokens`, a harness param, `--max-model-len` headroom). See `harness-vs-engine-attribution.md` |
| `finish_reason: stop` with `completion_tokens` in the tens | The **model** emitted EOS: generation degeneration, and every flag the reporter toggled at the same time is a suspect |
| `content: null`, the text inside `reasoning` | Output routing, not the engine — `reasoning-output-routing.md` |

One request separates the first two: same prompt plus
`{"temperature": 0, "max_tokens": 128, "ignore_eos": true}`. Generation continues
past the old stop point (content drifting or repeating) ⇒ early EOS. Stops in the
same place ⇒ sampler/stop-token/engine-side condition. Ask for the two raw
response bodies with `finish_reason` and `usage`; a screenshot of the text alone
leaves both lanes open.

## Step 2 — when a switch was toggled, audit the switch's wiring, not the sampler

Rule: `rg -n "<key>"` the pinned tree and check **each reader's condition
separately**. A single user-facing switch usually has a compile-pass half and an
env half, and they are not the same predicate. Recorded family: Sequence
Parallelism in vllm-ascend.

| Reader | Predicate | Effect |
|---|---|---|
| `vllm_ascend/utils.py:759-760` `enable_sp_by_pass()` | `not enforce_eager and pass_config.enable_sp` | the pass-side switch; `--enforce-eager` turns it off regardless of config |
| `vllm_ascend/worker/model_runner_v1.py:1822-1830`, called `:1902` | `enable_sp(vllm_config) or enable_sp_by_pass(vllm_config)` | pads `num_scheduled_tokens` up to a multiple of TP |
| `vllm_ascend/compilation/graph_fusion_pass_manager.py:72-75` | `config.compilation_config.pass_config.enable_sp` — no model-class and no quantization condition | registers `AscendSequenceParallelismPass` (AllReduce+RMSNorm → ReduceScatter+RMSNorm+AllGather) |
| `vllm_ascend/compilation/passes/sequence_parallelism.py:12-20`, `:184`, `:192-198` | `get_sp_threshold()`: **`1` when `is_moe_model(config)`**, else `additional_config["sp_threshold"]`, default `1000`; applicability = compile-range start ≥ threshold | on a MoE model the pass applies at every batch size; on dense models it needs ≥ 1000 tokens per range |
| `vllm_ascend/ascend_config.py:201-207` | `pass_config.enable_sp and not is_moe_model(...)` | appends the `sp_threshold` split point — MoE models are excluded here |
| `vllm_ascend/utils.py:763-781`, `vllm_ascend/envs.py:74` | env `VLLM_ASCEND_ENABLE_FLASHCOMM1` or legacy `VLLM_ASCEND_ENABLE_FLASHCOMM`, plus `enable_shared_expert_dp` | the runtime FC1 half. Consumers: `vllm_ascend/patch/worker/patch_qwen3_5.py:75,240-250` (GDN slicing — Qwen3.5 is a GDN hybrid) |

## Step 3 — read the shipped support matrix before inventing a mechanism

`docs/source/user_guide/feature_guide/sequence_parallelism.md` (v0.17.0rc1):

- `:12` "implemented Sequence Parallelism for **VL-class models** based on the
  Inductor pass"; `:21` "SP relies on graph mode … not supported in eager mode".
- `:41-47` matrix, without quantization: SP = `graph` for **VL + Dense** only;
  `x` for VL + MoE, non-VL + Dense, non-VL + MoE. FC1 = `eager/graph` for both
  non-VL columns.
- `:52` "SP currently does not support quantization".
- `:32-39` FC1 is a *different* optimisation (Custom OP based, hard for VL-class
  models), and "FC1 and SP are complementary" — **not** two names for one switch.
- `docs/source/user_guide/release_notes.md:97` the key was renamed back
  (`enable_flash_comm_v1` → `enable_sp`, #6883), so wording in an older guide is
  not evidence about the current key.

So for a non-VL MoE quantized model (w4a8/w8a8), `pass_config.enable_sp: true` is
**outside the documented matrix on two counts**, while the code still registers the
pass (no model or quantization gate, `graph_fusion_pass_manager.py:72`) and
`get_sp_threshold` returns `1` for MoE — i.e. the excluded combination is
activated, at every batch size, with padding switched on. That is the lead claim
for this shape; it is source-verified, and the doc matrix is what makes it
actionable.

Label the tempting story carefully: "`pass_config.enable_sp` alone half-enables
SP, so logits are wrong" is **not** established by the code. The pass rewrite is
self-contained, the doc's own VL + Dense recipe sets only `pass_config` (no env
var), and the runtime FC1 branches are a different feature. What is established is
the documented-scope mismatch above plus the fact that the key is still honored —
so "this parameter is no longer supported / silently dropped" (a very common
reporter conclusion) is false at the pinned commit and should be corrected
explicitly.

## Step 4 — the ablation rows to hand back

| Row | Change | What it proves |
|---|---|---|
| 0 | capture `finish_reason`, `usage`, and one `ignore_eos: true` run | early EOS vs budget (step 1) |
| 1 | delete only the `enable_sp` key, keep the rest of `--compilation-config` | isolates the switch from the rest of the config |
| 2 | drop `--compilation-config` entirely | isolates the graph-mode change itself |
| 3 | row 1 + `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | the documented non-VL/MoE switch (FC1), not a repair of SP |
| 4 | row 1 + `--enforce-eager` | `enable_sp_by_pass` is false in eager, so the pass half is off — same observable as row 1 if the pass is the trigger |
| 5 | row 1 + `--additional_config '{"sp_threshold": 1000}'` | dense-only knob; on a MoE model the threshold is 1, so a change here means something else |
| 6 | MTP off, then TP/TBT matched against the second node | the recorded pair differed by node, so shape/topology, not the flag alone |
| 7 | `VLLM_LOGGING_LEVEL=DEBUG … 2>&1 \| grep -E "Replaced .* patterns\|compile_ranges_split_points\|sequence parallel"` | whether the pass rewrote the graph at all (`sequence_parallelism.py:186-190` logs the match count, `ascend_config.py:207` the split point) |

## Recorded run: #12253 (vllm-ascend v0.18.0-a3, Qwen3.5-397B-A17B-w4a8-mtp, 8×910C)

Reporter-confirmed facts: works on v0.17.0rc1; breaks on v0.18.0-a3; the *same*
image and flags on a second 910C node do **not** reproduce; deleting
`"pass_config": {"enable_sp": true}` from the start script restores normal output;
their diagnosis is "the parameter is no longer supported in v0.18.0+, confirmed by
the compute team", with `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` as the workaround.

The verified reframe (pinned tree v0.17.0rc1, `e20f0b1`): the key **is** read, at
the six sites in step 2, so it is honored rather than dropped; the documented
matrix excludes this model class (non-VL + MoE) and every quantized model; the
registrations have no model gate; and FC1 is a separate, complementary feature —
so the reporter's temporary fix is a different switch, not the same one renamed.
"Another node does not reproduce" fits a model-class/quantization-agnostic graph
rewrite whose match depends on shapes and batch sizes, so the second node is a
data point to match (TP, `max-num-batched-tokens`, images) rather than a
refutation.

Gaps: the snapshot carries v0.17.0rc1 only — no v0.18.0-a3 tree — so *what changed
between the working and failing release* is unverified. Ask for that image's
`vllm_ascend/utils.py` (`enable_sp*`), `compilation/graph_fusion_pass_manager.py`
and `compilation/passes/sequence_parallelism.py`, plus both response bodies. Say
which half is verified and which is a hypothesis; do not let the version-flip
story inherit the confidence of the source-verified matrix.

Delivery note: this run is the calibration row for the shape — 14 assistant turns,
16 tool calls, a complete answer delivered on the last turn because
`/workspace/scratch/findings.md` was written one turn earlier, which is how an
answer survives to the end. It still led with the unverified half-gate mechanism; leading
with the doc matrix would have been one call cheaper and true at the pinned commit.
