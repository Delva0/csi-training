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
| `finish_reason: length` and `usage.completion_tokens == max_tokens` | Budget: the effective `max_tokens` is smaller than the request thinks (`--default-max-tokens`, `generation_config.json`'s `max_new_tokens`, a harness param, `--max-model-len` headroom). See `harness-vs-engine-attribution.md` — and see the checkpoint lane below, which makes "只在大 `max_tokens` 时正常" literal |
| `finish_reason: stop` with `completion_tokens` in the tens | The **model** emitted EOS: generation degeneration, and every flag the reporter toggled at the same time is a suspect |
| `content: null`, the text inside `reasoning` | Output routing, not the engine — `reasoning-output-routing.md` |

**The checkpoint lane has one grep and one log line.** At vllm v0.22.1
`ModelConfig.get_diff_sampling_param()` (`vllm/config/model.py:1480-1522`) reads
`generation_config.json` and renames `max_new_tokens` to `max_tokens`, so a
quantised checkpoint that copied `max_new_tokens` sets the *default* budget while an
explicit `max_tokens` in the request overrides it. Confirm with
`cat <model>/generation_config.json | grep -E "max_new_tokens|eos_token_id"` and the
serve log's `warning_once` "Default vLLM sampling parameters have been overridden
by the model's `generation_config.json`: {...}. If this is not intended, please
relaunch vLLM instance with `--generation-config vllm`". If that line is present the
reporter's sentence is explained **without** early EOS — but the body they pasted
still has to say `length`, not `stop`; never merge the two lanes into one story.

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

Two refs are recorded and the names moved between them: v0.17.0rc1 (`e20f0b1`,
used for #12253) and v0.22.1rc1 (`5f6faa0`, used for #12278). The table is the
v0.22.1rc1 wiring; the last column keeps the v0.17.0rc1 name so an older report is
still readable.

| Reader (v0.22.1rc1) | Predicate | Effect | Was (v0.17.0rc1) |
|---|---|---|---|
| `vllm_ascend/compilation/graph_fusion_pass_manager.py:74-79` | `config.compilation_config.pass_config.enable_sp` — no model-class and no quantization condition | registers **both** `SequenceParallelismPass` and `SequenceParallelismMoePass` | same site, `AscendSequenceParallelismPass` only (`:72-75`) |
| `vllm_ascend/compilation/passes/sequence_parallelism.py:12-20` | `get_sp_min_token_num()`: **`1` when `is_moe_model(config)`**, else `SP_MIN_TOKEN_NUM_DEFAULT = 1000` | per-range applicability `compile_range.start >= min_tokens` | `get_sp_threshold()` reading `additional_config["sp_threshold"]` (`:12-20`) |
| `vllm_ascend/platform.py:249-255` | `pass_config.enable_sp and pass_config.sp_min_token_num is None` → `get_sp_min_token_num(...)`; logs `Set sp_min_token_num. sp_min_token_num=%s` | fills the default. `sp_min_token_num` is now **upstream's** field (`vllm/config/compilation.py:175`), and this runs from `apply_config_platform_defaults` (`vllm/config/vllm.py:1081`) **before** upstream's own heuristic (`:1135-1161`), so the Ascend number wins | the split point was appended off `sp_threshold` in `ascend_config.py:201-207`; `sp_threshold` no longer exists in the tree |
| upstream `vllm/config/vllm.py:1134-1161` | `fuse_gemm_comms` ⇒ `enable_sp`; **TP==1 ⇒ `WARNING … Sequence Parallelism requires TP>1, disabling`**; hidden-size heuristic returning `None` ⇒ disabled ("Model hidden_size too small for the SP threshold heuristic") | the flag can be turned off upstream before any Ascend code reads it — **check TP>1 first**, the cheapest way to rule this switch out | not present |
| upstream `vllm/config/vllm.py:1805-1830` | `min_token_num < max_num_batched_tokens and min_token_num > 1` ⇒ append endpoint `min_token_num - 1` to `compile_ranges_split_points` | with the MoE value `1` no split point is added, so SP applies to the whole compile range | — |
| `vllm_ascend/ascend_config.py:260-264` (read via `utils.py:843-844 enable_sp_by_pass()`) | `model_config is not None and not enforce_eager and pass_config.enable_sp` | the pass-side switch as other code sees it; `--enforce-eager` turns it off regardless of config | `utils.py:759-760`, same predicate |
| `vllm_ascend/utils.py:847-873` `enable_sp()` | `additional_config["enable_flashcomm1"]`, else `get_ascend_config().enable_flashcomm1`, else env `VLLM_ASCEND_ENABLE_FLASHCOMM1`; **and `enable_shared_expert_dp` forces it True** (`:869-871`, logs `shared_expert_dp requires enable_sp=True. enable_sp has been set to True.`) | the runtime FC1 half — a *different* feature from `pass_config.enable_sp` | `utils.py:763-781` + `envs.py:74` |
| `vllm_ascend/platform.py:648-650` | `if not vllm_config.compilation_config.pass_config.enable_sp: parallel_config.all2all_backend = "flashinfer_all2allv"` (comment: "this is a tricky way to disable `use_sequence_parallel_moe` in vllm") | **the key also selects the MoE all2all backend**, so "silently ignore this key" would change behaviour for existing users instead of removing it | not recorded |

Two consequences to hand a reporter who proposes the fix: `--enforce-eager` or TP=1
makes the whole pass half inert (so "another node does not reproduce" has a cheap
first discriminator), and deleting `pass_config.enable_sp` is **not** the same as
turning SP off when `enable_shared_expert_dp` / `multistream_overlap_shared_expert`
is set — grep the log for `shared_expert_dp requires enable_sp=True` before
believing the fix took.

## Step 3 — read the shipped support matrix before inventing a mechanism

`docs/source/user_guide/feature_guide/sequence_parallelism.md` — quote the ref you
read, because **this matrix changed between the two recorded pins**:

- `:12` "implemented Sequence Parallelism for **VL-class models** based on the
  Inductor pass"; `:20` "Since SP relies on graph mode, it is not supported in
  eager mode".
- Matrix **without** quantization (v0.22.1rc1 `:44-47`): SP = `graph` for VL + Dense
  **and VL + MoE**; `x` for both non-VL columns; FC1 = `eager/graph` for both
  non-VL columns. At v0.17.0rc1 `:41-47` the VL + MoE cell was `x` — do not carry
  the old cell onto a newer report.
- Matrix **with** quantization (v0.22.1rc1 `:49-56`): SP = `x` in all four columns,
  "SP currently does not support quantization and is under adaptation" (`:51`).
- `:21` documents `sp_min_token_num` as "from upstream vllm's `pass_config`",
  "**The default value is 1000 on Ascend**", later appended into
  `compile_ranges_split_points`. That sentence is **wrong on a MoE model** at
  v0.22.1rc1 (`platform.py:251-255` + `get_sp_min_token_num()` give `1`), so its
  own "SP will only take effect when `num_tokens >= sp_min_token_num`" caveat does
  not hold there. Call that a documentation defect, not a code bug.
- `:32-39` FC1 is a *different* optimisation (Custom OP based, hard for VL-class
  models) and "FC1 and SP are complementary" — **not** two names for one switch.
- `docs/source/user_guide/release_notes.md:97` the key was renamed back
  (`enable_flash_comm_v1` → `enable_sp`, #6883), so wording in an older guide is
  not evidence about the current key.

So for a non-VL MoE quantized model (w4a8/w8a8), `pass_config.enable_sp: true` is
**outside the documented matrix on two counts at v0.22.1rc1** (non-VL, quantized),
and on one count plus a stale cell in the v0.17.0rc1 file, while the code still
registers the passes (no model or quantization gate, `graph_fusion_pass_manager.py:74-79`)
and the effective threshold is `1` for MoE — i.e. the excluded combination is
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
| 4 | row 1 + `--enforce-eager` | `enable_sp_by_pass` is false in eager (`ascend_config.py:260-264`), so the pass half is off — same observable as row 1 if the pass is the trigger; TP=1 is the even cheaper version, upstream disables `enable_sp` itself and logs it |
| 5 | row 1 + `--compilation-config '{"pass_config": {"enable_sp": false, "sp_min_token_num": 1000}}'` (>=v0.22: `sp_min_token_num` is an upstream `pass_config` field; the old `sp_threshold` key is dead in the tree) | dense-only knob; on a MoE model the default is 1, so a change here means something else |
| 6 | MTP off, then TP/TBT matched against the second node | the recorded pair differed by node, so shape/topology, not the flag alone |
| 7 | `VLLM_LOGGING_LEVEL=DEBUG … 2>&1 \| grep -E "Replaced .* patterns\|compile_ranges_split_points\|sequence parallel"` | whether the pass rewrote the graph at all (`sequence_parallelism.py:186-190` logs the match count, `ascend_config.py:207` the split point) |

## Recorded run: #12253 (vllm-ascend v0.18.0-a3, Qwen3.5-397B-A17B-w4a8-mtp, 8×910C)

Reporter-confirmed facts: works on v0.17.0rc1; breaks on v0.18.0-a3; the *same*
image and flags on a second 910C node do **not** reproduce; deleting
`"pass_config": {"enable_sp": true}` from the start script restores normal output;
their diagnosis is "the parameter is no longer supported in v0.18.0+, confirmed by
the compute team", with `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` as the workaround.

The verified reframe (pinned tree v0.17.0rc1, `e20f0b1`): the key **is** read, at
the six sites recorded for that pin in step 2, so it is honored rather than dropped; the documented
matrix excludes this model class (non-VL + MoE) and every quantized model; the
registrations have no model gate; and FC1 is a separate, complementary feature —
so the reporter's temporary fix is a different switch, not the same one renamed.
"Another node does not reproduce" fits a model-class/quantization-agnostic graph
rewrite whose match depends on shapes and batch sizes, so the second node is a
data point to match (TP, `max-num-batched-tokens`, images) rather than a
refutation.

Gaps: for #12253 the snapshot carried v0.17.0rc1 only — no v0.18.0-a3 tree — so
*what changed between the working and failing release* is unverified. Ask for that
image's `vllm_ascend/utils.py` (`enable_sp*`),
`compilation/graph_fusion_pass_manager.py` and
`compilation/passes/sequence_parallelism.py`, plus both response bodies. Say which
half is verified and which is a hypothesis; do not let the version-flip story
inherit the confidence of the source-verified matrix.

## Recorded run: #12278 (same model family, pinned at v0.22.1rc1)

The opening body is the discriminator on its own: `finish_reason: "stop"`,
`stop_reason: 248044`, `completion_tokens: 97`, `prompt_tokens: 15`, and the
thinking text inside `content` with `"reasoning": null`. That is the model-issued
EOS lane **plus** a missing `--reasoning-parser qwen3` (`reasoning-output-routing.md`)
— two separate findings from one body, and the run said so.

The reporter's later follow-up is the same outcome as #12253 verbatim: works on
v0.17.0rc1, fails on v0.18.0-a3, a second 8×910C node with the same image and flags
does not reproduce, deleting `"pass_config": {"enable_sp": true}` restores output,
and their fix list is "delete the key" + "`export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`
开启SP能力", with the claim that the parameter "在 v0.18.0 及以上已经不支持".

Three corrections to hand back, all source-verified at v0.22.1rc1 (and the first is
the whole reason this page exists):

1. the key is **not** unsupported: it is read at eight sites (step 2) with no
   model-class and no quantization gate, so the documented-scope violation is the
   finding — propose a `warning_once`/validation at
   `NPUPlatform.apply_config_platform_defaults` (`platform.py:249-255`) or a line in
   the model tutorial, not "silently drop unknown keys";
2. FC1 is a **different, complementary** optimisation (`:38`), so
   `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` is the documented route, not the same switch
   renamed — writing it as an equivalent replacement misleads the next reader;
3. a hard block on the key would silently change the MoE all2all backend for
   everyone else (`platform.py:648-650`), so "屏蔽该参数" is riskier than a loud
   warning.

The tree's own validated command for this model
(`docs/source/tutorials/models/Qwen3.5-397B-A17B.md:117-137`) contains **no**
`pass_config.enable_sp`, keeps `--compilation-config '{"cudagraph_mode":
"FULL_DECODE_ONLY"}'`, and gets its communication win from
`--additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1,
"enable_flashcomm1":true}'` (`:701` lists FlashComm1 as the knob); every curl in
that doc pins an explicit `max_tokens` (`:593` = 50), so "default `max_tokens` was
never exercised by the recipe" is a documentation finding you can state, not a
theory.

Delivery note for this shape: the opening answer (lane + the
`enable_sp`/`reasoning-parser` pair + the two greps) fits in one turn; the recording
run spent ten assistant turns re-deriving the pin's line numbers, then answered the
follow-up in two. Correct, and one turn cheaper if this page is read first.

Delivery note for #12253: this run is the calibration row for the shape — 14 assistant turns,
16 tool calls, a complete answer delivered on the last turn because
`/workspace/scratch/findings.md` was written one turn earlier, which is how an
answer survives to the end. It still led with the unverified half-gate mechanism; leading
with the doc matrix would have been one call cheaper and true at the pinned commit.
