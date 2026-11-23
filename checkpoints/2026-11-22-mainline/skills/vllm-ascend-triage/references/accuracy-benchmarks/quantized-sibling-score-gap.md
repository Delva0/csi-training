# A quantized sibling scores lower: quant format, recipe and model are three variables

## Scope

Reports of the shape "`<model>`-w8a8c8 scores N points below `<model>`-w4a8c8"
(also "w4a8 beats w8a8", "the 8-bit checkpoint is 18 points worse on <benchmark>")
on a GLM-5.x / DSA-sparse family model, usually in a multi-machine PD deployment
and often with `reasoning_effort` set in the request body. The deliverable is an
attribution verdict: which of **quant format, serve recipe, model/checkpoint and
release** owns the gap, plus the one ablation that decides each. Verified pins:
vllm-ascend **v0.26.0rc1 `f2f74a1c`**, vllm **v0.26.0 `568afb3a`** (SOURCES
`15925`; the C8 hang/garbage sibling is SOURCES `15927`).

Not this page: a hang, a stopped prefill side or a garbled first request on the
same flags (`../dsa-sparse-attention/c8-flags-and-kv-store-path.md`); a loss
table printed by the converter (`quantized-checkpoint-conversion.md`); knob
alignment on one model/one server (`harness-vs-engine-attribution.md`); run-to-run
nondeterminism (`repeated-run-nondeterminism.md`).

## Step 1 — Count the variables in the reporter's own table

Recorded #15925, all four metrics of an ifbench-style strict/loose benchmark:

| row | strict prompt/inst | loose prompt/inst |
|---|---|---|
| 0.23.0 + GLM-5.3-w8a8c8 | 55.67 / 59.5 | 65.33 / 68.33 |
| 0.23.0 + GLM-5.2-w4a8c8 | 73.67 / 76.5 | 77 / 79.33 |
| 0.23.0rc1 + GLM-5.2-w8a8 | 73.67 / 76.67 | 76.67 / 79.67 |

Rows 2 and 3 are the **within-model control**: the same model at two quant widths
agrees to ≤0.34 on all four metrics, which retires "8-bit / c8 weights are
inherently ~18 points worse" for that model and leaves the gap attached to the
row that changed model, recipe and release label at once. Say that in the first
paragraph — it is arithmetic on the reporter's own numbers, not a source claim.

Two more variables ride along and are easy to miss: the environment dump says
vllm-ascend **0.26.0rc1** / vllm 0.26.0 while the table rows are labelled 0.23.0
(rule 5's title-vs-env mismatch — say which ref you verified and give the
one-command check), and the two quant formats are shipped for *different node
counts* (Step 2), so a PD comparison rarely holds parallelism constant.

## Step 2 — The two formats' documented recipes are not the same attention path

`docs/source/tutorials/models/GLM5.2.md` at the pin ships two combinations of the
sparse C8 flags inside `--additional-config`, and they are not interchangeable
(the flag algebra and the write path are
`../dsa-sparse-attention/c8-flags-and-kv-store-path.md` Rule 3 — that page is
the authority, this is the summary):

| recipe | lines | c8 flags |
|---|---|---|
| **w4a8c8**, 2 nodes | `:211`, `:261` | `enable_sparse_sfa_c8: false`, `enable_sparse_li_c8: true`, `c8_enable_reshape_optim: false` |
| **w8a8c8**, 4 Atlas 800 A3, TP16 × PP2 | `:505`, `:579` | all three `true` |

So a reporter who followed each format's own tutorial compared a *sparse*
attention approximation on one side against a dense one on the other, over
different parallelism (the validated 4-node w8a8c8 PD e2e splits
`prefiller: [0,1] / decoder: [2,3]`, `dp_size 4`, `tp_size 8` —
`tests/e2e/nightly/multi_node/external_dp/config/GLM-5.2-W8A8C8-128k-1k-90-50-PD.yaml`).
The quant width is then the least isolated variable in the table. Ask for both
`--additional-config` blocks verbatim before ranking a kernel-level cause.

## Step 2b — The model tutorial is the published-weight inventory, and its node counts are the validated parallelism

Before ranking a width, read what the pin actually ships for that model:
`rg -n 'w[0-9]a[0-9]|QuaRot|require [0-9]+ Atlas|node' docs/source/tutorials/models/<Model>.md`.
A width the page does not list has **no validated recipe at the pin**, and saying so
with the line is the scope sentence of the answer. Recorded #8179: for
`Qwen3.5-397B-A17B` at v0.18.0rc1 the Model Weight block lists only BF16 ("2 Atlas
800 A3 (64G × 16) nodes or 2 Atlas 800 A2 (64G × 8) nodes") and `…-w8a8`
("1 Atlas 800 A3 … or 2 Atlas 800 A2", `Qwen3.5-397B-A17B.md:21-22`) — no w4a8 and
no QuaRot — so the reporter's community/self-converted w4a8 is outside every
validated recipe, and their BF16-vs-w4a8 rows also ran on **different node counts**
by the page's own arithmetic. That is Step 1's confound in its cheapest form: the
width and the TP/DP/EP layout moved together, so ask which layout each row used
before pricing any quant loss.

Two riders for the **VL / agent-grounding** metric (action and mouse-click
coordinate correctness, 93.34 % vs 94.86 % in #8179): a 1.5-point single-run gap
sits inside this lane's noise band (Step 3) until the same weights are scored twice,
so the first rung is a repeat run, and the input side is not free either — the same
image through `AutoProcessor` and through the engine must agree on `image_grid_thw`
and `prompt_tokens` (`vl-grounding-coordinate-drift.md` step 2 squeezes
`max_pixels`/`smart_resize`). Then the published sibling (`…-w8a8-mtp`) on the same
serve line decides converter-versus-engine. If the same report also complains that
**spec-decode acceptance fell after the rotation was added**, that is a different
owner and a different page: `../spec-decode-mtp/quarot-rotation-basis-draft-acceptance.md`.

## Step 3 — This lane's own documented noise floor is 2–3 points

At the pin (`docs/source/user_guide/release_notes.md`):

- `:270` GLM-5.2-**W4A8C8** PD with `enable_sparse_sfa_c8` moves GPQA by ~2-3
  points **run to run** (#14378).
- `:260` for SFA models (DeepSeek-V3.2, GLM-5.1, GLM-5.2) DCP must be enabled on
  prefiller and decoder alike; asymmetric DCP **causes accuracy issues** (#14322).
- `:262` v0.23.0 + GLM-5.2 DCP × `enable_sparse_sfa_c8` has known issues and is
  not recommended (#14011); `:268` the 64K/1K sfa+li regression (#14329).

Two consequences for the answer: single-run differences below ~3 points are not
evidence of anything on this lane, and an 18-point gap is a *different animal* —
do not spend the answer on quantization noise, spend it on the ablation matrix.

## Step 4 — Is the model validated at the pin at all?

`rg -n 'GLM-5\.3' -l` over the v0.26.0rc1 tree returns **nothing**, and
`release_notes.md:5` says the release is "model‑restricted … Fully validated
models include Kimi K3, GLM‑5.2, DeepSeek V4 Flash 0731, DeepSeek V4 Pro 0813.
Availability is not guaranteed for other models." #15088 is the open request to
adapt GLM-5.3(-Flash). That is a quotable verdict for the *model* half ("running is
not validation") and it does not by itself explain the numbers — keep it as the
scope sentence, not the cause.

## Step 5 — Arithmetic on a strict/loose metric pair (hypothesis-grade)

Subtract the two columns of Step 1: loose − strict is 9.66 / 8.83 for the
suspected row and 3.33 / 2.83 (5.2-w4a8c8) and 3.0 / 3.0 (5.2-w8a8) for the
controls. A **widened** spread with a uniformly lower strict level is what
constraint-violating answers look like — an answer cut before the instruction is
satisfied, or the answer sitting in the reasoning stream instead of `content` —
not what uniformly wrong answers look like. Label it hypothesis and make the
confirming artefact per-sample: `finish_reason`, `completion_tokens`, and whether
the answer text is in `content` or `reasoning`
(`harness-vs-engine-attribution.md` dump step, `reasoning-output-routing.md`).

## Step 6 — `reasoning_effort` at this pin is the checkpoint's template, not the engine

vllm v0.26.0 forwards the request field into the chat-template kwargs and derives
`enable_thinking` from it (`rust/src/chat/src/renderer/hf/mod.rs:1100-1122`,
`docs/features/reasoning_outputs.md:323-329`); the string reaches the checkpoint's
own `chat_template.jinja`, which is in no pinned tree. Inside vllm-ascend, the only
`reasoning_effort` code at the pin is the **DeepSeek-V4** thinking patch
(`vllm_ascend/patch/platform/patch_deepseek_v4_thinking.py`) and the Kimi-K3
renderer; the GLM patch does usage accounting only
(`vllm_ascend/patch/platform/patch_glm_reasoning_usage_accounting.py:17-52`). So
"does `<model>`-5.3 honour `max`" is settled by one request pair (default vs
`max`, compare `completion_tokens` and the rendered prompt), never by a
vllm-ascend source walk — and a max-effort run that hits the `max_tokens` cap is
Step 5's truncation row, not a quantisation row.

## Delivery order

1. Verdict: the reported comparison changes model, quant format, recipe and
   release label at once; the reporter's own same-model rows close the quant-width
   hypothesis; `<model>` is not a validated model at the pin.
2. The two `--additional-config` blocks and the parallelism, against the table in
   Step 2 — the difference, not a generic "check your config".
3. Ablation ladder, cheapest first: (a) same serve flags across the two
   checkpoints if both exist — a format-only comparison; (b)
   `enable_sparse_sfa_c8: false` with `enable_sparse_li_c8: true` on **both**
   sides; (c) DCP identical on P and D; (d) `reasoning_effort` default vs `max`
   with `max_tokens` headroom; (e) per-sample dump (`finish_reason`,
   `completion_tokens`, `content` vs `reasoning`).
4. Missing artefacts as gaps: both P and D serve lines, the benchmark's per-sample
   records, and which node count each row ran on.
5. Open question, labelled: whether `<model>`-5.3's checkpoint or the w8a8c8
   recipe owns the residual gap. Nothing in the snapshot decides it.
