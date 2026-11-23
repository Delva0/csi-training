# Acceptance drops on a rotated (QuaRot) quantized target while the target's text stays right

## Scope

A **self-converted, rotation-suppressed** quantized checkpoint — "加了 QuaRot
离群值抑制并重新量化后" / a `…-QuaRot` directory the reporter built — served with
spec decode (`mtp`, `qwen3_5_mtp`, `eagle3`, `draft_model`), where the **target's
own answers are correct** and only the acceptance rate falls, usually *low but
non-zero* (<40 %, per-position rates differ) and normal before the requantize.
Verified pins: vllm-ascend **v0.18.0rc1 `99e1ea0`**, vllm **v0.18.0 `bcf2be96`**
(SOURCES `8179`). Recorded #8179 (Qwen3.5-397B-A17B w4a8 + QuaRot, 8 cards,
"mtp 的采纳率非常低（<40%)").

Not this page: acceptance pinned at `0.000` on every position with drafts arriving
(that is bookkeeping — `acceptance-collapse.md`, and `acceptance-zero-at-high-k.md`
when a K threshold rides along); a capacity-thrash acceptance drop
(`../performance-throughput/cross-run-ratio-normalization.md`); a *converter loss
table* with no serve at all (`../accuracy-benchmarks/quantized-checkpoint-conversion.md`);
a score gap with acceptance unmentioned
(`../accuracy-benchmarks/quantized-sibling-score-gap.md`).

## Step 1 — read the shape, then decide whether the rotation is even the suspect

| In the report | Reading |
|---|---|
| acceptance low but **non-zero**, per-position rates differ, target text correct, **started with the requantize** | this page: the draft computes on a basis the target already left |
| `Accepted: 0`, every per-position `0.000`, `Mean acceptance length: 1.00`, drafts arriving | bookkeeping / positions, not the basis — `acceptance-collapse.md` |
| acceptance was already low **before** QuaRot, or falls with a `num_speculative_tokens` boundary | the K is the classification — `acceptance-zero-at-high-k.md` |
| acceptance falls only when an unrelated feature is switched on (FC1/SP, prefix caching) | that feature is the axis — `../feature-enable-recipes/sequence-parallel-flashcomm.md`, `hybrid-mtp-prefix-cache-unvalidated.md` |

The one-launch confirmation of the lane is row 1 of every ladder here: delete
`--speculative-config`. Text still correct ⇒ the target is innocent and the draft
is the surface; nothing about quantisation of the target's linears is left to rank.

## Step 2 — QuaRot is fused at export; only one runtime path un-rotates anything

QuaRot rotates the residual stream and fuses the rotation into adjacent weights
*when the artifact is produced*. The matrix does not come from the engine: it is
named by the checkpoint's own description,
`quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]`
(`vllm_ascend/patch/worker/patch_draft_quarot.py:42-53`). At v0.18.0rc1 nothing on
the **target** loading path reads it — the file above is the only consumer in the
tree (`rg -n 'rotation_map|global_rotation' vllm_ascend/`).

So there are exactly two owners, and the report names which one is at fault:

- **The export** owns the target *and everything shipped inside the checkpoint
  directory*, including a `mtp.*` head. Those tensors were rotated along with the
  decoder or they were not; the engine cannot tell.
- **The runtime** owns a draft loaded from a **separate** checkpoint that was
  trained in the unrotated basis. That is the only case the patch covers, and it is
  EAGLE3-class-keyed by design: `patch_draft_quarot.py:88` assigns
  `Eagle3LlamaForCausalLM.load_weights`; inside, `:111-114` right-multiplies every
  `fc.` weight by `block_diag(Q, Q, Q)` (EAGLE3 concatenates **three** aux hidden
  states) and `:121-124` loads the target's `embed_tokens` and multiplies it by
  `Q.T`. The call site `model_runner_v1.py:2573` is gated only on `self.drafter`
  plus `quant_config is not None`, so an MTP drafter runs the patcher and it does
  nothing — that is not a missing branch, it is the split above.

## Step 3 — the counter-example that keeps "unsupported" out of the answer

The same pin serves **MTP over a QuaRot target as a documented recipe**:
`docs/source/tutorials/models/DeepSeek-V3.1.md:28-29` publishes
`DeepSeek-V3.1-w8a8-mtp-QuaRot` and `…-Terminus-w4a8-mtp-QuaRot`, and every tab
serves one with `--speculative-config '{"num_speculative_tokens": 3, "method":
"mtp"}'` (`:111-124`, repeated at `:175`, `:227`, `:303`, `:380`, `:455`, `:531`;
`DeepSeek-V3.2.md:536` likewise). Release notes scope the *runtime* work to Eagle3
(`release_notes.md:74` "Eagle3 now supports QuaRot quantization without embedding"
#7038, `:132` "Fix Eagle3 acceptance rate for QuaRot quantized models" #6914, both
in the `## v0.16.0rc1` section) — consistent, not contradictory: when the head
ships inside the rotated artifact there is nothing to anti-rotate at load.

Consequence for delivery: **do not answer "MTP × QuaRot is structurally
unsupported" from the EAGLE3-only patch.** The recorded #8179 run did exactly that
and read the class keying as a gap — the correct statement is "the rotation
coverage of *your* export is the open question", with the published DeepSeek pair
as the control that proves the engine lane. Equally, do not propose adding
`Qwen3_5MoeMTP`/`mtp` to `patch_draft_quarot.py`: for an in-checkpoint head the
rotation is already in the weights, so a runtime anti-rotation would *break* a
correct export.

## Step 4 — the check that answers it is a census of the artifact, not a source walk

1. `cat <ckpt>/quant_model_description.json` (or `scripts/checkpoint-key-prefix.py`):
   does it carry `optional.quarot.rotation_map`, and do `mtp.*` rows appear in the
   scheme map at all (also worth reading: `visual.*` and `lm_head`). A head that is
   in the file but not in the rotation is the finding.
2. Ask the converter side one two-sided question: were the `mtp.*` tensors —
   the `fc`-equivalent projection **and** the embedding half — rotated with the R1
   matrix the description points at? That is the same provenance rule as
   `../accuracy-benchmarks/quantized-checkpoint-conversion.md` rule 2, applied to
   the rotation instead of the scheme.
3. If a rotation has to be built for this family, get the block count from the
   model, not from EAGLE3: `Qwen3_5MTP.forward` (`vllm/model_executor/models/qwen3_5_mtp.py:126-128`)
   concatenates `pre_fc_norm_embedding(inputs_embeds)` with
   `pre_fc_norm_hidden(hidden_states)` and feeds `self.fc` (`prefix=f"{prefix}.fc"`,
   `:85`) — **two** halves, i.e. `in_features = 2 × hidden_size` and a
   `block_diag(Q, Q)` matrix, while EAGLE3's is three. The two RMSNorms are
   rotation-invariant, so `fc` is the single site (hypothesis-grade arithmetic:
   label it as the shape a converter-side fix must use).
4. Coverage, quoted as coverage: `tests/e2e/multicard/2-cards/spec_decode/test_quarot_eagle.py`
   runs **eagle3** drafters only (`RedHatAI/Qwen3-32B-speculator.eagle3`, and
   `AngelSlim/Qwen3-a3B_eagle3` — the second is the shares-the-target-embedding
   case) against `vllm-ascend/Qwen3-*-W8A8-QuaRot` targets, K=4, `TOL = 0.06` on
   acceptance parity. No MTP × self-rotated artifact case exists. And at the pin
   `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:21-22` lists only BF16 and
   `…-w8a8` — no w4a8 and no QuaRot sibling — so for that model there is no
   published control: say "no validated QuaRot recipe for this weight set", not
   "the lane is broken".

## Step 5 — ladder

1. `--speculative-config` removed: text still correct.
2. The description census of Step 4.1 (reporter's box, one `cat`).
3. A checkpoint whose draft side is known-good at the same K — a published
   `-mtp-QuaRot` sibling of another family, or the same model's non-QuaRot quant
   export — which separates "the engine lane" from "this artifact".
4. K sweep 1 → 3: a basis fault is K-flat, a padding/lookahead fault is not.
5. Only then `--no-enable-prefix-caching` and `--async-scheduling` off
   (`hybrid-mtp-prefix-cache-unvalidated.md`), which the hybrid families need
   anyway before any acceptance number is trusted.

## What to send back

The checkpoint's rotation map and whether `mtp.*` appears in the scheme map; the
serve line and K; the metric lines (`Drafted`, `Accepted`, `Mean acceptance
length`, the per-position row); who produced the rotation (their pipeline or a
published artifact) and the converter's rotation recipe; and whether the
pre-QuaRot export had normal acceptance. A traceback is not owed — this lane
raises nothing.
