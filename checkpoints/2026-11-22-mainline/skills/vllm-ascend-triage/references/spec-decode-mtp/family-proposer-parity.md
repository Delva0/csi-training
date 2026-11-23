# Family parity: the model family's spec-decode branch exists upstream, not downstream

Scope: spec decode is on, the **target model's own output is correct**, request
throughput is fine, and the metrics read `Accepted: 0` / every per-position rate
`0.000` / `Drafted: N > 0` — on a model **family** whose upstream spec decode runs
through a *dedicated* proposer or speculator. The deliverable is a parity verdict
(does the downstream tree have that branch?), the family mechanisms the generic
path cannot serve, the one ablation that separates draft-side from verify-side, and
the **support-plan paragraph** the reporter will ask for next. Not a positions /
`num_computed_tokens` walk: `acceptance-collapse.md` assumes the family is served by
the same machinery as Qwen3.5, and that assumption is exactly what fails here.

## Step 1 — one grep decides the class, before any block-table read

```bash
rg -n 'use_<family>_mtp|<family>_mtp' <vllm_ascend tree>/vllm_ascend | head
```

Zero hits while upstream has `use_<family>_mtp()` and a `<Family>Proposer` class
means the family is served by the **generic** proposer. Confirm on the dispatch, and
read which families *do* have a branch — that list is the port template:

| Anchor (vllm-ascend v0.23.0, `5cb98caa`) | What it says |
|---|---|
| `vllm_ascend/spec_decode/__init__.py:34-47` `get_spec_decode_method` | `mtp`/`eagle`/`eagle3` → `AscendStep3p5MTPProposer` **only** when `use_step3p5_mtp()`, else `AscendEagleProposer`. A family absent from this `if` is not "unsupported" — it is silently served by the eagle path |
| `vllm_ascend/spec_decode/step3p5.py:757-759` | the downstream precedent: `validate_same_kv_cache_group` returns immediately ("Step 3.5 MTP draft layers may span multiple KV cache groups") |
| `vllm_ascend/spec_decode/step3p5.py:761-800` | and the matching per-group `initialize_attn_backend`: `layer_to_gid`, `UniformTypeKVCacheSpecs`, the `kv_sharing_target_layer_name` fallback, `AttentionGroup(..., kv_cache_group_id=gid)` |
| `vllm_ascend/spec_decode/llm_base_proposer.py:1456-1459` (also `:624, :911, :1630-1648, :1687`) | what the generic proposer does instead, in the tree's own words: `# Use the first draft attention group's kv_cache_spec for block_size` / `# (all draft layers share the same kv-cache group)` beside `self.draft_attn_groups[0].kv_cache_spec.block_size` — one geometry and one `CommonAttentionMetadata` for every draft layer |

## Step 2 — why that degrades to exactly 0 % and stays silent

| Upstream fact (vllm v0.23.0, `0fc695fc`) | Consequence on the generic path |
|---|---|
| `vllm/v1/spec_decode/llm_base_proposer.py:1556-1574` — `validate_same_kv_cache_group` asserts every drafting layer is in **one** KV cache group, "to ensure all drafting layers can use the same AttentionMetadata"; called from the base `initialize_attn_backend` (`:1594`) and from `gpu_model_runner.py:7330` | the base class *declares* the assumption the family violates; every family proposer that serves multi-group draft layers must override it (see Step 3) |
| `vllm/v1/spec_decode/gemma4.py:70-105` (method + docstring): "Gemma4 has multiple KV cache groups (sliding vs full attention) with different block tables. The base class receives a single common_attn_metadata whose block_table belongs to one group. We swap in the correct block table for each draft attention group." | a draft layer of the non-group-0 type reads/writes someone else's blocks: draft logits are noise, the target rejects every draft, and nothing raises |
| `vllm/v1/spec_decode/gemma4.py:195-197` overrides that assertion to `return` | the tree's own statement that a family can legitimately span groups |

So the symptom shape is: correct target text, healthy `Drafted:` count, uniformly
`0.000` acceptance, no traceback. That is *not* the capacity signature and *not* a
numerics signature — do not rank either first.

**Open question to leave open, with its check**: if the reporter's engine *starts*,
the single-group assertion did not fire. Either the draft layers collapse into one
group on that path or the assertion is never reached by the downstream runner. One
line of the startup log (the KV-group / block-size listing) settles it; do not assert
which it is.

## Step 3 — the three family mechanisms to grep for upstream

Read the family proposer for these; each one is a candidate mechanism and each has a
grep-shaped statement:

1. **Per-group block table.** `vllm/v1/spec_decode/gemma4.py:51,61-62` (`_per_group_block_tables`,
   `set_per_group_block_table(gid, …)`) fed by the runner at
   `vllm/v1/worker/gpu_model_runner.py:2439-2447`, which calls
   `set_per_group_block_table` for `Gemma4Proposer` and `set_per_group_attn_metadata`
   for `Step3p5MTPProposer`. Downstream has neither call site.
2. **Constant draft positions.** `vllm/v1/spec_decode/gemma4.py:45-47`
   (`constant_draft_positions = True`, "All draft steps predict from the same
   position"), consumed by the base proposer at `llm_base_proposer.py:541,594,606`.
   A single-position assistant served by a proposer that advances positions per step
   produces drafts for the wrong position — same 0 % signature, different fix.
3. **Draft→target KV sharing.** `vllm/v1/spec_decode/gemma4.py:180-193,280-…`:
   `_setup_gemma4_kv_sharing` maps each draft layer to the *last non-KV-shared target
   layer of the same `layer_types` entry*, and the config rewrite already zeroes the
   draft's own `num_kv_shared_layers` (`vllm/config/speculative.py:512-521`). No
   downstream equivalent; without it the draft attends its own empty KV.

Also note what does **not** port for free: `gemma4.py:193` calls
`_setup_centroids_cuda_graphs()` — a CUDA-graph structure. An NPU port needs its own
path there, so "port the class" is a plan, not a rename.

## Step 4 — the ablation that separates draft-side from verify-side

Row order, cheapest first; row 1 is the one this class needs (not the MTP-off row):

| Row | Change | What it proves |
|---|---|---|
| 1 | `--speculative-config '{"method": "ngram", "num_speculative_tokens": 1}'` (no family draft model) | acceptance > 0 ⇒ verify/sampler is healthy and the *draft side* owns the 0 %, which is the family-parity verdict; still exactly 0 ⇒ the verify path is broken and this page does not apply |
| 2 | delete `--speculative-config` | owner split, as `acceptance-collapse.md` row 1 |
| 3 | the family's own upstream-validated launcher (if one exists in the tree) | whether the failure is the family or this flag set |

An explicit `--speculative-config` `method` is also worth one line: the upstream
auto-detection chain ends in `raise NotImplementedError(f"Unsupported speculative method: '{self.method}'")`
(`vllm/config/speculative.py:729-731`), and a family draft reaches the `mtp` branch
only because `hf_config_override` rewrote its `model_type` (`:512-521`, applied at
`:692`). So "omit `method`" is a workaround for *a* gate there, not evidence about
acceptance — say which gate, and ask for the draft `config.json`'s `model_type`
rather than repeating the claim.

## Step 5 — the support-plan paragraph belongs in the first answer

The reporter's next message after a parity verdict is "支持计划是什么样的 / when will
family F's MTP work on my SoC". It has no date in the tree (`../feature-enable-recipes/index.md`
item 9), but it does have four evidence-backed parts. Ship them together with the
verdict:

1. **Three levels, stated separately** — the base family (docs/release notes/support
   matrix), the spec-decode branch (the Step-1 grep), the coverage (tests/docs). A
   family can be *experimentally supported* at level 1 while level 2 is absent; never
   compress that into "the family is unsupported".
2. **Status tier**: supported / experimental / present-but-single-group / absent. Cite
   one line per level.
3. **Prerequisites as a list with anchors**, in port order: dispatch branch, the
   `validate_same_kv_cache_group` override, per-group `AttentionGroup`s, the runner's
   per-group block-table hook, KV sharing, the positions rule, then the SoC-specific
   gates (for the worked case: A2/A3 FIA TND has no 512-dim global-head path —
   `vllm_ascend/attention/utils.py:71-73,105-112` and `device/device_op.py:73`, both
   `TODO: Remove this fallback when A2/A3 FIA TND supports Gemma4's 512-dim global
   attention heads`).
4. **The flip test** — the grep whose result changes: `use_gemma4_mtp` appearing under
   `vllm_ascend/`, the dispatch gaining a branch, or a release-note line naming the
   family *and* MTP. Say "no published date; re-check by these three", never a guess.

## Worked case: #16980 — gemma4 (Gemma4MTPModel) on 910B, vllm-ascend v0.23.0

Reporter facts: vllm/vllm-ascend 0.23.0, Atlas 800T A2 (910B3×8), `gemma-4-12B-it` /
`gemma-4-31B-it` + the matching `-assistant` draft, `num_speculative_tokens: 1`,
`--enforce-eager`, TP1 and TP2 both reproduced; target output correct, `Accepted: 0
tokens, Drafted: 40 tokens, Avg Draft acceptance rate: 0.0 %`; spec decode was a pure
slowdown. The reporter's own title named the mechanism ("draft layers on multiple KV
cache groups share group[0]'s block table") — which Steps 1-3 turn into citable
evidence.

What the tree says at the pin: upstream has `gemma4_mtp` (`vllm/config/speculative.py:52,512-521,1054-1060`),
a dedicated `Gemma4Proposer` on the V1 runner (`vllm/v1/worker/gpu_model_runner.py:589-590`)
and `Gemma4Speculator` on V2 (`vllm/v1/worker/gpu/spec_decode/__init__.py:11-16`), with
per-group block tables, constant draft positions and KV sharing (Step 3). Downstream has
**zero** `gemma4_mtp` / `use_gemma4_mtp` hits, so the assistant checkpoint is served by
`AscendEagleProposer`. Levels 1 and 3 for the support paragraph:
`docs/source/user_guide/release_notes.md:170` "Added experimental Gemma4 support on A2
and Ascend 950. [#11091] [#10643]", the matrix carries Gemma-2/Gemma-3 rows only
(`docs/source/user_guide/support_matrix/supported_models.md:84-85,185`), and no
`gemma4` test or tutorial exists (`tests/`, `docs/source/tutorials/models/`).

Delivery lesson: the recorded answer carried the parity verdict, the upstream evidence
table, the ngram/MTP-off rows and the port plan, but not the support-plan paragraph —
the reporter's reply thanked it and asked for exactly that ("gemma4 mtp 910B 上的支持计划
是什么样的"). Level 1 was also under-read: it said "downstream has no gemma4 support",
while the release notes document *experimental* Gemma4 on A2 with a named A2/A3
attention fallback. Read the family's release-note line and matrix row **before**
writing "no support", and separate the three levels.

## Stop rule

After (a) the Step-1 grep, (b) one read of the family proposer for the three Step-3
mechanisms, (c) the two 0-hit/coverage greps, and (d) `scripts/sources-summary.sh`
confirmation that the pin equals the reporter's release, write the answer in one
message: verdict, the family's mechanisms table, the ngram row, the port prerequisites,
and the support-plan paragraph. Do not read the KV-manager or block-table
implementations — the verdict is decided above them.
