# The model's only validated recipe (and its only published weights) are on another SoC

Scope: the engine is up and the report is a *slow* claim or a *weights* request
for a model already deployed — "910B2 + <model> 速度很慢，能否提供量化版本，社区只看到
310P 的量化版本，能否支持 910B" — where the model's own tutorial, support-matrix row
and quantized exports all belong to a different SoC than the reporter's. The
deliverable is the validation verdict plus the two or three deviations that are
provably meaningless, not a tuning ladder transplanted from the other SoC's page.

Inherit `index.md` (verdict first, constraint errors second, staged experiment
third) and, because "very slow" is attached, `../performance-throughput/index.md`
for the metric block. Verified at vllm-ascend **v0.23.0** (`5cb98ca`).

## Read the support matrix by TAB, not by row

`docs/source/user_guide/support_matrix/supported_models.md` is three tab-sets,
each with its own table and its own hardware: `Ascend 950 Products`, `A2/A3`,
`Atlas 300I DUO` (`:20`, `:31`, `:55`, then `:133`/`:142`/`:156` for the
generative set). A model row that appears **only** inside one tab is the
project's statement for that version: validated on that hardware, absent from
the other. Two facts decide such a report:

- the row's own `Supported Hardware` cell (e.g. `Qwen3.5-Dense` =
  Qwen3.5-2B/4B/9B at `:161`: `Atlas 300I DUO`, W8A8 ✅, BF16 ❌), and
- the **absence** of that model name from the `A2/A3` table (`:142-152` holds
  Qwen3.5-397B-A17B, Qwen3.5-27B/Qwen3.6-27B, Qwen3.6-35B-A3B and nothing
  smaller), which is why "the community only published 310P quantized weights"
  is not an accident — grep the model name in the whole file and print which tab
  each hit sits in, then say which tab it is missing from.

The model's tutorial repeats the same scope in prose and is the sentence to
quote: `docs/source/tutorials/models/Qwen3.5-Dense.md:9` "deployment and
verification of these models on **Atlas 300I DUO** and **Atlas 200I Pro**".
Absence from a tab is "not validated", not "cannot run" — the runtime path is not
SoC-gated for A2: `vllm_ascend/patch/worker/patch_qwen3_5.py:201` applies the
Ascend GDN forward only on the non-310P branch, i.e. A2 keeps the Ascend op. Say
both halves, or the reporter reads "unsupported" as "impossible".

## Take the launch command from the SAME SoC's test, not from the tutorial

The tutorial that documents the model belongs to the other SoC, so its serve
line is a cross-SoC transplant when handed to an A2 reporter. Compare
`--dtype`, the cache-dtype flag and the spec method before quoting anything:

| | 310P document (`Qwen3.5-Dense.md`, `single_npu_multimodal.md`-style command) | non-310P one-card e2e |
|---|---|---|
| dtype | `--dtype float16` (fp16-only hardware) | `dtype="bfloat16"` |
| mamba cache | `--mamba-ssm-cache-dtype float16` (310P e2e sets it too: `tests/e2e/pull_request/one_card/_310p/test_spec_decode_mtp_310p.py:27,29`) | absent |
| spec method | `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'` | `{"method":"mtp","num_speculative_tokens":1}` (`tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py:59-60`) |
| graph | `--cudagraph_mode FULL_DECODE_ONLY`, `enable_npugraph_ex: false` (310P cannot) | `FULL_DECODE_ONLY`, capture sizes `[2,4,6,8]` (`:55-56`) |

Routing rule for "which lane runs on which SoC" is in the tree:
`.github/workflows/scripts/test_config.yaml:22-24` — `pull_request/{one_card,
two_card,four_card}/` are matched by card count (A2/A3) while `_310p/` and
`*_310p.py` go to 310P. So a non-`_310p` one-card test **is** the A2-class
recipe, and it is stronger evidence than a tutorial paragraph. When the two
disagree, answer with the same-SoC test and mark the other direction untested.

## Prune the flags that provably do nothing before ranking levers

At v0.23.0 several tutorial/CI spellings are stale, and a stale key cannot
explain a slowdown *or* a rejection:

- `enable_weight_nz_layout` (still in `docs/source/user_guide/feature_guide/large_scale_ep.md:167,468`,
  `InternVL3.5.md`, `ucm_deployment.md` and in nightly yaml labels) is read by
  **no** code: `vllm_ascend/ascend_config.py:249-253` reads `weight_nz_mode`
  (0/1/2), renamed per `docs/source/user_guide/configuration/additional_config.md:25`
  (`VLLM_ASCEND_ENABLE_NZ` → `weight_nz_mode`, "field name changed"). The default
  is 1 = NZ for quantized weights only.
- `multistream_overlap_shared_expert` is MoE-with-shared-experts only
  (`additional_config.md:78`), so on a dense model it is not a lever.
- An `export` block is the same question, one level up: reporters copy it from the tab
  of some *other* SoC or model, so most of it can be provably inert before it is even a
  candidate. At v0.23.0rc1 (`f4a08bbd`) `USE_MULTI_GROUPS_KV_CACHE` is a module constant
  (`vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:35` `= True`), not an env
  read, so exporting it changes nothing; `USE_MULTI_BLOCK_POOL` has no reader anywhere
  (its only occurrence is a commented `#export` in `docs/source/tutorials/models/GLM5.2.md:316`);
  `ACL_OP_INIT_MODE` occurs in tutorial shell blocks and CI yaml only; `HCCL_BUFFSIZE` is
  set in docs/CI but read by no Python line (the consumer is CANN/HCCL — `utils.py:1109-1111`
  only *comments* on it when fitting pg_options to an `mc2` group). Classify a whole block
  in one call: `uv run --no-project python <skill_root>/scripts/env-knob-liveness.py
  --root <vllm-ascend> --vllm-root <vllm> <NAME> …` (or `--from-paste launch.sh`), and see
  `../runtime-memory/allocator-expandable-segments-unmap-crash.md` §*If the same reporter
  returns …* for the recorded 310P block (#12534) — the same diff that shows the model's
  own 310P tutorial tab carries none of those exports.

Both belong in the answer as "this line does nothing on your model", with the
`path:line`. Do not turn the silent no-op into a cause: on refs whose schema is
strict the same key is a startup `ValidationError` instead
(`../startup-failures/additional-config-schema.md`), and at v0.23.0 it is
silently ignored (`../runtime-memory/hbm-growth-and-oom.md` Step 4).

## Quantization availability, answered per card

- W8A8 exists as a *capability* on this stack and is self-service:
  msmodelslim (recommended) or LLM-Compressor, then `--quantization ascend`
  (`docs/source/user_guide/feature_guide/quantization.md`; the
  `feature_matrix.md` W8A8 row is feature × feature, **not** hardware — do not
  cite it as a card claim).
- A published `-310P` / `310P` export is a 310P artefact, not a portable one:
  the SoC shapes the quantization tensors (the 310P export carries an int32
  `quant_bias` that the 310P dynamic path is built around —
  `../soc-platform-paths/310p-quant-checkpoint-soc-match.md`). There is no
  evidence it loads on A2, and the reporter's question "能不能支持 910B" is about a
  conversion, not a download.
- A 9B dense hybrid model on a 64 GB A2 card is not memory-bound at bf16, so
  quantization is not the lever for "慢"; say that plainly instead of promising a
  speedup, and ask for the metric block (per-request inter-token rate,
  `Running`, `Avg generation throughput`) to price the claim at all.

## Recorded case: #16600 (v0.23.0 tree, 910B2 single card, Qwen3.5-9B, `--async-scheduling`, `FULL_DECODE_ONLY`, three `additional-config` keys)

Title "910B2 + Vllm ascend 0.23.0 + Qwen3.5 9B 速度很慢", body asks for a
quantized 910B version because only 310P ones are published. The session reached
exactly the three findings above in ~12 calls (support-matrix tab at `:161` and
`A2/A3` at `:142-152`; `Qwen3.5-Dense.md:9`; the same-SoC e2e at
`tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py:45,55-60`; the rename table
and `ascend_config.py:249`), emitted a provisional verdict at turn 9 — and then
spent the remaining budget inside `/workspace/repos/**` probing capture-mode gates
(`platform.py:505-660`), the hybrid-KV gate and issue bodies, and was cut by the
output cap **before the ranked answer existed**. The lesson is delivery, not
evidence: for this lane the support-table + same-SoC-test + stale-key triple *is*
the answer, and the metric block is the question to send back.
