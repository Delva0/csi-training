# `glm_moe_dsa` (GLM-5/5.1) w4a8: output collapse on a newer tag, older tag fine

## Report shape

The **same weights, launch scripts and hardware** answer correctly on one image tag
and emit repeated junk on another (`相比于 / zez / scarf / dep / alse / 笑语 / 斩`), from
the **first** generated token, at `temperature: 0`, on a W4A8 `--quantization ascend`
GLM-5/5.1 (`architectures: ["GlmMoeDsaForCausalLM"]`, `model_type: glm_moe_dsa`).
The reporter's own control is a **plain-text** `/v1/completions` prompt, which is
correct on the broken tag. By the time the report arrives the cheap lanes are usually
already retired in the body: the chat template renders, `<|user|>`/`<|assistant|>` ids
resolve, `tokenizer.json` is byte-identical to the working side, the special-token rows
of `embed_tokens`/`lm_head` are finite, removing `--speculative-config` changes nothing,
and weight shards hash-match the published set.

This is the same class as a family-wide concurrency crash answered by a config diff
(`index.md` last rule): the deliverable is *one request that splits the reporter's two
variables*, the launch diff against the pinned tutorial row, and the model-scoped sites
at the broken pin labelled as candidates. It is **not** a walk through the MoE, the
quantization config and the attention impl in one turn.

## What the snapshot can and cannot answer

A two-tag report asks "what changed between `old` and `new`". Check `SOURCES.json`
first: it carries **one** tree per report, and for this shape it may say so itself
(recorded #10984: `vllm-ascend v0.21.0rc1 80610e44`, `reported_ref_status:
conflicting_versions`, `version_role: dated_reference_only`,
`runtime_commit_verified: false`; the reporter's working commit `367b8e62`
(v0.20.2rc1) has **no** tree). Then:

- Do not diff: `one_commit_git` exports have no history to walk.
- Enumerate the delta from the pinned `docs/source/user_guide/release_notes.md`
  section for the newer tag (model-family and DSA entries) plus the in-tree
  provenance comments — `vllm_ascend/patch/__init__.py` records the upstream PR number
  and reason for each patch it copies.
- Say which candidates are unproven because of that gap. A ranked list of sites at the
  broken pin is a legitimate answer; a named culprit is not.

## Verified at vllm-ascend v0.21.0rc1 (`80610e44`) + vllm v0.21.0 (`ad7125a4`)

| site | fact |
|---|---|
| `vllm_ascend/attention/sfa_v1.py:467-478` | the **only** `model_type`-keyed branch in the SFA/DSA impl: for `glm_moe_dsa` it sets `is_rope_neox_style = False` and `use_torch_npu_lightning_indexer = True`. Both defaults are the opposite (`True` / `False`) for every other model on this backend |
| `vllm_ascend/device/device_op.py:383-424` | that flag picks between `torch_npu.npu_lightning_indexer` and `torch.ops._C_ascend.npu_lightning_indexer` (same `sparse_count=2048, sparse_mode=3`) — a *different kernel and possibly a different toolkit build* for the same math, chosen by model name, with the file's own TODO saying two branches are kept temporarily. The rope-style flag reaches `rope_forward_triton_siso` for `k_li`/`q_li` inside `indexer_select_pre_process` (`sfa_v1.py:976,1039`; also `context_parallel/sfa_cp.py:423`), so it moves the *scores* that choose the top-k set, not the attention arithmetic |
| `vllm_ascend/ops/fused_moe/fused_moe.py:386,748-795` | `multistream_overlap_shared_expert` runs the quantized shared-expert chain (`npu_dynamic_quant` → `npu_quant_matmul` → `npu_dequant_swiglu_quant` → `npu_quant_matmul`, gated `has_quantized_shared and quant_type in (W8A8, W4A8)`) on a **separate NPU stream**, ordering each step against the routed-experts stream with events (`before_routed_experts`, `after_routed_experts`, `before_gmm2`, `before_combine`). A mis-ordered or premature `maybe_wait_event` gives plausible-shaped, wrong hidden states — corruption from token 0 with no fault |
| `vllm_ascend/ops/fused_moe/fused_moe.py:481-535` | the load-time guard **does not cover that path**: `_validate_shared_expert_consistency` (wrapped around `process_weights_after_loading`) compares integrated vs split math on one `rand(10, hidden)` input on the default stream, then raises if `allclose` fails. So `Multistream overlap shared expert is enabled.` + `Shared expert split computation validation passed.` in the startup log proves the *split arithmetic*, never the stream ordering — do not read it as a clearance, and do not read its absence as one either (it is an `info_once`) |
| `vllm_ascend/patch/worker/patch_deepseek_mtp.py:66-76`, `patch/__init__.py:743-745` | this class is replaced with `AutoWeightsLoader(..., skip_prefixes=["rot.weight"])`. A silently skipped prefix is a "runs clean, numerics wrong" lane, so it stays on the list — last, behind a hash-matched weight check |
| `vllm_ascend/patch/worker/patch_deepseek_mtp.py:32,47`, `patch/platform/patch_speculative_config.py:17` | the QuaRot path is **draft-only by construction**: `rot` is built and applied only under `is_rot_used and target_model_type == "glm_moe_dsa"` inside the MTP layer, and this family's spec config is rewritten to `model_type: deepseek_mtp` / `DeepSeekMTPModel`. A `rot.safetensors`/QuaRot story therefore cannot survive the MTP-off run — if the reporter already did that, drop the lane instead of ranking it |
| `vllm_ascend/quantization/modelslim_config.py:108` | the `--quantization ascend` weight-name map has a `glm_moe_dsa` entry that is **identical** to `deepseek_v32`'s and `deepseek_v4`'s (same `gate_up_proj`/`experts`/`fused_qkv_a_proj` lists), so this table has no GLM-specific row to blame: a mapping error would have to come from the checkpoint's own `quant_description`, which is where the next grep goes if this lane is taken |

`docs/source/user_guide/release_notes.md` also carries **dated** multistream accuracy
rows (the `v0.10.0rc1` Known Issues section: "accuracy issue on W8A8 dynamic quantized
DeepSeek with multistream enabled … will be fixed in the next release", #2232; and the
`mc2_mask` row, #2681) in **old** sections. Quote them as dated history, not as a
verdict about the pinned tag.

## The one-request discriminator: match the prompt *length*

The reporter's two rows change **two** variables at once — token content (plain text vs
`[gMASK]<sop><|user|>…`) **and** length (a handful of tokens vs ~20). One request
settles it, with no restart and no flag:

> send a **plain-text** prompt padded with filler until `usage.prompt_tokens` equals the
> failing prompt's `usage.prompt_tokens`, at the same `temperature: 0`.

- It garbles too ⇒ prompt content is irrelevant; the lane is numeric/flag and the
  ladder below decides it.
- It stays clean ⇒ the carrier is the special-token region itself, and the owners
  narrow to id-keyed paths (embedding rows — already refuted numerically in the
  recording, a rendered-template difference, an id-keyed branch in the forward).

Put this **before** any flag row: it costs one request, while every ablation below costs
a restart of a 16-NPU serve. The recorded #10984 run offered it as its third check, after
a flag ablation and a local code patch.

## Launch diff against the pinned tutorial — first-answer material

`docs/source/tutorials/models/GLM5.md` ships this exact topology: the **multi-node A2
w4a8** row (`:412`+ per-node block) is `--data-parallel-size 2 --data-parallel-size-local 1
--data-parallel-address … --tensor-parallel-size 8 --quantization ascend
--enable-expert-parallel --max-num-seqs 2 --max-model-len 131072
--max-num-batched-tokens 4096 --gpu-memory-utilization 0.95`,
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`,
`--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true,
"ascend_compilation_config": {"enable_npugraph_ex": true}}'`,
`--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'`, and an
env block: `HCCL_OP_EXPANSION_MODE="AIV"`, `HCCL_IF_IP`/`GLOO_SOCKET_IFNAME`/
`TP_SOCKET_IFNAME`/`HCCL_SOCKET_IFNAME`, `OMP_PROC_BIND=false`, `OMP_NUM_THREADS=1`,
`HCCL_BUFFSIZE=200`, `VLLM_ASCEND_BALANCE_SCHEDULING=1`,
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`.
Neither the A2 single-node (`:230-265`) nor that multi-node row passes
`--enable-prefix-caching`, `--enable-chunked-prefill` or `--async-scheduling`
(`--async-scheduling` appears zero times on the page).

So a reported launch that differs on these axes is **not a matched flag set**, and each
difference is an ablation row, not a cause:

1. `--async-scheduling` added — spec-decode bookkeeping axis, absent from every row.
2. the env block missing, `VLLM_ASCEND_ENABLE_FLASHCOMM1` included: `enable_sp()` reads
   `additional_config["enable_flashcomm1"]` else `VLLM_ASCEND_ENABLE_FLASHCOMM1`
   (`vllm_ascend/utils.py:847-876`, no other default), so SP — and the shared-expert
   replication its log line announces — is **off** in a launch that omits it.
3. `--enable-prefix-caching` (+ `--enable-chunked-prefill`) added.
4. `fuse_muls_add` / `ascend_compilation_config.enable_npugraph_ex` dropped from
   `--additional-config`.
5. `--max-num-seqs` / `--max-model-len` moved (harmless on their own; state them, do not
   rank them).

Keep the two evidence types apart: the diff explains **why this deployment is off the
validated matrix**, while the cross-tag question is the un-diffable delta above. A
report whose commands are identical on both tags cannot be explained by rows 1-5 alone;
they are where a *tag-introduced* bug would become visible, not what changed.

## Ladder

1. Length-matched plain prompt (above) — free, splits content from numerics.
2. The **same greedy body twice**: byte-identical ⇒ deterministic (weights, rope style,
   quant mapping); differing ⇒ ordering/race (shared-expert overlap, async scheduling).
3. `--additional-config` with `"multistream_overlap_shared_expert": false` — one key,
   same image. Confirm the path was live first: grep the per-rank log for
   `Multistream overlap shared expert is enabled`.
4. `--async-scheduling` off, then `--no-enable-prefix-caching`.
5. `--enforce-eager` (moves the run off the graph path entirely, `FULL_DECODE_ONLY` →
   none): separates capture-mode numerics from eager numerics without touching the tag.
6. Only then the code rows: the `torch_npu` vs `_C_ascend` indexer branch needs a local
   patch, so it is a *confirmation* run, not a row the reporter can execute.

## Not this page

- An unresolvable chat template / `--tokenizer-mode` question where
  `/v1/chat/completions` never reaches the model:
  `../serving-frontend-faults/chat-template-missing-and-tokenizer-mode.md`.
- 乱码 on a **long** input with sparse c8 / DCP / PD / KV-offload flags:
  `../accuracy-benchmarks/garbled-long-context-output.md`; with DCP > 1 **and**
  `enable_sparse_sfa_c8` on, an accuracy-only or garble report is
  `dcp-c8-accuracy-garble.md` (this topic) before anything else.
- `glm_moe_dsa` is DSA/MLA **sparse attention**, not a mamba hybrid: `mamba_cache_mode`,
  `mamba_block_size` and `preprocess_mamba` never enter, so the hybrid
  block-size/prefix-caching rows (`../accuracy-benchmarks/hybrid-block-size-corruption.md`,
  `../spec-decode-mtp/hybrid-mtp-prefix-cache-unvalidated.md`) do **not** apply even
  though the launch carries `--enable-prefix-caching` and `deepseek_mtp`.
- A score gap between two quantized formats of this family:
  `../accuracy-benchmarks/quantized-sibling-score-gap.md`.

## Recorded case

**#10984** — GLM-5.1-w4a8, A2 2-node × 8 NPU, TP8 DP2 EP, `--async-scheduling`,
`FULL_DECODE_ONLY`, `deepseek_mtp` n=3, `multistream_overlap_shared_expert: true`;
broken `v0.21.0rc1 80610e44`, working `v0.20.2rc1 367b8e62` (not in the snapshot).
The report pasted the whole launch, a three-row prompt table and its own refutations, and
MTP-off was already run by the reporter — so rung 1 of the general spec-decode ladder is
spent before the first turn. The run ranked the W4A8 shared-expert overlap first and the
indexer branch second (both correct as *sites*, both unproven as *causes*), and offered
the length-matched plain prompt third; the tutorial diff above was not taken, though the
page ships a row with this exact topology.
