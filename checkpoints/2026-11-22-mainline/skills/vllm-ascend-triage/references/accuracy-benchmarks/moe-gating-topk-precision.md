# Fluent but wrong on a MoE serve: the gating/top-k numeric path

Scope: an MoE ("A3B"-class, sparse) serve whose answers are **wrong but fluent** — the
reasoning text is mixed-language or keeps re-deciding ("中英混杂，带着思考的内容一直思考，
不能给正确的结果"), the final JSON/answer contradicts the prompt's own rule, while
tokens/s, `finish_reason` and the engine log are normal. Reporter-confirmed shape
**#10892** (300I Duo / 310P, Qwen3.6-35B-A3B-w8a8, vllm-ascend v0.21.0rc1 `80610e4`,
vllm v0.21.0 `ad7125a`).

Inherit the topic rules in `index.md` and the delivery gates in
`../evidence-workspace/delivery-discipline.md`.

## Step 0 — correctness decides the lane, not the field the text sits in

One line from the reporter's own paste decides everything else: **is the text after
`</think>` right?**

| what the paste shows | lane | where |
|---|---|---|
| `content` carries the CoT but the answer after `</think>` is correct; or `content: null` with everything in `reasoning` and `finish_reason: length` | routing/display only — the upstream OpenAI frontend | `reasoning-output-routing.md` |
| the answer itself is wrong, or the reasoning contradicts itself / stays in the wrong language after the CoT | **numerics**; routing is a display layer | this page |

A `--reasoning-parser` / `enable_thinking: false` change **never touches logits**, so it
cannot repair a wrong answer. Say that in the first sentence and stop ranking the
routing row as the cause: the recorded #10892 run answered this shape with the routing
rows and opened by ruling numerics out ("这不是 300I Duo（310P）的硬件/量化精度问题",
`presence_penalty: 1.5` ranked only fourth), while the maintainer's later resolution was
a **precision defect in the MoE gating top-k softmax op** — a downstream fix that the pinned tree does not carry.
The reporter's own follow-up is the second half of the evidence: "输出很快", i.e. the
perf and truncation lanes are retired while the text is still wrong.

## Why the gating op, and why CI does not see it

On 310P the routing computation is a single fused op; on A2/A3 it is a different one.
Which one runs is decided by the checkpoint's routing knobs, so read the branch before
blaming an op:

| claim | anchor (v0.21.0rc1 `80610e4`) |
|---|---|
| 310P fast path: the whole route becomes one `torch_npu` op when `scoring_func == "softmax"`, `use_grouped_topk` false and no `custom_routing_function` — the branch a Qwen3-MoE/Qwen3.6-A3B checkpoint takes when its config has no expert groups (confirm from the checkpoint, last section) | `vllm_ascend/_310p/fused_moe/experts_selector.py:61`; renorm applied afterwards in `vllm_ascend/ops/fused_moe/experts_selector.py:195-201` |
| otherwise the 310P selector drops to the portable reference (`_native_select_experts`), and the generic path goes through `DeviceOperator.moe_gating_top_k` | `_310p/fused_moe/experts_selector.py:63-76`; `ops/fused_moe/experts_selector.py:140,296`; `device/device_op.py:79` (`torch.ops._C_ascend.moe_gating_top_k`), `:831` (`torch_npu.npu_moe_gating_top_k`) |
| the op the 310P path uses has **no numeric gate at the pin**: its e2e test computes `torch.allclose(...)` and throws the result away (no `assert`), feeds uniform `torch.rand` in fp16 with `D ≤ 128` while the model routes over 256 experts, and never spans more than one seed | `tests/e2e/nightly/single_node/ops/singlecard_ops/test_gating_top_k_softmax.py:28-37` |
| the `_310p` unit test **mocks** the op, so unit coverage is zero for its values | `tests/ut/_310p/fused_moe/test_experts_selector_310.py` |
| by contrast the generic op has an fp32 numpy reference and a tolerance | `tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_moe_gating_top_k.py` (`moe_gating_top_k_numpy_ref`, `RTOL_TOLERANCE 1e-5`) |

So a small weight/top-k error in the 310P path is invisible in CI and is applied once per
MoE layer per token — enough to bend the routing distribution and produce
plausible-but-wrong text without any crash, 乱码 or repeated token. Quote that as the
mechanism shape; do not derive the kernel (the repair is downstream, so a walk of the
AscendC source at the pin cannot find it — the same trap as
`../hccl-and-hardware/aclgraph-full-decode-runtime-hang.md`).

## Cheap discriminators, cheapest first

1. **Probe the op on the reporter's card**, one command, no restart:
   `uv run --no-project python <skill_root>/scripts/moe-gating-topk-probe.py --experts 256 --top-k 8`
   — compares `npu_moe_gating_top_k_softmax` against an fp32 softmax+topk reference over
   several seeds and near-tie spreads, and prints id-mismatch counts plus the weight
   max-abs-diff. Non-zero id mismatches or a weight error above fp16 noise ⇒ the gating
   lane is live and the deliverable becomes "report it with this output", not a config
   ladder. Exit 2 = cannot run here (no NPU / no `torch_npu`), which is a gap, not a pass.
2. **`return_token_ids: true` on the same body.** Tokens that the client can decode back
   to the same text prove the CoT is the model's own output, retiring every
   parser/delta/detokenizer theory before any of them is argued.
3. **A dense model, or the same checkpoint on an A2/A3 host**, answers the same prompt
   correctly ⇒ the box and the template are fine and the lane is the SoC routing/quant
   path. On the 310P host a dense control is the cheaper of the two.
4. **BF16 vs the W8A8 artifact** (only if both exist for the checkpoint) separates
   "gating op alone" from "gating op × quantised weights". Label it a hypothesis row
   until it runs.
5. Keep `presence_penalty`/`temperature` as a *secondary* row at most: they can amplify
   rambling, but they cannot explain a rule-following answer that is wrong while the
   same sampling was accepted as normal elsewhere in the report.

## What to put in the answer

Verdict line in the report's language ("不是思考路由/parser 问题：正文本身就是模型算错的
输出，`--reasoning-parser` 只改变显示位置"), then the correctness gate above with the
reporter's own quote, then the ranked lanes (gating op numerics first, quant artifact
second, sampling a distant third), then the probe command, then the coverage note: the
variant may be absent from `docs/source/user_guide/support_matrix/supported_models.md`
and only preview-listed on 310P (`docs/source/tutorials/hardwares/310p.md`) — that is
coverage, never the cause. State which artefacts settle it: one response printing
`content` + `reasoning` + token ids, the probe output, and the same prompt on a dense
model if one is available.

Open items to name rather than guess: whether the pinned image's op is the fixed build
(the fix landed downstream of this pin), and whether the checkpoint takes the 310P fast
branch at all (`use_grouped_topk` / `scoring_func` from the checkpoint config).
