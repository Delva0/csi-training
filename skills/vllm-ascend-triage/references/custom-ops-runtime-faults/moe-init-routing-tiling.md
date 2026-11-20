# MoE init-routing: `aclnnMoeInitRoutingV3 … 561103` is a tiling check on an argument

Scope: a MoE deployment dies at its first real batch — often right after a healthy
engine start — with `call aclnnMoeInitRoutingV3 failed, error code is 561103`,
`launch failed for MoeInitRoutingV2, errno:561103`, or the same code without the op
name; frequently with a parallel-layout question attached ("DP2 TP4 有没有限定？能否在
启动时校验拦截，而不是报算子错"). Recorded #8826. Not this page: an
`aclnnMoeDistributeDispatch*` / `…Combine*` error on the MC2 path
(`../hccl-and-hardware/aiv-mc2-env-ablation.md`,
`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md`), an
`EZ1001 … Get regInfo failed` registration error, or an init-routing *precision*
report with no error text.

## Decode the code before theorising: 561103 carries no operator identity

The same code appears as `errno:561103` on `aclnnFlashAttentionScore` (#115),
`error code is 561103` on `aclnnInplaceNormal` (#2892) and on
`aclnnIndexPutImpl` (#4430). It is "this launch's tiling rejected an argument", so
it never means bad card, bad HCCL, or "DP is unsupported" — grade the *argument*.

#3855 pastes the same error string **with its EZ9999 body**, which is the whole
diagnosis (vllm-ascend `0.11.0rc1.dev53` / `9eb62935b`, CANN `8.3.RC1.alpha003`,
vllm `0.11.0`, Qwen3-30B-A3B with `tensor_parallel_size=2` +
`enable_expert_parallel`, `load_format="dummy"`):

```text
NPU function error: call aclnnMoeInitRoutingV3 failed, error code is 561103
ERR00100 PTA call acl api failed.
EZ9999: Inner Error!
EZ9999: The first dim of expertTokensCountOrCumsum should be 128.
        [FUNC:CheckTokenCount][FILE:moe_init_routing_v2_tiling.cpp][LINE:131]
        Tiling failed
        Kernel Run failed. opType: 37, MoeInitRoutingV2
        launch failed for MoeInitRoutingV2, errno:561103.
```

Two facts follow. (a) The check names an **argument**: the first dim of the
expert-token count tensor must equal the expert count the op was tiled for, so the
question is which expert count the caller declared. (b) The three op names in one
block are three layers of one call — the caller's `torch_npu` function, the aclnn
entry point of that CANN line, and the tiling/kernel opType — so "aclnnMoeInitRoutingV3"
is **not** a grep key in vllm-ascend. At v0.13.0rc3 the whole tree contains no
`MoeInitRoutingV3` string: only `torch_npu.npu_moe_init_routing_v2`
(`ops/fused_moe/token_dispatcher.py:338`) and the downstream backport custom op
`aclnnMoeInitRoutingCustom` (`csrc/torch_binding.cpp:1134`). Ask for the EZ9999
block — `CheckTokenCount` / `moe_init_routing_v2_tiling` in the host log or plog —
instead of reading kernels.

## Which caller parameters the tiling sees

`vllm_ascend/ops/fused_moe/token_dispatcher.py:324-350` at v0.13.0rc3
(`TokenDispatcherWithAllGather.token_dispatch`) passes `expert_num=global_num_experts`,
`active_expert_range=[first, last]` built from the EP rank and `num_experts_local`
(`:329-333`), `active_num=num_tokens * top_k`, `expert_tokens_num_type=1` (count
mode), and `quant_mode=1` under w8a8-dynamic: those are exactly the numbers the
check compares, and every one of them comes from the **parallel layout plus the
checkpoint**, not from a kernel defect.

Which dispatcher is even reachable is arithmetic, one call:
`ascend_forward_context.py:246-255` (EP off → `ALLGATHER`; on A2 `MC2` only when
`num_tokens <= capacity` **and** `world_size_across_dp / pp >= 16`, else
`ALLGATHER`). DP2×TP4 = 8 lands on `ALLGATHER` by that inequality, so on a
single A2 node the reporter's layout can never reach MC2. Take the selection table
from `../feature-enable-recipes/moe-expert-parallel-with-dp.md` (it also carries
the v0.12→v0.13 move of the selector and the capacity formula) rather than
re-deriving it.

The "please validate it at start-up instead" request has three answerable parts,
and none of them is a DP×TP matrix:

- EP **off** with `DP > 1`: every DP rank replicates all experts — a warning the
  tree does not emit.
- The `ALLGATHER`-instead-of-`MC2` choice below the ≥16 threshold is silent by
  default; the DEBUG `num_tokens: N, moe_comm_type: …` line is the only trace.
- An aclnn-code wrapper at the op call that prints `num_tokens / top_k /
  expert_num / active_expert_range / quant_mode` turns 561103 into a readable
  argument mismatch. The constraint itself is a comparison on capacity and world
  size, so a fixed table cannot express it.

## The documented layout and the operator-availability axis

- `docs/source/tutorials/DeepSeek-V4.md` (**not** `tutorials/models/`) at
  v0.13.0rc3: `:5` "temporarily only supports DeepSeek-V4-FLASH"; the A2
  single-node block `:149-167` is `--data-parallel-size 1 --tensor-parallel-size 8
  --enable-expert-parallel`, MTP `num_speculative_tokens: 1`, `FULL_DECODE_ONLY`.
  DP2×TP4 appears in no example, in no `tests/e2e/**` file and nowhere in `tests/`
  (grep for `DeepSeek-V4` in `tests/` → nothing), so that layout has no CI
  envelope; say that as scope, not as the cause.
- Operator availability is a release axis and belongs in the ranked list, but
  **the V3 name does not fingerprint the release**: #3855 printed it on CANN
  `8.3.RC1.alpha003`. What the RFCs say is that CANN 8.5 ships an *enhanced*
  `aclnnMoeInitRoutingV3` and that 8.3 needed a downstream backport custom op
  (`#5501: RFC Support MoeInitRoutingCustom for CANN 8.3`, PRs #5251/#5332;
  `#5494: Upgrade CANN from 8.3 to 8.5` still lists "Need replace the
  aclnnMoeInitRoutingV3" as an open item). So keep the pairing as a hypothesis and
  settle it with `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`,
  `pip show vllm-ascend torch-npu`, and the image tag suffix — not by reading the
  error string.

## Action ladder

1. Re-run the documented A2 command (DP1 TP8 + EP + MTP 1, `:149-167`). Still
   failing ⇒ the layout is not the variable; go to rung 3.
2. Keep DP2 if required: add `--enable-expert-parallel` when absent, then
   `--enforce-eager`, then drop the MTP `--speculative-config` — one change per run.
3. `ASCEND_LAUNCH_BLOCKING=1` + plog, and grep the host log for `CheckTokenCount`
   or `moe_init_routing_v2_tiling`: the EZ9999 line names the required first dim.
4. Compare that number with the checkpoint's `num_experts` and with
   `num_experts / ep_world` (`config.json`) — the count tensor and `expert_num` are
   built from those two.

Ask for: the serve command with the EP flag visible, CANN and `torch-npu` versions,
`num_experts` / `top_k` / `moe_quantize` from the checkpoint config, the EZ9999
tiling line, and the rank log around the failure.

## Limits to keep stated

- #8826's own body is one sentence. Nothing in it names the argument, so its
  answer is a ranked list plus this ladder, never a confirmed mechanism. The
  recorded answer also left the "does `npu_moe_init_routing_v2` reach a V3 aclnn
  entry point?" hop open — #3855's block closes that hop for the same error string,
  but from a different release; re-read the wording in the reporter's own plog.
- The operator's source is not in the snapshot: `Ascend/op-plugin` (complete, 107
  files) contains no `MoeInitRouting`, and no `cann/ops-*` repo is exported at all,
  so no in-tree source can settle the tiling check.
