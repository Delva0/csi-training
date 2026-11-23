# Fused-MC2 MoE variants: aliased workspace sizing and GMM1 row stride

## Trigger

`enable_fused_mc2=1` (usually with `--quantization ascend`, W8A8 = int8 weights +
per-expert int64 scales) on an MoE model whose GMM1 width `N = 2 × moe_intermediate_size`
exceeds the hidden size `K`:

- `N` moderately `> K`: **garbled tokens from the first token**, every MoE layer, both
  eager and graph mode (a data-layout bug, so `--enforce-eager` changes nothing);
- `N ≫ K` (e.g. `K=256, N=1536`): AI Core fault, `MTE accesses an invalid GM address`
  / `507xxx`, because the last row's write leaves the allocation.

The report may arrive **already diagnosed**, with `path:line` and a proposed patch
(recorded #16561, vllm-ascend v0.26.0rc1 `f2f74a16c`, DeepSeek-V2-32B topology
`K=2048 / N=3072 / E=40 / topk 7`, TP2+EP2 on A2). Then the job is claim-check plus
fix review, not a cause hunt — see the last section.

## The three variants and the aliasing rule

`csrc/mc2/dispatch_ffn_combine{,_bf16,_w4_a8}/`, each with `op_host/*_tiling.cpp` and
`op_kernel/*_kernel.hpp`. The bf16 variant allocates **four separate regions**, one per
row width (`…bf16_tiling.cpp:232-243`: `maxOutputSize × {N, n2=K, K, k2=N/2}`), with a
commented-out `max()` pair right below. W8A8 and W4A8 instead **alias** two pairs at one
offset each — `ptrC`/`ptrC2` and `ptrA`/`ptrPermutedToken` (`…_kernel.hpp:1125-1133`),
advancing once per pair. The aliasing is intentional and safe (GMM1's C is dead before
GMM2 writes C2; `SyncAll` separates the A lifetimes), so each region must be sized to
the **wider** user: `max(N, K)` and `max(K, N/2)` — not the sum, and not `K`.

## State at v0.26.0rc1 `f2f74a16c` (pre-fix, verified)

| Site | Line | What it does |
|---|---|---|
| W8A8 host tiling | `dispatch_ffn_combine_tiling.cpp:297-304` | `n2 = info.K` (right as GMM2's width) is then reused as the **C region row width** (`:303 maxOutputSize * n2 * sizeof(int16_t)`) and the A region stays `:304 maxOutputSize * info.K * sizeof(int8_t)` |
| W8A8 GMM1 C layout | `…_kernel.hpp:473` | `LayoutC(m, n, problemShape.k())` — N columns written with `ldm = K` |
| W8A8 group advance | `:522` | `gmGroupOffsetC += m * k()` |
| epilogue path 1 | `:927-931` | `LayoutC{rows, n()}` (correct) but the last arg of `blockEpilogue1(...)` is `problemShape.k()`, and that arg is the **row stride**: `utils/block_epilogue_pertoken_swiglu.hpp:189 gmC[loopIdx * blockK]` |
| epilogue path 2 | `:946-950` | `LayoutC{dequantLen, k()}` and the same `k()` tile stride, so `gmOffsetC` is K-strided too — a size-independent offset bug that fires whenever the epilogue splits the GMM1 loop |
| kernel region advance | `:1129`/`:1133` | duplicates the K-based sizing (`n2`, then `k()`), must stay in step with host |
| W4A8 host tiling | `…w4_a8_tiling.cpp:253-260` | **already fixed**: `max(N, n2)` / `max(K, k2)`, with the maintainer comment "ptrC/ptrC2 alias reuse: allocate max(N, n2), not N + n2 … same pattern as W8A8 ptrA/ptrPermutedToken" |
| W4A8 kernel | `…w4_a8_kernel.hpp:585, 637, 1054, 1059-1061` | still K-strided → silent corruption *within* the allocation (no OOB crash) |

Thresholds, in the order they bite: C region at `N > K`; A region only at `N > 2K`;
epilogue path 2 at any shape where the split happens and `N ≠ K`. `max == K` when
`N ≤ K`, so the correct patch is bit-identical in the old regime.

That W4A8 comment is the strongest in-tree evidence available: the rule is already
written down for the sibling op, which means the W8A8 defect is a **partial repair** —
check for an open PR before proposing a duplicate.

## Sites where `k()` is correct — do not sweep them

- `…_kernel.hpp:859/861`: `gmOffsetPeer = rowSrc * (k() + UB_ALIGN)` and
  `CopyGMToGMPerToken(..., k(), ...)` address the **remote peer A buffer in the HCCL
  window**, whose size is pinned by the `HCCL_BUFFSIZE` check in tiling (`:285` in the
  W8A8 tiling). Changing them forces a buffer-size change the fix does not need.
- `:892` `n2 = problemShape.k()` for the epilogue's output width (genuinely K);
  `:469`/`:581` build `GemmCoord`s whose `n`/`k` are per-pass and correct.

## Discriminators that isolate the W8A8 kernel

1. Flip **only** `enable_fused_mc2` 0↔1 on one checkpoint: coherent ↔ token soup.
2. Keep the flag and dequantize **only the MoE experts** back to bf16: coherence
   returns, which names the variant and rules out the MC2 dispatch/Combine orchestration
   (round-robin dispatch, `CombineV2` capacity units, `#8320`-family garbling).
3. Op level, no model: `aclnnDispatchFFNCombine` with `n ≠ k` and non-zero scales, plus
   a `K=256 / N=1536` case for the OOB. Compare against dequantized GMM1→SwiGLU→GMM2.

## Why CI did not catch it — read the test before believing "nightly passes"

- `tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dispatch_ffn_combine.py`
  uses `k = 1024; n = 1024` (`:90-91`, `:144-145`) — `N > K` is never exercised — and
  `scale1/scale2 = torch.randint(0, 1, …)` (`:105-106`, `:159-160`) where the high bound
  is **exclusive**, so the dequant scales are all zero and both sides are identically
  zero: a pure stride/size error cannot show. `#7189` is exactly that all-zero output
  being reported as a test problem.
- Only one e2e model config enables the feature:
  `tests/e2e/nightly/single_node/models/configs/Minimax_m2.7_w8a8_A3.yaml` (A3);
  `MiniMax-M2.5-w8a8-QuaRot-A2.yaml` exists and does not, so no A2 W8A8 coverage.
- The fix's own test needs `n > k` **and** non-zero scales; a shape-only A/B is not
  enough.

## Reviewing a self-diagnosed defect report (the general shape)

- Verify every cited `path:line` at the `SOURCES.json` pin and say which are off —
  #16561's cited `933`/`952` are really `931`/`950`. Line numbers are version-scoped:
  quote the pin and give the reporter a `sed -n 'A,Bp'` to run on theirs.
- Quote the sibling the reporter used as reference precisely. bf16 does not "use an
  N-wide buffer instead of sharing"; it allocates four regions and never aliases, so
  `max()` (not a sum) is the right analogue.
- Name the sites that must **not** change (above) and the memory delta for review:
  `workSpaces[0]` grows by `maxOutputSize·(max(N,K)−K)·2 + maxOutputSize·(max(K,k2)−K)`
  bytes; `HCCL_BUFFSIZE` is untouched.
- Separate source-verified code claims from reporter-measured numbers (relative error,
  tok/s, eval parity). Report the latter as reporter-confirmed; no NPU exists in this
  workspace.
- If a newer pin in the snapshot already carries the repair, quote it. Here the pin *is*
  the reported ref and the defect is present.

## Precedents (same op family, different mechanisms — do not conflate)

`#7189`: this very op test printing all-zero output. `#10335`: `dispatch_ffn_combine_w4_a8`
dispatch 收 tokens 错误. `#14869`: W4A8 FUSED_MC2 `CombineV2` mixes packed-row and
token-row capacity units. `#8320`: `fused_mc2` garbling/empties with
FULL_DECODE_ONLY+DP8TP2 and saturated kvcache. `#9748`, `#7154`, `#13923`: fused-MC2
startup error, post-capture timeout, bf16 crash under CANN 9.1. `#9186`: the `bias1`
`Optional[List[Tensor]]` binding error.
