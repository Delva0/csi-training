#!/usr/bin/env python3
"""Compare torch_npu.npu_moe_gating_top_k_softmax with an fp32 reference.

Originating case: #10892 (300I Duo / 310P, Qwen3.6-35B-A3B-w8a8, vllm-ascend
v0.21.0rc1) — fluent but wrong answers traced to the MoE gating top-k softmax
op by the maintainer.  vllm_ascend/_310p/fused_moe/experts_selector.py:61 calls
this op for the whole routing computation, so one run on the reporter's card
decides whether the gating lane is live.

Reference (the same identity the in-tree e2e test asserts, but at fp32):
    ref_weights, ref_ids = softmax(x.float(), dim=-1).topk(k, dim=-1)

Run through uv, inside the vllm-ascend image, on the reported card:

    uv run --no-project python moe-gating-topk-probe.py --experts 256 --top-k 8

Exit codes: 0 = op matches the fp32 reference within tolerance (gating lane
weakened), 1 = mismatch (the gating lane is live — report the output), 2 = the
probe cannot run here (no torch / torch_npu / NPU).  Run it across several
seeds and both `--spread` values: a wide spread hides precision loss, a narrow
one concentrates the logits where top-k ties live.
"""

import argparse
import sys


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experts", type=int, default=256,
                    help="num_experts of the checkpoint (Qwen3.6-35B-A3B routes over 256)")
    ap.add_argument("--top-k", type=int, default=8, help="top_k routed experts per token")
    ap.add_argument("--rows", type=int, default=64, help="tokens per case")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="router_logits dtype as the engine sees it")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per spread value")
    ap.add_argument("--spread", type=float, nargs="+", default=[1.0, 0.05],
                    help="logit std; small values force near ties between experts")
    ap.add_argument("--device", default="npu", help="torch device string, e.g. npu")
    ap.add_argument("--atol", type=float, default=1e-3, help="weight tolerance")
    ap.add_argument("--rtol", type=float, default=1e-3, help="weight relative tolerance")
    return ap.parse_args(argv)


def call_op(op, x, k):
    """Call the op across the two argument forms seen in tree (finished may be required)."""
    errors = []
    for kwargs in ({"k": k}, {}):
        try:
            if kwargs:
                out = op(x, k=k)
            else:
                out = op(x, None, k=k)
            return out[0], out[1]
        except TypeError as exc:  # signature mismatch, try the next form
            errors.append(str(exc))
    raise RuntimeError("no accepted call form: " + "; ".join(errors))


def main(argv):
    args = parse_args(argv)

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - any import failure means "cannot run here"
        print(f"SKIP: torch not importable here ({exc})")
        return 2

    try:
        import torch_npu  # noqa: F401
        op = torch_npu.npu_moe_gating_top_k_softmax
        available = torch.npu.is_available()
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: torch_npu.npu_moe_gating_top_k_softmax unavailable ({exc})")
        return 2
    if not available:
        print("SKIP: torch.npu.is_available() is False on this host")
        return 2

    if args.top_k < 1 or args.top_k > args.experts:
        print(f"BAD INPUT: --top-k {args.top_k} outside 1..{args.experts}")
        return 2

    dtype = getattr(torch, args.dtype)
    worst_weight_diff = 0.0
    weight_failures = 0
    set_mismatch_rows = 0
    rows_total = 0
    order_mismatch_rows = 0

    for spread in args.spread:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            logits = torch.randn(args.rows, args.experts, dtype=torch.float32) * spread
            x = logits.to(dtype).to(args.device)
            ref_weights, ref_ids = x.float().softmax(dim=-1).topk(args.top_k, dim=-1)

            weights, ids = call_op(op, x, args.top_k)
            weights = weights.float().cpu()
            ids = ids.cpu().long()
            ref_weights = ref_weights.cpu()
            ref_ids = ref_ids.cpu()

            diff = (weights - ref_weights).abs()
            tol = args.atol + args.rtol * ref_weights.abs()
            bad = int((diff > tol).sum().item())
            weight_failures += bad
            worst_weight_diff = max(worst_weight_diff, float(diff.max().item()))

            order_bad = int((ids != ref_ids).any(dim=1).sum().item())
            set_bad = 0
            for row in range(ids.shape[0]):
                if set(ids[row].tolist()) != set(ref_ids[row].tolist()):
                    set_bad += 1
            order_mismatch_rows += order_bad
            set_mismatch_rows += set_bad
            rows_total += ids.shape[0]

            print(f"spread={spread:<6} seed={seed}  max|dw|={float(diff.max().item()):.3e} "
                  f"weights_out_of_tol={bad}  rows_with_id_set_mismatch={set_bad}  "
                  f"rows_with_order_mismatch={order_bad}")

    print()
    print(f"cases: {rows_total} rows over {len(args.spread) * args.seeds} seeds/spreads, "
          f"experts={args.experts} top_k={args.top_k} dtype={args.dtype} atol={args.atol} rtol={args.rtol}")
    print(f"worst weight abs diff: {worst_weight_diff:.3e}")
    print(f"rows whose selected expert SET differs from the fp32 reference: "
          f"{set_mismatch_rows}/{rows_total}")
    print(f"rows whose top-k ORDER differs (ties only, benign when the set matches): "
          f"{order_mismatch_rows}/{rows_total}")
    if weight_failures or set_mismatch_rows:
        print("VERDICT: mismatch vs the fp32 reference — the gating/top-k lane is live "
              "(MoE routing weights are applied once per layer per token).")
        return 1
    print("VERDICT: op matches the fp32 reference on this card and dtype — the gating "
          "lane is weakened for these shapes; try larger --experts / other --dtype "
          "before discarding it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
