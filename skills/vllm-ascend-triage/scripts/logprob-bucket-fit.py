#!/usr/bin/env python3
"""Fit one logit-space perturbation to a per-bucket per-token logprob-delta table.

Why: a cross-platform accuracy report (Ascend vs an H100/H200 golden, teacher-forced
`prompt_logprobs`) usually arrives bucketed by the golden token's own probability:

    | GT token probability | n    | mean abs delta | p95   |
    | p > 0.99             | 7747 | 0.0002         | 0.0012|
    | 0.9-0.99             | 2009 | 0.0099         | 0.034 |
    | 0.4-0.7              | 1305 | 0.0834         | 0.243 |
    | p < 0.4              | 1330 | 0.1705         | 0.513 |

One mechanism explains the whole table: a zero-mean perturbation of the *logits*.
For the scored token y, d(log p_y) = d(logit_y) - sum_i p_i d(logit_i), so the
observed delta shrinks as (1-p_y) while the residual mass sits on one or two close
competitors. The OCR-stable statistic is therefore the ratio

    mean|d log p| / (1 - p_mid)        ("how much of the confidence margin moved")

being FLAT across buckets below ~0.99 - that is one perturbation scale and one
mechanism. The ratio in the p > 0.99 bucket is expected to come out *lower* than the
flat value (the scorer's margin is large, so the same logit perturbation moves its
logprob less), so a quiet top bucket is not a second mechanism. A bucket whose ratio
is ABOVE the flat value is the one that is not roundoff - dump top_logprobs there.

The script also inverts the ratio into a logit-space delta bracket (concentrated vs
uniform residual mass). Quote the bracket as an order of magnitude, label the
assumption, and never present it as a measured per-op error.

Usage (run through uv):
  uv run --no-project python scripts/logprob-bucket-fit.py \
      --row '0.99,1.0,7747,0.0002' --row '0.9,0.99,2009,0.0099' \
      --row '0.4,0.7,1305,0.0834' --row '0.0,0.4,1330,0.1705'

Row = P_LOW,P_HIGH,N,MEAN_ABS. Two fields (P,MEAN_ABS) are also accepted. P_LOW may
be 0 or P_HIGH 1. `--vocab` (default 152064) is only used for the uniform-tail bound.
`--spread` (default 2.0) is the factor inside which the buckets are called one
mechanism. Prints a table plus the verdict. Exit 0; 2 on bad input.
"""
from __future__ import annotations

import argparse
import math
import sys

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def parse_row(text: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 4:
        lo, hi, n, mean = (float(x) for x in parts)
    elif len(parts) == 2:
        lo = hi = float(parts[0])
        n = float("nan")
        mean = float(parts[1])
    else:
        raise ValueError(f"row {text!r}: need 2 or 4 comma-separated fields")
    if not 0.0 <= lo <= 1.0 or not 0.0 <= hi <= 1.0 or hi < lo:
        raise ValueError(f"row {text!r}: p must satisfy 0 <= low <= high <= 1")
    if not mean >= 0.0:
        raise ValueError(f"row {text!r}: mean abs delta must be >= 0")
    return lo, hi, n, mean


def implied(mean: float, p: float, vocab: int) -> tuple[float, float]:
    """Return (delta_concentrated_tail, delta_uniform_tail)."""
    q = max(0.0, 1.0 - p)
    factor = SQRT_2_OVER_PI
    if q == 0.0 and mean > 0.0:
        # p == 1: no Jacobian shrinkage, the whole delta is the logit perturbation
        return (mean / factor, mean / factor)
    conc = math.sqrt(q * q + q * q)
    unif = math.sqrt(q * q + q * q / max(1, vocab - 1))
    lo = mean / (factor * conc) if conc > 0 else float("nan")
    hi = mean / (factor * unif) if unif > 0 else float("nan")
    return lo, hi


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--row", action="append", required=True,
                    metavar="P_LOW,P_HIGH,N,MEAN_ABS")
    ap.add_argument("--vocab", type=int, default=152064)
    ap.add_argument("--spread", type=float, default=2.0,
                    help="factor inside which the ratio counts as flat")
    args = ap.parse_args(argv)

    rows = []
    for raw in args.row:
        try:
            rows.append(parse_row(raw))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    print(f"{'bucket':>16} {'n':>7} {'p_mid':>6} {'mean|d|':>9} "
          f"{'mean/(1-p)':>11} {'delta(c)':>9} {'delta(u)':>9}")
    out = []
    for lo, hi, n, mean in rows:
        p_mid = (lo + hi) / 2.0
        d_c, d_u = implied(mean, p_mid, args.vocab)
        ratio = mean / (1.0 - p_mid) if p_mid < 1.0 else float("nan")
        label = f"{lo:g}-{hi:g}"
        n_txt = "-" if math.isnan(n) else f"{int(n)}"
        print(f"{label:>16} {n_txt:>7} {p_mid:>6.3f} {mean:>9.5f} "
              f"{ratio:>11.4f} {d_c:>9.4f} {d_u:>9.4f}")
        out.append((label, mean, p_mid, d_c, d_u))

    if len(out) < 2:
        print("\n(one bucket: nothing to compare; add the other rows of the table)")
        return 0

    # flat-ratio test over the buckets that are not the saturated top bucket
    body = [r for r in out if r[2] < 0.99 and r[1] > 0]
    if not body:
        body = [r for r in out if r[1] > 0]
    if not body:
        print("\nAll deltas are zero: the two sides agree within the print precision.")
        return 0
    ratios = [(r[0], r[1] / (1.0 - r[2])) for r in body if r[2] < 1.0]
    r_lo = min(ratios, key=lambda t: t[1])
    r_hi = max(ratios, key=lambda t: t[1])
    spread = r_hi[1] / r_lo[1] if r_lo[1] > 0 else float("inf")
    dcs = [r[3] for r in out if not math.isnan(r[3]) and r[3] > 0]
    if dcs:
        print(f"\nimplied logit-space delta: {min(dcs):.3f} .. {max(dcs):.3f} "
              f"(residual-mass bracket; quote as an order of magnitude)")
    print(f"flat-ratio test over {len(ratios)} bucket(s) below p=0.99: "
          f"{r_lo[1]:.3f} ({r_lo[0]}) .. {r_hi[1]:.3f} ({r_hi[0]}), "
          f"spread {spread:.2f}x")
    if spread <= args.spread:
        print("VERDICT: one scale. Quote 'mean|d log p| ~= "
              f"{math.sqrt(r_lo[1] * r_hi[1]):.2f} x (1-p)' as the platform floor and "
              "do not rank a per-bucket cause; a quieter p>0.99 row is expected.")
    else:
        print(f"VERDICT: not one scale - {r_hi[0]} moves "
              f"{spread:.1f}x more of its margin than {r_lo[0]}. That bucket is not "
              "roundoff: dump top_logprobs (5) at the scored token on both sides "
              "before ranking causes.")
    print("\nAssumptions to state with the numbers: zero-mean independent per-logit "
          "perturbation, fp32 log_softmax on both sides (so the arithmetic of the "
          "softmax is not the variable), bucket p taken as its midpoint, and the "
          "bucket's own mean |d| as the estimate. Bucket width matters: a 0.4-0.7 row "
          "is not a point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
