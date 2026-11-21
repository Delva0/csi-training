#!/usr/bin/env python3
"""Reconcile a serving-benchmark summary table (EvalScope / vllm bench serve /
AISBench, pasted as text or as a screenshot OCR) and turn two runs into rates.

Why: the summary block reports `Time taken for tests (s)`, concurrency, request
count, `Output token throughput (tok/s)` and `Total token throughput (tok/s)`.
The two throughput fields are the only ones that survive an OCR pass, because
they are large unique numbers. `Average input/output tokens per request` and the
per-token latency columns are small and get transposed (`4094` -> `4994`, `100`
-> `199`, `0.1046` -> `0.1946`). Everything the answer needs is therefore
*derived*: per-request token counts, the decode step time at that concurrency,
and the prefill cost per input token.

Definitions used (all derived from the five reliable fields):
  output_tokens_total = output_tok_s * wall_s
  per_req_out         = output_tokens_total / requests
  per_req_in          = (total_tok_s * wall_s - output_tokens_total) / requests
  step_s              = concurrency / output_tok_s      # one decode step of a full batch
  rate_per_req        = output_tok_s / concurrency      # tokens/s per active sequence
  prefill_ms_per_tok  = 1000 * ttft_s / per_req_in      # only when --ttft is given

usage:
  benchmark-table-normalize.py --run NAME OUT_TOK_S WALL_S REQS CONC
                               [--total-tok-s RUN=TOTAL] [--ttft RUN=SECONDS]
                               [--compare A B]

  --run        repeatable; the five required numbers positionally
  --total-tok-s  RUN=value, repeatable (gives per-request input tokens)
  --ttft         RUN=value, repeatable (gives prefill ms per input token)
  --compare      A B: print the pairwise ratios and say which axis moved more

Exit 0 on success, 2 on unusable input (non-numeric, zero output throughput or
zero requests). Ratios are printed as B/A. The script does not decide the cause:
it fixes which axis (decode step vs prefill per token) the reporter's number is
about, and whether the two runs are comparable at all.
"""
import argparse
import sys


def parse_run(raw):
    parts = raw.split()
    if len(parts) < 5:
        raise ValueError(f"--run needs NAME OUT_TOK_S WALL_S REQS CONC, got {raw!r}")
    name = parts[0]
    try:
        out_tok_s, wall, reqs, conc = (float(x) for x in parts[1:5])
    except ValueError as exc:
        raise ValueError(f"--run {name}: non-numeric field ({exc})") from exc
    if wall <= 0 or out_tok_s <= 0 or reqs <= 0 or conc <= 0:
        raise ValueError(f"--run {name}: wall/output-tok-s/requests/concurrency must be > 0")
    return name, out_tok_s, wall, reqs, conc


def parse_named(raw, flag):
    if "=" not in raw:
        raise ValueError(f"{flag} wants RUN=VALUE, got {raw!r}")
    name, value = raw.split("=", 1)
    try:
        return name.strip(), float(value)
    except ValueError as exc:
        raise ValueError(f"{flag} {name}: non-numeric value ({exc})") from exc


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--total-tok-s", action="append", default=[])
    ap.add_argument("--ttft", action="append", default=[])
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    try:
        runs = [parse_run(r) for r in args.run]
        totals = dict(parse_named(r, "--total-tok-s") for r in args.total_tok_s)
        ttfts = dict(parse_named(r, "--ttft") for r in args.ttft)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    derived = {}
    for name, out_tok_s, wall, reqs, conc in runs:
        out_total = out_tok_s * wall
        per_req_out = out_total / reqs
        step = conc / out_tok_s
        row = {
            "wall": wall,
            "reqs": reqs,
            "conc": conc,
            "out_tok_s": out_tok_s,
            "out_total": out_total,
            "per_req_out": per_req_out,
            "step_s": step,
            "waste": wall / (reqs / conc * per_req_out * step) if step > 0 else 0.0,
        }
        if name in totals:
            per_req_in = (totals[name] * wall - out_total) / reqs
            row["total_tok_s"] = totals[name]
            row["per_req_in"] = per_req_in
            row["prefill_share"] = (totals[name] - out_tok_s) / totals[name]
        if name in ttfts and "per_req_in" in row and row["per_req_in"] > 0:
            row["ttft"] = ttfts[name]
            row["prefill_ms_per_tok"] = 1000.0 * ttfts[name] / row["per_req_in"]
        derived[name] = row

        print(f"== {name}")
        print(f"   wall={wall:.4g}s  requests={reqs:.6g}  concurrency={conc:.6g}")
        print(f"   output tokens total = {out_total:.5g}  -> per request = {per_req_out:.5g}")
        print(f"   decode step (full batch of {conc:.4g}) = {step*1000:.4g} ms"
              f"   per-sequence rate = {out_tok_s/conc:.4g} tok/s")
        if reqs / conc > 1:
            print(f"   waves = {reqs/conc:.4g}; wall / (waves x per-request tokens x step) = "
                  f"{row['waste']:.3f}  (>>1 means queueing, warmup, or a per-request fixed cost"
                  " on top of the decode loop)")
        if "per_req_in" in row:
            print(f"   total tok/s = {row['total_tok_s']:.5g}  -> per request input = "
                  f"{row['per_req_in']:.5g}  (prefill share of tokens = {row['prefill_share']*100:.1f}%)")
        if "prefill_ms_per_tok" in row:
            print(f"   TTFT = {row['ttft']:.4g}s -> prefill = "
                  f"{row['prefill_ms_per_tok']:.4g} ms per input token")
        derived_avgs = [f"avg out {row['per_req_out']:.4g}"]
        if "per_req_in" in row:
            derived_avgs.append(f"avg in {row['per_req_in']:.4g}")
        print(f"   quote these, not the table's own average columns: {', '.join(derived_avgs)}")
        print("   (an OCR'd average or per-token column is a small number and gets transposed;"
              " the two large throughput fields do not)")

    if args.compare:
        a, b = args.compare
        if a not in derived or b not in derived:
            print(f"error: --compare needs two --run names, got {a!r}/{b!r}", file=sys.stderr)
            return 2
        step_ratio = derived[b]["step_s"] / derived[a]["step_s"]
        print(f"== {b} / {a}  (cost ratios: >1 means {b} is slower)")
        print(f"   decode step cost = {step_ratio:.3g}x  (output tok/s ratio = "
              f"{derived[b]['out_tok_s']/derived[a]['out_tok_s']:.3g}x at equal concurrency)")
        if "prefill_ms_per_tok" in derived[a] and "prefill_ms_per_tok" in derived[b]:
            pre_ratio = derived[b]["prefill_ms_per_tok"] / derived[a]["prefill_ms_per_tok"]
            print(f"   prefill cost per input token = {pre_ratio:.3g}x")
            import math
            dev = {"decode": abs(math.log(step_ratio)), "prefill": abs(math.log(pre_ratio))}
            worse = max(dev, key=dev.get)
            slower = b if (step_ratio if worse == "decode" else pre_ratio) > 1 else a
            print(f"   the axis that moved more is {worse}, and it is worse on {slower}:"
                  f" rank causes on the {worse} side first and do not spend turns on the"
                  " other side's flags")
        else:
            print("   pass --total-tok-s and --ttft for both runs to split decode from prefill")
        if "per_req_out" in derived[a] and "per_req_out" in derived[b]:
            ra, rb = derived[a]["per_req_out"], derived[b]["per_req_out"]
            if max(ra, rb) / min(ra, rb) > 1.2:
                print(f"   WARNING per-request output differs ({ra:.4g} vs {rb:.4g}): the"
                      " *totals* are not comparable; the step ratio above still is at equal"
                      " concurrency, and on a spec-decode run divide it by (1+num_speculative_tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
