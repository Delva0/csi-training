#!/usr/bin/env python3
"""opcount-reconcile.py — turn a profiler op-statistics table (operator name ->
call count) into the collection window's step count and each op's per-step
count, and reconcile the table against the window's own clock.

Why: an op-count report ("<Op> 被调用了 98 万次", "NPU 上 matmul 调用次数是 GPU
的 50 倍") is a cumulative count over a *collection window*. The window's step
count is recoverable from the table alone when three or more ops with different
per-step multiplicities agree on one quotient; the total-time column then has to
reconcile with that window (`sum(count * avg) <= elapsed`). If nothing agrees,
the table is not one window and the first question is the collection, not the
engine -- see references/performance-throughput/profiler-opcount-evidence.md.

Usage:
  uv run --no-project python scripts/opcount-reconcile.py \
      --count QuantBatchMatmulV3_ND=988224 --count FusedInferAttentionScore=329408 \
      --count MatMulV2_FP16=329408 --count AddRmsNormQuant=653669 \
      --count AscendQuantV2=334555 --layers 64
  uv run --no-project python scripts/opcount-reconcile.py --table ops.txt \
      --steps 5147 --elapsed-s 50.5 --avg-us QuantBatchMatmulV3_ND=51.1

  ops.txt lines: "<name> <count>" / "<name>,<count>" / "<name>\t<count>"
  (thousands separators and a trailing total-duration column are ignored;
  lines without a positive integer count are listed as skipped).

Exit status: 0 = the table reconciles (one step count agreed by every count, or
a single count given), 1 = no common step count (treat the table as mixed
windows / a broken collection), 2 = input unusable.
"""

import argparse
import re
import sys


COUNT = re.compile(r"^(\d[\d,_]*)")


def parse_pair(text, flag, kind=int):
    if "=" not in text:
        sys.exit(f"{flag} expects NAME=VALUE, got {text!r}")
    name, value = text.split("=", 1)
    value = value.strip().replace(",", "").replace("_", "")
    try:
        return name.strip(), kind(float(value) if kind is float else int(value))
    except ValueError:
        sys.exit(f"{flag} {name}: {value!r} is not a number")


def read_table(path):
    counts, skipped, notes = {}, [], []
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8", errors="replace")
    with stream:
        for line in stream:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            line = line.strip()
            parts = re.split(r"[\t|]+|\s+", line, maxsplit=1)
            if len(parts) < 2 or "," in parts[0]:
                parts = line.split(",", 1)
            if len(parts) < 2:
                skipped.append(line)
                continue
            name, rest = parts[0].strip(), parts[1].strip()
            m = COUNT.match(rest)
            if not name or not m:
                skipped.append(line)
                continue
            digits = m.group(1)
            after = rest[m.end():]
            # "653 669": a single-space thousands separator, joined with a note
            while re.match(r"^\s\d{3}$", after):
                digits += after.strip()
                after = ""
                notes.append(f"joined a space-separated thousands group in {line!r}")
            counts[name] = int(re.sub(r"[,_]", "", digits))
    return counts, skipped, notes


def quotient_sets(counts, max_factor):
    """{op: {quotient: [factors]}} for every divisor up to max_factor."""
    out = {}
    for name, count in counts.items():
        d = {}
        for f in range(1, min(max_factor, count) + 1):
            if count % f == 0:
                d.setdefault(count // f, []).append(f)
        out[name] = d
    return out


def main(argv):
    ap = argparse.ArgumentParser(description="reconcile profiler op counts with a collection window")
    ap.add_argument("--count", action="append", default=[], metavar="NAME=N",
                    help="one operator and its call count; repeatable")
    ap.add_argument("--table", help="file (or - for stdin) of '<name> <count>' lines")
    ap.add_argument("--max-factor", type=int, default=512,
                    help="largest per-step multiplicity to consider (default 512; "
                         "use ~4x the model's layer count)")
    ap.add_argument("--layers", type=int, help="model layer count, for count/layers")
    ap.add_argument("--steps", type=int, help="a step count to check against the table")
    ap.add_argument("--elapsed-s", type=float, help="collection window length in seconds")
    ap.add_argument("--avg-us", action="append", default=[], metavar="NAME=US",
                    help="average per-call duration in microseconds; repeatable")
    a = ap.parse_args(argv)

    counts = {}
    skipped = []
    notes = []
    for item in a.count:
        name, value = parse_pair(item, "--count")
        counts[name] = value
    if a.table:
        tcounts, skipped, notes = read_table(a.table)
        counts.update(tcounts)
    if not counts:
        print(__doc__)
        return 2

    print("== counts ==")
    for name, count in counts.items():
        print(f"  {count:>12,}  {name}")
    for line in notes:
        print(f"  note: {line}")
    for line in skipped:
        print(f"  skipped (no integer count): {line[:80]}")

    status = 0
    sets = quotient_sets(counts, a.max_factor)
    if len(counts) >= 2:
        common = set.intersection(*[set(s) for s in sets.values()])
        common = {q for q in common if q >= 2}
        print(f"\n== step counts all {len(counts)} ops agree on "
              f"(per-step factor <= {a.max_factor}) ==")
        if common:
            for q in sorted(common)[:5]:
                factors = ", ".join(f"{n}={sets[n][q][0]}" for n in counts)
                print(f"  {q:,} steps   -> per step: {factors}")
            if len(common) > 5:
                print(f"  (+{len(common) - 5} more candidates)")
            print("  Read the per-step factors against the model: for a dense "
                  "transformer they should be the layer count times a small "
                  "integer (one per matmul / norm / attention per layer).")
        else:
            status = 1
            print("  none: this table is not one collection window (mixed step "
                  "counts, a partially broken collection, or ops from two runs).")
            print("  pairwise agreements:")
            names = list(sets)
            for i, n1 in enumerate(names):
                for n2 in names[i + 1:]:
                    shared = (set(sets[n1]) & set(sets[n2])) - {1}
                    for q in sorted(shared)[:3]:
                        print(f"    {q:>10,} steps: {n1}/{sets[n1][q][0]} = "
                              f"{n2}/{sets[n2][q][0]}")
    else:
        print("\n== one count: no cross-check possible, use --steps or add ops ==")

    if a.steps:
        print(f"\n== per step at --steps {a.steps:,} ==")
        for name, count in counts.items():
            per = count / a.steps
            flag = "" if abs(per - round(per)) < 1e-6 else "  (not an integer)"
            print(f"  {per:>10.3f}  {name}{flag}")

    if a.layers:
        print(f"\n== count / layer over the whole window (--layers {a.layers}) ==")
        for name, count in counts.items():
            print(f"  {count / a.layers:>10.3f}  {name}")

    if a.elapsed_s:
        print(f"\n== window ({a.elapsed_s:g} s) ==")
        for name, count in counts.items():
            print(f"  {count / a.elapsed_s:>12,.1f} calls/s  {name}")

    if a.avg_us:
        print("\n== total-time reconciliation (count x avg-us vs elapsed) ==")
        total = 0.0
        for item in a.avg_us:
            name, us = parse_pair(item, "--avg-us", float)
            if name not in counts:
                print(f"  {name}: not in the table, ignored")
                continue
            seconds = counts[name] * us / 1e6
            total += seconds
            print(f"  {seconds:>10.3f} s  {name} ({counts[name]:,} x {us:g} us)")
        if a.elapsed_s:
            share = total / a.elapsed_s * 100
            print(f"  {total:>10.3f} s  summed over the ops given")
            print(f"  {share:>10.1f} % of the {a.elapsed_s:g} s window")
            if total > a.elapsed_s:
                status = 1
                print("  over 100 %: the per-call durations and the counts were "
                      "not sampled in one window, or the counts are inflated.")
            elif share < 5:
                print("  under 5 %: those ops cannot be the window's cost; check "
                      "whether the collection mostly covers idle time.")
        else:
            print("  (add --elapsed-s to compare with the window)")

    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
