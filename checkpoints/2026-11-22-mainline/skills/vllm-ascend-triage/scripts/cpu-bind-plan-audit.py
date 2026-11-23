#!/usr/bin/env python3
"""cpu-bind-plan-audit.py — invert the `The CPU allocation plan is as follows:` blocks
of one or more serve logs and report, per NPU, the CPU pool `cpu_binding.py` really
built plus every cross-plan overlap.

Why: `CpuAlloc.allocate()` trims the pool before printing, so a plan line
`NPU0: main=[146 … 189]  acl=[190]  release=[191]` does **not** say "the pool is
146-189": it says `pool[0]`, `pool[1]` were reserved for the sq/cq IRQs, `main` is
`pool[2:-2]`, `acl = pool[-2]` and `release = pool[-1]`. Inverting that recovers the
pool (= the card's own topo `CPU Affinity` base plus `extend_numa`'s whole next NUMA
node), which is what distinguishes the three readings of a multi-instance "it got
slower" report: two instances bound to the identical band (real overlap), an
intra-process split that worked (two halves of one band), or no binding at all (no
plan line, only `Skip binding cpu`). See
references/performance-throughput/cpu-binding-multi-instance-overlap.md (#8600,
source-verified at vllm-ascend v0.18.0rc1, cpu_binding.py:352-372).

Usage:
  uv run --no-project python scripts/cpu-bind-plan-audit.py server_a.log server_b.log
  bash -c 'grep -h "cpu_binding\\|CPU allocation plan\\|main=\\[\\|migrate\\]" *.log' | \
      uv run --no-project python scripts/cpu-bind-plan-audit.py -

Reads the log lines that carry `[cpu_bind_mode]`, `The CPU allocation plan is as
follows:`, `NPU<n>: main=[…]  acl=[…]  release=[…]` and `[migrate] NPU:n -> NUMA [k]`;
everything else is ignored, so a full serve log can be passed as-is.

Exit status: 0 = at least one plan parsed and printed; 2 = nothing parsed (say which
lines you searched for rather than reporting "no overlap").
"""

import re
import sys

MODE = re.compile(r"\[cpu_bind_mode\]\s+mode=(\S+)\s+rank=(\d+)\s+visible_npus=\[([^\]]*)\]")
# print_plan writes  NPU<n>: main=[a b …]  acl=[x]  release=[[y]]  — the release
# field is `str([y])`, so it carries one pair of brackets inside the outer pair.
PLAN_NPU = re.compile(r"NPU(\d+)\s*:\s*main=\[([^\]]*)\]")
PLAN_ACL = re.compile(r"acl=\[([^\]]*)\]")
PLAN_REL = re.compile(r"release=\[+([^\]]*)\]")
MIGRATE = re.compile(r"\[migrate\]\s*NPU:(\d+)\s*->\s*NUMA\s*\[(\d+)\]")
SKIP = re.compile(r"Skip binding cpu|Bind cpus failed")


def ints(text):
    return [int(tok) for tok in text.replace(",", " ").split() if tok.lstrip("-").isdigit()]


def read(lines, label):
    """Return the plan records found in `lines` (two passes: `[migrate]` may follow
    the plan block, since bind_threads runs after print_plan)."""
    lines = list(lines)
    records, mode, rank, visible, numa = [], None, None, None, {}
    for line in lines:
        m = MODE.search(line)
        if m:
            mode, rank = m.group(1), int(m.group(2))
            visible = ints(m.group(3))
            continue
        m = MIGRATE.search(line)
        if m:
            numa[int(m.group(1))] = int(m.group(2))
            continue
        if SKIP.search(line):
            print(f"  [skip] {label}: {line.strip()[:160]}")
            continue
        m = PLAN_NPU.search(line)
        if not m:
            continue
        main = ints(m.group(2))
        if not main:
            continue
        npu = int(m.group(1))
        ma, mr = PLAN_ACL.search(line), PLAN_REL.search(line)
        acl, rel = ints(ma.group(1)) if ma else [], ints(mr.group(1)) if mr else []
        # IRQ cores are pool[0], pool[1]; main starts at pool[2] for a contiguous pool.
        irq = [main[0] - 2, main[0] - 1] if main[0] >= 2 else []
        records.append(
            {
                "log": label,
                "npu": npu,
                "mode": mode or "?",
                "rank": rank,
                "visible": visible or [],
                "irq": irq,
                "main": main,
                "acl": acl,
                "rel": rel,
                "pool": irq + main + acl + rel,
                "numa": None,  # filled by the second pass below
            }
        )
    for rec in records:
        rec["numa"] = numa.get(rec["npu"])
    return records


def span(cpus):
    return f"{min(cpus)}..{max(cpus)}" if cpus else "-"


def contiguous(cpus):
    return bool(cpus) and len(set(cpus)) == len(cpus) and max(cpus) - min(cpus) + 1 == len(cpus)


def main(argv):
    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        paths = ["-"]
    records = []
    for path in paths:
        label = "<stdin>" if path == "-" else path
        stream = sys.stdin if path == "-" else open(path, encoding="utf-8", errors="replace")
        with stream:
            records.extend(read(stream, label))

    if not records:
        print(
            "no `NPU<n>: main=[…]  acl=[…]  release=[…]` plan line found.\n"
            "search the log for: '[cpu_bind_mode]', 'The CPU allocation plan is as follows:', "
            "'[migrate] NPU:', 'Skip binding cpu'.\n"
            "A log with no plan line and no mode line means binding never ran (non-ARM host, "
            "`enable_cpu_binding: false`, or `Bind cpus failed ... Skip binding cpu`) — that "
            "is a finding, not an empty result."
        )
        return 2

    print(f"{len(records)} plan(s) parsed\n")
    for rec in records:
        n = len(rec["pool"])
        flags = []
        if not contiguous(rec["pool"]):
            flags.append("pool not contiguous — main/acl/release may come from other lines")
        if len(rec["main"]) < 2:
            flags.append("main<2 cores")
        print(
            f"  {rec['log']}  NPU{rec['npu']}  mode={rec['mode']} rank={rec['rank']}"
            f"  visible={rec['visible']}\n"
            f"      pool {span(rec['pool'])} ({n} cpus)  irq={rec['irq']}"
            f"  main={span(rec['main'])} ({len(rec['main'])})"
            f"  acl={rec['acl']} release={rec['rel']}  numa={rec['numa']}"
            + (f"\n      !! {'; '.join(flags)}" if flags else "")
        )

    print("\ncross-plan comparison:")
    overlaps = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            shared = sorted(set(a["pool"]) & set(b["pool"]))
            irq_shared = sorted(set(a["irq"]) & set(b["irq"]))
            if not shared:
                verdict = "disjoint"
            elif set(a["pool"]) == set(b["pool"]):
                verdict = f"IDENTICAL pool ({len(shared)} cpus), irq shared={irq_shared}"
                overlaps += 1
            else:
                verdict = f"overlap {len(shared)} cpus {span(shared)}, irq shared={irq_shared}"
                overlaps += 1
            print(
                f"  {a['log']}:NPU{a['npu']} vs {b['log']}:NPU{b['npu']}: {verdict}"
                f"   [{span(a['pool'])} vs {span(b['pool'])}]"
            )

    same_visible = {}
    for rec in records:
        if len(rec["visible"]) == 1:
            same_visible.setdefault(rec["visible"][0], set()).add(rec["log"])
    for npu, logs in same_visible.items():
        if len(logs) > 1:
            print(f"  note: {len(logs)} logs each bind NPU{npu} ({sorted(logs)})")

    print(
        f"\n{overlaps} overlapping pair(s)."
        + (
            " Identical pools across separate logs are the multi-instance topo_affinity"
            " overlap: read the band from `npu-smi info -t topo` and pair cards from"
            " different bands, or disable `enable_cpu_binding` and partition by hand."
            if overlaps
            else " No cross-plan overlap in this paste."
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
