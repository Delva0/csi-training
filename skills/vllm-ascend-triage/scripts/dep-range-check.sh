#!/usr/bin/env bash
# dep-range-check.sh -- one reported dependency line from an env dump, against the
# requirement ranges the pinned trees declare. Answers "is this package version
# inside the contract for this pair", which comparison by eye gets wrong once one
# side is floor-only and the other caps a major (vllm-ascend v0.17.0rc1
# `requirements.txt:37` `transformers>=4.57.4` vs vllm v0.17.0
# `requirements/common.txt:10` `transformers >= 4.56.0, < 5`).
#
# Usage:
#   bash dep-range-check.sh <package> <reported-version> <tree-root> [<tree-root> ...]
#   bash dep-range-check.sh transformers 5.5.0.dev0 \
#        /workspace/repos/<issue>/vllm-project/vllm \
#        /workspace/repos/<issue>/vllm-project/vllm-ascend
#
# Reads, per tree root, when present: requirements.txt, requirements-*.txt,
# requirements/*.txt, pyproject.toml, setup.py. Prints every line that constrains
# <package> as path:line, then one verdict per tree. Exact package name only
# (sentence-transformers is not transformers).
# Exit 0 = inside every range found, 1 = outside at least one, 2 = nothing found.
set -u
pkg=${1:?usage: dep-range-check.sh <package> <reported-version> <tree-root>...}
ver=${2:?usage: dep-range-check.sh <package> <reported-version> <tree-root>...}
shift 2
[ $# -ge 1 ] || { echo "need at least one tree root"; exit 2; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
pat="(^|[^A-Za-z0-9_.-])${pkg}[[:space:]]*(==|>=|<=|!=|~=|<|>)"
n=0
for root in "$@"; do
  [ -d "$root" ] || { echo "!! $root not a directory"; continue; }
  n=$((n + 1))
  : > "$tmp/hits.$n"
  for f in "$root"/requirements.txt "$root"/requirements-*.txt "$root"/requirements/*.txt \
           "$root"/pyproject.toml "$root"/setup.py; do
    [ -f "$f" ] || continue
    grep -nE "$pat" "$f" 2>/dev/null | sed "s|^|$f:|" >> "$tmp/hits.$n"
  done
  echo "=== tree $n: $root"
  if [ -s "$tmp/hits.$n" ]; then sed -n '1,12p' "$tmp/hits.$n"
  else echo "  (no ${pkg} constraint found)"; fi
done

uv run --no-project python - "$pkg" "$ver" "$tmp" "$n" <<'PY'
import sys, os, re
pkg, ver, tmp, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def key(v):
    m = re.match(r"^v?(\d+(?:\.\d+)*)(.*)$", v.strip())
    if not m:
        return None, v
    return tuple(int(x) for x in m.group(1).split(".")), m.group(2)

def cmp(a, b):
    ka, sa = key(a); kb, sb = key(b)
    if ka is None or kb is None:
        return None
    w = max(len(ka), len(kb))
    ka += (0,) * (w - len(ka)); kb += (0,) * (w - len(kb))
    if ka != kb:
        return -1 if ka < kb else 1
    # same numeric part: a bare release > its pre-release ("5.5.0.dev0" < "5.5.0")
    if not sa and sb:
        return 1
    if sa and not sb:
        return -1
    return 0

def ok_one(spec):
    m = re.match(r"^(==|!=|>=|<=|~=|>|<)\s*(.+)$", spec.strip())
    if not m:
        return True
    op, want = m.group(1), m.group(2).strip()
    if op == "~=":
        parts = want.split(".")
        op, want = ">=", want
        r = cmp(ver, want)
        if r is None or r < 0:
            return False
        prefix = ".".join(parts[:-1]) if len(parts) > 1 else want
        return ver.strip().startswith(prefix + ".") or ver.strip() == prefix
    if op == "==" and want.endswith(".*"):
        return ver.strip().startswith(want[:-2])
    r = cmp(ver, want)
    if r is None:
        return True
    return {"==": r == 0, "!=": r != 0, ">=": r >= 0, "<=": r <= 0,
            ">": r > 0, "<": r < 0}[op]

spec_re = re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(pkg) + r"\s*((?:==|>=|<=|!=|~=|<|>).*)$")
found = False
outside = []
for i in range(1, n + 1):
    by_file = {}
    for line in open(os.path.join(tmp, f"hits.{i}")):
        parts = line.rstrip("\n").split(":", 2)
        if len(parts) < 3:
            continue
        path, _, text = parts
        for whole in spec_re.findall(text.split("#")[0].rstrip()):
            for spec in whole.split(","):
                spec = " ".join(spec.split()).strip("'\" ;")
                if spec:
                    by_file.setdefault(path, set()).add(spec)
    for path, specs in by_file.items():
        found = True
        bad = [s for s in sorted(specs) if not ok_one(s)]
        print(f"{path}: {ver} {'INSIDE' if not bad else 'OUTSIDE'} {'; '.join(sorted(specs))}")
        if bad:
            outside.append((path, bad))
for path, bad in outside:
    print(f"!! {path} excludes {ver} via {'; '.join(bad)}")
if outside:
    print("before walking the consuming code path, read the tree's own per-model statement"
          " (tests/models/registry.py min_transformers_version, release_notes.md Known Issue):"
          " the requirement may be the family's own, not a defect")
if not found:
    print(f"no {pkg} range found in the given trees")
    sys.exit(2)
sys.exit(1 if outside else 0)
PY
