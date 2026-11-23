#!/usr/bin/env python3
"""Sweep every quant-matmul scale argument in a vllm-ascend tree.

Usage:
  uv run --no-project python scripts/quant-scale-dtype-audit.py <vllm-ascend-root>
  uv run --no-project python scripts/quant-scale-dtype-audit.py /workspace/repos/<issue>/vllm-project/vllm-ascend

Answers the three questions a `EZ1001 Tensor scale not implemented for DT_FLOAT16`
report needs before an answer is written:
  1. which *weight-side* scale arguments reach a quant / grouped matmul, per call
     site (`npu_quant_matmul`, `npu_grouped_matmul`, `npu_grouped_matmul_gmm2`,
     `fused_experts`; per-token/activation scales are a different contract and are
     left out), together with the local alias the argument is bound to — that alias
     is where the MoE twin hides (`fused_experts(w2_scale=w2_scale)` at `:251`,
     `w2_scale = [layer.w2_weight_scale]` at `:249`);
  2. for each such parameter, its float32 state *in the file that owns the scheme*:
     companion read in the same file / companion **created but never read** (a
     repair nobody wired up) / no companion in that file / cast at the call site;
  3. which call sites take the op's *output* dtype from a scale tensor — the second
     operand of the same dtype gate.

The report is ordered candidate-first: `START HERE` names the quant-method sites
whose scale parameter is in a broken float32 state, then the full inventory of
every site, then every `_fp32` binding. Read it whole (it is ~100 lines at
v0.17.0rc1): the inventory and companion list are the sweep, and a `| head -60`
cuts them.

It reproduces the sweep, it does not decide the cause: quote its output as the
check and keep the claim-strength label (SKILL.md rule 6). Every site is
file:line at the tree you point it at, so re-run it per release.
"""
import ast
import sys
from pathlib import Path

QUANT_CALLS = ("npu_quant_matmul", "npu_grouped_matmul", "npu_grouped_matmul_gmm2", "fused_experts")
DTYPE_KW = ("output_dtype", "fallback_output_dtype")
NOT_WEIGHT_SCALE = ("per_token", "pertoken", "activation", "swiglu", "token", "type", "bias", "x_scale")
SKIP_LEAF = ("flag", "is_", "has_", "scale_type", "scale_dtype")
SKIP_DIRS = {".git", "build", "docs"}
FP32_SUFFIXES = ("_fp32_list", "_fp32")


def leaf(node):
    """Last attribute/name component of an expression such as layer.w2_weight_scale."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def walk(root):
    for path in sorted(root.rglob("*.py")):
        if SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        try:
            yield path, ast.parse(path.read_text(errors="replace"))
        except SyntaxError as exc:
            print(f"  ! unparsed {path}:{exc.lineno} (skipped)")


def main(argv):
    if len(argv) != 2:
        print("usage: uv run --no-project python scripts/quant-scale-dtype-audit.py <vllm-ascend-root>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not (root / "vllm_ascend").is_dir():
        print(f"not a vllm-ascend root (no vllm_ascend/): {root}", file=sys.stderr)
        return 2
    root = root / "vllm_ascend"  # the package only; sibling dirs are not shipped code

    created, call_sites, dtype_sites = [], [], []
    reads, aliases = {}, {}  # both keyed by rel path: a companion is a same-file fact
    for path, tree in walk(root):
        rel = str(path.relative_to(root.parent))
        targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    targets.update(id(n) for n in ast.walk(tgt))
                    text = ast.unparse(tgt)
                    if any(s in text for s in FP32_SUFFIXES):
                        created.append((rel, node.lineno, text.split(".")[-1], text, ast.unparse(node.value)))
                tgt0 = node.targets[0]
                if len(node.targets) == 1 and isinstance(tgt0, ast.Name):
                    aliases.setdefault(rel, []).append((node.lineno, tgt0.id, ast.unparse(node.value)))
                if "_output_dtype" in ast.unparse(tgt0) and "scale" in ast.unparse(node.value):
                    dtype_sites.append((rel, node.lineno, "_output_dtype", ast.unparse(node.value)))
            if isinstance(node, ast.Call):
                fname = leaf(node.func) or ""
                if fname in QUANT_CALLS:
                    for kw in node.keywords:
                        arg, expr = kw.arg or "", ast.unparse(kw.value)
                        if "scale" in arg and not any(t in arg for t in NOT_WEIGHT_SCALE):
                            call_sites.append((rel, node.lineno, fname, arg, expr))
                        if arg in DTYPE_KW and "scale" in expr:
                            dtype_sites.append((rel, node.lineno, arg, expr))
                    # npu_quant_matmul takes the weight scale as its 3rd positional arg
                    if fname == "npu_quant_matmul" and len(node.args) >= 3:
                        pos = ast.unparse(node.args[2])
                        if "scale" in pos:
                            call_sites.append((rel, node.lineno, fname, "arg[2]", pos))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)) and id(node) not in targets:
                name = leaf(node)
                if name:
                    reads.setdefault(rel, {})[name] = reads.setdefault(rel, {}).get(name, 0) + 1

    companions = {}  # rel -> {name: (line, state, target_text)}
    for rel, line, name, target, value in created:
        state = "read elsewhere in this file" if reads.get(rel, {}).get(name, 0) else "CREATED BUT NEVER READ"
        companions.setdefault(rel, {})[name] = (line, state, target)

    def scale_leaf(name):
        return bool(name) and "scale" in name and not any(s in name for s in SKIP_LEAF)

    def resolve(rel, line, expr):
        """One hop of local aliasing for a bare argument name."""
        if not expr.isidentifier():
            return None
        prior = [a for a in aliases.get(rel, []) if a[0] < line and a[1] == expr]
        return max(prior) if prior else None

    def notes(rel, line, expr, seen=None):
        """float32 state of every scale-bearing leaf in expr, one alias hop deep."""
        seen = set() if seen is None else seen
        out = []
        alias = resolve(rel, line, expr)
        body = alias[2] if alias else expr
        for node in ast.walk(ast.parse(body, mode="eval")):
            name = leaf(node)
            if not scale_leaf(name) or name in seen:
                continue
            seen.add(name)
            if "float32" in body:
                out.append((name, "cast to float32 at the call site", "ok"))
                continue
            if any(name.endswith(s) for s in FP32_SUFFIXES):
                out.append((name, "is itself a float32 companion", "ok"))
                continue
            known = companions.get(rel, {}).get(name + "_fp32") or companions.get(rel, {}).get(name + "_fp32_list")
            if known and known[1] != "CREATED BUT NEVER READ":
                out.append((name, f"`{name}_fp32` at {rel}:{known[0]} read elsewhere in this file", "ok"))
            elif known:
                out.append((name, f"`{name}_fp32` at {rel}:{known[0]} {known[1]} — pass that tensor", "never-read"))
            else:
                out.append((name, "no `_fp32` companion in this file (read the parameter's "
                                   "allocation before calling it a defect)", "missing"))
        return out

    rows = []
    for rel, line, fname, arg, expr in call_sites:
        alias = resolve(rel, line, expr)
        rows.append((rel, line, fname, arg, expr, alias, notes(rel, line, expr)))

    def fmt(row, pad=""):
        rel, line, fname, arg, expr, alias, ns = row
        head = f"{pad}{rel}:{line}  {fname}({arg}={expr})"
        if alias:
            head += f"   [{expr} ← :{alias[0]} {alias[2]}]"
        return [head] + [f"{pad}    {n[0]}: {n[1]}" for n in ns]

    print("== START HERE — quant-method sites whose scale is not a float32 tensor")
    never = [(rel, c) for rel, comps in companions.items() for c in comps.values() if c[1] == "CREATED BUT NEVER READ"]
    if never:
        print("  [a] companion created and never read (the repair exists; the launch ignores it)")
        for rel, (line, state, target) in sorted(never):
            name = target.split(".")[-1]
            twin = next((name[: -len(s)] for s in FP32_SUFFIXES if name.endswith(s)), name)
            print(f"      {rel}:{line}  {target}   [NEVER READ]")
            users = [r for r in rows if r[0] == rel and any(n[0] == twin for n in r[6])]
            for r in users:
                print(f"        its non-fp32 twin is still passed at {r[0]}:{r[1]}  {r[2]}({r[3]}={r[4]})")
            if not users:
                print("        no launch in this file — the scheme delegates, so follow the delegate")
    broken = [r for r in rows if "quantization/" in r[0] and any(n[2] in ("missing", "never-read") for n in r[6])]
    if broken:
        print("  [b] quant-method launch sites whose scale is not float32 (read apply() for each)")
        for r in broken:
            for ln in fmt(r, pad="    "):
                print(ln)
    if dtype_sites:
        print("  [c] output / result dtype taken from a scale tensor (the gate's second operand)")
        seen = set()
        for rel, line, arg, expr in dtype_sites:
            if (rel, line, expr) in seen:
                continue
            seen.add((rel, line, expr))
            print(f"      {rel}:{line}  {arg} = {expr}")
    if not (never or broken or dtype_sites):
        print("      none — every weight-side scale argument is a float32 companion or a call-site cast")

    print(f"\n== full inventory ({len(call_sites)} weight-side scale arguments)")
    for row in rows:
        for ln in fmt(row, pad="  "):
            print(ln)

    print(f"\n== float32 companions created ({len(created)})")
    for rel, line, name, target, value in created:
        state = companions[rel][name][1]
        print(f"  {rel}:{line}  {target} = {value}   [{state}]")

    print("\nReminder: a companion that exists is not a companion that is used —")
    print("read apply() (the launch) for each site above, and both operands of the")
    print("gate (scale dtype AND output dtype), before answering.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
