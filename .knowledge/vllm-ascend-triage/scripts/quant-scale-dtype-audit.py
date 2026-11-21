#!/usr/bin/env python3
"""Sweep every quant-matmul scale argument in a vllm-ascend tree.

Usage:
  uv run --no-project python scripts/quant-scale-dtype-audit.py <vllm-ascend-root>
  uv run --no-project python scripts/quant-scale-dtype-audit.py /workspace/repos/<issue>/vllm-project/vllm-ascend

Answers the three questions a `EZ1001 Tensor scale not implemented for DT_FLOAT16`
report needs before any answer is written:
  1. which *weight-side* scale arguments reach a quant / grouped matmul, per call
     site (`npu_quant_matmul`, `npu_grouped_matmul`, `npu_grouped_matmul_gmm2`,
     `fused_experts`; per-token/activation scales are a different contract and are
     left out);
  2. which of those parameters have a float32 `_fp32` / `_fp32_list` companion in
     the package, which have none, and which companions are created but read
     nowhere else;
  3. which call sites derive the op's *output* dtype from a scale tensor — the
     second operand of the same dtype gate.
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
SKIP_DIRS = {".git", "build", "docs"}


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

    created, call_sites, dtype_sites, reads = [], [], [], {}
    for path, tree in walk(root):
        rel = path.relative_to(root.parent)
        targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    targets.update(id(n) for n in ast.walk(tgt))
                    text = ast.unparse(tgt)
                    if "_fp32" in text:
                        created.append((rel, node.lineno, text, ast.unparse(node.value)))
                if "_output_dtype" in ast.unparse(node.targets[0]) and "scale" in ast.unparse(node.value):
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
                    reads[name] = reads.get(name, 0) + 1

    def companions(expr):
        notes = set()
        for node in ast.walk(ast.parse(expr, mode="eval")):
            name = leaf(node)
            if not name or "scale" not in name:
                continue
            if "float32" in expr:
                notes.add(f"{name}: cast to float32 at the call site")
            elif name.endswith("_fp32") or name.endswith("_fp32_list"):
                notes.add(f"{name}: is itself a float32 companion")
            elif f"{name}_fp32" in reads or f"{name}_fp32_list" in reads:
                notes.add(f"{name}: fp32 companion present")
            else:
                notes.add(f"{name}: NO fp32 companion anywhere in the package")
        return sorted(notes)

    print("== weight-side scale arguments handed to a quant / grouped matmul")
    for rel, line, fname, arg, expr in call_sites:
        print(f"  {rel}:{line}  {fname}({arg}={expr})")
        for note in companions(expr):
            print(f"      {note}")

    print("\n== float32 companions created")
    for rel, line, target, value in created:
        name = target.split(".")[-1]
        state = "read elsewhere" if reads.get(name, 0) else "CREATED BUT NEVER READ"
        print(f"  {rel}:{line}  {target} = {value}   [{state}]")

    print("\n== output / result dtype taken from a scale tensor")
    seen = set()
    for rel, line, arg, expr in dtype_sites:
        if (rel, line, expr) in seen:
            continue
        seen.add((rel, line, expr))
        print(f"  {rel}:{line}  {arg} = {expr}")

    print("\nReminder: a companion that exists is not a companion that is used —")
    print("read apply() (the launch) for each site above, and both operands of the")
    print("gate (scale dtype AND output dtype), before answering.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
