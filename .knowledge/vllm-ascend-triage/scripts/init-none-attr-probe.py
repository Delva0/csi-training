#!/usr/bin/env python3
"""List `self.<x> = None` attributes that are never assigned again in the same file.

Usage: uv run --no-project python <skill_root>/scripts/init-none-attr-probe.py FILE [FILE ...]

Why: a fork/copied impl that declares `self.key_cache = None` in __init__ and then
passes `self.key_cache` into a `torch_npu`/ATB op sends an uninitialised tensor to the
op, which reports it as `dimNum 0` / `CheckTensorShape failed` — the crash names the op,
not the missing binding. Run this on the failing impl and on its sibling impl (the eager
one, or the upstream copy) and compare the two outputs: the sibling's extra assignment
line is the fix. Read-only; exits 1 if any unbound attribute is found, else 0.
"""
import ast
import sys

NONE = ast.Constant


def _self_name(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == "self":
        return node.attr
    return None


def probe(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, path)
    declared = {}      # attr -> lineno, `self.x = None` inside __init__
    assigned = set()   # attrs assigned to something other than None
    reads = {}         # attr -> [lineno] of `self.x` reads outside __init__

    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        init = next((f for f in cls.body
                     if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
        if init is None:
            continue
        for node in ast.walk(init):
            if isinstance(node, ast.Assign) and isinstance(node.value, NONE) \
                    and node.value.value is None:
                for tgt in node.targets:
                    name = _self_name(tgt)
                    if name:
                        declared[name] = node.lineno
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign) and not (
                    isinstance(node.value, NONE) and node.value.value is None):
                for tgt in node.targets:
                    name = _self_name(tgt)
                    if name:
                        assigned.add(name)
                    if isinstance(tgt, ast.Tuple):
                        for elt in tgt.elts:
                            if _self_name(elt):
                                assigned.add(_self_name(elt))
            if isinstance(node, ast.Attribute) and _self_name(node) \
                    and isinstance(node.ctx, ast.Load):
                reads.setdefault(node.attr, []).append(node.lineno)

    bad = {a: ln for a, ln in declared.items() if a not in assigned}
    for attr, line in sorted(bad.items(), key=lambda kv: kv[1]):
        where = ",".join(str(n) for n in reads.get(attr, [])) or "-"
        print(f"{path}:{line}: self.{attr} = None in __init__, never reassigned; "
              f"read at line(s) {where}")
    return len(bad)


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[1], file=sys.stderr)
        return 2
    total = 0
    for path in argv[1:]:
        try:
            total += probe(path)
        except (OSError, SyntaxError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
