#!/usr/bin/env python3
"""Print a vllm pin's `tl` placeholder attributes beside the module-level reads that need them.

Usage: uv run --no-project python <skill_root>/scripts/triton-placeholder-probe.py VLLM_ROOT [PATH ...]

VLLM_ROOT is the *pinned vllm tree* (`…/vllm-project/vllm` at the reported ref); PATHs
default to `vllm/model_executor/layers/fla/ops`, so one call covers the FLA/GDN import
chain that Qwen3-Next and other hybrid models pull in at module scope. Read-only.

Why: with `HAS_TRITON` false — no triton in the image (the Ascend `Dockerfile` uninstalls
it) or an installed one that reports no active driver (`importing.py:42-49`) —
`vllm/triton_utils/__init__.py` binds `tl`/`tldevice` to `TritonLanguagePlaceholder`, a
`types.ModuleType` with a **fixed, version-dependent** attribute list and no
`__getattr__`. A module-level `tl.<x>` it does not define raises
`AttributeError: module 'triton.language' has no attribute '<x>'` at import, i.e. the same
text a *mispaired* triton raises. That list is not stable: at vllm v0.11.0 it has no `exp`,
so `fla/ops/op.py:26 exp = tl.exp` kills `import vllm_ascend.models.qwen3_next`
(recorded #3325/#4206 inspection crash); at v0.16.0 it has `exp/log/log2` and op.py guards
`gather` with `hasattr`, so the same import survives. Run this before repeating either
answer from memory.

Each reported read is tagged [present] or [MISSING] against the pin's own placeholder and
prints its enclosing top-level statement, because a read inside an `if not
is_gather_supported:`-style branch is not reached at runtime — read the branch, this
script does not evaluate it. Exits 1 when at least one read is MISSING, else 0.
"""
import ast
import sys
import types
from pathlib import Path

DEFAULT_PATHS = ["vllm/model_executor/layers/fla/ops"]
LANGUAGE_NAMES = {"tl", "tldevice"}
MODULE_NAMES = {"triton"}
DUNDER = ("__", "_")


def load_placeholders(vllm_root):
    path = vllm_root / "vllm" / "triton_utils" / "importing.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in ("TritonPlaceholder", "TritonLanguagePlaceholder")]
    ns = {"types": types}
    for cls in classes:                                   # file order: the module class
        body = ast.get_source_segment(src, cls) or ""      # names its language sibling
        exec(compile(ast.parse(body), f"{path}:{cls.lineno}", "exec"), ns)  # noqa: S102
    found = {}
    for cls in classes:
        obj = ns[cls.name]()
        found[cls.name] = (
            cls.lineno,
            sorted(k for k in vars(obj) if not k.startswith(DUNDER)),
            any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in cls.body),
        )
    return path, found


def aliases_of(tree):
    lang, mod = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "triton_utils" in node.module:
            for a in node.names:
                name = a.asname or a.name
                if a.name in LANGUAGE_NAMES:
                    lang.add(name)
                elif a.name in MODULE_NAMES:
                    mod.add(name)
    return lang, mod


def reads_of(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, str(path))
    lang, mod = aliases_of(tree)
    hits = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.If):
            test = " ".join((ast.get_source_segment(src, stmt.test) or "").split())[:76]
            branches = [(f"then of `if {test}`", stmt.body),
                        (f"else of `if {test}`", stmt.orelse)]
        else:
            head = (ast.get_source_segment(src, stmt) or "").strip().splitlines()
            branches = [((head[0][:96] if head else ""), [stmt])]
        for label, nodes in branches:
            for sub in nodes:
                for node in ast.walk(sub):
                    if not isinstance(node, ast.Attribute) \
                            or not isinstance(node.value, ast.Name):
                        continue
                    base = node.value.id
                    if base in lang:
                        hits.append((node.lineno, f"{base}.{node.attr}", "language", label))
                    elif base in mod:
                        hits.append((node.lineno, f"{base}.{node.attr}", "module", label))
    hits.sort()
    return hits


def main(argv):
    if not argv:
        print(__doc__.strip())
        return 2
    vllm_root = Path(argv[0]).resolve()
    if not (vllm_root / "vllm").is_dir():
        print(f"not a vllm tree: {vllm_root}", file=sys.stderr)
        return 2
    importing, found = load_placeholders(vllm_root)
    print(f"placeholder source: {importing}")
    for name, (lineno, attrs, has_getattr) in found.items():
        print(f"  {name} (:{lineno}): {', '.join(attrs) or '<none>'}"
              f"{'  [has __getattr__]' if has_getattr else '  [no __getattr__ → undefined attr raises]'}")
    lang_attrs = set(found.get("TritonLanguagePlaceholder", (0, [], False))[1])
    mod_attrs = set(found.get("TritonPlaceholder", (0, [], False))[1])

    targets = [Path(p) for p in argv[1:]] or [vllm_root / p for p in DEFAULT_PATHS]
    files = []
    for t in targets:
        files.extend(sorted(t.glob("*.py")) if t.is_dir() else [t])
    if not files:
        print("no python files under the given paths", file=sys.stderr)
        return 2

    missing = 0
    for f in files:
        hits = reads_of(f)
        if not hits:
            continue
        print(f"\n{f}")
        for lineno, expr, kind, head in hits:
            provided = (lang_attrs if kind == "language" else mod_attrs)
            ok = expr.split(".", 1)[1] in provided
            if not ok:
                missing += 1
            print(f"  :{lineno} {expr:<24} {'present' if ok else 'MISSING':<8} {head}")
    print("\nMISSING reads raise AttributeError at import only if that line is evaluated;"
          "\nread the branch shown beside it (e.g. a `hasattr`/`is_gather_supported` guard).")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
