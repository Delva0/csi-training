#!/usr/bin/env bash
# Downstream forks of ONE upstream base class, compared on a constructor keyword
# argument: which subclasses declare it, which forward **kwargs, which still carry
# the base's older signature (the bug set). Subclasses of a matched fork are
# included too (an `Async…` twin inherits the same drift).
#
#   scripts/fork-ctor-sweep.sh <repo-root> --base '<regex on the base class name>'
#                               [--kwarg NAME] [--glob '*.py']
#
# Example (vllm-ascend v0.23.0, issue #18688):
#   scripts/fork-ctor-sweep.sh /workspace/repos/18688/vllm-project/vllm-ascend \
#       --base '^Scheduler$' --kwarg hash_block_size
#
# Status per subclass:
#   OK       the ctor declares <kwarg> (its line number is printed: copy that line
#            as the in-tree fix pattern)
#   IMMUNE   the ctor takes `**kwargs`, so a new base kwarg binds by itself
#   MISSING  neither the kwarg nor `**kwargs`: this one raises
#            `TypeError: <Class>.__init__() got an unexpected keyword argument`
#            as soon as the upstream caller passes it
#   NO-CTOR  no own `__init__`: inherits the base signature, so it cannot drift
# Matching uses the tail name of each base expression, so
# `(AsyncScheduler, RecomputeScheduler)` counts for both. Run it in the
# reporter's vllm-ascend tree; the same call on the vllm tree with --base on the
# upstream class shows the caller side. Exit 0 always; 2 on bad usage.
set -u

if [ "$#" -lt 1 ]; then
  sed -n '2,20p' "$0"; exit 2
fi

uv run --no-project python - "$@" <<'PY'
import ast, os, re, sys, warnings

warnings.simplefilter("ignore", SyntaxWarning)

argv = sys.argv[1:]
root = argv[0]
base_re = kwarg = None
suffix = ".py"
i = 1
while i < len(argv):
    a = argv[i]
    if a == "--base" and i + 1 < len(argv):
        base_re = argv[i + 1]; i += 2
    elif a == "--kwarg" and i + 1 < len(argv):
        kwarg = argv[i + 1]; i += 2
    elif a == "--glob" and i + 1 < len(argv):
        suffix = argv[i + 1].replace("*", "") or ".py"; i += 2
    else:
        print(f"unknown argument: {a}"); raise SystemExit(2)

if base_re is None:
    print("missing --base '<regex>'; see the header for the invocation"); raise SystemExit(2)
if not os.path.isdir(root):
    print(f"not a directory: {root}"); raise SystemExit(2)
bre = re.compile(base_re)

skip_dirs = {".git", "node_modules", "__pycache__", "build", "dist", ".venv"}
classes = []            # (rel, lineno, name, bases, initsig-or-None, kwline)
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    for fn in sorted(filenames):
        if not fn.endswith(suffix):
            continue
        path = os.path.join(dirpath, fn)
        try:
            src = open(path, encoding="utf-8", errors="replace").read()
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        rel = os.path.relpath(path, root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = []
            for b in node.bases:
                tail = b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", None)
                if tail:
                    bases.append(tail)
            init = next((n for n in node.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name == "__init__"), None)
            if init is None:
                classes.append((rel, node.lineno, node.name, bases, None, None))
                continue
            a = init.args
            names = [x.arg for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
                     if x.arg not in ("self", "cls")]
            sig = ", ".join(names)
            if a.vararg:
                sig += (", " if sig else "") + "*" + a.vararg.arg
            if a.kwarg:
                sig += (", " if sig else "") + "**" + a.kwarg.arg
            kwline = None
            if kwarg and kwarg in names:
                lines = src.splitlines()
                for off, line in enumerate(lines[init.lineno - 1:init.end_lineno or init.lineno]):
                    if re.search(rf"\b{re.escape(kwarg)}\b", line):
                        kwline = init.lineno + off
                        break
            flags = (a.kwarg is not None, a.vararg is not None)
            classes.append((rel, node.lineno, node.name, bases, (sig, names, flags), kwline))

selected = [c for c in classes if c[3] and any(bre.search(b) for b in c[3])]
names_seen = {c[2] for c in selected}
for _ in range(6):                      # transitive: forks of a fork
        grown = [c for c in classes
                 if c not in selected and c[3] and any(b in names_seen for b in c[3])]
        if not grown:
            break
        selected += grown
        names_seen |= {c[2] for c in grown}

selected.sort(key=lambda c: (c[0], c[1]))
if not selected:
    print(f"no subclass of /{bre.pattern}/ found under {root}")
    raise SystemExit(0)

print(f"base /{bre.pattern}/  kwarg={kwarg or '-'}  root={root}")
rows = []
for rel, ln, name, bases, initinfo, kwline in selected:
    if initinfo is None:
        rows.append(("NO-CTOR", rel, ln, name, bases, "", None))
        continue
    sig, names, (has_kwargs, _has_varargs) = initinfo
    status = "OK" if (kwarg and kwarg in names) else ("IMMUNE" if has_kwargs else ("MISSING" if kwarg else "REF"))
    rows.append((status, rel, ln, name, bases, sig, kwline))
    print(f"{status:<8} {rel}:{ln} {name}({', '.join(bases)})")
    print(f"{'':9}    __init__({sig})" + (f"   kwarg at :{kwline}" if kwline else ""))

if kwarg:
    missing = [r for r in rows if r[0] == "MISSING"]
    print(f"\nsummary: {len(rows)} fork(s); {len(missing)} MISSING {kwarg} -> "
          + (", ".join(f"{r[3]} ({r[1]}:{r[2]})" for r in missing) or "none"))
    ok = [r for r in rows if r[0] == "OK"]
    if ok:
        print("fix pattern already in this tree: " + ", ".join(f"{r[3]}:{r[6]}" for r in ok))
PY
