# Classify environment-variable names from a launch block against a pinned checkout:
# which of them is actually READ by the Python trees, which is only set (docs, tests,
# CI yaml) and consumed elsewhere (CANN/HCCL), which is a module constant that merely
# shares the name, and which exists only inside a tutorial's commented line.
#
# Usage:
#   uv run --no-project python env-knob-liveness.py \
#       --root /workspace/repos/<issue>/vllm-project/vllm-ascend \
#       [--vllm-root /workspace/repos/<issue>/vllm-project/vllm] \
#       USE_MULTI_BLOCK_POOL USE_MULTI_GROUPS_KV_CACHE ACL_OP_INIT_MODE
#   uv run --no-project python env-knob-liveness.py --root <ascend> --from-paste launch.sh
#
# Why one call: an export block pasted by a reporter is usually copied from another
# model's or another SoC's tutorial tab, so "the reporter's env vars" are frequently
# not levers at the pin being triaged (recorded on #12534, v0.23.0rc1 f4a08bbd: two of
# five exports were inert and a third is read for one model only). A name that no code
# reads cannot explain a hang or a slow step; classify it before ranking it.
#
# Output, one line per name, then up to --max-hits evidence lines each:
#   LIVE        read by code at this pin (os.environ.get / getenv / environ[...])
#   SET-ONLY    never read by Python here; set in docs, tests or CI only, i.e. the
#               consumer is CANN/HCCL/another component, or the name is stale
#   CONSTANT    only a Python identifier/module constant shares the name (not an env)
#   COMMENTED   only inside a comment (docs/tutorial/CI yaml) at this pin
#   ABSENT      not found in the scanned trees
# Exit 0 unless arguments are unusable (2). It reports sites; it does not decide cause.
import argparse
import os
import re
import sys

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
CODE_EXT = {".py", ".pyi", ".sh", ".bash", ".yaml", ".yml", ".cmake", ".txt"}
DOC_EXT = {".md", ".rst", ".po"}
MAX_BYTES = 4 * 1024 * 1024


def classify_dirs(root):
    """Return (code_dirs, doc_dirs) relative tags for reporting."""
    return root


def walk(root, exts, skip_tests_docs=False):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = os.path.relpath(dirpath, root)
        if skip_tests_docs and (rel.split(os.sep)[0] in {"tests", "docs"}):
            continue
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts or name.startswith("Dockerfile") or name == "requirements.txt":
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > MAX_BYTES:
                        continue
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        yield path, fh.read().splitlines()
                except OSError:
                    continue


def scan(root, names, max_hits):
    """Return {name: [(kind, path, lineno, text)]}."""
    out = {n: [] for n in names}
    pats = {}
    for n in names:
        q = re.escape(n)
        pats[n] = {
            "read": re.compile(rf"(getenv\(\s*[\"']{q}[\"']|environ\.get\(\s*[\"']{q}[\"']|"
                               rf"environ\[\s*[\"']{q}[\"']\s*\](?!\s*=))"),
            "write": re.compile(rf"environ\[\s*[\"']{q}[\"']\s*\]\s*="),
            "assign": re.compile(rf"^\s*{q}\s*(:[^=]*)?=\s*\S"),
            "any": re.compile(rf"\b{q}\b"),
        }
    for path, lines in walk(root, CODE_EXT | DOC_EXT):
        for i, line in enumerate(lines, 1):
            for n in names:
                if not pats[n]["any"].search(line):
                    continue
                p = pats[n]
                if p["read"].search(line):
                    kind = "READ"
                elif p["write"].search(line):
                    kind = "WRITE"
                elif p["assign"].search(line) and path.endswith((".py", ".pyi")):
                    kind = "ASSIGN"
                elif re.match(r"^\s*#", line):
                    kind = "COMMENTED"
                else:
                    kind = "SET"
                out[n].append((kind, path, i, line.strip()))
                if len(out[n]) >= max_hits * 8:
                    break
    return out


KIND_RANK = {"READ": 0, "WRITE": 1, "ASSIGN": 2, "SET": 3, "COMMENTED": 4}


def sort_key(hit):
    """Show the site that carries the verdict, not the first line in file order."""
    kind, path, _lineno, _text = hit
    return (KIND_RANK.get(kind, 9), path.endswith(".po"), path)


def verdict(hits):
    kinds = {h[0] for h in hits}
    if not kinds:
        return "ABSENT"
    if "READ" in kinds:
        return "LIVE"
    if "WRITE" in kinds:
        return "SET-ONLY"
    if "ASSIGN" in kinds and kinds <= {"ASSIGN", "COMMENTED"}:
        return "CONSTANT"
    if kinds <= {"COMMENTED"}:
        return "COMMENTED"
    return "SET-ONLY"


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("names", nargs="*", help="environment variable names to classify")
    ap.add_argument("--root", default=os.environ.get("VLLM_ASCEND_ROOT", ""),
                    help="vllm-ascend checkout to scan (required unless --from-paste gives names)")
    ap.add_argument("--vllm-root", default=os.environ.get("VLLM_ROOT", ""),
                    help="optional vllm checkout; upstream readers are reported too")
    ap.add_argument("--from-paste", default="", help="file holding a launch block; names are the "
                                                    "exported/assigned identifiers in it")
    ap.add_argument("--max-hits", type=int, default=3, help="evidence lines per name (default 3)")
    args = ap.parse_args()

    names = list(args.names)
    if args.from_paste:
        try:
            with open(args.from_paste, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError as exc:
            print(f"cannot read paste: {exc}", file=sys.stderr)
            return 2
        names += re.findall(r"(?:export\s+|^\s*-\s*)([A-Z][A-Z0-9_]{2,})\s*[:=]", body, re.M)
        names += re.findall(r"[\"']([A-Z][A-Z0-9_]{2,})[\"']\s*:", body)
    names = list(dict.fromkeys(names))
    if not names:
        print("no names given (see --help)", file=sys.stderr)
        return 2
    if not args.root or not os.path.isdir(args.root):
        print("need --root <vllm-ascend checkout>", file=sys.stderr)
        return 2

    down = scan(args.root, names, args.max_hits)
    up = scan(args.vllm_root, names, args.max_hits) if args.vllm_root and os.path.isdir(args.vllm_root) else {}

    print(f"== env-knob liveness: root={args.root}"
          + (f" vllm-root={args.vllm_root}" if up else " (no --vllm-root: upstream unread)"))
    for n in names:
        hits = down[n]
        up_hits = up.get(n, [])
        v = verdict(hits)
        if v == "ABSENT":
            v = "LIVE-UPSTREAM" if "READ" in {h[0] for h in up_hits} else "ABSENT"
        if v == "SET-ONLY" and "READ" in {h[0] for h in up_hits}:
            v = "LIVE-UPSTREAM"
        print(f"\n{n}: {v}")
        if v == "LIVE":
            print("  a code path reads this name at this pin -> it can change behaviour")
        elif v == "LIVE-UPSTREAM":
            print("  read only in the upstream vllm tree -> the downstream patch may bypass it; "
                  "check patch/** before quoting it")
        elif v == "SET-ONLY":
            print("  never read by Python here: the consumer is CANN/HCCL/another component, "
                  "or the spelling is stale -> do not rank it as the cause of a Python-phase stall")
        elif v == "CONSTANT":
            print("  only a Python identifier/module constant shares the name; exporting it "
                  "changes nothing")
        elif v == "COMMENTED":
            print("  only inside comments at this pin -> inert as written")
        else:
            print("  not found in the scanned trees -> inert here (re-grep at your pin)")
        shown = 0
        for kind, path, lineno, text in sorted(hits, key=sort_key):
            if shown >= args.max_hits:
                break
            print(f"  [{kind}] {os.path.relpath(path, args.root)}:{lineno}: {text[:160]}")
            shown += 1
        if shown == 0 and up_hits:
            for kind, path, lineno, text in up_hits[:args.max_hits]:
                print(f"  [upstream {kind}] {os.path.relpath(path, args.vllm_root)}:{lineno}: {text[:160]}")
    print("\nRule: a name that no code reads cannot explain a hang, an OOM or a slow step at "
          "this pin. Quote the verdict and the site, then move to the phase that stalled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
