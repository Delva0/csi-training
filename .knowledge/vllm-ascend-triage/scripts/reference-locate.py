#!/usr/bin/env python3
"""Grade every reference page against the paste's distinctive strings.

    uv run --no-project python scripts/reference-locate.py TOKEN [TOKEN ...] \
        [--root DIR] [--max N] [--min-grade N] [--regex] [--case] [--quiet]

One token per argument: the paste's own error sentence, attribute / function /
op / kernel name, numeric code, flag, or symptom phrase. The script greps
`references/` once per token and prints, in this order:

1. one line per token with its file count (so the rarest token is visible);
2. the **all-token** grade — pages that contain every token, i.e. the owning
   page (membership in one list is a shared word, membership in all is the case);
3. the next grades down (`N-1`, `N-2`, …) capped, because an empty top grade is
   not "no page" and a near-miss pair is what a renamed spelling looks like;
4. when no page carries two tokens, the two shortest lists, which is what has to
   be read by hand at that point.

`[case]` marks a `references/evidence-workspace/recorded-cases*` hit, which is
opened first (its entry is the case history); `[index]` marks a topic
`index.md`, a route, not an owning page; `[dispatch]` marks a wording table.
Matching is case-insensitive and fixed-string unless `--regex` / `--case` is
given. Exit 0 always (2 only on a bad root or an unusable rg); a thin or empty
result is a fact about the tokens, not an error.
"""

import argparse
import os
import shutil
import shlex
import subprocess
import sys

MAX_LINE = 200


def run_rg(pattern, root, regex, case):
    rg = shutil.which("rg")
    if rg:
        args = [rg, "-l", "--no-messages", "--hidden", "-g", "*.md"]
        args += ["-e", pattern] if regex else ["-F", "-e", pattern]
        if not case:
            args.append("-i")
    else:  # no ripgrep on this host: plain grep takes the same flags
        args = ["grep", "-rl", "--include=*.md"]
        if not case:
            args.append("-i")
        args += ["-e", pattern] if regex else ["-F", "-e", pattern]
    args.append(root)
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr[-500:])
        sys.exit(2)
    files = {ln.strip() for ln in proc.stdout.splitlines() if ln.strip()}
    if not regex:
        # A page is keyed on the reporter's wording by its *filename* too: the
        # owning page rarely repeats its own name in its body, but every sibling
        # links to it as a path, so a content-only match ranks the owner below
        # the pages that point at it.
        needle = pattern if case else pattern.lower()
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(".md"):
                    continue
                full = os.path.join(dirpath, name)
                hay = full if case else full.lower()
                if needle in hay:
                    files.add(full)
    return files


def tag(path):
    if "recorded-cases" in path:
        return " [case]"
    if path.endswith("/index.md"):
        return " [index]"
    if "/dispatch/" in path:
        return " [dispatch]"
    return ""


def heading(abs_path, width=88):
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#"):
                    text = line.lstrip("#").strip()
                    return text if len(text) <= width else text[: width - 1] + "\u2026"
                if line and not line.startswith(("---", "name:", "description:")):
                    return ""
    except OSError:
        return ""
    return ""


def order(path):
    return (0 if "recorded-cases" in path else 1,
            1 if path.endswith("/index.md") else 0,
            1 if "/dispatch/" in path else 0,
            path)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("tokens", nargs="+")
    ap.add_argument("--root", default=None,
                    help="skill root (default: the parent of this script's directory)")
    ap.add_argument("--max", type=int, default=8, help="lines per grade (default 8)")
    ap.add_argument("--min-grade", type=int, default=2,
                    help="lowest grade printed (default 2)")
    ap.add_argument("--regex", action="store_true", help="treat tokens as regexes")
    ap.add_argument("--case", action="store_true", help="match case-sensitively")
    ap.add_argument("--quiet", action="store_true", help="suppress the token summary")
    args = ap.parse_args()

    root = args.root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    refs = os.path.realpath(os.path.join(root, "references"))
    if not os.path.isdir(refs):
        sys.stderr.write(f"no references/ under {root!r}\n")
        sys.exit(2)

    sets = {}
    for tok in args.tokens:
        sets[tok] = run_rg(tok, refs, args.regex, args.case)

    if not args.quiet:
        print(f"skill references: {refs}")
        for tok, files in sorted(sets.items(), key=lambda kv: len(kv[1])):
            print(f"  {len(files):3d}  {tok}")
        print()

    n = len(sets)
    counts = {}
    for files in sets.values():
        for f in files:
            counts[f] = counts.get(f, 0) + 1

    lo = min(args.min_grade, n)
    for grade in range(n, lo - 1, -1):
        hits = sorted((f for f, c in counts.items() if c == grade), key=order)
        if not hits:
            if grade > lo:
                print(f"{grade}-of-{n}: none")
            continue
        head = "ALL" if grade == n else f"{grade}-of-{n}"
        print(f"{head} ({len(hits)} file{'s' if len(hits) != 1 else ''})")
        show_head = grade >= n - 1
        for f in hits[: args.max]:
            rel = os.path.relpath(f, refs)
            line = "  " + rel + tag(rel)
            h = heading(f) if show_head else ""
            if h:
                line += "  \u2014 " + h
            print(line)
        if len(hits) > args.max:
            print(f"  +{len(hits) - args.max} more")
        print()

    if n >= 2 and not any(c >= 2 for c in counts.values()):
        thin = sorted(sets.items(), key=lambda kv: len(kv[1]))[:2]
        print("no page carries two tokens — read the shortest list(s) by hand:")
        for tok, files in thin:
            print(f"  {tok} ({len(files)})")
            for f in sorted(files, key=order)[: args.max]:
                print("    " + os.path.relpath(f, refs) + tag(os.path.relpath(f, refs)))
            if len(files) > args.max:
                print(f"    +{len(files) - args.max} more")
        print("A token spelled differently in the library (op-name suffix, Chinese")
        print("phrasing) is the usual reason for a thin grade: drop or shorten it.")


if __name__ == "__main__":
    main()
