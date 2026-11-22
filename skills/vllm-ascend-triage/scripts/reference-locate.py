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
   This grade is printed **in full** whenever it is a decision list (<= 25
   members), whatever `--max` says: `[case]` aggregate pages sort first, so the
   topical owner is often its last line, and a truncated list is where that page
   disappears (recorded #6972: 13 members, `startup-failures/graph-mode-not-applied.md`
   fell into `+5 more` of a capped 8);
2b. a **[case-first]** block whenever a `references/evidence-workspace/
   recorded-cases*` entry shares a token: the entry to open, with a ready-to-run
   `sed -n 'A,BP'` range. Printed when no page carries every token and some entry
   shares two of them; its `(n/m)` is not a filter — a recorded entry quotes the
   reporter's own wording, so a real match grades 2-of-4 while the family words sit
   in 100+ pages;
3. the next grades down (`N-1`, `N-2`, …) capped, because an empty top grade is
   not "no page" and a near-miss pair is what a renamed spelling looks like;
4. a **family-word guard**: when no page carries all the tokens and the widest
   one alone hits 40+ pages, the grade cannot name the owner — it says so and
   names the moves that do (the shape's row in `references/dispatch/
   classified-shapes.md`, a `[case]` file opened by the report's own wording, or,
   for a paste that is only a serve command, the shape grade plus the pin's own
   defaults, `references/dispatch/index.md` rule 4);
5. when no page carries two tokens, the two shortest lists, which is what has to
   be read by hand at that point.

A token also matches a page by **filename**, with `-`, `_` and spaces folded onto
`-` on both sides: `one_boundary_per_chunk` finds
`tool-call-streaming/one-boundary-per-chunk.md`, which no page body was needed for.

A **flag is a token as it stands**: `--compilation-config`, `--enforce-eager` and
`--additional-config` go in as they appear in the paste, and an unknown dashed
argument is taken as a token rather than rejected (argparse did reject it, which
cost a call in recorded #6972); a dashed token is also matched without its
leading dashes, because a page may spell the same knob `compilation-config`.

`[case]` marks a `references/evidence-workspace/recorded-cases*` hit, which is
opened first (its entry is the case history). Its line carries the recorded entries
that share most of the query tokens — `→ #12985:488 (3/3)` is the case id, the line
of its record heading and how many tokens its entry contains — so the entry is one
`sed` away without a second grep. When no entry matches a token, the line falls back
to the file's number of `**#<id>**` entries, so a many-case page is greped for the
report's id rather than read whole. `[index]` marks a topic `index.md`, a route, not
an owning page; `[dispatch]` marks a wording table; `[row]` marks a
`first-checks/signature-routes/` page, whose matching **row** is the decision itself — it
is printed right after the `[case]` lines and, when both appear on the all-token grade,
the script says to read the row with the case arrow, because several recorded cases can
share three or four tokens and the arrow then names a neighbour in another lane.
Matching is case-insensitive and fixed-string unless `--regex` / `--case` is
given. Exit 0 always (2 only on a bad root or an unusable rg); a thin or empty
result is a fact about the tokens, not an error.
"""

import argparse
import os
import re
import shutil
import shlex
import subprocess
import sys

MAX_LINE = 200

# Filenames use `-`, a paste or a guessed page name may use `_` or a space: fold
# every separator run onto one `-` so the same words compare equal.
_SEP = re.compile(r"[-_ ]+")


def _sep_norm(text):
    return _SEP.sub("-", text)


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
        # the pages that point at it. Filenames are dash-separated while a paste
        # (and a caller guessing the page) spells the same words with underscores
        # or spaces, so both sides are folded onto `-` before comparing: a run
        # that graded `one_boundary_per_chunk` got 0 files, lost the owner page
        # `one-boundary-per-chunk.md` off the all-token grade, and fell back to
        # the family-word guard (recorded #5596).
        needle = _sep_norm(pattern if case else pattern.lower())
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(".md"):
                    continue
                full = os.path.join(dirpath, name)
                hay = _sep_norm(full if case else full.lower())
                if needle in hay:
                    files.add(full)
    return files


# The two record-heading forms in `references/evidence-workspace/`: the detail
# page opens an entry with `**#12985** vllm-ascend …`, the condensed index with
# `- **#12985** — …`, and a page whose entry is titled by its mechanism carries
# the id in its `#` heading (`# Device fault under MTP + … (#10700)`). A `#NNN`
# quoted mid-entry cites another case, so it never starts an entry.
CASE_HEADING = re.compile(r"^\s*(?:-\s+)?\*\*#(\d{3,6})\*\*|^#{1,4}\s.*#(\d{3,6})\b")


def case_pointers(abs_path, tokens, regex, case, limit=2):
    """`#<id>:<line>` for the recorded entries that share most of the paste.

    A `recorded-cases*` file holds dozens of entries, so the file name alone does
    not locate this report's. Split it at its record headings, score every entry
    by how many of the query tokens its span contains, and print the best two
    with the heading's line number — turn 1 then names the entry to read instead
    of the file to open.
    """
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    pats = []
    for tok in tokens:
        pats.append(re.compile(tok if regex else re.escape(tok),
                               0 if case else re.IGNORECASE))
    spans = []
    for idx, line in enumerate(lines):
        m = CASE_HEADING.match(line)
        if m:
            spans.append((m.group(1) or m.group(2), idx))
    if not spans:
        return []
    spans.append((None, len(lines)))
    scored = []
    for (cid, first), (_nxt, stop) in zip(spans, spans[1:]):
        body = "\n".join(lines[first:stop])
        hits = sum(1 for p in pats if p.search(body))
        if hits:
            scored.append((-hits, first, cid))
    scored.sort()
    total = len(pats)
    out = []
    for neg, first, cid in scored[:limit]:
        mark = "#{}:{}".format(cid, first + 1)
        if total > 1:
            mark += " ({}/{})".format(-neg, total)
        out.append(mark)
    return out


def tag(path):
    if "recorded-cases" in path:
        return " [case]"
    if "/signature-routes/" in path:
        return " [row]"
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


CASE_ENTRY = re.compile(r"^\s*(?:-\s*)?\*\*#\d+")


def case_entries(path):
    """How many `**#<id>**` case entries a recorded-cases page carries."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if CASE_ENTRY.match(line))
    except OSError:
        return 0


PTR = re.compile(r"#(\d+):(\d+)(?: \((\d+)/(\d+)\))?")


def case_first(refs, counts, tokens, regex, case, best_grade, ntok):
    """Print the recorded entry to open, with the sed range that reaches it.

    Printed only when no page carries every token (a thick all-token grade is a
    decision list already) and some entry shares two of them: one shared token is
    noise, and a thick grade's arrows are listed in place.
    """
    if best_grade >= ntok:
        return
    best = []
    for path in counts:
        rel = os.path.relpath(path, refs)
        if "recorded-cases" not in rel:
            continue
        for ptr in case_pointers(path, tokens, regex, case):
            m = PTR.match(ptr)
            if not m:
                continue
            best.append((-int(m.group(3) or 0), rel, m.group(1), int(m.group(2))))
    if not best or -best[0][0] < 2:
        return
    best.sort()
    print("[case-first] a recorded case carries this paste \u2014 open it before any")
    print("  /workspace/repos grep:")
    seen = set()
    for _neg, rel, cid, line in best:
        if (rel, cid) in seen:
            continue
        seen.add((rel, cid))
        print("  sed -n '{},{}p' <skill_root>/references/{}   (#{}, line {})".format(
            max(1, line - 30), line + 40, rel, cid, line))
        if len(seen) >= 2:
            break
    print("  The (n/m) beside an arrow is not a filter: an entry quotes the reporter's own")
    print("  wording, so a real match can grade 2-of-4 while family words fill 100+ pages.")
    print("  Confirm the versions, model and stack before adopting its ladder \u2014 an arrow")
    print("  can name a neighbour in another lane (recorded #2984). Re-deriving a recorded")
    print("  ladder from source spends the budget (#5643: this arrow was printed, its file")
    print("  name greped twice, the entry never opened, `expert_load_balancer.py` re-read")
    print("  for 14 calls).")
    print()


def order(path):
    rank = 0 if "recorded-cases" in path else 1 if "/signature-routes/" in path else 2
    return (rank,
            1 if path.endswith("/index.md") else 0,
            1 if "/dispatch/" in path else 0,
            path)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("tokens", nargs="+")
    ap.add_argument("--root", default=None,
                    help="skill root (default: the parent of this script's directory)")
    ap.add_argument("--max", type=int, default=8,
                    help="lines per near-miss grade (default 8); the all-token grade "
                         "prints up to 25 members")
    ap.add_argument("--min-grade", type=int, default=2,
                    help="lowest grade printed (default 2)")
    ap.add_argument("--regex", action="store_true", help="treat tokens as regexes")
    ap.add_argument("--case", action="store_true", help="match case-sensitively")
    ap.add_argument("--quiet", action="store_true", help="suppress the token summary")
    args, extra = ap.parse_known_args()
    # A flag is a first-class token here: the paste's own `--compilation-config`
    # begins with dashes, so argparse must not eat it as an unknown option.
    if extra:
        args.tokens = list(args.tokens) + list(extra)

    root = args.root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    refs = os.path.realpath(os.path.join(root, "references"))
    if not os.path.isdir(refs):
        sys.stderr.write(f"no references/ under {root!r}\n")
        sys.exit(2)

    sets = {}
    for tok in args.tokens:
        files = run_rg(tok, refs, args.regex, args.case)
        stripped = tok.lstrip("-")
        if stripped and stripped != tok and not args.regex:
            files = files | run_rg(stripped, refs, args.regex, args.case)
        sets[tok] = files

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

    widest_tok, widest_files = max(sets.items(), key=lambda kv: len(kv[1]))
    widest = len(widest_files)
    best = max(counts.values(), default=0)
    if n >= 2 and best < n and widest >= 40:
        print(f"family words, not a case: '{widest_tok}' alone hits {widest} pages and no")
        print(f"page carries all {n}, so this grade cannot name the owner. These moves do:")
        print("  - re-grade the two or three shortened tokens in one more call, and open")
        print("    nothing absent from this printout: a page whose *name* matches the")
        print("    paste but which grades on no list is periphery \u2014 the skill_view")
        print("    `linked_files` inventory is a listing, not a search result;")
        print("  - take the shape's row in references/dispatch/classified-shapes.md "
              "(symptom wording, not title words);")
        print("  - open a [case] file below by the report's own wording \u2014 an entry whose")
        print("    versions, model and stack match is the case history, and its ladder is")
        print("    the answer. A report with no distinctive token at all (a benchmark-table")
        print("    or screenshot comparison) has no grade to run: go straight to these two.")
        print("  - a paste that is only a serve command is that same shape: its flags are")
        print("    the pin's tutorial flags, so grade the shape instead (SoC + phase + the")
        print("    version pair + the reporter's own control, e.g. 'the older -310p tag")
        print("    works'), or skip the grade and rank what the pin's own defaults and")
        print("    status document fix before asking for the log \u2014 references/dispatch/")
        print("    index.md rule 4.")
        print()

    case_first(refs, counts, args.tokens, args.regex, args.case, best, n)

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
        # The all-token grade is the decision list: print every member while it is
        # one, because `[case]` pages sort first and the topical owner is often its
        # last line. Past 25 members the tokens are family words, so cap it and say so.
        cap = len(hits) if grade == n and len(hits) <= 25 else args.max
        for f in hits[:cap]:
            rel = os.path.relpath(f, refs)
            line = "  " + rel + tag(rel)
            if "recorded-cases" in rel:
                ptr = case_pointers(f, args.tokens, args.regex, args.case)
                if ptr:
                    line += "   \u2192 " + ", ".join(ptr)
                else:
                    ncase = case_entries(f)
                    if ncase:
                        line += f" ({ncase} case entries \u2014 grep '#<id>')"
            h = heading(f) if show_head else ""
            if h:
                line += "  \u2014 " + h
            print(line)
        if (grade == n and any("recorded-cases" in h for h in hits)
                and any("/signature-routes/" in h for h in hits)):
            print("  (a [row] member is a signature table: several recorded cases can share")
            print("   three or four tokens, so a [case] arrow may name the neighbour in another")
            print("   lane \u2014 read that row's own line together with the entry. Recorded #2984:")
            print("   the arrow pointed at #3494, a `\u2026-w8a8` export another section owns,")
            print("   while the row held this report's verdict and its pin lines.)")
        if len(hits) > cap:
            print(f"  +{len(hits) - cap} more")
            if grade == n:
                print("  (too wide for a decision list: these tokens are family words — "
                      "shorten them to the flag, code or attribute the paste names)")
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
