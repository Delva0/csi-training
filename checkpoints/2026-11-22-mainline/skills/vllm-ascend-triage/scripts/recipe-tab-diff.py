#!/usr/bin/env python3
"""recipe-tab-diff.py — which hardware tab does this serve command actually belong to?

Tutorial model pages duplicate one deployment section per SoC
(`::::{tab-item} A3 series` / `A2 series` / `Ascend 950DT`, paired by `:sync:`), and
the tabs are NOT interchangeable: different `--data-parallel-size` /
`--tensor-parallel-size`, different `--max-model-len`, sometimes a flag that exists
in only one tab (`enable_dsa_cp`) and a different image tag (`-a3` suffix vs the
unsuffixed A2 tag). A reporter who runs one tab's block on the other generation's
hardware is a recurring startup-failure cause (#11300: "已解决，A3机器跑了A2的命令"),
and the tree itself cannot tell you which tab they used — the pasted command can.

This script reads the pinned page and prints, per tab: the `vllm serve` line, the
flags that differ from the other tabs, and the env-var / IMAGE lines. With
`--command FILE` (the reporter's script, or the command block they pasted) it also
diffs their launch against each tab and names the closest one.

Usage:
  uv run --no-project python scripts/recipe-tab-diff.py \
      --page <vllm-ascend-root>/docs/source/tutorials/models/DeepSeek-V4-Flash.md
  uv run --no-project python scripts/recipe-tab-diff.py --page <page> \
      --command reporter_start.sh [--section 5.1] [--tab A3]

Output is one line per (tab, fact); read the `DIFF` block last. Exit code 0 on a
successful read even when the command matches no tab (that is a finding, not an
error). Options --section takes the page's own heading number ("5.1") so a page
with several tab-sets is not merged into one.
"""
from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

TAB_OPEN = re.compile(r"^(?P<fence>:+)\{tab-item\}\s*(?P<name>.+?)\s*$")
TAB_CLOSE = re.compile(r"^(?P<fence>:+)\}\s*$")
HEADING = re.compile(r"^#{2,4}\s+(?P<num>\d+(?:\.\d+)*)\.?\s*(?P<title>.*)$")
SYNC = re.compile(r"^:sync:\s*(?P<sync>.+?)\s*$")
EXPORT = re.compile(r"^\s*export\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>.*)$")
IMAGE = re.compile(r"^\s*(?:export\s+)?IMAGE=(?P<val>.+?)\s*$")
SERVE = re.compile(r"^\s*vllm\s+serve\b")
FLAG = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")


def join_continuations(lines: list[str], start: int) -> tuple[str, int]:
    """Join a backslash-continued block beginning at lines[start]; return (text, end)."""
    parts: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i].rstrip("\n")
        cont = line.rstrip().endswith("\\")
        parts.append(line.rstrip().rstrip("\\").rstrip())
        i += 1
        if not cont:
            break
    return " ".join(p for p in parts if p), i


def split_flags(command: str) -> dict[str, str]:
    """Return {flag: value} for a serve command; bare flags get ''."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    out: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if FLAG.fullmatch(tok):
            val = ""
            if "=" in tok and not tok.startswith("--="):
                name, _, val = tok.partition("=")
                out[name] = val
            elif i + 1 < len(tokens) and not FLAG.fullmatch(tokens[i + 1]):
                out[tok] = tokens[i + 1]
                i += 1
            else:
                out[tok] = ""
        i += 1
    return out


def parse_page(path: Path, section: str | None) -> list[dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tabs: list[dict] = []
    heading = ("", "")
    current: dict | None = None
    in_scope = section is None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEADING.match(line)
        if m:
            heading = (m.group("num"), m.group("title"))
            in_scope = section is None or heading[0].startswith(section)
            if not in_scope:
                # leaving the requested section ends the block we were collecting:
                # the next tab-set belongs to another deployment mode.
                current = None
        open_m = TAB_OPEN.match(line)
        if open_m and in_scope:
            current = {
                "name": open_m.group("name"),
                "sync": "",
                "section": heading[0],
                "line": i + 1,
                "serve_line": None,
                "serve": "",
                "flags": {},
                "exports": {},
                "image": "",
            }
            tabs.append(current)
            i += 1
            continue
        if current is not None:
            # a tab closes on `::::` (colons only) or on `::::}`; both are fences,
            # not content, and the next tab-set may open right after with the same
            # env lines for a different deployment mode.
            stripped = line.strip()
            if (TAB_CLOSE.match(stripped) or (stripped and set(stripped) == {":"})) and len(stripped) >= 3:
                current = None
                i += 1
                continue
            sync_m = SYNC.match(line.strip())
            if sync_m:
                current["sync"] = sync_m.group("sync")
            exp_m = EXPORT.match(line)
            if exp_m and exp_m.group("var") != "IMAGE":
                current["exports"][exp_m.group("var")] = exp_m.group("val")
            img_m = IMAGE.match(line)
            if img_m:
                current["image"] = img_m.group("val")
            if SERVE.match(line) and not current["serve"]:
                text, nxt = join_continuations(lines, i)
                current["serve_line"] = i + 1
                current["serve"] = text
                current["flags"] = split_flags(text)
                i = nxt
                continue
        i += 1
    return tabs


def key_flags(tab: dict) -> list[tuple[str, str]]:
    return sorted(tab["flags"].items())


def print_tab(tab: dict, flag_sets: dict[str, set[str]]) -> None:
    label = f"[{tab['section']}] {tab['name']}" + (f" (:sync: {tab['sync']})" if tab["sync"] else "")
    where = f"  — serve block at :{tab['serve_line']}" if tab["serve_line"] else ""
    print(f"\n== {label}{where}")
    if tab["image"]:
        print(f"   IMAGE={tab['image']}")
    if tab["exports"]:
        print("   env: " + " ".join(f"{k}={v}" for k, v in sorted(tab["exports"].items())))
    if not tab["serve"]:
        print("   (no `vllm serve` block in this tab)")
        return
    print(f"   serve: {tab['serve'][:300]}{'…' if len(tab['serve']) > 300 else ''}")
    for flag, val in key_flags(tab):
        others = sorted(flag_sets.get(flag, set()) - {val})
        if others:
            print(f"   * {flag} = {val!r}      (other tab(s): {others})")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--section", default=None, help='page heading number, e.g. "5.1"')
    ap.add_argument("--command", default=None, help="file holding the reporter's serve command")
    ap.add_argument("--tab", default=None, help="force the comparison tab (substring match)")
    args = ap.parse_args()

    page = Path(args.page)
    if not page.is_file():
        print(f"recipe-tab-diff.py: no such page: {page}", file=sys.stderr)
        return 2
    all_tabs = parse_page(page, None)
    tabs = parse_page(page, args.section) if args.section else all_tabs
    if not tabs:
        sections = sorted({t["section"] for t in all_tabs})
        if not all_tabs:
            print(
                "recipe-tab-diff.py: this page has no `{tab-item}` block at all — one "
                "block serves every SoC, so the tab question does not apply here",
            )
        else:
            where = ", ".join(sections) or "(no heading above the tab-set)"
            print(
                f"recipe-tab-diff.py: no `{{tab-item}}` under section {args.section} — the "
                f"tab-sets on this page sit under: {where}. Re-run with one of those, or "
                "without --section to see all of them",
            )
        return 0

    flag_sets: dict[str, set[str]] = {}
    for tab in tabs:
        for flag, val in tab["flags"].items():
            flag_sets.setdefault(flag, set()).add(val)

    print(f"page: {page}   tabs: {len(tabs)}" + (f"   section: {args.section}" if args.section else ""))
    for tab in tabs:
        print_tab(tab, flag_sets)

    if len(tabs) > 1:
        print("\n== flags present in only one tab")
        for tab in tabs:
            others = set().union(*(set(t["flags"]) for t in tabs if t is not tab))
            only = sorted(set(tab["flags"]) - others)
            if only:
                print(f"   {tab['name']}: {' '.join(only)}")

    if args.command:
        cmd_path = Path(args.command)
        if not cmd_path.is_file():
            print(f"recipe-tab-diff.py: no such command file: {cmd_path}", file=sys.stderr)
            return 2
        text = cmd_path.read_text(encoding="utf-8", errors="replace")
        serve_line = ""
        for idx, line in enumerate(text.splitlines()):
            if SERVE.match(line):
                serve_line, _ = join_continuations(text.splitlines(), idx)
                break
        if not serve_line:
            print(f"\n== reporter command: no `vllm serve` line found in {cmd_path}")
            return 0
        rep_flags = split_flags(serve_line)
        rep_exports = {
            m.group("var"): m.group("val") for m in (EXPORT.match(l) for l in text.splitlines()) if m
        }
        print(f"\n== reporter command ({cmd_path})")
        print(f"   dp={rep_flags.get('--data-parallel-size', '(unset)')} "
              f"tp={rep_flags.get('--tensor-parallel-size', '(unset)')} "
              f"max-model-len={rep_flags.get('--max-model-len', '(unset)')} "
              f"api-server-count={rep_flags.get('--api-server-count', '(unset)')}")
        scored: list[tuple[int, dict]] = []
        for tab in tabs:
            tab_flags = tab["flags"]
            common = set(tab_flags) & set(rep_flags)
            same = sum(1 for f in common if tab_flags[f] == rep_flags[f])
            missing = sorted(set(tab_flags) - set(rep_flags))
            extra = sorted(set(rep_flags) - set(tab_flags))
            differ = sorted(f for f in common if tab_flags[f] != rep_flags[f])
            score = len(differ) + len(extra) + len(missing)
            scored.append((score, tab))
            print(f"\n   vs tab {tab['name']}  (distance {score})")
            if differ:
                print("     differing values:")
                for f in differ:
                    print(f"       {f}: reporter {rep_flags[f]!r} vs tab {tab_flags[f]!r}")
            if extra:
                print("     only in reporter's command: " + " ".join(extra))
            if missing:
                print("     only in the tab: " + " ".join(missing))
            if not (differ or extra or missing):
                print("     identical flag set")
            if not rep_exports:
                print("     env: the reporter's file carries no `export` lines — compare the")
                print("          pasted env block by hand, or ask for the launcher script")
                rep_exports = dict(tab["exports"])
            env_missing = sorted(set(tab["exports"]) - set(rep_exports))
            env_diff = sorted(v for v in set(tab["exports"]) & set(rep_exports)
                              if tab["exports"][v] != rep_exports[v])
            if env_missing or env_diff:
                print("     env: " + ("unset " + " ".join(env_missing) if env_missing else "")
                      + (" " if env_missing and env_diff else "")
                      + ("different " + " ".join(env_diff) if env_diff else ""))
        scored.sort(key=lambda s: s[0])
        best_score, best = scored[0]
        label = "closest tab"
        if args.tab:
            forced = next(((sc, t) for sc, t in scored if args.tab.lower() in t["name"].lower()), None)
            if forced is None:
                print(f"\n== --tab {args.tab!r} matches no tab on this page; showing the closest one")
            else:
                best_score, best = forced
                label = f"tab you forced ({args.tab})"
        print(f"\n== {label}: {best['name']} (distance {best_score})")
        print("   If the reporter's cards are NOT that tab's generation, that mismatch is the")
        print("   first thing to fix before reading any native/version lane — and if the same")
        print("   two lines come back on the matching tab, the tab is closed as a discriminator.")
        rep_dp = rep_flags.get("--data-parallel-size", "1")
        print(f"   Cross-check with the log: `EngineCore_DP<N>` process names exist only when")
        print(f"   data_parallel_size > 1, so a paste with EngineCore_DP2 next to"
              f" --data-parallel-size {rep_dp} is two different launches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
