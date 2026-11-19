#!/usr/bin/env python3
"""Version archaeology for a bug report against a released tag.

Answers, in one call: which refs correspond to the reported version, how far main is ahead,
and where the report's own tokens live at that version versus in later history.

Usage:
    uv run --no-project python version_archaeology.py \
        --repo /workspace/repos/vllm-project/vllm-ascend \
        --version 0.26.rc1 \
        --keyword enable_sparse_li_c8 --keyword dspark

Every git call is read-only (tag/branch/grep/log/show). Nothing is checked out.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

GIT_TIMEOUT = 120


def git(repo: str, *args: str, check: bool = False) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ""
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


def norm_version(text: str) -> str:
    """Normalise a version-ish string to comparable components.

    '0.26.rc1', 'v0.26.0rc1', 'release-0.26.0-rc1' -> '0.26.0rc1'
    """
    s = text.strip().lower().lstrip("v")
    s = re.sub(r"[^0-9a-z.]", ".", s)
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.?(rc|post)(\d+))?", s)
    if not m:
        return s
    major, minor, patch, kind, kind_n = m.groups()
    out = f"{major}.{minor}.{patch or '0'}"
    if kind:
        out += f"{kind}{kind_n}"
    return out


def collect_refs(repo: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for tag in git(repo, "tag", "--list").splitlines():
        refs.append(("tag", tag.strip()))
    for line in git(repo, "branch", "-r", "--format=%(refname:short)").splitlines():
        name = line.strip()
        if name and "HEAD" not in name:
            refs.append(("branch", name))
    return refs


def short(repo: str, ref: str) -> str:
    return git(repo, "log", "-1", "--format=%h %ci %s", ref, "--")


def pick_version(repo: str, version: str) -> tuple[str, list[str]]:
    want = norm_version(version)
    family = ".".join(want.split(".")[:2])
    exact: list[str] = []
    same_family: list[str] = []
    for kind, ref in collect_refs(repo):
        n = norm_version(ref.rsplit("/", 1)[-1])
        if n == want:
            exact.append(ref)
        elif n.startswith(family + "."):
            same_family.append(ref)
    return want, exact or same_family


def keyword_at_ref(repo: str, ref: str, keyword: str, pathspec: str) -> str:
    args = ["grep", "-n", "-I", "-E", keyword, ref]
    if pathspec:
        args += ["--", pathspec]
    out = git(repo, *args)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return "    (no hits)"
    head = "\n".join("    " + ln for ln in lines[:args_limit])
    extra = "" if len(lines) <= args_limit else f"\n    ... {len(lines) - args_limit} more"
    return head + extra


args_limit = 6


def keyword_history(repo: str, keyword: str, depth: int = 8) -> str:
    blocks = []
    bodies = git(repo, "log", "--all", "--oneline", f"--grep={keyword}", f"-{depth}")
    if bodies:
        blocks.append("  message matches (--grep):\n" + indent(bodies.splitlines()))
    pickaxe = git(
        repo,
        "log",
        "--all",
        "--oneline",
        f"-S{keyword}",
        f"-{depth}",
        "--",
        ".",
        ":(exclude)**/locale/**",
    )
    if pickaxe:
        blocks.append("  content added/removed (-S):\n" + indent(pickaxe.splitlines()))
    return "\n".join(blocks) if blocks else "  (no commit in any ref)"


def indent(lines: list[str]) -> str:
    return "\n".join("    " + ln for ln in lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help="local clone path, or a clone name such as vllm-ascend "
        "(resolved by scripts/repo_root.sh; exit 3 if it names no clone)",
    )
    parser.add_argument("--version", required=True, help="reported version string")
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="config key / op name / error string from the report (repeatable)",
    )
    parser.add_argument(
        "--pathspec",
        default="*.py",
        help="restrict at-ref grep; use '' to grep all tracked files (default: *.py)",
    )
    parser.add_argument(
        "--main",
        default="origin/main",
        help="ref that is 'later than the report' (default: origin/main)",
    )
    parser.add_argument("--docs", action="store_true", help="also grep docs/ for each keyword")
    ns = parser.parse_args()

    # A bare clone name is as common as a path; treating it as a path makes every git call fail and
    # then reads as "no such tag" — resolve it through repo_root.sh first, and keep exit 3 distinct.
    repo_root = pathlib.Path(__file__).resolve().parent / "repo_root.sh"
    try:
        resolved = subprocess.run(
            ["bash", str(repo_root), ns.repo], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - defensive
        print(f"repo_root.sh failed: {exc}", file=sys.stderr)
        return 3
    if resolved.returncode != 0:
        print(resolved.stderr.strip() or f"not a git clone: {ns.repo}", file=sys.stderr)
        return 3
    repo = resolved.stdout.strip().splitlines()[-1]
    if git(repo, "rev-parse", "--git-dir") == "":
        print(f"not a git clone: {repo}", file=sys.stderr)
        return 2

    want, candidates = pick_version(repo, ns.version)
    print(f"repo: {repo}")
    print(f"reported version -> {want}")
    if not candidates:
        print("  no tag or branch matches this version; list refs manually")
    for ref in candidates[:8]:
        marker = "  <- use this" if ref == candidates[0] else ""
        print(f"  {ref}: {short(repo, ref)}{marker}")

    if not candidates:
        return 0
    tag = candidates[0]
    print(f"\nreported tree ref: {tag}")
    print(f"HEAD: {short(repo, 'HEAD')}")

    ahead = git(repo, "rev-list", "--count", f"{tag}..{ns.main}")
    if ahead:
        print(f"commits in {ns.main} but not in {tag}: {ahead}")

    if ns.keyword:
        print("\nfix candidates after the reported version (subject filter):")
        pattern = "|".join(re.escape(k) for k in ns.keyword)
        log = git(repo, "log", "--oneline", f"{tag}..{ns.main}", f"--grep={pattern}", "-i", "-30")
        print(indent(log.splitlines()) if log else "  (none)")

    for kw in ns.keyword:
        print(f"\n=== keyword: {kw} ===")
        print(f"  at {tag}:")
        print(keyword_at_ref(repo, tag, kw, ns.pathspec))
        if ns.docs:
            print(f"  docs at {tag}:")
            print(keyword_at_ref(repo, tag, kw, "docs"))
        print("  history:")
        print(keyword_history(repo, kw))

    print(
        "\nnext: confirm a fix with `git -C <repo> merge-base --is-ancestor <sha> <tag>` "
        "(exit 0 = shipped in the reported version), then read the fix body with "
        "`git log -1 --format=%B <sha>`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
