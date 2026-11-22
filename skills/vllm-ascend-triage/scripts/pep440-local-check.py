#!/usr/bin/env python3
"""Which side of a pip conflict is unsatisfiable: PEP 440 local-label matching.

Built for the "Cannot install <a> and <b> because these package versions have
conflicting dependencies" paste (e.g. `The user requested torch==2.9.0` /
`torch-npu 2.9.0 depends on torch==2.9.0+cpu`, #7089, #4621). pip's own advice
("loosen the range of package versions") assumes the compatibility row; the
local-label rules decide whether the two specs are compatible at all.

    uv run --no-project python <skill_root>/scripts/pep440-local-check.py '==2.9.0' '==2.9.0+cpu'
    uv run --no-project python <skill_root>/scripts/pep440-local-check.py \
        '==2.9.0' '==2.9.0+cpu' --candidates 2.9.0,2.9.0+cpu,2.10.0

Rules printed and exercised:
  * a specifier without a local label ignores the candidate's local label, so
    ==X accepts X+cpu;
  * a specifier with a local label requires it, so ==X+cpu rejects plain X.

The prerelease case is the other verdict a *build floor* turns on: `torch_npu >= 2.5.1rc1`
does **not** admit the `2.5.1.dev20250320` wheel the same release documents (a dev build
sorts below rc1), which pushes a build env onto the released `2.5.1rc1` and its `+cpu`
torch pin (#446):

    uv run --no-project python <skill_root>/scripts/pep440-local-check.py \
        '>=2.5.1rc1' '>=2.5.1' --candidates 2.5.1.dev20250320,2.5.1rc1,2.5.1,2.5.1+cpu

Prereleases are judged by packaging's own policy unless forced, because a specifier that
names a prerelease admits prereleases — a bare `contains(v, False)` would report
`2.5.1rc1` as failing `>=2.5.1rc1`, which pip does not.

`packaging` is usually absent from the no-project environment and installing it
needs network, so fall back to pip's / setuptools' vendored copy (verified
present: `python -c "import pip._vendor.packaging"` under uv 26.2 pip).
"""

import argparse
import re
import sys


def _api():
    for mod in ("packaging", "pip._vendor.packaging", "setuptools._vendor.packaging"):
        try:
            __import__(mod)
            spec_mod = __import__(mod + ".specifiers", fromlist=["Specifier"])
            ver_mod = __import__(mod + ".version", fromlist=["Version"])
            return spec_mod.Specifier, ver_mod.Version, mod
        except ImportError:
            continue
    return None, None, None


def public_parts(specs):
    """Candidate versions implied by the specs: strip the operator, keep the rest."""
    out = []
    for s in specs:
        body = re.sub(r"^(===|==|~=|>=|<=|!=|>|<)\s*", "", s.strip())
        body = body.split(".*")[0].split(",")[0]
        if body and body not in out:
            out.append(body)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="PEP 440 local-label matching probe")
    ap.add_argument("specifiers", nargs="+", help="version specifiers from the paste, e.g. '==2.9.0'")
    ap.add_argument("--candidates", default=None,
                    help="comma list of concrete versions to test (default: the versions the specs name)")
    ap.add_argument("--prereleases", action="store_true", default=None,
                    help="force prerelease candidates on")
    ap.add_argument("--no-prereleases", dest="prereleases", action="store_false",
                    default=argparse.SUPPRESS,
                    help="force them off (what a spec naming no prerelease does)")
    a = ap.parse_args(argv)

    Specifier, Version, source = _api()
    if Specifier is None:
        print("no packaging implementation importable (packaging / pip._vendor / setuptools._vendor);"
              " do not install one over the network, reason from PEP 440 instead", file=sys.stderr)
        return 2

    raw = [c.strip() for c in (a.candidates or "").split(",") if c.strip()]
    cands = raw or public_parts(a.specifiers)
    policy = ("forced on" if a.prereleases
              else "forced off" if a.prereleases is False
              else "packaging default: a specifier that itself names a prerelease admits prereleases")
    print(f"# packaging implementation: {source}")
    print(f"# prereleases: {policy}")
    print("rules: ==X accepts X+<local>;  ==X+<local> rejects plain X\n")

    header = "candidate".ljust(18) + "".join(s.ljust(16) for s in a.specifiers)
    print(header)
    verdicts = {}
    for c in cands:
        try:
            v = Version(c)
        except Exception as e:  # noqa: BLE001 - report, do not trace
            print(f"{c.ljust(18)} not a PEP 440 version: {e}")
            continue
        row = c.ljust(18)
        for s in a.specifiers:
            try:
                ok = Specifier(s).contains(v, prereleases=a.prereleases)
            except Exception as e:  # noqa: BLE001
                row += f"ERR {e}".ljust(16)[:15] + " "
                continue
            verdicts.setdefault(c, []).append(ok)
            row += ("MATCH" if ok else "no").ljust(16)
        print(row)

    print()
    uniq = {tuple(v) for v in verdicts.values()}
    if len(verdicts) == 1 or not uniq:
        print("VERDICT: one candidate under test - add '--candidates' with both labelled and plain versions.")
    elif any(all(v) for v in verdicts.values()):
        print("VERDICT: at least one candidate satisfies EVERY specifier -> the pin set is satisfiable;"
              " a pip failure is the visibility row (index/metadata), not a version range.")
    else:
        print("VERDICT: no candidate satisfies every specifier -> genuine compatibility conflict;"
              " name the two torch lines and change one of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
