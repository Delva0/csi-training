#!/usr/bin/env bash
# The SOURCES.json version contract (one line per repo) plus the `unavailable`
# repos, without dumping 20-54 KB of JSON into the transcript.
#
#   scripts/sources-summary.sh [SOURCES.json ...]
#
# With no arguments it discovers every SOURCES.json under the case workspace.
set -u

uv run --no-project python - "$@" <<'PY'
import glob, json, sys

paths = sys.argv[1:]
if not paths:
    paths = (glob.glob('/workspace/repos/SOURCES.json')
             + glob.glob('/workspace/repos/*/SOURCES.json')
             + glob.glob('/workspace/SOURCES.json'))
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)
    paths = uniq
if not paths:
    print('no SOURCES.json found; ls /workspace and adapt')
    raise SystemExit(0)

for path in paths:
    try:
        doc = json.load(open(path))
    except Exception as exc:                      # noqa: BLE001 - report, do not fail
        print(f'{path}: unreadable ({exc})')
        continue
    print(f'--- {path}  opened_at={doc.get("opened_at")}')
    for r in doc.get('repositories', []):
        files, regular = r.get('files'), r.get('regular_files')
        complete = r.get('complete_regular_files')
        coverage = 'n/a' if regular is None else f'{files}/{regular}'
        if complete is False:
            coverage += ' TRUNCATED'
        elif complete is True:
            coverage += ' complete'
        print('  {repo:<52} {ref:<24} {commit} {mode:<15} files={coverage} runtime_verified={rv}'.format(
            repo=str(r.get('repo'))[:52], ref=str(r.get('ref'))[:24],
            commit=str(r.get('commit'))[:12], mode=str(r.get('mode')),
            coverage=coverage, rv=r.get('runtime_commit_verified')))
    missing = doc.get('unavailable') or []
    if missing:
        print('  unavailable (no verified public source in this snapshot):')
        for entry in missing:
            if isinstance(entry, dict):
                print('    {repo:<50} {reason}'.format(
                    repo=str(entry.get('repo'))[:50], reason=entry.get('reason')))
            else:
                print(f'    {entry}')
PY
