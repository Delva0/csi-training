# Case layout: concrete commands

Orientation (cheap, one call):

```bash
ls /workspace
uv run --no-project python - <<'PY'
import json,glob
p=glob.glob('/workspace/repos*/SOURCES.json')+glob.glob('/workspace/repos/*/SOURCES.json')
for f in p:
    d=json.load(open(f))
    print(f, d.get('opened_at'))
    for r in d['repositories']:
        print(' ', r['repo'], r.get('ref'), r['commit'][:12], r['mode'],
              r.get('basis'), 'files=%s/%s'%(r.get('files'), r.get('regular_files')),
              'runtime_verified=%s'%r.get('runtime_commit_verified'))
PY
```

Source reading at the pinned commit (when the repo is a one-commit git):

```bash
git -C /workspace/repos/<...>/vllm-ascend log --oneline -3     # confirm you are at ref
rg -n "aclnnAddRmsNormBias" /workspace/repos/<...>/vllm-ascend -g '!*.po' | head
sed -n '240,300p' path/to/file.cpp
```

`git log --oneline -3` on these exports usually shows a single commit (the release); there
is no divergence history, so "when was this introduced" cannot be answered from the export.
Say that instead of guessing.

History precedent:

```bash
rg -l "aclnnAddRmsNormBias|DDR address of the MTE" /workspace/history | head
rg -l "evalscope|mmlu|思考|精度" /workspace/history | head
sed -n '1,60p' /workspace/history/<issue>.md
```

Quote precedent as `#<id>: <one-line fact>` (e.g. "upstream parser fix tracked in vllm PR
34779, referenced by #7061/#7094"). Distinguish "the same symptom was reported" from "the
same cause was confirmed".

Checklist for the final answer: repo+ref+commit cited once; every code claim has
`path:line`; precedent ids quoted; missing artefacts listed; OCR/snapshot limits stated.
