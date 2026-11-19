# Version pairing: vllm-ascend ↔ vLLM ↔ runtime pins

vllm-ascend is a plugin for vLLM and tracks one vLLM version per branch; a mismatch changes which
patches and model code apply. Establish both halves before reading behaviour.

## Paired vLLM tag

From the v0.21 era onward each commit records its vLLM release in `.github/vllm-release-tag.commit`
(plain text, one tag):

| vllm-ascend ref | paired vLLM |
| --- | --- |
| `v0.21.0rc1` | `v0.21.0` |
| `v0.23.0`, `v0.23.0rc1` | `v0.23.0` |
| `v0.26.0rc1` | `v0.26.0` |
| `v0.30.0` | `v0.30.0` |
| main (observed HEAD `1bd23a0a9`) | `v0.32.0` |

**Read that file at the ref, never with a bare `cat`.** It is a one-line file, so `cat`/`grep` looks
harmless and answers for `main`: at HEAD it says `v0.32.0` while `v0.23.0`'s own copy says `v0.23.0`. A
recorded run `cat`-ed it for the `v0.23.0` report #16365, concluded that "vllm-ascend v0.23.0 pairs with
vllm v0.32.0", re-issued its `vllm`-clone greps at `v0.32.0` and ended `output_truncated` with no reply —
after `ref_snapshot.sh` had already printed the correct `v0.23.0` in batch 1. The wrong tag was not even
obviously wrong: the API it was looking for (`max_admission_blocks_per_request`) exists at `v0.23.0` too,
so the false pairing read as a finding. Overriding a script's ref-pinned fact with a working-tree read is
the most expensive form of this mistake: it corrupts the ref every later `vllm/` claim depends on. The
one-line form is `git -C .../vllm-ascend show <tag>:.github/vllm-release-tag.commit`.
`scripts/open_answer.sh` and `scripts/ref_snapshot.sh` already print it ref-pinned
(`== paired vLLM tag ==`); use theirs instead of a second read.

The file is absent before `v0.21`, so for older releases read the **Release Compatibility Matrix** in
`docs/source/community/versioning_policy.md` at that tag — it is authoritative and it rarely follows
the same-rc rule. Verified rows: `v0.11.0rc0` → vLLM `v0.11.0rc3`, torch/torch_npu
`2.7.1 / 2.7.1.dev20250724`, **stable CANN `8.2.RC1`**; `v0.10.2rc1` → `v0.10.2`; `v0.9.2rc1` →
`v0.9.2`. The FAQ ("How vllm-ascend work with vllm" in `docs/source/faqs.md`) states the policy, and
the issue's `collect_env` block is the authority for the machine.


```bash
git -C /workspace/repos/vllm-project/vllm-ascend show v0.26.0rc1:.github/vllm-release-tag.commit
git -C /workspace/repos/vllm-project/vllm-ascend show v0.26.0rc1:docs/source/user_guide/release_notes.md \
  | grep -n -A8 '### Dependencies'
# v0.26.0rc1 -> Upstream vLLM: v0.26.0 ; v0.20.2rc1 -> vLLM 0.18.0
```

The upstream clone exists too, with its own tags (`git -C .../vllm tag | sort -V`), so upstream
behaviour is readable at the paired tag: `git -C .../vllm show v0.7.3:vllm/config.py`.

## Runtime pins per vllm-ascend tag (`requirements.txt`)

| ref | torch | torch-npu | triton-ascend |
| --- | --- | --- | --- |
| `v0.7.3.post1` | torch>=2.5.1 | 2.5.1 | — |
| `v0.18.0rc1` | 2.9.0 | 2.9.0 | 3.2.0 |
| `v0.21.0rc1` | 2.10.0 | 2.10.0 | 3.2.1 |
| `v0.23.0rc1` | 2.10.0 | 2.10.0.post2 | 3.2.1 |
| `v0.26.0rc1` | 2.10.0 | 2.10.0.post4 | 3.2.2 |

`requirements.txt` is authoritative for the tag; `collect_env.py` output in the issue is
authoritative for the machine (e.g. an issue on `v0.18.0rc1` reported
`torch_npu==2.9.0.post1+gitee7ba04`, a build the tag does not pin). Operator availability follows
the installed torch-npu/CANN, so a gate that tests `hasattr(torch_npu, ...)` can differ between two
users of the same vllm-ascend release — state the runtime assumption.

**The image can disagree with the tag's own pin, and the Dockerfile wins.** `requirements.txt` and
`pyproject.toml` carry the same pin at a tag (both `2.8.0` at `v0.13.0rc2`, both `2.8.0.post2` at
`v0.13.0rc3`), so a disagreement between *them* is not a lead; the one place a tag can contradict itself
is a hard-coded `pip install … 'torch-npu==X.postN'` in `Dockerfile*`, which replaces the wheel *after*
the editable install resolved the pin. At `v0.13.0rc2` — `pyproject.toml:21`/`requirements.txt:31`
`torch-npu==2.8.0`, `Dockerfile:64` editable install, `Dockerfile:76` (and `Dockerfile.310p:61`,
`.310p.openEuler:56`) then pip-installing `2.8.0.post1` — every rc2 image ships a `torch_npu` that was
never the build target. #6287 ("[CI] Update pta to 2.8.0.post2", merged) deletes that line from all five
variants, and `v0.13.0rc3`/`v0.13.0` move the pin to `2.8.0.post2` with no Dockerfile install line: the
rc2 → rc3 **tree** diff is the release-inclusion proof, no ancestry query needed.

Symptom shape (worked report #6250, `v0.13.0rc2`, Qwen3-Next-80B-A3B-Instruct on 800I A2, TP4): weight
load succeeds, then the engine dies during graph warm-up — `capture_model` → `_dummy_run` →
`_generate_dummy_run_hidden_states` → the compiled block (`acl_graph.py:113 __call__`) — with no frame
that names the wheel. The thread's lever is a maintainer sentence in the *linked* issue: torch-npu
`2.8.0.post1` in the image, vllm-ascend compiled against `2.8.0`. Levers, in order:
`pip install torch-npu==2.8.0` inside the container (matches the build pin, loses the `post1` fixes),
`pip install -v -e .` after pinning `pyproject.toml` to `2.8.0.post1` (a full local build), or an image
that already carries `post2` (`releases-v0.13.0`/`-a3`, `v0.13.0rc3`+). The ABI/op-signature mechanism is
`hypothesis`: falsify it with the wall's own versions (`python -c "import torch_npu; print(torch_npu.__version__)"`
against the two pins) before repeating it as fact, and never cite the `pip install` line's *comment* as
the pin. Full detail: `dockerfile-env.md` §D.

`scripts/ref_snapshot.sh <ref> [repo-path-or-name]` prints all of this for one ref in one call: paired vLLM tag,
runtime pins, package layout, path existence and commits since.

## Per-image CANN base

A report that quotes an *image* rather than a wheel is quoting the CANN build baked into it, and that
base is pinned in the repo: `Dockerfile`, `Dockerfile.310p`, `Dockerfile.a3`, `Dockerfile.a5` (+
`.openEuler` variants) each start from a `quay.io/ascend/cann:<version>-<arch>` image. At
`v0.23.0rc1`, `Dockerfile.310p:18` is `FROM quay.io/ascend/cann:9.1.0-beta.1-310p-ubuntu22.04-py3.12`,
which is the CANN the 310P report named. Use it both ways:

```bash
git -C /workspace/repos/vllm-project/vllm-ascend show v0.23.0rc1:Dockerfile.310p | grep -n '^FROM'
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:requirements.txt | grep -E 'torch|triton'
```

So "which CANN ships with tag X's image" is answerable from the tag; "which CANN your operator-level fix
needs" is a CANN release question, not a vllm-ascend tag question (`../symptoms/device-kernel-faults.md`).

**Read the tag's own supported-version table before calling a reported CANN version a skew.** The same ref
publishes both halves: `v0.22.1rc1:Dockerfile:18` is `FROM quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12`
and `v0.22.1rc1:docs/source/installation.md:14-18` lists `CANN == 9.0.0` (HDK and NNAL the same) beside
`torch-npu == 2.10.0`; at `v0.23.0` the table says `CANN == 9.1.0`. So a report titled
"v0.22.1rc1 配套 CANN 9.0.0 …" describes the **documented pairing**, not a mismatch: a CANN
downgrade/upgrade is not a workaround there, because the pinned `torch_npu` is built for that CANN. Check
`Dockerfile*` (image), `docs/source/installation.md` (wheel/docs) and `requirements.txt` together, and only
claim a skew when the reporter's `collect_env` disagrees with all three. The docs path moves on `main`
(`docs/source/getting_started/installation/base_environment.inc.md`), so read it at the ref with `git show`.

```bash
git -C /workspace/repos/vllm-project/vllm-ascend show v0.22.1rc1:Dockerfile | grep -n '^FROM'
git -C /workspace/repos/vllm-project/vllm-ascend show v0.22.1rc1:docs/source/installation.md | sed -n '10,20p'
```

## Version mismatch symptoms

- `InvalidVersion` at import: an editable/dev vLLM reports a version the plugin cannot parse;
  `VLLM_VERSION=X.Y.Z` overrides it (`docs/source/community/versioning_policy.md`,
  `docs/source/faqs.md` item 14, the version check in `vllm_ascend/utils.py`).
- **`VLLM_VERSION` is a switch, not a label.** `vllm_version_is()` reads it *first* and only falls back to
  `vllm.__version__` (`vllm_ascend/utils.py:609-618` at `v0.23.0rc1`), and the fork's compatibility layer
  is `if vllm_version_is("0.23.0")` — `patch/platform/__init__.py:34`/`:53`,
  `ops/fused_moe/fused_moe.py:55`/`:193`/`:290`, `models/deepseek_v4.py:89`, `_310p/*`. Exporting an older
  value on a newer image silently takes every "old vLLM" branch, which is a candidate cause for a startup
  death no traceback explains. The fork's own docs do it: the GLM-5.2 tutorial's A2 two-node block ships
  `export VLLM_VERSION=0.21.0` (`docs/source/tutorials/models/GLM5.2.md:321`, `:378` at `v0.23.0rc1`; 13
  occurrences at `v0.24.0rc1`, gone by `v0.26.0rc1`) although that tag pairs with vLLM `v0.23.0`. Check the
  report's env dump for the line before reading model code; falsifier for "the doc line is stale" — a
  named `glm5.2` image that really ships vLLM `0.21.0`. A reporter who *commented the line out* runs the
  default install, so it is not their fault.
- **A fork override does not inherit the upstream method's fixes.** `vllm_ascend/worker/model_runner_v1.py`
  (and several `patch/` files) replace upstream methods wholesale, so a fix landed in the *paired* vLLM tag
  can still be absent from the same-named method in the fork until it is cherry-picked into the override.
  Before blaming dependency pins for a race or a stale CPU mirror, run
  `git grep -n "def <method>" <ascend-ref> -- vllm_ascend` and, when it is found, compare the override's
  body against `git show <paired-vllm-ref>:<upstream path>`. Worked case: `_build_attention_metadata` had
  no `num_accepted_tokens_event.synchronize()` at `v0.18.0rc1` although vLLM `v0.18.0rc0` already carried
  #36013; the fork got it only in `v0.18.0` (#8764, `9a0b786f2`; `../symptoms/spec-decode-and-parallelism.md`
  §1b).
- A feature documented in the tutorial for release N may not exist in release N-2: check
  `git ls-tree` at the reporter's tag before assuming the flag was honoured.
- The CANN row of that matrix matters for behaviour, not only for install: `v0.11.0rc0` pins
  CANN `8.2.RC1`, and code in the same tag tests `torch.version.cann.startswith("8.3")` — so the
  shipped default takes the *other* branch from a user who upgraded CANN. Pair the tag with the
  reporter's `collect_env`, then read the gate (`../symptoms/attention-mask/index.md`).
- The environment dump is evidence about what ran; the tag is evidence about what shipped. When
  they disagree (a torch-npu build the tag never pinned), keep both and say so instead of silently
  picking one.
