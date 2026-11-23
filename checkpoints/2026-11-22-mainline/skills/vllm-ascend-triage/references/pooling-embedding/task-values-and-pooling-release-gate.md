# `--task rerank` is not a task value, and pooling has a release gate

Verified at vllm-ascend **v0.9.1** (`0740d10`) + vllm **v0.9.1** (`b6553be`);
recorded for #4796 (`bge-reranker-v2-m3`, `vllm serve … --task rerank`, A2/A3,
image built from the `v0.9.1-dev` branch). This is the *launch-line* class of the
pooling topic — nothing was built, no worker existed, and the whole paste is
argparse plus an environment dump.

## The signature

```
usage: vllm serve [model_tag] [options]
vllm serve: error: argument --task: invalid choice: 'rerank' (choose from 'auto',
'classify', 'draft', 'embed', 'embedding', 'generate', 'reward', 'score', 'transcription')
```

beside `pip show` lines like `vllm 0.9.1+empty` / `vllm-ascend 0.9.1`, and a
title of the form "不支持 <reranker>？". Two independent facts are being reported
and both belong in the first message: the **value** is wrong, and the release may
not support the **task** at all.

## Verdict first

1. **`rerank` is not a `--task` value in this line.** The valid set is the
   `TaskOption` literal, `vllm/config.py:82-83` at v0.9.1 — `score` is the
   cross-encoder (reranker) task, and the HTTP side already has every endpoint the
   reporter wants: `/score` (`entrypoints/openai/api_server.py:688`), `/v1/score`
   (`:716`), `/rerank` (`:777`), `/v1/rerank`. The correct launch is
   `--task score` (or `--runner pooling`, `config.py:88`). The architecture is not
   the gap either: `bge-reranker-v2-m3` is `XLMRobertaForSequenceClassification`,
   which the registry rewrites to `RobertaForSequenceClassification`
   (`model_executor/models/registry.py:171-172`).
2. **On vllm-ascend v0.9.1, pooling does not exist, so the corrected flag alone
   still will not serve it.** The project says so itself:
   `docs/source/user_guide/support_matrix/supported_features.md:13` →
   `| Pooling | 🟡 Planned | v0.9.1 doesn't support pooling. Use higher version instead |`,
   and the same release's note retires the engine that used to carry pooling:
   `docs/source/user_guide/release_notes.md:25` — *"V0 Engine is not supported
   from this release. Please always set `VLLM_USE_V1=1` to use V1 engine."*
   Upstream vllm v0.9.1 then closes the V1 door for every non-generate task:
   `vllm/engine/arg_utils.py:1347-1351` calls
   `_raise_or_fallback(feature_name=f"--task {model_config.task}")`, so with
   `VLLM_USE_V1=1` **exported** (which the 0.9.1 note tells users to do) it raises
   `NotImplementedError: VLLM_USE_V1=1 is not supported with --task score.`
   (`arg_utils.py:1633-1636`), and unset it only warns and falls back to V0.
   State both branches — otherwise the reporter re-runs, gets a *different*
   error, and the thread restarts.
3. **Minimum version that carries it: v0.9.2rc1.** Same file,
   `release_notes.md:183-188`: *"Pooling model works with V1 engine now. You can
   take a try with Qwen3 embedding model [#1359]"*.

## Why the reporter expects it to work (and is not wrong)

`release_notes.md:440`, under `## v0.7.3rc1` (`:427`): *"Added initial support for
pooling models. Bert based model, such as `BAAI/bge-base-en-v1.5` and
`BAAI/bge-reranker-v2-m3` works now. [#229]"*. The exact model was named as
working in the V0 era, and the 0.9 line is V1-only, so this is a **version
regression in the model-applicability story**, not "the model was never
supported" and not a defect to bisect. Quote the three lines and the thread is
closed.

## Procedure (this shape is a 3-call answer)

1. Read the choices list **out of the paste** — argparse prints the entire valid
   set, so the correct value needs no source read, no repository and no version
   check. One sentence: `rerank` → `score` (`/v1/rerank` unchanged).
2. Support at *their* release, two greps in the vllm-ascend tree:
   `rg -n "Pooling" docs/source/user_guide/support_matrix/supported_features.md`
   and `sed -n '1,40p' docs/source/user_guide/release_notes.md` (V0/V1 statement)
   plus `rg -n "TaskOption" vllm/config.py` in the vllm tree when the paste's
   choice list must be confirmed rather than trusted. The release-note **history
   below the current section** is where the "which version added/removed it"
   answer lives — read the headings (`grep -n '^## v' release_notes.md`) instead
   of the whole file.
3. Emit: corrected launch line, the minimum version, the line they will see at
   0.9.1 after the fix (both `VLLM_USE_V1` branches), and the endpoint to test
   with (`/v1/rerank`, not `/v1/completions`). Stop there; the pooling *runtime*
   classes (`index.md` 1-5) only apply once they are on a release that has it.

## Traps

- **Do not investigate the model or the checkpoint.** Both are registered and
  documented at this ref; the gate is the release.
- **A choice list in the error is version data.** A later doc, blog or tutorial
  may offer `rerank`; re-grep `TaskOption` at the reporter's ref before repeating
  their list. Same rule as `SKILL.md` rule 5 for any quoted CLI surface.
- **Do not take `pip show`'s version as the release.** Dev images install both
  projects editable from source, so the number is a build-time derivation
  (`0.1.dev1+g<sha>`, `0.9.1+empty`) — ask for the image tag or
  `git -C /vllm-workspace/vllm-ascend rev-parse HEAD`
  (`../image-dependency-faults/installed-version-vs-checkout.md`).
- **`--runner pooling` is the same gate.** It is the runner half of the same
  config (`config.py:88`, `_RUNNER_TASKS`); switching to it changes the message,
  not the version verdict.

## Recorded run (#4796) and how it ended

32 assistant turns, 1.77 M cumulative prompt tokens, **zero message text on every
turn**, killed at the 16 384-token cap. The two decisive facts were in hand by
turn 15 of the transcript (`TaskOption` by the 4th turn, the support-matrix row
in the 31st), and the
session had already derived both `VLLM_USE_V1` branches — nothing was missing but
the message. Treat an invalid-choice paste as evidence-in-hand: the answer is owed
before the second grep, not after the transport walk
(`../evidence-workspace/delivery-discipline.md`, `SKILL.md` rule 1 and rule 14).

## Recheck conditions

- A release whose `TaskOption` includes `rerank`, or whose `_raise_or_fallback`
  list admits non-generate tasks on V1, removes the second half of the verdict.
- The support-matrix wording is per-release: `🟡 Planned` may become
  `🟢 Functional` — quote the row at the reporter's ref, never the `latest` docs.
- Sibling shapes: a pooling serve that starts and then fails is
  `index.md` classes 1-5; an architecture string that routes to the wrong model
  class is `seq-cls-arch-routing.md`.
