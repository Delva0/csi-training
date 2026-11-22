# Triton-Ascend kernel compile failure: the installed wheel/compiler pair

## Scope

The engine starts, the weights load, and then the **first real request** dies
inside a Triton-Ascend kernel compile:

```
triton.compiler.errors.MLIRCompilationError:
///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc("/tmp/tmpXXXX/kernel.ttadapter.mlir":42:20): error: operation's operand is unlinked
subprocess.CalledProcessError: Command '['/usr/local/Ascend/tools/bishengir/bin/bishengir-compile',
  '/tmp/tmpXXXX/kernel.ttadapter.mlir', '--enable-auto-multi-buffer=True', ...]' returned non-zero exit status 1
```

Recorded on a hybrid GDN model (Qwen3-Next-80B-A3B) with the deepest
`vllm_ascend` frames in the FLA chunked-delta-rule kernels
(`vllm_ascend/ops/triton/fla/chunk.py` → `chunk_scaled_dot_kkt.py`), under
`triton/runtime/autotuner.py`: the traceback runs through the autotuner and the
error escapes it into `execute_model`, so the request dies on the first config it
tries — the failure is not confined to one bench candidate.

Owner: the **triton-ascend wheel × BiSheng / AscendNPU-IR compiler pair that was
installed into the image**, not vllm-ascend Python and not the kernel's shapes.
`bishengir-compile` is a separate executable fed the backend's `*.ttadapter.mlir`;
a wheel and a compiler from different builds is exactly how that conversion step
rejects valid IR. Sibling pages cover AscendC/aclnn device faults
(`../custom-ops-runtime-faults/index.md`); this page is the Triton compiler lane.

## Step 0 — read the `[ConvertLinalgRToBinary]` *detail* line, not the header

The banner `///------------------[ERROR][Triton][BEG]------------------` +
`[ConvertLinalgRToBinary] encounters error:` is emitted by the Triton compile wrapper for
**any** exception raised inside that stage, so the header alone does not name this lane.
What follows it does:

- `operation's operand is unlinked`, a `bishengir-compile … returned non-zero exit status 1`
  line, or IR diagnostics → the wheel/compiler pair, this page.
- **`/tmp/tmpXXXX/libkernel.so: failed to map segment from shared object`** → the loader:
  `ctypes.CDLL` in the backend's metadata-callback read returned ENOMEM from `mmap`. That is
  the mapping-count lane on a worker that has been up for days, **not** a mismatched wheel
  — `../runtime-memory/dlopen-map-count-exhaustion.md` (reporter-confirmed on #10551, where
  the convert block was explicitly a red herring; the worker's own traceback carries the
  `OSError` first and the `MLIRCompilationError` only as the re-raise in
  `During handling of the above exception`). Reinstalling or upgrading the pair fixes
  nothing there.
- **`loc("…kernel.ttadapter.mlir":2:3): error: ub overflow, requires N bits while M bits
  available!`** (also `Failed to run BiShengHIR pipeline`) → **not this page**: a budget
  refusal of the kernel's own tile, so reinstalling or moving the pair changes nothing —
  `../custom-ops-runtime-faults/triton-ub-overflow-tiling-budget.md`. It is the one detail
  line under this banner where the kernel's shapes *are* the owner, which is why the
  blanket "do not compute tiling here" below carries this exception.
- Any other detail — read it as the exception class it is; the pair is implied by an IR
  diagnostic or `operation's operand is unlinked`, never by a budget number.

## Reading the report correctly

- **"Failed to compile custom operators with Triton Ascend" is usually the
  reporter's paraphrase, not a library string.** In the recorded #5750 paste the
  phrase does not appear anywhere (`grep -c "Failed to compile"` = 0); the real
  error is the `MLIRCompilationError` block above. Do not go looking for the
  AscendC extension build (`vllm_ascend_C`, `Failed to register custom ops`):
  nothing in this failure touches it. Same family as SKILL.md rule 11.
- **It is not a startup failure.** The engine initialised, KV cache was allocated,
  then the request entered `execute_model` (`vllm/v1/worker/worker_base.py` →
  `vllm_ascend/worker/worker.py` → `model_runner_v1.py`, line numbers move per
  release; #5750 was on the 0.13.0rc1 line). So "the server came up but any
  inference kills it" is the signature — the opposite of the lane A/B workers in
  `../startup-failures/index.md`.
- Which kernel is chosen (`chunk_scaled_dot_kkt_fwd`, `chunk_gated_delta_rule`,
  `layer_norm_fwd_kernel`, …) dates the release but does not identify the defect: it
  is release code that the release's own pair is exercised with, so a per-kernel MLIR
  rejection says nothing about its shapes. Do not compute tiling or shapes here — unless
  the detail line is `ub overflow, requires N bits …`, which is the one case where the
  tile *is* the finding (`../custom-ops-runtime-faults/triton-ub-overflow-tiling-budget.md`).

## The install axis — the release's own pin decides which build is even installable

The project prescribes a **versioned pair**, never "latest from a mirror". At
vllm-ascend v0.11.0 (`2f1aed9`), `docs/source/tutorials/multi_npu_qwen3_next.md`:

| Lines | Step |
|---|---|
| `:48-54` | `wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/Ascend-BiSheng-toolkit_aarch64.run`, `--install`, then `source /usr/local/Ascend/8.3.RC2/bisheng_toolkit/set_env.sh` |
| `:60-61` | `triton_ascend-3.2.0.dev20250914-cp311-…whl` — an exact dev build, not a range |
| `:78` | `set_env.sh` is sourced again after the wheel install |

The recorded reporter's container instead logged (from its own attachment):

```
[AscendNPU-IR] ... AscendNPU-IR-1.0.0 install success       # into /usr/local/Ascend/tools
Downloading .../triton_ascend-3.2.0rc4-cp311-…whl
Successfully installed triton-ascend-3.2.0rc4
...
bishengir-compile: returned non-zero exit status 1
```

i.e. a standalone AscendNPU-IR package plus whatever the mirror served. Nothing in
that log says `Attempting uninstall: triton-ascend` / `Found existing installation:
triton-ascend`, so the distribution it built was **new**, not a replacement — an
image of that release carried none (see the pin table below). The differencing items
to ask for: `pip show triton-ascend`, `which bishengir-compile`, and whether
`set_env.sh` from the *doc's* toolkit root was sourced (a `RUN source …` in a
`Dockerfile` does not persist into later layers or the container).

**The reporter's `pip install triton-ascend` is not the variable; that release's pin
is.** One grep classifies it: `rg -n 'triton-ascend' requirements.txt pyproject.toml`
plus `rg -n 'Install Triton Ascend|triton_ascend-3' docs/source/` in the tree at the
reported ref, and the log's own `Downloading …/triton_ascend-<version>-…whl` line
(names the build actually installed):

| Pin at that ref | What it means | Route |
|---|---|---|
| **absent**, with a tutorial `Install Triton Ascend` section | the pair is a **hand install** for that release, and the only valid build is the `.dev` wheel its own docs name in the `vllm-ascend.obs.cn-north-4.myhuaweicloud.com` bucket | install *that wheel file* verbatim (plus `set_env.sh`), or move to a release whose pin is final |
| `.dev…` / `.rc…` pin | the index cannot be trusted to serve it, so `pip install triton-ascend` resolves some *other* build | same |
| **final** pin (`triton-ascend==3.2.0`) | pip — and therefore the image — installs the pair; there is **no** manual step, and a manual install only shadows/downgrades it | drop the manual install, keep the image |

Verified pins and the flip date: v0.11.0 (`2f1aed9`, the table above) and v0.12.0rc1
both hand-install — the latter names `triton_ascend-3.2.0.dev2025110717-…whl` and
"triton is not installed and enabled by default"
(`docs/source/user_guide/release_notes.md:215`); **v0.13.0rc2 flips it** —
"triton-ascend is upgraded to 3.2.0" (`release_notes.md:98`; the v0.13.0rc1 section
`:107-163` still names no triton-ascend anywhere) — and at v0.14.0rc1
`requirements.txt:37` / `pyproject.toml:33` pin `triton-ascend==3.2.0`, installed by
`Dockerfile:60-64` through `PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi`.
That is *why* the maintainer says v0.14.0rc1 "does not require additional
installation", and why 0.13.0rc1 — one release below the flip — is the trap: there is
no final pin to resolve, so any simple index (`mirrors.xfusion.com/pypi/simple` in the
recorded case) hands out a build that release was never tested with. When the snapshot
does not hold the reported ref, these neighbour pins plus the log's own download URL
are the evidence; say which one you read.

## Maintainer-confirmed resolution (#5750, 2026-01-29) and the ladder

> 0.13.0rc1 uses the triton-ascend dev package. Installing it via `pip install
> --trusted-host mirrors.xfusion.com -i https://example.invalid/pypi/simple
> triton-ascend` will result in an unusable version of triton-ascend. We
> recommend using our latest version, v0.14.0rc1, which works correctly and does
> not require additional installation of triton-ascend.

1. **Rung 1 is the release move, and it goes first in the answer.** v0.14.0rc1 (or
   anything at/after the v0.13.0rc2 pin flip) both fixes the pair and deletes the
   install step. This is the maintainer's own recommendation and the reporter
   accepted it ("OK, I'll give it a try"), so do not bury it under a local repair.
2. **If the reporter must stay on a hand-install release** (≤ v0.13.0rc1):
   `pip uninstall triton-ascend`, then install the *wheel file* that release's own
   release notes / tutorial name, verbatim — the obs
   `…/triton_ascend-3.2.0.dev<date>-cp311-…whl` — and source the doc's `set_env.sh`.
   Never `pip install triton-ascend` against any index, and never a "latest".
3. **"Re-run the untouched image" is conditional, not the universal cheapest step.**
   It is only a valid ablation where that release installs the pair itself (final
   pin, per Step 1): `pip show triton-ascend` on the untouched image must show a
   version. On a hand-install release the untouched image has none, and a
   Triton-required model (Qwen3-Next, DeepSeek-V3.2) then fails or silently loses the
   kernels — so the recorded #5750 container (rc4 installed over an image that had no
   triton-ascend) could not have been fixed that way. Do not rank it above rung 1.

The mechanism sentence to keep labelled as hypothesis: the log shows the IR
rejection, not which of the two artefacts is older or newer; "mispaired
wheel/compiler" is the reading that fits both artifacts, and the maintainer's
answer settles the *action*, not the ABI detail.

## Answer shape

1. Name the lane in the first sentence: Triton-Ascend kernel compile, engine up,
   first request dies — not an AscendC custom-op build.
2. Give the release pin class in one line (Step 1's table), then **rung 1 = move to
   the release whose `requirements.txt` pins a final `triton-ascend`** (v0.14.0rc1 at
   the recorded date; the flip is v0.13.0rc2) — maintainer-confirmed, and it also
   removes the install step.
3. If the reporter must stay: install that release's own `.dev` wheel from the obs
   URL its docs name, after `pip uninstall triton-ascend`; never an index
   `pip install triton-ascend`. Offer the untouched-image re-run only after
   `pip show triton-ascend` on that image returns a version; print
   `pip show triton-ascend` / `which bishengir-compile` either way as evidence.
4. Missing artefacts to ask for: the `pip list | grep -i triton`, the toolkit root,
   and whether the failing request is the very first one after startup.
5. Close with the version gap when it applies: the pinned snapshot tree may not
   contain the release under test (for #5750 it is v0.11.0, which has no
   `vllm_ascend/ops/triton/` directory at all), so `path:line` for the 0.13 line
   can only come from the reporter's own traceback — say so.

## The same pairing's numeric lane: NaN in quantized operators

Not every bad triton-ascend build fails to compile. The recorded #5606 (Qwen3-Next
W8A8 at vllm-ascend v0.12.0rc1, `--quantization ascend`) had a *serving* engine whose
answers were a run of `!` to `finish_reason: length`, and the maintainer's statement in
that issue names the wheel: the triton-ascend `1107` build "will led to the occurrence
of **NAN** problems for some operators in the quantization scenario, just like all the
answers here are full of exclamation marks"; the reporter then changed the build and the
deployment served.

- One command decides it: `pip show triton-ascend` (or `pip list | grep -i triton`).
  `collect_env` filters `pip list` by the substring `triton`
  (`vllm/collect_env.py:81-92`, applied at `:540-543`), so a paste with no `triton*`
  line is a fact to ask about, not a version you can assume.
- Compare against the *model's* tutorial, not a global pin: at vllm-ascend v0.12.0rc1
  `docs/source/tutorials/Qwen3-Next.md:58-59` installs
  `triton_ascend-3.2.0.dev2025110717` — i.e. the build the maintainer calls known-bad
  for quantization. The repair direction is a **newer** build than that page's pin.
- A NaN here does not raise: it collapses logits to one token, so the symptom looks
  like a sampler collapse. Route on it from
  `../accuracy-benchmarks/qwen3-next-gdn-numeric-path.md` §*triton-ascend build*,
  which also fixes the ordering against the bf16 control (a bf16 run on the same bad
  wheel can be clean).

## Related

- Same banner, loader detail (`libkernel.so: failed to map segment from shared object`):
  `../runtime-memory/dlopen-map-count-exhaustion.md`.

- Same stack, **hand-installed**: `Failed to import Triton kernels … Error: No
  module named 'triton.language.target_info'` (a logged, non-fatal vllm line) beside
  the reporter's own `no attribute 'next_power_of_2'` / `'function' object is not
  subscriptable` ladder → `triton-kernels-import-error.md`; the fix is removal of the
  foreign `triton`/`triton_kernels`, never another triton version.
- Same pair at **startup**, before any kernel runs: an `Error in inspecting model
  architecture '<Arch>'` whose logged tail is an import-time
  `AttributeError: module 'triton.language' has no attribute '<x>'` (a triton is
  installed and it is the wrong one — an absent triton only gives vllm's
  placeholders) → `../startup-failures/model-type-not-recognized.md`, inspection
  section.
- Same pair, performance side: `../performance-throughput/qwen3-next-decode-throughput.md`
  (a hybrid GDN model is slow when the Triton FLA kernels are absent/replaced).
- `../custom-ops-runtime-faults/index.md` for the aclnn/AscendC lane that this
  page is *not*.
- `offline-source-build.md` for a wheel that has to be built, rather than
  installed, on a host without public network.
