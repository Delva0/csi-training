# Atlas 300I / 310P: what the platform changes, and what the docs admit

Verified at vllm-ascend **v0.11.0rc3** (`b6d63bbd`) with vllm **v0.11.0**
(`b8b302cd`); the `-310P` image is built from `Dockerfile.310p.openEuler`
(`FROM quay.io/ascend/cann:8.3.rc2-310p-openeuler24.03-py3.11:18`,
`SOC_VERSION=ASCEND310P3:51`, `VLLM_TAG=v0.11.0:37`). Re-grep per version: the
gate list and line numbers move, the shape does not.

For "there is no 310P artefact for my machine" (arm64 host, an arm64 tag that pulls
amd64, an aarch64 wheel) the owner is the image pipeline, not this page:
`../image-dependency-faults/image-arch-and-tag-availability.md`.

Two *runtime* shapes on this SoC are not this page's either: an op that lands on the
CPU/AICPU fallback (`507018` "aicpu execution is abnormal" / `funcName=RunCpuKernel` /
`IndexPut` / `<Op>Operation setup failed!`) is `310p-cpu-fallback-op-fault.md`, and a
non-fp16 KV cache (`--kv-cache-dtype fp8`) refused at the first store is
`310p-kv-cache-dtype.md`. This page owns the gates, the docs and the startup lanes.

## The documented status (quote it, then move on)

- `docs/source/faqs.md:18`: "[Experimental] Atlas 300I inference series (Atlas
  300I Duo). Currently for 310I Duo, the **stable version is vllm-ascend
  v0.10.0rc1**."
- `docs/source/tutorials/single_node_300i.md:3-7`: 300I is "currently
  experimental", "only supports eager mode and the float16 data type", known
  issues `#3316` / `#2795`, "You can use v0.10.0rc1 version first."
- **The same file contradicts itself**, and this is the trap in #4812: its
  Qwen2.5-VL-3B tab (`:126-131`) ships exactly the failing command
  (`vllm serve … --tensor-parallel-size 1 --enforce-eager --dtype float16
  --compilation-config '{"custom_ops":["none","+rms_norm","+rotary_embedding"]}'`)
  while the image line at `:17` still says `v0.10.0rc1-310p`. When a reporter
  pastes a command that is verbatim from a tutorial, say so — it removes "user
  error" from the ranking and it is a documentation finding in its own right.
- Environment the tutorial requires and reporters routinely omit (`:43-49`):
  `VLLM_USE_MODELSCOPE=True` and
  `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`. That value is the project's
  300I fragmentation workaround, not general tuning, and from v0.18.0 on it also
  switches off vllm-ascend's own `expandable_segments:True` default
  (`vllm_ascend/platform.py:518-528` at v0.18.0; no such injection exists at
  v0.11.0rc3) — state both effects
  (`../runtime-memory/hbm-growth-and-oom.md`, step 3).

- **At a v0.14.0rc1 pin the tutorial is a flat `docs/source/tutorials/310p.md`**
  (`hardwares/310p.md` does not exist there), and the release that re-opened the lane
  also *scoped* it: `docs/source/user_guide/release_notes.md:9` — "310P support is back
  now. In this release, **only basic dense and vl models** are supported with eager
  mode" (#5776). A 310P **MoE** (A3B) report at that pin is outside the declared surface,
  and the same build carries **no custom aclnn op**: `csrc/build_aclnn.sh:5-11` reaches
  `exit 0` for `^ascend310` before `CUSTOM_OPS` is ever assigned, so
  `enable_custom_op()` (`vllm_ascend/utils.py:243-266`) is False and the routed-expert
  selector's `torch.ops._C_ascend.moe_gating_top_k`
  (`vllm_ascend/ops/fused_moe/experts_selector.py:221`) has no aclnn backend. Any
  `aclnn<Op>` in such a paste is therefore a CANN/`torch_npu` **built-in**.
- **That tutorial contradicts the release note**, and it is why a 310P report quotes two
  tags two releases apart: `310p.md:3-8` is still "experimental … only supports eager
  mode and the float16 data type … You can use v0.10.0rc1 version first", with
  `IMAGE=quay.io/ascend/vllm-ascend:v0.10.0rc1-310p` at `:17`. Quote both and name the
  inconsistency as a doc defect (recorded #6488: the reporter tried
  `v0.10.0rc1-310p` **and** `v0.14.0rc1-310p` because the page told them to).
- **An op name is a generation marker.** `aclnnMoeGatingTopKSoftmax` is
  `torch_npu.npu_moe_gating_top_k_softmax`: at v0.14.0rc1 vllm-ascend calls it nowhere
  (`rg -n npu_moe_gating_top_k_softmax vllm_ascend/` is empty; only
  `tests/e2e/nightly/single_node/ops/singlecard_ops/test_gating_top_k_softmax.py:29`)
  while its own `release_notes.md:681` records when the selector was wired to it
  (`#1902`) — i.e. the name dates the failing run to the ≤v0.10 selector generation. Grep
  the trees **once**, then stop: the caller is in no pinned export, and the deliverable is
  the standalone probe plus the container's op store (`rg -l MoeGatingTopKSoftmax
  /usr/local/Ascend/ascend-toolkit/latest/opp`, under `ASCEND_LAUNCH_BLOCKING=1`) —
  `kernel-package-vs-card.md`. #6488 spent 21 turns and 39 tool calls on that caller hunt
  and shipped **81 characters** of message text.

A VL or OCR checkpoint that **serves and answers garbage** on this SoC is the
other shape of the same platform question — support is a per-model doc statement
and document parsing is a pipeline: `310p-vl-ocr-garbage-output.md`.

## A capability question at an old pin: "does `v0.9.1rc1-310p` do 300I Duo + MRoPE (Qwen2.5-VL)?"

Verified at vllm-ascend **v0.9.1rc1** (`c30ddb8`) + vllm **v0.9.1** (`b6553be`),
recorded for #1684 (`vllm-ascend:v0.9.1rc1-310p`, Atlas 300I Duo, Qwen2.5-VL-7B
and -32B, no log — a pure status question). The reporter has no traceback, so the
verdict comes from the docs at that pin and from one code fact, not from a walk of
the rope implementation. Deliver it first: **no statement at this pin validates a
VL model on 300I, and the 32B does not fit one Duo card** — then the two
ablation-sized facts below.

At a v0.9.x pin **the docs tree has a different shape**, and grepping the layout
you remember wastes calls: `docs/source/tutorials/` is flat (`single_npu.md`,
`single_npu_multimodal.md`, `single_node_300i.md` — the last **only from v0.9.1rc1**,
see the next subsection — `multi_node.md`, …) with no
`models/` and no `hardwares/`; the support matrix is
`docs/source/user_guide/supported_models.md` with no `support_matrix/` directory;
and `vllm_ascend/` has **no `_310p/` subpackage**. Locate by a *content* token, not
a filename — `rg -l 'Qwen2.5-VL|qwen2_5_vl' docs/source` — because the filename of
the multimodal tutorial is `single_npu_multimodal.md` and
`rg --files docs/source | rg -i 'vl|vision|ocr'` returns only logo images.

The three status statements at that one pin, and they disagree:

| Statement | Anchor |
|---|---|
| 300I is "experimental supported in this release … this feature **will NOT be included in v0.9.1-dev branch** … We will improve this from 0.9.2rc1 and later" | `docs/source/user_guide/release_notes.md:3,9` |
| "Currently, **ONLY Atlas A2 series** … are supported … Below series are NOT supported yet: Atlas 300I Duo、Atlas 300I Pro (Ascend-cann-kernels-310p) might be supported on 2025.Q2" | `docs/source/faqs.md:12,17-18` |
| the multimodal matrix has **no hardware column at all** — `\| Model \| Supported \| Note \|` with `Qwen2-VL`/`Qwen2.5-VL` ✅, so the ✅ is a project-wide claim, not a 300I claim | `docs/source/user_guide/supported_models.md:39-42` |

The closest documented VL recipe is **not** the 300I page:
`docs/source/tutorials/single_npu_multimodal.md:1` is "Single NPU (Qwen2.5-VL
7B)", `:147` serves it with `--dtype bfloat16 --max-model-len 16384
--max-num-batched-tokens 16384`, and `:151` says only that the value depends on
the NPU series' HBM. The file names no SoC, while
`docs/source/tutorials/single_node_300i.md:46` lists what that channel validated —
Qwen3-0.6B 1 card, Qwen2.5-7B-Instruct 2 cards, Pangu-Pro-MoE-72B 8 cards, all
`--enforce-eager --dtype float16` — and `rg -n -i 'vl|vision|multimodal'` on it
returns nothing. So hand the tester that page plus the two deltas 300I forces
(fp16, eager), labelled as derived.

**Why it can still start (the code fact, not a kiss-off).** There is no 310P MRoPE
path to be missing at this pin: `rg -n 'MRotaryEmbedding|AscendMRotary'
vllm_ascend/` returns only the host-side position helpers
(`worker/model_runner_v1.py:45,486,797`, `worker/model_runner.py:45,724,879`), and
the one rope patch is `RotaryEmbedding.forward_oot = rope_forward_oot`
(`ops/rotary_embedding.py:277`), whose kernel branch is gated
`… and not is_310p()` (`:52`). Upstream `MRotaryEmbedding`
(`vllm/model_executor/layers/rotary_embedding.py:985`) overrides **`forward`**
itself (`:1009`), so the ascend patch never applies to it, and `_apply_rotary_emb`
takes the `_apply_rotary_emb_torch` branch off CUDA (`:84-88`). Consequence:
MRoPE runs a plain torch path and is *not refused* — "能跑起来但精度有问题"
(#118 is the same report on main) is the expected shape, and `qwen2_5_vl.py` is
registered unconditionally at this pin (`vllm_ascend/models/__init__.py:11-13,
23-31`, padded and without-padding variants). Grade the pair **unvalidated**, not
unsupported-by-code, and say which of the two you mean.

**Sizing decides the 32B before any rope question.** 7B at bf16 is ~15 GB and the
pin's own 300I tutorial serves Qwen2.5-7B-Instruct on **2 cards**; 32B at bf16 is
~64 GB, i.e. ~4× that footprint, so it needs several times that card count — a
memory verdict plus `--max-model-len` arithmetic (`scripts/card-budget.sh`,
`../startup-failures/max-model-len-budget.md`). Take the per-card HBM from the
reporter's `npu-smi info`, not from a datasheet this snapshot cannot check, and
state the arithmetic with the number you were given.

The one grep that would change the answer is on a *later* pin, so name it as the
recheck: a later `-310p` image carries the `_310p/` package (its gate list is in
this page) and its rope override answers a different question
(`310p-vl-ocr-garbage-output.md` §4), so "MRoPE is unpatched" is a version fact,
not a permanent one — re-grep before reusing this block.

### One release earlier the lane does not exist at all: v0.9.0rc2 has no 310P

Verified at vllm-ascend **v0.9.0rc2** (`8dd686d`) + vllm **v0.9.0** (`5873877`), recorded
for #1609 (Atlas 300I Duo, `qwen2.5-vl-7b`, no log, answered at its 9th assistant turn).
`git grep -n -i -E '300i|310p'` over the whole tree returns **one** hit —
`docs/source/faqs.md:18` "Below series are NOT supported yet: Atlas 300I Duo … might be
supported on 2025.Q2" (with `:12` "Currently, **ONLY Atlas A2 series**"). There is no
`docs/source/tutorials/single_node_300i.md` (that directory is `index.md`, `multi_node.md`,
`multi_npu.md`, `multi_npu_quantization.md`, `single_npu.md`, `single_npu_multimodal.md`),
no `Dockerfile.310p` (only `Dockerfile`, `Dockerfile.openEuler`) and no `is_310p` in any
`.py`. The VL class is still registered unconditionally (`vllm_ascend/models/__init__.py:11-13`,
`:24-26`, `:30-31`), so the engine resolves the architecture and a run *starts* — the
verdict is **"the lane is absent from this release"**, not "the model is unregistered".

**The doc the reporter quotes is itself the version marker.** #1609 quoted a 300I page
listing "Qwen3-0.6B: 1 card, Qwen2.5-7B-Instruct: 2 cards, Pangu-Pro-MoE-72B: 8 cards" —
the v0.9.1rc1 `single_node_300i.md:46` list above. At v0.9.0rc2 the string `Pangu` occurs
in **no** file (`grep -rni pangu` is empty), so one grep proves they are reading
`latest`/v0.9.1+ docs, and the first paragraph is "your pin predates 300I support; the page
you quote is a later release" plus the move to a `-310p` tag. Do not walk the VL/rope code
here: at this pin there is no 310P branch to read, which is why the same question has a
completely different answer one rc later (the MRoPE block above).

Do not promise that the upgrade fixes it. **#1383** on the first `-310p` release
(v0.9.1rc1, 310P3) could not start even `Qwen/Qwen2.5-0.5B-Instruct`:
`Optype [TransData] of Ops kernel [AIcoreEngine] is unsupported … op type TransData is not
found in this op store`. When the paste carries a card table, take capacity from it —
**#1141**'s `npu-smi` shows `310P3` ×4 at `1460 / 44280 MB` per die — and label it a
reporter run, never a datasheet.

## An image that carries triton-ascend cannot import the worker at all

Verified at vllm-ascend **v0.14.0rc1** (`52d4acfa` — the release whose notes say
"310P support is back now", `docs/source/user_guide/release_notes.md:9`, PR #5776)
with vllm **v0.14.1** (`d7de043d`), image `…:v0.14.0rc1-310p`. A different pin
from the rest of this page, because the 310P lane was re-opened there; re-grep
for the next 310P release.

Shape (#6309): `EngineCore failed to start` in the same second the worker class is
resolved — before any config check and before any weight load:
`resolve_obj_by_qualname` → `vllm_ascend/_310p/worker_310p.py:21` →
`worker/model_runner_v1.py:96` → `patch/worker/__init__.py:20-21`
(`if HAS_TRITON: import … patch_triton`) → `patch_triton.py:4` →
`ops/triton/fla/chunk.py:15` → upstream
`vllm/model_executor/layers/fla/ops/utils.py:141`, a **module-level**
`device = "cuda" if … else get_available_device()` →
`triton.runtime.driver.active.get_current_target()` →
`triton/backends/ascend/driver.py:188,91` `get_arch()`, ending
`SystemError: <built-in function get_arch> returned NULL without setting an exception`.

Two facts make this a one-call answer:

- The owner is the **triton-ascend build in the image**, not the model, not the
  310P capability table: nothing model-specific has run yet, and `get_arch()`
  returns NULL because that backend cannot map the SoC.
- Upstream `get_available_device()` catches only `(RuntimeError, AttributeError)`
  (`utils.py:118-123`), so `SystemError` escapes at import time — while the
  listener on the vllm-ascend side is `HAS_TRITON`-gated and 310P is eager-only
  (no Triton kernels), so removing triton removes the whole chain:
  `pip uninstall -y triton triton-ascend`, then
  `python -c "from vllm.triton_utils import HAS_TRITON; print(HAS_TRITON)"` must
  print `False`. The same frame can be probed in one line without vLLM:
  `python -c "import triton; print(triton.runtime.driver.active.get_current_target())"`.

Labels: source-verified from the reporter's own traceback plus this pin. The
`pip uninstall` action is **not** reporter-confirmed, and whether the shipped
`-310p` image carries that triton build is a registry fact this snapshot cannot
check — ask for `pip list | grep -i triton` before asserting it. Do not answer
this shape with the version survey the reporter asked for: at this pin
`v0.14.0rc1-310p` is the image that reintroduced 310P, so "which tag supports
310P" is the wrong question.

## A penalty parameter crashes the test run on 310P (400/500-class, no log)

Verified at vllm-ascend **v0.23.0** (`5cb98caa`) with vllm **v0.23.0** (`0fc695f`).
Shape (#15161, Qwen3.6-35B-A3B-w8a8, aisbench, `presence_penalty: 1.5`): the engine comes
up and every request without a penalty is fine; the first request carrying a penalty
"一到测试阶段就会崩溃", with no traceback pasted.

The branch is printed in the reporter's own log, so ask for that one line before any
source walk:

| branch | trigger | anchor |
|---|---|---|
| Triton-Ascend penalty kernels | `HAS_TRITON` true in that image | `vllm_ascend/sample/sampler.py:61-64` (`apply_all_penalties`, after `assert sampling_metadata.prompt_token_ids is not None` at `:63`), reached from `AscendSampler310` (`_310p/sample/sampler.py:125`, inherits `apply_penalties`) |
| upstream torch fallback | `HAS_TRITON` false — **the shipped 310P image's own configuration** | `sampler.py:54-59` prints `[sample/sampler] Triton not available, falling back to vLLM default penalty implementation…` once, then `Sampler.apply_penalties` |

The image difference is in-tree, not a registry guess: `Dockerfile.310p:63` runs
`pip uninstall -y triton-ascend triton` and does **not** reinstall, while `Dockerfile:70-71`
and `Dockerfile.a3:72-73` reinstall `triton-ascend==3.2.2` after the same uninstall. So on
a `-310p` image the warning line is expected and the fallback is the live path.

Two consequences for the answer:

- **The #2533/#2544 precedent is not this pin's mechanism.** `AttributeError: '_OpNamespace'
  '_C' object has no attribute 'apply_repetition_penalties_'` came from an older vLLM;
  at v0.23.0 `vllm/_custom_ops.py:387-411` dispatches `apply_repetition_penalties` to
  `apply_repetition_penalties_torch` whenever `not logits.is_cuda`, so the fallback runs
  pure torch on NPU. Quote the dispatch line, not the old issue.
- **A crash with the fallback live needs its traceback**: the remaining candidates are the
  `prompt_token_ids` assertion (only in the Triton branch, so excluded when the warning is
  present), an NPU-unsupported op inside the torch path, or something outside penalties
  entirely — and only the text decides. Ask for the traceback, the warning line, and
  `pip list | grep -i triton`; the one-request ablation (`presence_penalty: 0` on the same
  body) confirms the penalty is even the trigger.

Labels: the branch table and the Dockerfile split are source-verified at those commits;
which branch the reporter's image took is one log line away, and the crash owner is
**unresolved** in this snapshot — do not present either branch as the cause.

## `the socversion <A> of bin package does not match the current device socverison <B>`

Recorded #2725: `openEuler23.03 aarch64`, reported v0.10.0rc1, `vllm serve
Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --enforce-eager --dtype float16`
(the `single_node_300i` command verbatim). The paste shows `Loading model weights took
14.2488 GB`, then `the socversion Ascend910B1 of bin package does not match the
current device socverison Ascend310P3. Please modify default socversion in run.sh or
execute run.sh with socversion parameter.`

Read it as an **artifact** mismatch, not a code bug: weights load because that is
plain `torch_npu` copies, and the first AscendC/aclnn kernel out of the *bin package*
refuses because the package was built for `Ascend910B1` while the live card is
`Ascend310P3`. This is the kernel-package twin of
`device-type-build-vs-runtime.md`'s Python-side assertion, and the two SoC strings keep
the same direction rule: the **first** is the build, the **second** is the card.

Navigation cost to avoid: **the message is in no pinned tree.** `rg -n "of bin
package|socverison" /workspace/repos` returns nothing (checked at v0.9.1rc3
`763ed69`), and the history titles do not carry it either — its own wording names a
`run.sh` with a `socversion` parameter, i.e. the CANN custom-op build script that
ships with the ops package. One grep at most, then stop: the owner is the artifact
choice.

Two owners, cheapest first:

1. **Wrong image variant.** The generic and `openEuler` images are both the 910B line
   — `Dockerfile.openEuler:19` is `FROM quay.io/ascend/cann:8.2.rc1-910b-openeuler22.03-py3.11`
   — and `-310p` tags are separate builds. Fix: pull the `-310p` tag for the reported
   version (`docker manifest inspect` first, and pick `--platform` explicitly on an
   arm64 host), then confirm `npu-smi info` in the same container. #1905 runs 300I Duo
   on `v0.9.2rc1-310p`; #1684 / #1812 show `-310p` tags exist for other rcs.
2. **Self-built package with the default SoC.** `vllm_ascend/envs.py:58-61` defaults
   `SOC_VERSION` to `"ASCEND910B1"` and `setup.py:168` passes it as `-DSOC_VERSION=`,
   so a wheel or custom-op build made without `SOC_VERSION` is stamped 910B1 no matter
   which card builds it (`device-type-build-vs-runtime.md` step 3 for the stamp).

Quote the pinned support statement in the same answer: `docs/source/faqs.md:18` at
v0.9.1rc3 lists `Atlas 300I Duo、Atlas 300I Pro (Ascend-cann-kernels-310p)` under
"Below series are NOT supported yet … might be supported on 2025.Q2" — the separate
`310p` kernel package is exactly why a generic bin package cannot match.

## The 310P gates that change behaviour

| Gate | Code (v0.11.0rc3) | Consequence to check |
|---|---|---|
| SoC detection | `vllm_ascend/utils.py:66-71` (`:65-69` at v0.11.0rc1) — `is_310p()` reads `_build_info.__soc_version__` | every branch below keys off the build stamp, not the device name. At **v0.11.0rc1** there is no runtime guard at all — `check_ascend_device_type()` (the assert in `device-type-build-vs-runtime.md`) does not exist in that tree, so a generic unsuffixed image on a 310P host silently keeps `is_310p() == False` and takes the A2 path; the tag suffix is the only signal |
| Custom ops not activated | `vllm_ascend/platform.py:287-288` — `# Activate custom ops for v1, except on 310P` / `if not is_310p(): compilation_config.custom_ops = ["all"]` | on 310P the reporter's `--compilation-config custom_ops` list is the only one in play; on other SoCs vllm-ascend overwrites it. A "custom op choice made no difference" observation is expected here |
| Global NZ format | `vllm_ascend/worker/model_runner_v1.py:146-154` — `torch.npu.config.allow_internal_format = True`; `set_compile_mode(jit_compile=False)`; `ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ` on 310P (ND otherwise) | every `_convert_torch_format` call site becomes a real conversion on 310P and a no-op elsewhere |
| Weight NZ conversion after load | `model_runner_v1.py:2685-2694` converts every `MergedColumnParallelLinear` / `QKVParallelLinear` / `RowParallelLinear` weight | load-time failures on 310P can be format casts, not op bugs |
| Cache NZ conversion | `model_runner_v1.py:2716-2721` (`_convert_torch_format`) called from the KV-init paths (`:2826`, `:3076` for `k_cache`) | see `../runtime-memory/kv-init-and-format-cast-alloc.md` — this frame is an allocation site |
| dtype gate on NZ | `vllm_ascend/utils.py:74-84` — `is_enable_nz()` returns **False for float16/bfloat16**, else `VLLM_ASCEND_ENABLE_NZ` (default 1, `vllm_ascend/envs.py:168-169`) and False for `qwen3_next` | at this pin an fp16 cache is returned unchanged, so a traceback whose deepest frame really executed `torch_npu.npu_format_cast` implies a non-fp16/bf16 cache (int8 / quantised) **or** a tree whose gate differs from this pin. Say which you verified |
| Attention mask NZ | `vllm_ascend/attention/attention_v1.py:253-262` (`nd_to_nz_2d` / `nd_to_nz_spec` + `npu_format_cast`) and further `is_310p()` branches at `:72,347,403,531` | mask-shape/format errors on 310P belong to this branch, not to the model's attention code |
| Separate 310P patch module | added later than this pin: at v0.18.0 `patch/platform/__init__.py:21-26` imports `patch_mamba_config_310` when `is_310p()` (the file does not exist at v0.11.0rc3) | a 310P-only patch can drift independently of the A2 one, and the gate list itself moves between releases — re-run the grep |

`patch_mamba_config_310.py` is also the entry to the 310P **mamba-state lane**: it
rewrites `mamba_block_size` for `align`/`all` cache modes, `patch_mamba_utils.py` swaps
the Triton state copy and align postprocess for torch/numpy fallbacks on 310P, and the
GDN op forces an fp16 SSM state — the mechanism map plus the MTP x prefix-cache coverage
verdict is `../accuracy-benchmarks/mtp-prefix-cache-310p.md`.

Two of these gates combine into the 310P LoRA answer: there is no graph mode
to amortise small launches — readable as "300I is eager-only" in the ≤v0.12 doc,
while the v0.18.0 doc below ships a `FULL_DECODE_ONLY` block with TP>1 capture
limits, so re-check the pin before reusing this — **and** `PunicaWrapperNPU` binds
the PyTorch `vllm.lora.ops.torch_ops` callables whenever the device type is
310P — at v0.9.2rc1 unconditionally, at v0.13.0rc2 by that explicit predicate
even though the AscendC `vllm_ascend/lora/lora_ops.py` exists in the same tree
(`vllm_ascend/lora/punica_npu.py:27-39`). So "LoRA is 4× slower here" is
structural on this SoC, and the lever is merging the adapter or moving the
release, not a flag — `../lora-adapters/decode-overhead-vs-base-model.md`.

## At v0.12.0rc1: the gate is renamed, the tutorial path moved, and "when will 310 get graph mode" has no answer

Verified at vllm-ascend **v0.12.0rc1** (`42ceaf0`) with vllm **v0.12.0**
(`4fd9d6a8`) while triaging #5075 (300I Pro, reporter on vllm-ascend 0.10.0rc1,
Qwen2.5-VL-3B, `--enforce-eager`, "310系列什么时候支持图模式和量化", 8 s vs T4 3 s
end-to-end on an ID-card OCR pipeline):

- **`is_310p()` does not exist at this ref.** `rg -n "is_310p" vllm_ascend/` returns
  nothing — a version fact, not a missing gate. The predicate is spelled
  `get_ascend_device_type() == AscendDeviceType._310P` (`utils.py:677` `_310P = 2`,
  `get_ascend_device_type()` at `:710`; call sites `platform.py:297`,
  `worker/model_runner_v1.py:138,2342`, `quantization/w8a8.py:51,163,320,349,366,386`,
  `lora/punica_npu.py:9`). Grep `_310P` before concluding "this tree has no 310P path".
- **The tutorial path moved, and the reporter's command is copied from whichever one
  their tag ships.** `docs/source/tutorials/single_node_300i.md` (≤v0.11, quoted
  above) → `docs/source/tutorials/310p.md` (v0.12.0rc1: eager-only note at `:5`,
  Qwen2.5-VL-3B-Instruct tab at `:120-131` whose serve line is
  `--tensor-parallel-size 1 --enforce-eager --dtype float16 --compilation-config
  '{"custom_ops":["none","+rms_norm","+rotary_embedding"]}'` verbatim) →
  `docs/source/tutorials/hardwares/310p.md` (already at v0.18.0 — see the v0.18.0
  block below; v0.12 spelled it `docs/source/tutorials/310p.md`). A paste that matches a tab
  byte-for-byte is the doc's own recipe, so "user error" leaves the ranking; the old
  path still exists only as `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/single_node_300i.po`,
  which dates a paste that cites it.
- **Quantization already has a 310P path at this ref.** `--quantization ascend` plus
  the 310P branch of `npu_quant_matmul` (`quantization/w8a8.py:163`, transpose
  decision at `:51`, `fused_experts_310p` at `:320`), so "310 量化什么时候支持" is
  answered "already — with a 310-targeted checkpoint", not "wait"
  (`310p-quant-checkpoint-soc-match.md`). Graph mode is **not** supported: `310p.md:5`
  says the 310I series "only supports eager mode and the float16 data type".
- **Establish the coverage verdict from test *paths*, then from the workflow.** At
  this pin the 310P suite is one file, `tests/e2e/310p/test_offline_inference_310p.py`,
  and every case in it is `enforce_eager=True` + fp16 (Qwen3-0.6B,
  Qwen2.5-7B-Instruct, and a Qwen2.5-VL-3B case — the reporter's own model):
  content-grepping `310p|_310P` inside `tests/` misses it because the SoC is only in
  the **directory name**, so use `rg --files tests/ | rg -i 310`. The wiring is
  `.github/workflows/vllm_ascend_test_310p.yaml` (label-gated, not per-PR):
  `:112-115` runs that file on `linux-aarch64-310p-1` and
  `tests/e2e/310p/test_offline_inference_parallel_310p.py` on `-4` — **a file that
  does not exist at this commit** (`git ls-tree -r --name-only HEAD | rg 310p` lists
  only the one test), so the 4-card leg cannot run and the only real 310P coverage at
  this ref is single-card eager.
- **"什么时候支持" is answered from the project's own staging, not from a guess.** The
  RFC that owns this platform is #1309 `[RFC]: Support Altlas 300I series`:
  `Stage1: E2E workflow work` (310P CI runner, e2e, wheels — several unchecked) and
  **`Stage2: Performance improvement` with an empty body**. That empty heading is the
  citable statement that graph mode/quantization work on 300I has no committed scope
  or date; a documented 310P `FULL_DECODE_ONLY` block is already in v0.18.0's
  `docs/source/tutorials/hardwares/310p.md:7-11` (verified, see that block below), and
  which release between v0.12 and v0.18.0 landed it is unanswerable here
  (`one_commit_git` per repo — say so, do not guess). Roadmap
  issues #448 (Q2 2025) and #1168 (Q3 2025) carry a "Graph mode" line with no 310P
  date; find them with `scripts/history-titles.sh -i '300I|310P'`.
- **Do not import the v0.22 mask trap backwards.** `attention/attention_v1.py` has
  **no** `_310P` branch at v0.12.0rc1 (`rg -c _310P` = 0), so the
  `O(max_model_len²)` fp16 mask allocation quoted for v0.22.1rc1 below cannot be
  attributed to a v0.12 paste.

## At v0.18.0: the 310P doc is a Qwen3 tutorial, and Qwen3.5/3.6 are "preview … follow #7394"

Verified at vllm-ascend **v0.18.0** (`e18643f8a4d5bd9990727654318ad069ea0b56e2`) with vllm
**v0.18.0** (`bcf2be96120005e9aea171927f85055a6a5c0cf6`) while triaging #9376 (Atlas 300I Duo,
`quay.io/ascend/vllm-ascend:v0.18.0-310p`, host CANN 9.0.0,
`vllm serve /opt/models/Qwen3.5-4B --tensor-parallel-size 2 --max-model-len 4096
--enforce-eager --dtype float16`, and the only log an external `user-attachments` URL the
snapshot cannot fetch). Re-grep per release; the shape below — *the doc's own closing note is
the support verdict* — is what does not move.

- **The doc path moved before v0.22, and the v0.12 block above is wrong about it.** At
  v0.18.0 the 310P document is already `docs/source/tutorials/hardwares/310p.md`, and
  `docs/source/tutorials/single_node_300i.md` no longer exists. A paste citing the old name
  dates the reporter's tag, not this release; `docs/source/tutorials/310p.md` is the
  v0.12-era spelling.
- **Quote these lines, in this order.** `:7-11` — `FULL_DECODE_ONLY` **is** supported on 300I
  Duo, with the TP>1 capture-count limit ("Qwen3-32B can capture and replay 2 graphs", none at
  TP=1), so "300I is eager-only" is a ≤v0.12 statement and not this pin's. `:13-21` — the card
  "does not support `triton` or `triton-ascend`", with the uninstall line. `:26-40` — never omit
  `--max-model-len` (`AttentionMaskBuilder310` builds the `[max_model_len, max_model_len]` fp16
  causal mask, then converts to FRACTAL_NZ). `:94-205` — the validated set is
  **Qwen3-{8B,14B,32B} w8a8s/w8a8sc only** (TP1/TP1/TP4, `--quantization ascend --dtype float16
  --no-enable-prefix-caching`, `FULL_DECODE_ONLY`). `:207-209`, the closing note — "For early
  access to Qwen3-MoE, Qwen3-VL, and **preview support for Qwen3.5 and Qwen3.6** with
  performance acceleration, follow #7394 for updated deployment guidance."
- **"Does vllm-ascend support <new family> on 310P, or only the POC image?" is answered by that
  sentence, in the turn you read it.** The reporter's own later comment on #9376 is verbatim
  that question (`是vllm-ascend不支持Qwen3.5-4B，只支持#7394中的POC镜像吗？`), asked after a
  session that had the doc line by its 13th tool result and shipped no answer before the
  output cap. #7394 is a *future*
  issue here — `/repos/vllm-project/vllm-ascend/issues/7394` returns 404
  `snapshot_unavailable` and it is not in `history/` — so the doc line is the entire local
  evidence: quote it, say the referenced guidance is not retrievable at this pin, and ask the
  reporter for the #7394 content instead of deriving a launcher from the code.
- **Coverage at the same pin agrees with "preview".**
  `tests/e2e/310p/singlecard/test_dense_model_singlecard.py:20-57` runs only `Qwen/Qwen3-8B`
  (fp16) and `vllm-ascend/Qwen3-8B-W8A8` (`tensor_parallel_size=1, enforce_eager=True,
  max_model_len=16384`), and `rg -l 'Qwen3\.5|Qwen3_5' tests/e2e/310p` is empty: no hybrid model
  was validated on 310P at this release. Absence is quotable as coverage, never as the cause.

### The triton-less hybrid trap (rank it, do not try to prove it)

A GDN / linear-attention family on this SoC reaches a triton kernel that is not installed:

- `vllm_ascend/patch/worker/__init__.py:18-22` imports `patch_triton` **only** `if HAS_TRITON`,
  while `:38` imports `patch_qwen3_5` unconditionally; `patch_triton.py:1-10` is what rebinds
  `causal_conv1d_update` / `causal_conv1d_fn` onto the upstream module.
- `patch_qwen3_5.py:25` binds `causal_conv1d_update` **by name at import time** and calls it on
  both decode paths (`:163` spec decodes, `:194-200` plain decodes); only the prefill branch
  (`:181`) uses the AscendC `torch.ops._C_ascend.npu_causal_conv1d_custom`.
- With triton absent that name is upstream's
  `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:1069`, whose body computes
  `triton.next_power_of_2` / `triton.cdiv` and launches `_causal_conv1d_update_kernel[grid]`
  (`:1189`). Upstream substitutes `TritonPlaceholder`
  (`vllm/triton_utils/importing.py:72-93`), which defines only `jit/autotune/heuristics/Config/
  language` — so the first decode step dies as a **Python** `AttributeError: 'TritonPlaceholder'
  object has no attribute 'cdiv'` (or `next_power_of_2`), with no device op and no plog entry.
- The 310P pytorch fallbacks exist but are **test-only at this pin**:
  `vllm_ascend/_310p/ops/fla/{chunk_gated_delta_rule,fused_gdn_gating,
  fused_recurrent_gated_delta_rule}.py` are imported by
  `tests/ut/_310p/ops/test_chunk_gated_delta_rule_310.py:3` and the two nightly triton op tests,
  and the `is_310p()` registry (`vllm_ascend/utils.py:661-673`) lists only
  fused_moe / activation / conv / layernorm / mm_encoder_attention / rotary / vocab_parallel. A
  pytorch path in the tree is not the same as a wired one.

Labels: the family×SoC status is source-verified (doc line plus coverage); the AttributeError
chain is source-verified code but **not confirmed** for #9376, whose log was never readable. Rank
it first with "send the traceback's deepest frame" as its check, not as the answer.

### Then the config defects in the paste (one paragraph, not a section)

- `--compilation-config '{"custom_ops":["none","+rms_norm","+rotary_embeding"]}'` is the 300I
  tutorial line with a typo. `+<name>` passes the only syntax check
  (`vllm/config/compilation.py:889-896`), and `custom_op_log_check` (`:1099-1140`) then logs
  **once**, at warning level, `Op 'rotary_embeding' doesn't exist (or wasn't imported/
  registered), enabling with '+rotary_embeding' has no effect`. So it does not fail the launch;
  it quietly leaves rope on the default path — and `+rotary_embedding` could not have moved rope
  off ATB on 310P anyway (the `TransData` section below).
- On 310P the user's list is still the only one in play at this pin
  (`vllm_ascend/platform.py:447-449` overwrites `custom_ops = ["all"]` only for non-310P), so
  the typo is the whole custom-op configuration — and `tests/ut/test_platform.py:635` pins that
  behaviour.
- `--tensor-parallel-size 2` on a 4B model and `--max-model-len 4096` are not errors here; TP>1
  only brings in the capture-count limit above. Say so rather than ranking them as causes.

## At v0.22.1rc1: the `_310p/` subpackage, the doc's validated table, two traps

Verified at vllm-ascend **v0.22.1rc1** (`5f6faa0`) + vllm **v0.22.1**
(`0decac0d`) while triaging #12015 (`v0.22.1rc1-310p-openeuler` image); five
distinctive log strings of that paste appear in the pinned tree with line numbers
within ±1 of the log's own `[file:line]` tags (`platform.py:60-61` breakable
cudagraph, `:526` eager-mode default, `:759` allocator injection, `:1036-1037`
GPU-only parameter, `ascend_config.py:772` Dynamic EPLB), so the pin is citable
for that release even though the image is not byte-verified.

- **The 310P code is now a subpackage.** `vllm_ascend/_310p/` holds
  `worker_310p.py`, `model_runner_310p.py`,
  `attention/{attention_v1,attention_mask,metadata_builder}.py`,
  `kv_block_zeroer.py`, `quantization/modelslim_config.py`, `ops/…`. The
  substitutions are `platform.py:651-652` (`worker_cls =
  "vllm_ascend._310p.worker_310p.NPUWorker310"`), `:807-810`
  (`AscendAttentionBackend310`) and `:197-201`
  (`AscendModelSlimConfig310`, which also prints the
  `quantization method 'ascend' … will be overwritten` warning). The v0.11-style
  `worker/model_runner_v1.py` line numbers in the gate table above no longer
  exist at this ref — re-grep inside `_310p/`.
- **The doc's validated table is narrow.** `docs/source/tutorials/hardwares/310p.md:95-176`
  validates only `Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310-vllm` (TP1, TP1, TP4),
  every row carrying `--dtype float16 --quantization ascend --load_format
  sharded_state --no-enable-prefix-caching --max_model_len 16384` (20480 for 32B)
  `--gpu_memory_utilization 0.90 --max_num_seqs`. `:86-87` warns that the
  attention mask is `O(max_model_len^2)` and asks for an explicit, conservative
  `--max-model-len`. A paste serving anything else — a VL checkpoint, an fp16
  dense model, prefix caching left on (the doc passes
  `--no-enable-prefix-caching`; #12015's own config echo says
  `enable_prefix_caching=True`) — is outside the validated set: say so, and make
  "run the doc's row on a small model" rung 1.
- **Trap: the fusion flag in those very doc commands is inert on 310P.** All three
  rows pass `--additional-config '{"ascend_compilation_config":
  {"fuse_norm_quant": false}}'`, but
  `vllm_ascend/compilation/graph_fusion_pass_manager.py:54` and `:69` read that
  key with `and not is_310p()`, so the pass is never added here. The
  `Enabled custom fusions: norm_quant, act_quant` line is **upstream**
  (`vllm/config/compilation.py:313`) and reports
  `compilation_config.pass_config`, which the upstream *optimization level* fills
  (`vllm/config/vllm.py:99-107` `enable_norm_fusion`, the O2 table at `:188-…`,
  default level `:353`, applied at `:1110`) — not the `additional_config` key. Under
  `--enforce-eager` the mode is `NONE` (`platform.py:525-528`) and no pass runs at
  all. So "the flag I set is not what the log lists" is a two-registry artifact,
  not a smoking gun: do not walk it (#12015 spent ~8 calls on it and ran out of
  output before answering).
- **Trap: the 310P mask allocation is real, but it is sized by `max_model_len`.**
  `_310p/attention/metadata_builder.py:80` constructs
  `AttentionMaskBuilder310(self.device, max_model_len)` and
  `_310p/attention/attention_mask.py:68` allocates
  `torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)`
  (class default `max_seqlen = 16384`, `:38`), and `gen_causal_additive_mask`
  (`:66-70`) holds a `bool` `tril`, its `~tril` complement and the fp16 mask at
  once — **≈4·N² bytes per rank**, so `--max-model-len 100000` is ~40 GB peak on a
  43 GiB die before any weight (60 K → ~14 GB, 32768 → ~4.3 GB). That is what the
  doc's `:86-87` warning is about, and it is an argument for an explicit
  `--max-model-len` — never a licence to read a **CPU** `torch.zeros` top frame in
  a segfault dump as this allocation
  (`../hccl-and-hardware/worker-segfault-no-python-frames.md`,
  `## The startup variant`).
- **Trap: the 310P KV budget charges device-global HBM, so a small model can be told
  there is no room.** `_310p/worker_310p.py:96-102,106,128-130` derives
  `non_torch_memory_cleared_by_empty_cache` from `mem_get_info()` (total − free −
  reserved, i.e. every process on the die) and subtracts it *in addition to* the
  instance-local `non_kv_cache_memory` (`#14274`: `Available KV cache memory: 0.09
  GiB (halved for workspace)` for Qwen3-0.6B at `gmu=0.5` with ≈23 GiB free). The
  `(halved for workspace)` suffix identifies this override; `gpu_memory_utilization`
  is a per-instance limit, so raising it is only a bounded workaround, and
  `--kv-cache-memory` is inert on 310P even though the inherited log recommends it →
  `../runtime-memory/kv-budget-per-instance-vs-device-global.md`.

## At v0.22+/v0.23: a quantized model that loads and then dies in an op

When the weights load and the failure is an operator-level dtype sentence
(`Bias dtype should be INT32, actual dtype is DT_FLOAT16`) rather than a Python
traceback, the owning variable is the **checkpoint conversion**, not this page's
gate table: 310P needs its own `-310`/`-310P` artifact, and the `_310p/`
`W8A8_DYNAMIC` scheme passes the model's fp16 bias into `npu_quant_matmul` while
the static/SC schemes pass the checkpoint's int32 `quant_bias` —
`310p-quant-checkpoint-soc-match.md` (v0.23.0 line set, plus the doc trap that
the 27B tutorial's 300I tab points at the generic `-w8a8` repo).

## The `TransData` / `EZ3002` rejection at `profile_run` (the bf16 form)

Verified at vllm-ascend **v0.9.1rc1** (`c30ddb8`) with vllm **v0.9.1** (`b6553be`),
recorded for #1383 (310P3, `Qwen/Qwen2.5-0.5B-Instruct`, no `--dtype` so the
checkpoint's bf16 is used, V0 engine, `profile_run` at
`determine_num_available_blocks`). The block is long; only four lines decide it:

```
EZ3002: Optype [TransData] of Ops kernel [AIcoreEngine] is unsupported. Reason:
  [tbe-custom]:op type TransData is not found in this op store.
  [Dynamic shape check]: The format and dtype is not precisely equivalent to
  format and dtype in op information library
Optype [TransData] of Ops kernel [aicpu_ascend_kernel] is unsupported. Reason:
  Transdata op, groups should be greater than 1, but now is 1.
build op model failed, result = 500002
```

Read it as a **dtype/format rejection**, not as a missing kernel: the op and the
engine were found and the *input combination* was refused. The missing-kernel shape
is different text — `EZ9999 … Parse dynamic kernel config fail` /
`AclOpKernelInit failed opType` / `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE failed` →
`kernel-package-vs-card.md`. The `aicpu … groups should be greater than 1` line is the
CPU fallback's own complaint about the same node: a red herring, not a second cause.

Why bf16 is cause 1 at this pin (all citable):

| Fact | Where |
|---|---|
| The checkpoint's dtype reaches the device: the engine line prints `dtype=torch.bfloat16` and the tree has no 310P dtype gate | the report's own `Initializing a V0 LLM engine …` line |
| On 310P the rope **always** takes the ATB fallback: the custom-op branch is `custom_rotary_embedding_enabled(...) and not is_310p()`, else `torch_npu._npu_rotary_embedding` — where the TransData is inserted | `vllm_ascend/ops/rotary_embedding.py:52`, fallback `:69-76` |
| So the tutorial's `custom_ops: ["+rms_norm","+rotary_embedding"]` **cannot** move rope off ATB here — "add the flag" is not an action, and the doc/code disagreement is a finding | same file, plus `docs/source/tutorials/single_node_300i.md:175` |
| The same tag's tutorial states the constraint outright | `single_node_300i.md:175`: `dtype="float16", # IMPORTANT cause some ATB ops cannot support bf16 on 300I series` (repeated `:224`, `:274`) |
| Precedent on the same SoC | #752 (310P3, CANN 8.0.0, Qwen3): `RopeOperation_0 Atlas inference products can not support bf16` / `setup check fail` |

Ladder: `--dtype float16` first (with `--enforce-eager`, the tutorial's other
mandatory flag); then the image/kernel line — `Dockerfile.310p:18` bases on
`cann:8.1.rc1-310p-…` while the generic `Dockerfile:18` bases on `…-910b-…`, so a
generic image plus a missing `Ascend-cann-kernels-310p` is cause 2; then
`ASCEND_LAUNCH_BLOCKING=1` + `ASCEND_GLOBAL_LOG_LEVEL=1` for the node's real
dtype/format. State the doc contradiction as it stands at the pin:
`docs/source/faqs.md:18` still says 300I/310P "might be supported on 2025.Q2" while
`single_node_300i.md` ships a working fp16 recipe — experimental channel, not a
status answer.

## Ranked actions for a 310P startup report

1. Confirm the SoC and the real versions: `python -c "from vllm_ascend import
   _build_info; print(_build_info.__soc_version__)"`, `pip show vllm vllm-ascend`,
   `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`,
   `cat /usr/local/Ascend/driver/version.info`, `npu-smi info`. The image pins
   CANN `8.3.rc2-310p`; a host driver older than that pairing is a real candidate.
2. Apply the 300I environment (`PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`),
   `--enforce-eager`, `--dtype float16` — the tutorial treats all three as
   mandatory, and the reporter usually has only the last two. Write an explicit
   value: `unset` re-enables vllm-ascend's `expandable_segments:True` injection, and
   on 310P (v0.23.0rc1 `-310p` image, CANN 9.1.0-beta.1, torch_npu 2.10.0.post2)
   that value crashed every rank in `torch.npu.empty_cache()` with `unmapHandles …
   507899` while `expandable_segments:False` deployed (reporter-confirmed #12534 —
   `../runtime-memory/allocator-expandable-segments-unmap-crash.md`). Also quote the
   graph budget when TP > 1: `FULL_DECODE_ONLY` is supported but "the number of
   capturable graphs is limited and depends on the model depth … Qwen3-32B can
   capture and replay 2 graphs", with no limit at TP=1
   (`docs/source/tutorials/hardwares/310p.md:20-30` at v0.23.0rc1) — so a
   "2 cards fine, 4 cards stall" report takes `--enforce-eager` and
   `cudagraph_capture_sizes: [1]` before any HCCL theory. The same stall across a
   **different checkpoint** retires every model lane (`#12985`: Qwen3.5-35B-A3B,
   then Qwen3-VL-32B, reporter asking "是硬件的问题么" — the answer is these
   documented platform limits, not a broken card). The cheapest discriminator is
   already in the report: the reporter's own *working* command, usually a
   screenshot whose OCR carries it (`../evidence-workspace/index.md`). The recorded
   pair differed on exactly three axes — `--enforce-eager` + 60 K
   `--max-model-len` + `gpu_memory_utilization 0.4` (worked) vs
   `FULL_DECODE_ONLY` + `enable_npugraph_ex` + 100 K + 0.6 at TP 8 (stalled behind
   the 60 s `shm_broadcast` warning) — so diff that pair instead of theorising from
   the pasted traceback. Co-tenancy (`mindie_llm_backend … --npu_device_id 2`
   beside `Worker_TP0..3` in that screenshot's `ps`, while `npu-smi` util% reads
   idle) is visible only there, never in `npu-smi`.
3. Shrink the budget (`--max-model-len`, `--gpu-memory-utilization`) before
   blaming the platform: 310P cards are small and the documented KV/profile
   arithmetic in `../startup-failures/max-model-len-budget.md` applies unchanged.
4. Then the documented-status sentence: v0.11.0rc2-310P is experimental, the
   project itself points at v0.10.0rc1 for 300I, and `#3316` / `#2795` are the
   open known issues the tutorial cites. Say explicitly that this is the
   *documented* state and not a diagnosis of the reporter's host.
5. Only then propose code ownership, and name the gate you think is responsible
   from the table above.

## Limits

- The snapshot carries **one commit** per repo (`mode: one_commit_git`), so
  "which commit introduced this between v0.10.0rc1 and v0.11.0rc2" is
  unanswerable here; say so instead of guessing. `SOURCES.json` also marks the
  reported tag `v0.11.0.rc2-310P-openeuler` as `no_named_release` — the pinned
  reference is v0.11.0rc3.
- Title/command mismatches (7B in the title, 3B in the command) are common in
  this family; ask which one failed before reasoning about memory.
