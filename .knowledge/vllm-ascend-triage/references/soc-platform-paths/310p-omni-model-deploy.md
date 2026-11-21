# Deploying a vllm-omni model (stage-based / TTS / audio-out) on 310P

Scope: the question is *"310P 上能用 `<quay.io/ascend/vllm-omni:tag>` 这个镜像部署
Qwen3-TTS / Qwen3-Omni / 语音类模型么?"* — an omni-modality or TTS model, a vllm-omni
image tag, a 310P (Atlas 300I Duo) host, and no crash yet. The deliverable is three
things in this order: **there is no prebuilt 310P omni image** (one artefact read),
**the route that does work is a source build with `SOC_VERSION=ascend310p1`**, and
**the deploy-config contract** (per-stage fp16 + `enforce_eager`), which is what the
reporter actually needs to move. Not this topic: a stage layout that is memory-
infeasible on A2 (`../deploy-sizing/card-budget-and-device-visibility.md`), the
`--omni`/graph-mode flag not appearing in any tree
(`../startup-failures/graph-mode-not-applied.md`), or a 310P model that is in-tree and
crashes (`atlas-300i-310p.md`, `310p-sibling-class-drift.md`).

Recorded case: **#10303** (pin vllm-ascend **v0.20.2rc1** `367b8e62`, vllm **v0.20.2**
`bc150f50`; vllm-omni absent from `SOURCES.json`). The reporter's later public comments
of 2026-07-28 are the source of the working recipe; label them reporter-reported, not
source-verified.

## Step 1 — settle the artefact question from the attachment, not from tag lore

The post's own screenshot is the vendor install page; its `.ocr.txt` under
`/workspace/attachments/<issue>/` reads as a table **`name | image-tag | image-tag-a3 |
Hardware Atlas A2 | Atlas A3 | OS Ubuntu`** with "verified on 4 x NPUs". That is the
whole answer to "is there an omni image for 310P": **no — the project publishes A2 and
A3 variants only**, and the unsuffixed `image-tag` is the A2 build (suffix rules:
`../image-dependency-faults/image-arch-and-tag-availability.md`).

Read that OCR first. In the recorded run a 22-call session reconstructed the same
conclusion from another issue's `collect_env` (`镜像：vllm-omni:v0.18.0（from
quay.io/ascend/vllm-ascend:v0.18.0rc1）`) and never opened the attachment; the
attachment was one call and stronger. Asking the reporter to run
`docker buildx imagetools inspect quay.io/ascend/vllm-omni:<ver>-310p` is a fine
*rung*, but state the doc fact first — a reporter on an air-gapped host cannot run it.

## Step 2 — the build route that was reported working on 310P

Reported 2026-07-28 on one 310P die, Qwen3-TTS-12Hz-1.7B-Base, image built locally as
`qwen3-tts-310p:v0.22.0` from `Dockerfile.qwen3_tts_310p`:

1. **Base on a 310P CANN image**, not a generic one:
   `FROM quay.io/ascend/cann:9.1.0-beta.1-310p-openeuler24.03-py3.11`. A generic
   `cann:…-910b-…` / `…-a3-…` base is the wrong kernel package whatever the tag says.
2. **vLLM core first, CPU-targeted:** `VLLM_TARGET_DEVICE=empty pip install -e
   /vllm-workspace/vllm/[audio]` (the `[audio]` extra is what the TTS stages import),
   then `pip uninstall -y triton`.
3. **vllm-ascend second, with the 310P build stamp baked in:**
   `SOC_VERSION=ascend310p1 COMPILE_CUSTOM_KERNELS=1 pip install -e
   /vllm-workspace/vllm-ascend/`, then `pip uninstall -y triton-ascend triton`. The
   value is the tree's own documented string at this pin (`setup.py:143` "Atlas 300I:
   export SOC_VERSION=\"ascend310p1\"", map `setup.py:169-176` → `_310P`). Without it,
   a build on a host with no visible card raises the `Could not determine chip type
   automatically via 'npu-smi'` `RuntimeError` (`setup.py:134-147`) — i.e. an image
   build must pass it explicitly; the auto-detect path needs `npu-smi`
   (`device-type-build-vs-runtime.md`, step 3).
4. **vllm-omni last**, from the revision that carries the fix the reporter pointed at
   (their text: "vllm-omni main 已合入 PR #4283"; the earlier `#2323`/`#2353` pair is a
   weight left off the NPU): `pip install -e /vllm-workspace/vllm-omni/
   --no-build-isolation`.
5. Host-side pre-download of the three trees (`COPY build-deps/…`) and
   `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.22.0` — the offline/fetch leg is
   `../image-dependency-faults/offline-source-build.md`; a source build of vllm-ascend
   pulls catlass plus four cmake fetches unless they are pre-seeded.

Two `pip uninstall triton` lines are not noise: the 310P document states the card
"does not support `triton` or `triton-ascend`" (`docs/source/tutorials/hardwares/310p.md`
`:13-21` at v0.18.0), and the triton-less hybrid trap in `atlas-300i-310p.md` is the
decode-step form of the same rule.

## Step 3 — the deploy-config contract (the part a launcher cannot guess)

The serve line is `vllm serve <model> --deploy-config <stage>.yaml --host … --port …
--served-model-name … --trust-remote-code --dtype float16 --omni`. `--deploy-config`
and `--omni` exist in **no** tree in this snapshot (vllm-omni is absent, and `--omni`
is not an upstream flag), so the yaml is evidence the reporter must supply — never
synthesize it from the code. Ask for theirs, and compare it against the known-good
starter `templates/omni-310p-deploy-config.yaml` (the recorded 310P Qwen3-TTS config):

- **Two stages, both pinned to the same device** (`devices: "0"`), one model loaded
  twice: a talker/thinker stage and a codec/decoder stage.
- **`dtype: float16` in every stage and in the `platforms.npu.stages` override** —
  310P is fp16-only; a bf16 stage dies as `EZ1001 … DT_BF16 not support in current soc
  version` (the 310P variant of `atlas-300i-310p.md`'s TransData section).
- **`platforms.npu.stages` is the SoC-conditional hook**: it re-states `dtype` and
  `enforce_eager: true` per stage. This is the `platforms.npu` merge that
  `../deploy-sizing/card-budget-and-device-visibility.md` had to leave open; a reporter
  whose yaml has no `platforms:` block is running the A2/A3 defaults on 310P.
- **`enforce_eager: true` on the decode stage** plus `enable_prefix_caching: false`,
  `max_num_seqs: 4`, `gpu_memory_utilization: 0.35` per stage (two stages share one
  die — the per-device gmu sum, not one stage's value, is the budget).
- **An explicit `max_model_len`** (4096 talker / 65536 codec in the recorded yaml): the
  310P attention mask is `O(max_model_len²)` fp16→FRACTAL_NZ, so the unset default is
  the documented OOM (`atlas-300i-310p.md`, the mask paragraph).
- **A streaming codec connector**: `SharedMemoryConnector` with
  `codec_streaming: true`, `codec_chunk_frames`, `codec_left_context_frames`,
  `initial_codec_chunk_frames`, `decode_cudagraph_capture_sizes`, fed by
  `connectors:` + `output_connectors:`/`input_connectors:` and `async_chunk: true`.
  For a *streaming audio* complaint this block, not the model, is the variable.
- **Runtime env in the container**: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
  and `nnal/atb/set_env.sh`, `ASCEND_RT_VISIBLE_DEVICES=<die>`,
  `HCCL_OP_EXPANSION_MODE=AIV`, `OMP_NUM_THREADS=1`, jemalloc preloaded
  (`LD_PRELOAD=/usr/lib64/libjemalloc.so.2`) — a driver/toolkit mount list and the env
  contract are in `../image-dependency-faults/launch-env-and-library-path.md`.
- Readiness: poll `GET /v1/models`; an omni multi-stage start is slow (the recorded
  script waits up to 10 min) and a stage that never binds is a
  `docker logs` line, not a 310P verdict.

## Answer shape

1. **First sentence: no prebuilt 310P omni image** (doc table from the attachment), so
   the tag in the question cannot be the answer; the plain tag is an A2 build.
2. **Then the route**: source build on a `-310p` CANN base with
   `SOC_VERSION=ascend310p1`, install order vllm → vllm-ascend → vllm-omni, triton and
   triton-ascend removed.
3. **Then the deploy-config contract** above, offered as a starter file plus "send
   yours and I will diff it".
4. **Constraints with their error text**, so the reporter maps their own failure:
   bf16 → `EZ1001`; missing stamp → `check_ascend_device_type()` assertion
   (`device-type-build-vs-runtime.md`); no `platforms.npu` block → A2 graph/eager
   defaults on a card that cannot take them; unset `max_model_len` → mask OOM.
5. **Gaps as gaps**: vllm-omni is not in the snapshot, so `--omni`, `--deploy-config`,
   `platforms.npu` merging and `SharedMemoryConnector` cannot be source-verified; the
   recipe is reporter-reported at a later version (vllm 0.22.0-class build) than the
   pinned vllm-ascend v0.20.2rc1, and the reporter's last comment in the corpus is a
   screenshot with no readable text, so end-to-end success is the reporter's claim, not
   a verified result.

## Limits

- One 310P die, one model family, one image build: do not transfer `devices: "0"`,
  the 0.35 gmu or the codec frame numbers to another model or to TP>1.
- The `-310p` CANN base and the `9.1.0-beta.1` driver pairing are the reporter's, not a
  project-published recipe; check the pairing before blaming the yaml.
