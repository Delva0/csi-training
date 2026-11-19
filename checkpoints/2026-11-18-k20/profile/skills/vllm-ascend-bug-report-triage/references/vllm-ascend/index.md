# vllm-ascend: repository and release specifics

## Scope

Concrete anchors for triaging `vllm-project/vllm-ascend` bug reports and the upstream
`vllm-project/vllm` it plugs into: where each serving feature lives, which vLLM/torch-npu/CANN
release a tag pairs with, how `--additional-config` flags change meaning across releases, and which
subsystem a symptom points at. Method lives in `../workflow/index.md`; failure families in
`../symptoms/index.md`.

Versions and line numbers were verified against the local clones at `main` = `1bd23a0a9`
(2026-10-30, 6301 commits, tags `v0.20.2rc1` … `v0.30.0`, paired vLLM `v0.32.0`). They anchor the
search; they are not a contract.

## Common rules

- **Reports name a release tag; the clone is far ahead of it** — 1900–2400 commits ahead of the rc
  tags analysed so far. Resolve and read at the tag (`../workflow/version-pinning.md`) before
  trusting any path or line number here. This topic says *where* to look, not what a tag contains.
- **Flags in `--additional-config` are read and derived in `vllm_ascend/ascend_config.py`.** A key's
  presence does not mean it has an effect: check the derivation expression at the tag (many keys are
  `key and use_sparse`, `key and is_prefill_node`, or replaced by another config field entirely).
- **Module paths are not stable across releases.** Search for the symbol at the ref
  (`git grep -n -F '<symbol>' <ref> -- '*.py'`) instead of trusting a remembered path.
- **Tutorial docs under `docs/source/tutorials/models/` are what reporters copy verbatim** under the
  label "official configuration". Compare their command lines at the reported tag.
- **The same fix often appears twice** — a `main` commit plus a `[Cherry-pick]`/`[BugFix][vX.Y]`
  commit on `origin/releases/vX.Y.Z*`. Cite the pair once with the branches it landed on.
- **Repo-local skills are not harness skills.** `$REPO/.agents/skills/` contains
  `vllm-ascend-release` (release checklist, issue scan, test-coverage scripts) and
  `vllm-ascend-model-adapter` (model bring-up constraints). Both carry
  `disable-model-invocation: true`; `skills_list` does not list them and `skill_view` fails on
  them, so open them with `cat` if their reference pages help. Their scripts drive `gh`, which is
  unauthenticated in this environment — use the `github` tool instead. Read them only when the
  report itself is about a release or model onboarding.

## Routes

- `repo-map.md` — path/symbol → question table for the package, `csrc/`, tests, docs and recipes,
  including quantisation layout per ref. Start here to locate a frame or a flag.
- `code-layout.md` — package layout per release and the verified path drift (what moved or
  disappeared, 310P family, attention backends, parallelism plumbing).
- `version-pairing.md` — vllm-ascend tag → paired vLLM tag, torch/torch-npu/triton-ascend pins, the tag's
  own `Dockerfile*` + `docs/source/installation.md` supported-version table (so a reported CANN version is
  checked, not assumed to be skew), and version-mismatch symptoms.
- `install-and-image.md` — artefact questions: which chip a `Dockerfile*` family targets (910b/A2,
  `-a3`/910C, 310p, a5) per ref and the `AclOpKernelInit`/`aclnn<Op> failed` consequence of the wrong base
  image (#1696, §1); and what building produces (§2) — the `vllm_ascend_C.so` + `libvllm_ascend_kernels.so`
  pair from one CMake `install(TARGETS …)`, why `Failed to import vllm_ascend_C:libvllm_ascend_kernels.so`
  names the module's missing DT_NEEDED, the `COMPILE_CUSTOM_KERNELS` gate (`bool(int(...))`, default `1`,
  removed by `37db0844f`, first tag `v0.12.0rc1`), and the `setup.py` install-prefix / copy-back rule that
  decides whether a source tree keeps the `.so` (#1114); and (§3) the `catlass` git submodule a source
  install from `v0.12.0rc1` on must fetch — the offline-pod build abort with no `vllm_ascend/` frame
  (#8788), with the 310P exemption and the A3 `moe_distribute_base.h` second prerequisite; and (§5) a
  wheel that silently omits a package tree — `setup.py`'s `find_packages`, a parent directory without
  `__init__.py`, the reporter's `dev+g<sha>` branch build whose tree differs from the nearest tag, and the
  two branch fixes (#8686 → #8729 on `releases/v0.13.0`, #9649 on `main`).
- `npu-device-visibility.md` — the device-visibility layer rather than a compute path: which variable
  the container runtime reads (`ASCEND_VISIBLE_DEVICES` at `docker run`) versus the one vLLM and
  torch_npu map ranks through (`ASCEND_RT_VISIBLE_DEVICES`, `platform.py:101`
  `device_control_env_var`, upstream `Platform.device_id_to_physical_device_id`), the reporter-confirmed
  pair that fixed #6942, and the falsifiers that send the report back to a load class.
- `hardware-support-by-release.md` — "does/when will <hardware> support <feature>" questions, answered
  from the *hardware tutorial page at the ref* rather than a feature guide: the per-ref page path, the
  verified 310P timeline (eager-only + fp16 documented through `v0.19.1rc1`; decode-only ACL graph in code
  from `v0.16.0rc1` (`ddc78dbad` #6849) but documented only from `v0.20.2rc1` as `FULL_DECODE_ONLY` with
  the event-id/TP capture limit; torchair gated to `TORCHAIR_MODEL_LIST`, so no graph mode at all for a
  Qwen/VLM on `v0.10.0rc1`), the 310P ModelSlim static-W8A8 path versus the absent dynamic/W4A8 branches,
  and why `--compilation-config '{"custom_ops": …}'` is a no-op on 310P.
- `dockerfile-env.md` — what an image inherits with no shell: the `export ... >> ~/.bashrc` lines in
  the eight `Dockerfile*` variants (jemalloc `LD_PRELOAD`, `LD_LIBRARY_PATH`), why non-interactive
  startup never sees them, which commits removed which half (#17146 first tag `v0.27.1rc1`; the
  `LD_LIBRARY_PATH` half still open), the `-e`/`ENV` workaround (#12059), and (§C) a package the image
  never installed — `libjemalloc2` absent at `v0.12.0rc1`, so `from mooncake.store import …` aborts with
  `corrupted size vs. prev_size` (#5609); and (§D) a `torch-npu` `.post` build the image installs that the
  tag's `pyproject.toml` does not ask for — a crash inside the first captured graph whose lever is the
  image pin, the `#6287` fix and `v0.13.0rc3` (#6250).
- `config-flag-lifetimes.md` — `additional_config` parsing traps (silent-ignore vs reject; Trap D includes
  the `expert_tensor_parallel_size` algebra that makes the key dead under `--enable-expert-parallel` until
  its removal at `v0.10.0rc1`), the
  sparse-C8 / `c8_enable_reshape_optim` / StoreKVBlock history with PR numbers and per-tag states,
  other documented renames and removals, the operator-name instability, and (§6–7) `VLLM_ASCEND_*`
  environment variables: why upstream calls them "unknown", and why `weight_nz_mode` does nothing on
  310P.
