# Dockerfiles and image environment: what a container inherits, and which NPUs it can see

## Scope

**Family A — an inherited variable.** The report is about the *image*, not the model: a runtime
variable the docs assume
(`LD_LIBRARY_PATH`, `LD_PRELOAD`, `HCCL_*`, `MOONCAKE_*`) is missing when the container is started
non-interactively (`docker run <image> python -m vllm.entrypoints...`, `docker exec <c> bash -c ...`, a
k8s `command:`, systemd, or any entrypoint that is not an interactive login shell). The service then
fails to load a library or falls back to a different allocator. Worked case: #12059, filed as a
"Mooncake issue" but owned by the vllm-ascend `Dockerfile*`.

A second family shares the same owner — the run command, not the code: **which NPUs the container can
use at all** (`ASCEND_VISIBLE_DEVICES` at `docker run`, `ASCEND_RT_VISIBLE_DEVICES` inside), §B below.

A third family is the same owner again but the opposite failure: the image **never installed** something
the wheel needs, so an import aborts with a glibc heap error rather than a missing variable — `libjemalloc2`
absent from `v0.12.0rc1` and Mooncake's `from mooncake.store import …` dying with `corrupted size vs.
prev_size`, §C below.

Load this page when the report quotes a Dockerfile line, an image tag plus `echo $SOME_VAR`, says
"works in `bash -ic`, empty in `bash -c`", or shows a `docker run --device /dev/davinciN` line next to
"the model will not load while another container is running". For the image's CANN base see `version-pairing.md`
(per-image CANN base), which also explains why the `Dockerfile*` files are authoritative evidence at a
tag.

## Code facts (verified)

- Eight variants exist at `main`: `Dockerfile`, `Dockerfile.310p`, `Dockerfile.a3`, `Dockerfile.a5`
  plus `.openEuler` versions. Each starts from `quay.io/ascend/cann:<ver>-<arch>-ubuntu22.04-py3.12`
  (or `…-openeuler…`), e.g. `Dockerfile.a3:18` at `v0.22.1rc1`.
- Every variant appends **both** a jemalloc preload and a library path to the shell profile, never as
  `ENV`:
  `RUN echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" >> ~/.bashrc`
  and, under the comment "Append `libascend_hal.so` path (devlib)",
  `RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> ~/.bashrc`.
  The `.openEuler` variants use `/usr/lib64/libjemalloc.so.2`. Verified line numbers:
  `v0.18.0` `:98-99`; `v0.19.1rc1` `:78-79`; the reported build commit `05e095a20` `Dockerfile.a3:77-78`;
  `v0.22.1rc1` / `origin/releases/v0.22.1rc` `Dockerfile:83/84`, `Dockerfile.a3:85/86` (base image
  `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.12` at the reported commit),
  `Dockerfile.a5:77`, `Dockerfile.310p:67`; `v0.26.0rc2` `:92-93`. Quote the variant and the ref — the
  numbers drift by a few lines between tags.
- `Dockerfile*` sets `ENV` only for `DEBIAN_FRONTEND`, `SOC_VERSION`, `TASK_QUEUE_ENABLE`,
  `OMP_NUM_THREADS` and `PATH` — **there is no `ENV LD_LIBRARY_PATH`/`LD_PRELOAD`**. The two settings
  reach a process only if something reads `~/.bashrc`.
- The files end with `CMD ["/bin/bash"]`, so a bare `docker run <image>` starts a non-interactive bash
  and skips `~/.bashrc`; that is why an interactive `docker run -it` looks fine while `docker exec`,
  a k8s `command:` or a Python `ENTRYPOINT` does not.
- The jemalloc half is **gone** on `main`: `499d08699` "[BugFix] Remove jemalloc LD_PRELOAD from
  Dockerfiles (#17146)", 2026-09-21, a backport of #17145 to `releases/v0.27.1rc`; the PR states the
  reason (globally preloading jemalloc forces every process in the image onto it and can cause
  allocator compatibility issues) and keeps the packages installed. First mainline tag containing it:
  `v0.27.1rc1`. `#18392`/`#18414` (`06d7a2441`, `c21d38dfa`, 2026-10-12) removed the same advice from
  the docs.
- The jemalloc half arrived in `5ab6d124e` *"[Doc] Add a perf tune section"* (#5127, 2025-12-19), which
  adds `libjemalloc2` to the `apt-get install` line and the preload line to all five `Dockerfile*`
  variants that exist at that commit (the family grows to eight by `main`); first tag containing it is
  `v0.13.0rc1`, so a `v0.12.0rc1` image has neither and Mooncake aborts at import there (§C).
- The `LD_LIBRARY_PATH` half is **still present** on `main` (2026-10-30): all eight variants append it
  to `~/.bashrc` (`Dockerfile.a3:84` at `main`), and no `ENV LD_LIBRARY_PATH=` line exists. A report
  about `/usr/local/lib` missing from a non-interactive process is therefore *not* fixed by upgrading.

## Why it bites (mechanism)

`~/.bashrc` is read by **interactive non-login** bash only. `bash -c '…'`, `sh -c '…'`, a Python
`ENTRYPOINT` and the container runtime's own `CMD` never read it. On the Ubuntu-based images the
default `/root/.bashrc` also opens with the interactive guard

```sh
case $- in *i*) ;; *) return;; esac
```

so even an explicit `source ~/.bashrc` inside a script returns before reaching the appended lines
(appends land at the end of the file). That is why `bash -ic 'echo $LD_PRELOAD'` shows the value and
`bash -c 'source ~/.bashrc; echo $LD_PRELOAD'` does not. `[hypothesis]` for the guard's presence in
the built image — it is Ubuntu skel content, not repo content, and the image contents are not
inspectable from the clones — falsified or confirmed on the reporter's machine with the two-shell
probe:

```bash
docker run --rm --entrypoint /bin/bash <image> -c \
  'echo "noninteractive:[$LD_PRELOAD][$LD_LIBRARY_PATH]"; \
   bash -ic "echo interactive:[$LD_PRELOAD][$LD_LIBRARY_PATH]" 2>/dev/null; \
   grep -n "case \$-" /root/.bashrc | head -2'
```

`bash -lc 'env | grep LD_'` versus `bash -c 'env | grep LD_'` is the cheaper variant. #12059's own
transcript is the expected shape: `non-interactive: []` and
`interactive: [/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:]`.

The tutorials compound it: `docs/source/tutorials/models/*.md` tell users to `source /root/.bashrc`
before serving (e.g. `GLM5.md:192` at the refs where it appears), which does not work for the reason
above; `tools/mooncake_installer.sh:321-324` has the Mooncake toolchain's own `PATH` line. Quote the
tutorial line when a reporter says "I followed the official steps".

## Deliverable shape for this class

The mechanism is deterministic and needs no NPU, so give a real fix, not a suspect:

- Workaround: set the variables where every process inherits them — `docker run -e LD_LIBRARY_PATH=...
  -e LD_PRELOAD=...`, a `--env-file`, a `Pod` `env:` block, `ENV` in a derived image, or a wrapper
  entrypoint that exports and then `exec`s the server. Removing the jemalloc preload is only right if
  that release does not need it (pre-`v0.27.1rc1` images ship it deliberately); remember the cost:
  jemalloc affects allocator behaviour, so a swapped allocator changes memory-footprint baselines.
- Root cause: the `Dockerfile:<line>` appends (name the variant and ref) plus the startup path that
  never reads `~/.bashrc`; the fix is `ENV`/entrypoint-level setting, and for jemalloc the upstream
  decision was removal (#17146).
- Do not look for an NPU-side defect, and do not attribute the symptom to CANN or to Mooncake: the
  missing variable only changes an allocator or a library search path. Mooncake's own `docker/` files
  in `kvcache-ai/Mooncake` do not add these exports, so a report titled "Mooncake issue" can still be a
  vllm-ascend image defect — check the image the reporter ran, not the title.

## Status of #12059 (read 2026-11-17)

Open, created 2026-07-15, labels `ci/build` + `triaged`, three comments: a "please check this issue", a
contributor's plan, and that contributor announcing a draft PR **#15781** (minimal `Dockerfile.a3`
change; since un-drafted, still not merged as of 2026-10-12). No maintainer mechanism statement exists
in the thread. The jemalloc half has since been fixed on `main` by #17145/#17146; the
`LD_LIBRARY_PATH` half is open. Report both halves separately — "fixed" for the whole issue would be
wrong. Its version signals also conflict: the reproduce command names
`quay.nju.edu.cn/ascend/vllm-ascend:v0.22.1rc1-a3` while `collect_env` reports
`0.19.1rc2.dev849+g05e095a20 (git sha: 05e095a20)` with vLLM `0.24.0`; fingerprint the built commit
first (`../workflow/version-pinning.md`), since both refs carry the appends but at different lines.

## B. The container's device set: `ASCEND_VISIBLE_DEVICES` vs `ASCEND_RT_VISIBLE_DEVICES`

Worked case: vllm-ascend #6942 — image `quay.io/ascend/vllm-ascend:v0.14.0rc1-openeuler`, HDK 25.5,
Kylin V10 SP3; `docker run --device /dev/davinci6 --device /dev/davinci7 …`, then
`vllm serve /data/model/Qwen3-30B-A3B --tensor-parallel-size 2 --enable-expert-parallel`. It serves when
the box is idle and fails to load the model while another container holds other chips.

This is a **run-command/env class, not a code path**: `vllm_ascend/platform.py:101` sets
`device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"` (the only variable vLLM maps ranks through), and
`git grep -n ASCEND_VISIBLE_DEVICES v0.14.0rc1` in vllm-ascend returns nothing — that name is consumed
by the container runtime when the container is created. The reporter-confirmed fix (`reported`) passes
the device list **both** places: `-e ASCEND_VISIBLE_DEVICES=6,7` on `docker run` *and*
`export ASCEND_RT_VISIBLE_DEVICES=6,7` inside; either alone failed for them, and the maintainer's first
advice was the in-container export alone. Full class, verified anchors, hypothesis and falsifier:
`npu-device-visibility.md`. Do **not** answer it with a code path, an `additional_config` key or an
upgrade.

## C. A package the image must carry: `from mooncake.store import …` aborts (`corrupted size vs. prev_size`)

Worked report #5609 — "Mooncake is broken in vllm-ascend v0.12.0rc1 docker image":
`python -c "from mooncake.store import MooncakeDistributedStore"` prints `corrupted size vs. prev_size`
and `Aborted (core dumped)`. Same owner as §A — the image's own build lines — but the other failure
shape: not an unexported variable, a **package and preload missing from the build**.

Verified in `Dockerfile.a3` at the tags:

| ref | mooncake payload | allocator |
| --- | --- | --- |
| `v0.12.0rc1` | `ARG MOONCAKE_TAG=v0.3.7.post2` (`:21`), cloned and built with `-DUSE_ASCEND_DIRECT=ON` (`:35-42`) | **absent**: the install line is `apt-get install -y git vim wget net-tools gcc g++ cmake libnuma-dev` (`:35`) and no `LD_PRELOAD` line exists in the file |
| `v0.13.0rc1` | same `MOONCAKE_TAG=v0.3.7.post2` (`:21`) | `… libnuma-dev libjemalloc2` (`:37`) plus `RUN echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" >> ~/.bashrc` (`:70`) |

The Mooncake payload is identical between the two images, so the only delta is the allocator. It arrived
in `5ab6d124e` *"[Doc] Add a perf tune section"* (#5127, 2025-12-19) — a doc-titled PR that edits all
`Dockerfile*` variants present at that commit; `git tag --contains 5ab6d124e` starts at **`v0.13.0rc1`**, so no
`v0.12.0*` image has it. The same preload was later removed on `main` by #17146/#18392 (code facts
above): pre-`v0.27.1rc1` images ship jemalloc deliberately, so do not tell a `v0.13`–`v0.26` user to drop
it.

**Mechanism.** Mooncake's store is built against the system allocator — at `v0.3.7.post2`
`CMakeLists.txt:42` is `option(STORE_USE_JEMALLOC "Use jemalloc in mooncake store master" OFF)`, and the
Python binding is `mooncake-integration/store/store_py.cpp` — and the import aborts inside glibc's heap
bookkeeping before any inference code runs. Preloading jemalloc replaces the allocator for the whole
process and the abort disappears; that masks an allocation/free mismatch inside the Mooncake build, so
the exact owner is `hypothesis` (the store's own allocator versus an ABI-mismatched dependency cannot be
separated from the clones).

**Answer shape — all three confirmed on the actual image by maintainer `gcanlin` (2026-01-05).**

1. In place: `apt-get update && apt-get install -y libjemalloc2`, then
   `export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD`. Cost: a global
   allocator swap (the §A caveat), which is what the image itself does from `v0.13.0rc1` on.
2. Disable Mooncake in that container:
   `mv /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake{,.disabled}` — reported, and
   it names where the image puts the binding (the toolkit's own site-packages). Cost: no KV
   transfer/pool.
3. Move to `v0.13.0rc1`+ — "v0.13.0rc1 mooncake can work normally".
   Do **not** chase a Mooncake release: both images pin `v0.3.7.post2`.

**Discriminator against a Mooncake defect:** ask for the image tag and the import command; if
`apt list --installed | grep jemalloc` is empty on a `≤ v0.12.0rc1` image, this is the build gap.
`undefined symbol: …` from the same import is a different class — an ABI/wheel mismatch, not this one.

## D. The image's `torch-npu` build is not the pin the extension was compiled against

Report shape (#6250): image `vllm-ascend 0.13.0rc2`, Qwen3-Next-80B-A3B-Instruct on Atlas 800I A2 (TP4,
`FULL_DECODE_ONLY`, `--max-num-batched-tokens 4096`), and the engine dies **during warm-up** — `capture_model`
→ `worker/model_runner_v1.py _dummy_run` / `_generate_dummy_run_hidden_states` → the model block through
`compilation/acl_graph.py:113`. Nothing fails at weight load, no `vllm_ascend/` frame is deeper than the
runner, and the report's own exception text is truncated, which is why the thread — not the traceback — is
the deliverable here.

**Verified at `v0.13.0rc2`:** `pyproject.toml:21` declares `"torch-npu==2.8.0"` (the dependency the
extension's custom ops are built and validated against) while `Dockerfile:76` — and `Dockerfile.310p:61`,
`Dockerfile.310p.openEuler:56` — hard-code
`pip install modelscope 'ray>=2.47.1,<=2.48.0' 'protobuf>3.20.0' 'torch-npu==2.8.0.post1'`. So every rc2 image
ships a `torch_npu` that was never the build target. The fix is #6287 ("[CI] Update pta to 2.8.0.post2",
**merged**), whose diff deletes that pip install from `Dockerfile`, `Dockerfile.310p`, `Dockerfile.310p.openEuler`,
`Dockerfile.a3`, `Dockerfile.a3.openEuler` so the `pyproject.toml` pin governs; `v0.13.0rc3` and `v0.13.0` then
carry `torch-npu==2.8.0.post2` at `pyproject.toml:21` and no Dockerfile install line. That is a
**tree** comparison across three tags, which is stronger than an ancestry query and cheaper.

Maintainer answer (`wangxiyuan`, 2026-01-26, posted in the *linked* issue **#6242**, not in the report's own
thread) — the report only says "see #6242 comment 3797795676", so read the linked thread's comments, not just
the report's:

1. `pip install torch-npu==2.8.0` inside the container (matches the pin the wheel was built against); cost:
   loses the image's `post1` fixes;
2. edit `pyproject.toml` to `2.8.0.post1` and `cd /vllm-workspace/vllm-ascend && pip install -v -e .`; cost: a
   full local build inside the CANN container;
3. use the image rebuilt for the release: `quay.io/ascend/vllm-ascend:releases-v0.13.0` (or `-a3`), which ships
   `2.8.0.post2`; cost: a whole-stack move.

Mechanism (`hypothesis` on the fault site, `verified` on the pin skew): a `.post` bump is a *rebuilt wheel*
over the same version, and the extension's dispatcher is bound at build time, so the first op it runs inside
the captured region faults — hence a crash at `compile_or_warm_up_model` rather than at import or weight
load. Where exactly it lands is not the discriminator; the image's `torch-npu` version against the tag's
`pyproject.toml` is. Falsifier: `pip show torch-npu` inside the container reports `2.8.0` while
`grep torch-npu pyproject.toml` reports the same, and the crash persists.

One comparison settles it per tag:

```bash
R=/workspace/repos/vllm-project/vllm-ascend
for t in v0.13.0rc2 v0.13.0rc3 v0.13.0; do echo "== $t"; \
  git -C $R show $t:pyproject.toml | grep -n 'torch-npu'; \
  git -C $R show $t:Dockerfile | grep -n 'torch-npu'; done
```

Treat any `pip install … 'torch-npu==X.postN'` in a `Dockerfile*` whose `pyproject.toml` asks for a different
`X` as this defect, and say which tag first makes the two agree — do **not** answer with a CANN/HDK
statement: the CANN base of that image is the tag's own `FROM` line (§1) and is not what disagrees here.

## Where to look first next time

1. `git show <ref>:Dockerfile*` and `git grep -n 'bashrc\|^ENV ' <ref> -- 'Dockerfile*'` — the ground
   truth for what the image sets. Grep every variant, not just `Dockerfile`: 310P/openEuler images
   diverge in the library path (`/usr/lib64` vs `/usr/lib/$(uname -m)-linux-gnu`) and in which
   variables they set.
2. The docs at the same ref: tutorials that instruct `source /root/.bashrc` are the propagation path
   the reporter copied.
3. Whether the neighbouring repo matters: `kvcache-ai/Mooncake`'s `docker/` files do not add these
   exports, so check the image the reporter actually ran.
4. For the device-set family (§B): `git grep -n 'ASCEND_RT_VISIBLE_DEVICES\|ASCEND_VISIBLE_DEVICES'
   <ref> -- vllm_ascend` — which of the two names this build reads at all, and whether
   `platform.py`'s `device_control_env_var` is what the reporter set.
5. For a package the image must carry (§C): `git show <ref>:Dockerfile.<variant> | grep -n -i
   'jemalloc\|MOONCAKE_TAG\|apt-get install'` — a preload line without the package (or the reverse)
   is the build half, and the `MOONCAKE_TAG` tells you whether the payload changed at all.
6. For a wheel whose build the tag disagrees with (§D): `git show <ref>:pyproject.toml | grep -n 'torch-npu'`
   beside `git show <ref>:Dockerfile | grep -n 'torch-npu'` — a `pip install … 'torch-npu==X.postN'` in the
   image whose `pyproject.toml` asks for `X` is the defect, and the first tag where they agree is the fix.

## Check before reuse

```bash
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:Dockerfile | grep -n 'bashrc\|^ENV'
git -C /workspace/repos/vllm-project/vllm-ascend log --oneline -S 'libjemalloc.so.2:$LD_PRELOAD' \
  -- 'Dockerfile*'
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'ASCEND_RT_VISIBLE_DEVICES' <ref> -- vllm_ascend
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:Dockerfile.a3 | grep -n -i 'jemalloc'
git -C /workspace/repos/vllm-project/vllm-ascend tag --contains 5ab6d124e --list 'v0.*' | sort -V | head -3
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:pyproject.toml | grep -n 'torch-npu'   # §D
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:Dockerfile | grep -n 'torch-npu'       # §D
```

The appends, the `CMD` line and the `ENV` set change between releases; re-read them at the reported ref
rather than quoting this page's line numbers.
