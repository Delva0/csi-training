# Image and build-time reports: Dockerfile env, and wheel-build breakage

Scope: reports whose evidence is what the shipped image or the from-source wheel build does,
not the serving code path. Verified by reading tag-pinned `vllm-project/vllm-ascend` sources,
the bundled `pybind/pybind11` CLI sources, and the recorded report threads; nothing was built or
run on an NPU. Pin the reported tag before quoting any line below.
A wheel-pin defect inside the image (a forced `torch-npu==2.8.0.post1` install whose own
`AclOpCompileInterface.h` cannot compile) is tracked with its version table in
[deepseek-v3.2.md](deepseek-v3.2.md); read it before blaming a Triton kernel or CANN for a
`use of undeclared identifier` compile failure at start-up.
A startup that dies on a hardware/image *variant* mismatch rather than on the build itself
(`socversion Ascend910B1 of bin package does not match ... Ascend310P3`, or from `v0.12.0rc1` the
plugin's own "does not match the installed version's device type") is the wrong tag suffix, not a
build defect: [310p-atlas-300i-duo.md](310p-atlas-300i-duo.md) and the variant matrix in
[model-tutorial-recipes.md](model-tutorial-recipes.md). The A3/910C side has a date boundary the
variant matrix records: no `-a3` image (and no `Dockerfile.a3`) exists before `v0.10.0rc1`, so a
910C server on an earlier bare tag runs the 910B CANN base and dies at the first `aclnn` op.

## `~/.bashrc` is not the environment of the process

Symptom class: "env vars set in `~/.bashrc` are missing on non-interactive container startup"
(`LD_LIBRARY_PATH`, `LD_PRELOAD`, `PATH`), or a KV-transfer / Mooncake / custom-`.so` import that
works in an interactive shell fails under `docker run <image> <cmd>`, `kubectl exec ... -- cmd`,
ssh-with-command, or any supervisor that runs `bash -c`.

- At `v0.22.1rc1` every Dockerfile variant writes both exports into `~/.bashrc` and nothing else:
  `Dockerfile.a3:84-86` (comment "Append `libascend_hal.so` path (devlib) to LD_LIBRARY_PATH"),
  `RUN echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" >> ~/.bashrc`,
  `RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> ~/.bashrc`; the same two
  lines are at `Dockerfile:83-84`, `Dockerfile.310p:67-68`, `Dockerfile.a3.openEuler:81-82`,
  `Dockerfile.a5:77-78`, `Dockerfile.openEuler:81-82` and their openEuler/310P siblings. `CMD` is
  `["/bin/bash"]`, and the Dockerfile itself adds no `ENV LD_LIBRARY_PATH` — the only place the
  path is set for a plain container command is the `~/.bashrc` append.
- Mechanism: bash reads `~/.bashrc` only for non-login **interactive** shells. A container
  command, an exec, and a systemd/ray/launcher child get the image `ENV` plus whatever the caller
  passes, so the appended path is absent. Verify the reporter's own probe before interpreting it:
  `bash -c 'echo $LD_LIBRARY_PATH'` vs `bash -ic`, and `docker exec` vs `docker exec -it`.
- Which export matters is decided by what the image installed, not by the title. At `v0.22.1rc1`
  `Dockerfile.a3:29-43` clones Mooncake (`MOONCAKE_TAG=v0.3.9`), runs
  `tools/mooncake_installer.sh`, then `cmake .. -DUSE_ASCEND_DIRECT=ON && make -j && make install`
  in the Mooncake tree (default prefix `/usr/local`, so `/usr/local/lib`) — the flag is Mooncake's
  optional Ascend-transport switch, default OFF, falling back to Host RDMA/TCP
  ([kv-pool-ascend-store.md](kv-pool-ascend-store.md)) — and the tag's own PD
  tutorial `docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md` states "Ensure
  `/usr/local/lib` and `/usr/local/lib64` are in your `LD_LIBRARY_PATH`" — while the `export` it
  shows next only adds `.../site-packages/mooncake`. So the `LD_LIBRARY_PATH` line carries the
  functional consequence; the `LD_PRELOAD` jemalloc line is a performance knob *at this tag* (its
  lifecycle bullet below is what decides that, and before `v0.13.0rc1` the absence is functional).
- That jemalloc line's lifecycle bounds the report, and it starts later than the image line:
  `5ab6d124e` "[Doc] Add a perf tune section (#5127)" adds `libjemalloc2` to every Dockerfile's apt
  line and appends the `LD_PRELOAD` export, first release `v0.13.0rc1` — so on `v0.12.0rc1` and
  earlier the package and the export are both absent. Its *intent* was performance, but on those
  earlier tags the absence is functional, not just slower (next section). After that, `499d08699`
  "[BugFix] Remove jemalloc LD_PRELOAD from Dockerfiles (#17146)" first ships in `v0.27.1rc1`, and
  the package and every reference are deleted later (`c21d38dfa`, first tag `v0.30.0`; `06d7a2441`
  is main-only at the clone snapshot). At a tag that already carries it, such as the reported
  `v0.22.1rc1`, both exports are present, so "jemalloc preload missing" cannot be the whole cause of
  a Mooncake failure, and a fix that only removes the preload is not a fix for a `LD_LIBRARY_PATH`
  lookup failure. Read the tag before deciding which of the two regimes applies.
- Workaround class: put the variable where the launcher sees it — image `ENV`, `docker run -e`,
  a `/etc/profile.d/*.sh` sourced by `bash -lc`, or an explicit `export` in the start script —
  never `~/.bashrc` alone. Report it as the documented route, not as a reproduced fix: which
  export the reporter's Mooncake path needed was never confirmed.

## Mooncake import aborts inside the image: `corrupted size vs. prev_size`

Symptom class: an image-only probe — `python -c "from mooncake.store import
MooncakeDistributedStore"` prints `corrupted size vs. prev_size` and `Aborted (core dumped)` — while
the serving path is never exercised. That command imports neither `vllm` nor `vllm_ascend`, so the
plugin's code is not in the process: read the Dockerfile's Mooncake block at the reported tag and at
the next release, and quote the reporter's probe. Report it as an image-package defect.

- At `v0.12.0rc1` (`Dockerfile.a3:33-44`) Mooncake is built from `MOONCAKE_TAG=v0.3.7.post2` with
  `bash mooncake_installer.sh -y` and `cmake .. -DUSE_ASCEND_DIRECT=ON`, and no jemalloc package is
  installed. `v0.13.0rc1` differs only by `libjemalloc2` in the same apt line (`:37`) and
  `RUN echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" >> ~/.bashrc`
  (`:70`); the base image (`quay.io/ascend/cann:8.3.rc2-a3-ubuntu22.04-py3.11`), `MOONCAKE_TAG` and
  `tools/mooncake_installer.sh` are byte-identical, so the allocator is the only image delta. That
  change is `5ab6d124e` (#5127, first release `v0.13.0rc1`).
- Reporter-confirmed routes, both from the same thread: `apt install libjemalloc2` plus that
  `LD_PRELOAD` export ("After installing `libjemalloc2` manually, the error disappeared"), or drop
  Mooncake altogether with
  `mv /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake{,.disabled}` — the
  image's Mooncake Python package lives under the CANN toolkit's `site-packages`, not pip's. He also
  rebuilt `v0.12.0rc1` locally from that tag's own `Dockerfile.a3` and reproduced the abort, so the
  tag's package set, not one broken image build, is the variable.
- Mechanism stays a candidate: preloading jemalloc interposes `malloc`/`free` process-wide, so a
  heap-consistency abort raised by glibc's allocator can disappear without the invalid free being
  fixed. Mooncake does not hard-link it either — `STORE_USE_JEMALLOC` defaults `OFF`
  (`CMakeLists.txt:42`, `mooncake-store/src/CMakeLists.txt:57-61` at `v0.3.7.post2`) and the image
  never passes `-DSTORE_USE_JEMALLOC=ON`. Quote the workarounds as reporter-confirmed and the
  interposition as inference; the single check that would settle it is the same probe under
  `MALLOC_CHECK_=0`/valgrind against the unpatched image.
- Which invocation matters: the export lives only in `~/.bashrc` (section above), so an interactive
  `docker exec -it` shell gets the preload while `docker run <image> python -c ...` does not. A
  "`v0.13.0rc1` works normally" comment is evidence about one invocation, not about the tag.

## The image removes a package the plugin imports at module level: `No module named 'triton'` behind `Model architectures [...] failed to be inspected`

```text
ERROR [registry.py:449] Error in inspecting model architecture 'Qwen3NextForCausalLM'
ERROR [registry.py:449] RuntimeError: Error raised in subprocess:
ERROR [registry.py:449]   File "/vllm-workspace/vllm-ascend/vllm_ascend/ops/casual_conv1d.py", line 14, in <module>
ERROR [registry.py:449]     import triton
ERROR [registry.py:449] ModuleNotFoundError: No module named 'triton'
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Model architectures ['Qwen3NextForCausalLM'] failed to be inspected.
```

- Read the last `ModuleNotFoundError` line, not the `ValidationError`: vLLM inspects the
  architecture in a subprocess (`vllm/model_executor/models/registry.py:849 _run_in_subprocess`,
  `:867 returned.check_returncode()`, `:870 raise RuntimeError("Error raised in subprocess:")`,
  `:449 logger.exception("Error in inspecting model architecture ...")` at vLLM `v0.10.2`), so any
  import-time failure in the model's plugin module is re-raised as a `ModelConfig` validation
  error and the outer message never names the package. Those four line numbers reproducing the
  paste is also what pins the pasted tree to vLLM `v0.10.2`, the `ARG VLLM_TAG` of the reported
  plugin tag.
- The file is spelled `casual_conv1d.py` before the `ops/triton/` move — grep both spellings.
  At `v0.10.2rc1` (issue #3029, Qwen3-Next-80B-A3B, TP8, driver 25.5.0) `casual_conv1d.py:13-14`
  carries bare `import triton` / `import triton.language as tl`, reached from
  `vllm_ascend/models/qwen3_next.py:65` (the same model module also imports
  `vllm_ascend/ops/fla.py`, whose `:11` bare-imports triton); `d5f77f14d` ("mkdir triton package
  and move triton files", #4420) moves it to `vllm_ascend/ops/triton/mamba/causal_conv1d.py`,
  first tag `v0.12.0rc1`. `vllm_ascend/ops/__init__.py` did not import those files at
  `v0.10.2rc1`, so only models whose module does (Qwen3-Next/GDN hybrids) fail — a dense model on
  the same image starts. The recorded forwards and their tag boundaries are enumerated in
  [qwen3-next-family.md](qwen3-next-family.md).
- The same tag's image recipe is the cause: `Dockerfile:42-44` runs
  `python3 -m pip uninstall -y triton` right after installing vLLM, with the comment "In x86,
  triton will be installed by vllm. But in Ascend, triton doesn't work correctly. we need to
  uninstall it." The tag's own tutorial is the workaround: `docs/source/tutorials/multi_npu_qwen3_next.md`
  "### Install Triton Ascend" installs the Ascend BiSheng toolkit and then the pinned
  `triton_ascend-3.2.0.dev20250914-...whl`.
- Thread agreement (issue #3029): a contributor wrote "need install triton-ascend and bisheng
  first."; a reporter who installed a mismatched build then hit
  `ImportError: cannot import name 'OutOfResources' from 'triton.triton_patch.runtime.autotuner'`
  while torch loaded `torch_npu` -> `RuntimeError: Failed to load the backend extension: torch_npu`,
  and resolved it by installing a historical triton-ascend build; the closing maintainer comment
  points at the tutorial wheel. So quote the pinned wheel, not "install triton".
- Boundary: the shipped image starts installing triton-ascend at `v0.18.0rc1` (first tag whose
  `Dockerfile` pip-installs the wheel; `v0.17.0rc1` and earlier only uninstall it), and from
  `v0.23.0rc1` it pins `triton-ascend==3.2.1` from the Huawei mirror. The source side matches: no
  unguarded `^import triton` remains under `vllm_ascend/` at `v0.18.0rc1` (`from vllm.triton_utils
  import tl, triton`; `vllm_ascend/ops/__init__.py` gates its triton imports on `HAS_TRITON` from
  `v0.13.0` on). A `No module named 'triton'` on a later tag means a hand-built or host-mounted tree
  instead of the image, which is a different answer.

## In-container device init: `AclSetCompileopt ... 500001` / `GEInitialize failed`

Symptom class: the container starts, the model never loads, and the worker dies in `init_device`.

```text
ERROR 03-11 14:48:04 engine.py:387] Initialize: ... AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE,
  precision_mode), error code is 500001
ERROR 03-11 14:48:04 engine.py:387] EH9999: [Init][PlatformInfo]init runtime platform info failed,
  SocVersion = Ascend910B2
ERROR 03-11 14:48:04 engine.py:387] GEInitialize failed. ge result = 4294967295 / Init compiler failed
ERROR 03-11 14:48:04 engine.py:387]   ... vllm_ascend/worker.py:173 init_device -> platform.py:83 set_device
  -> torch.npu.set_device -> torch_npu._C._npu_setDevice
```

- The frames place it before any model or plugin logic: `set_device` cannot read the platform info for
  the SoC it can see, so this is driver/device-access inside the container, not a vllm-ascend bug and
  not a version mismatch with the checkpoint.
- The SoC string in the block is the reporter's hardware, not a different class: the same text appears
  with `SocVersion = Ascend310P3` in #1991 (`v0.9.2rc1-310p`, Qwen3-Embedding-4B). A 310P *title*
  routes to [310p-atlas-300i-duo.md](310p-atlas-300i-duo.md), but a pasted `500001` +
  `init runtime platform info failed` belongs here — and when one report carries both this and a
  model-support symptom, answer them as two layers (see
  [pooling-embedding-rerank.md](pooling-embedding-rerank.md) for that instance).
- Compare the reporter's `docker run` with the tag's own recipe before blaming the user. At
  `v0.7.1rc1` the reporter's device/mount set (davinci0, `davinci_manager`, `devmm_svm`, `hisi_hdc`,
  `/usr/local/dcmi`, `npu-smi`, `driver/lib64`, `driver/version.info`, `/etc/ascend_install.info`,
  `/root/.cache`, `-p 8000:8000`) is exactly the one in `docs/source/quick_start.md` at that tag,
  differing only by two `-e` settings the reporter added, and `grep -c privileged` on that file is 0
  at `v0.7.1rc1`, `v0.8.4rc1`, `v0.9.1rc1`, `v0.10.0rc1` and `v0.13.0`. So "off the documented path"
  is not available as an explanation for this class.
- The fix that the reporter reported: the only comment in #302 is the reporter's own
  "solved by adding `--privileged=true`". Treat it as a reporter-confirmed workaround (one variable
  changed, the failure went away), with the mechanism a candidate: `--privileged` widens the device
  cgroup, mounts all of `/dev`, and drops label checks on the bind-mounted driver files. Ask for
  `npu-smi info` inside the container and the driver/firmware version pinned by the image before
  calling it a driver-version problem.
- Do not attach `origin/main`'s FAQ 20 (`--shm-size` -> `--privileged=true`) to this symptom: that
  entry is about an `operation not permitted` failure, and finding a matching keyword in a newer tree
  is not documentation for the reported one.

## Container device selection: `--device /dev/davinciN` does not renumber

Symptom class: "the NPU card cannot be called / the model will not load" from a container started
with individual device nodes and no device-visibility variable, often with the note that it works
when no other container is running.

- `ASCEND_RT_VISIBLE_DEVICES` is the plugin's device-control variable:
  `NPUPlatform.device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"`
  (`vllm_ascend/platform.py:101` at `v0.14.0rc1`). Upstream vLLM turns it into the device id:
  `Platform.device_id_to_physical_device_id` returns
  `int(os.environ[device_control_env_var].split(",")[device_id])` when the variable is non-empty and
  the local `device_id` unchanged when it is empty or missing
  (`vllm/platforms/interface.py:202-214` at upstream `v0.14.0`, the paired release — the report does
  not paste its vLLM version). Without the variable, local ids therefore address physical cards 0,
  1, … even though only `davinci6`/`davinci7` were mapped into the container; with it, logical 0/1
  resolve to 6/7. On `main` the same contract gained
  `logical_device_id_to_visible_device_id` / `visible_device_id_to_physical_device_id` and the
  explicit `Physical device N for logical device M is not visible in ASCEND_RT_VISIBLE_DEVICES=...`
  error. The same tag also parses it in `vllm_ascend/cpu_binding.py:11,75-76`,
  and `utils.py:243-247` explains that the custom-op import stays lazy so the variable can still be
  changed before `torch.npu.set_device()`.
- `ASCEND_VISIBLE_DEVICES` has **zero hits in the whole `v0.14.0rc1` tree**
  (`git grep -n ASCEND_VISIBLE_DEVICES v0.14.0rc1`). It is a container-runtime/driver-level
  variable, so that emptiness is not evidence that setting it was ignored — it acts, if at all,
  when the container is created, before any plugin code runs.
- Recorded thread (#6942, `v0.14.0rc1` image, `--device /dev/davinci6 --device /dev/davinci7`,
  `vllm serve … --tensor-parallel-size 2 --enable-expert-parallel`): the maintainer's reply names
  occupancy of the cards by the container that was already running and suggests
  `export ASCEND_RT_VISIBLE_DEVICES=6,7` inside the container. The reporter then reported that
  `ASCEND_RT_VISIBLE_DEVICES=6,7` alone and `ASCEND_VISIBLE_DEVICES=6,7` alone each failed, and
  that the working combination was `-e ASCEND_VISIBLE_DEVICES=6,7` on `docker run` **plus**
  `export ASCEND_RT_VISIBLE_DEVICES=6,7` inside. Give that pair as the reporter-confirmed
  workaround (one environment, two variables) and keep the occupancy explanation labelled as the
  maintainer's hypothesis; the thread was closed `wait-feedback`/stale with no fix commit.
- Passing the variable at container start is documented by the tag itself:
  `docs/source/tutorials/PaddleOCR-VL.md:178` runs the image with
  `-e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"`, and the DeepSeek tutorials take
  `export ASCEND_RT_VISIBLE_DEVICES=$1`. So the general advice for this class is to pass the
  physical ids as `-e ASCEND_RT_VISIBLE_DEVICES=<ids>` at `docker run` **and** re-export them in
  the shell that starts the server, then confirm inside the container that the chosen chips are
  the free ones (`npu-smi info`, `ls /dev/davinci*` for the nodes actually mapped).
- `--device /dev/davinci6` maps that node into the container without renaming it to `davinci0`, so
  a process that enumerates devices from 0 with no variable set addresses cards it cannot see.

## Wheel build: `python -m pybind11 --cmake` vs pybind11 3

Symptom class: `pip install vllm-ascend` fails while building the sdist, ending in
`RuntimeError: CMake configuration failed: Command '[.../bin/python3.x', '-m', 'pybind11',
'--cmake']' returned non-zero exit status 2`, with `__main__.py: error: unrecognized arguments:
--cmake` above it.

- The chain is fully verifiable. `setup.py` at `v0.8.5rc1` (lines 152-158) runs
  `subprocess.check_output([python_executable, "-m", "pybind11", "--cmake"])` with
  `python_executable = sys.executable`, and wraps `CalledProcessError` in that `RuntimeError`.
  `--cmake` is quoted at `v0.8.4rc2`, `v0.8.5rc1`, `v0.9.0rc1`, `v0.9.0rc2` and becomes
  `--cmakedir` at `v0.9.1rc1` and later (`v0.9.1`, `v0.9.2rc1`, `v0.10.0rc1`). `git log -S` names
  `96fa7ff63` (PR #1235; the title is about DP rank set and a torch-npu bump, not pybind) with
  first tag `v0.10.0rc1`, and its release-branch backport `d798125b1` with first tags `v0.9.1`,
  `v0.9.1rc2`, `v0.9.1rc3`.
- Why an unpinned `pybind11` breaks it: `requirements.txt:7` and the `pyproject.toml`
  `[build-system] requires` both list `pybind11` bare, so any version satisfies it. In
  `pybind/pybind11`, `pybind11/__main__.py` at `v2.6.2`/`v2.9.2`/`v2.10.4`/`v2.13.6` uses a plain
  `argparse.ArgumentParser()` and therefore accepts `--cmake` as the unique prefix of
  `--cmakedir` (argparse abbreviation); at `v3.0.0` the parser is built with
  `functools.partial(argparse.ArgumentParser, allow_abbrev=False)`, so `--cmake` is rejected and
  `--cmakedir` is the only spelling. The "it worked last month" report is the pybind11 3.0
  release, not a vllm-ascend change.
- Which pybind11 matters: the report's failing interpreter is the **outer** one
  (`/usr/local/python3.10.17/bin/python3.10`), not the PEP 517 overlay under
  `/tmp/pip-build-env-*/`, because `sys.executable` is what `check_output` invokes. So pinning
  inside build isolation does nothing; install into the environment that runs `pip`.
- Routes, in order of directness: upgrade to a release whose `setup.py` says `--cmakedir`
  (`>= v0.9.1rc1`), or `pip install "pybind11<3"` in the outer interpreter before retrying
  (`<3` works because of the abbreviation above, not because 2.x has a `--cmake` flag — no
  pybind11 release in the checked range defines one). Patching `setup.py` in a local checkout
  is the same change as the upstream fix; label it as such rather than as a user-facing option.
- Boundary: this is a build-environment dependency break, so it is fixed by the user's
  environment or an upgrade, and it is not evidence about any runtime behaviour of the wheel.

## From-source build: the `csrc/build/compat/liboptiling.so` link is expected to dangle

Symptom class: an `[Installation]` report after `pip install -v -e .` from source — the build tree
contains `csrc/build/compat/liboptiling.so -> lib/linux/<arch>/libcust_opmaster_rt2.0.so` and
`cat` on it says "No such file or directory", while the same library is present under
`csrc/build/_CPack_Packages/.../packages/vendors/vllm-ascend/op_impl/ai_core/tbe/op_tiling/lib/linux/<arch>/`.

- The link is written deliberately, relative to its *install* destination, at
  `csrc/CMakeLists.txt:195-210` (`v0.19.1rc1`):
  `COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:optiling> ${compat_optiling_file}`
  with `install(FILES ${compat_optiling_file} DESTINATION packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling)`.
  The `optiling` target's `OUTPUT_NAME` is `cust_opmaster_rt2.0` (`:183-185`) and it is installed to
  `.../op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}` (`:191-193`), so in the packaged tree the
  sibling `lib/linux/<arch>/` directory exists and the relative link resolves. In
  `csrc/build/compat/` there is no such sibling, which is why the link dangles there.
- No release "fixes" it: the block is unchanged in behaviour on `main`, moved to
  `csrc/cmake/custom_build.cmake:212-229` with the target renamed `cust_opmaster` (`:198-199`).
- Several files carry the name, so resolve the path a report means before diagnosing: the
  post-build `TILING_CUSTOM_FILE` link (`csrc/cmake/config.cmake:93,135-136`, absolute
  `$<TARGET_FILE:...>`) inside the build's own package tree; `csrc/cmake/symbol.cmake:156-158`
  installing `${CMAKE_BINARY_DIR}/liboptiling.so`; and
  `csrc/cmake/scripts/custom/install.sh:240` **removing** `.../op_tiling/liboptiling.so` during a
  custom-op install (last three read on `main`).
- The report (#10792, open, `triage review`, no maintainer answer) only shows the link and the
  `cat` failure. Treat "the build is broken" as unestablished and ask for the line that actually
  consumes the library — a real failure names a missing `.so` while `import vllm_ascend` or
  `set_device` runs, not while a shell read follows a build-tree link.
- Boundary: CMake packaging layout, verified by reading the two trees; whether `install(FILES
  <symlink>)` copies the link or follows it into the target for the CMake version in use was not
  tested.

## Offline (no public network) from-source install: the `catlass` submodule fetch

Symptom class (`vllm-ascend#8788`, v0.13.0rc3-a3-openeuler / vllm-ascend `v0.13.0`, A3): an
`[Installation]` report says the custom-op build works on a networked host but fails inside a pod
with no public network (an internal iTask pod is typical), and blames the build refactor it noticed
in 0.13.

- The build entry point is `setup.py`: `cmake_build_ext.run` calls `self.run_command("build_aclnn")`
  (`v0.13.0` `setup.py:425-428`), and `build_and_install_aclnn.run` shells out to
  `bash csrc/build_aclnn.sh <ROOT_DIR> <envs.SOC_VERSION>` (`:200-217`).
- The public-network statement is in that script. Both 910 branches (`^ascend910b` A2 and
  `^ascend910_93` A3) set `CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include` and, when the
  directory is absent, print `depdendency catlass is missing, try to fetch it...` and run
  `git submodule update --init --recursive` (`fetch failed` -> `exit 1`). `.gitmodules` maps that
  path to `https://gitcode.com/cann/catlass.git`, branch `catlass-v1-stable`.
- Boundary correction: the first tag with `csrc/build_aclnn.sh` is `v0.12.0rc1` (absent at
  `v0.10.0rc1` and `v0.11.0rc1`), so a 0.12 report has the same cause; "starting from v0.13.0" is the
  reporter's dating, not the code's.
- The tag documents it: `docs/source/installation.md:169,174` at `v0.13.0` puts
  `git submodule update --init --recursive` in the from-source recipe and adds "If you are building
  custom operators for Atlas A3, you should run `git submodule update --init --recursive` manually,
  or ensure your environment has Internet access". The same guidance persists (`v0.18.0`;
  `v0.22.1rc1:224,229`) and moves to
  `docs/source/getting_started/installation/install_vllm_ascend.inc.md` on `main` — quote the tag's
  own copy, not a memory of it.
- Workaround, in order: pre-seed the checkout where the network exists (`git clone --recursive`, or
  `git submodule update --init --recursive`, or copy `csrc/third_party/catlass` from an internal
  mirror) and confirm `test -d csrc/third_party/catlass/include` before the offline build; the build
  then needs no network. `COMPILE_CUSTOM_KERNELS=0` also removes the fetch (`envs.py:44-45`, default
  `1`; `setup.py:448` then adds no extension and `:354 build_extensions` returns immediately), but it
  drops every custom op and the worker logs "COMPILE_CUSTOM_KERNELS is set to False. In most
  scenarios, without custom kernels, vllm-ascend will not function correctly."
  (`vllm_ascend/worker/worker.py:94-97`) — a last resort with that cost stated, not the fix.
- The A3 branch has a second, *local* prerequisite: it requires `moe_distribute_base.h` under
  `$ASCEND_TOOLKIT_HOME` (`cannot find moe_distribute_base.h file in CANN env` -> `exit 1`) and
  copies it in-tree. That failure is a toolkit mismatch, not a network fetch. `^ascend310` exits 0
  before both — 310P has no custom aclnn ops on these tags.
- Do not blame FetchContent: `setup.py:316` only carries upstream vLLM's inherited
  `FETCHCONTENT_BASE_DIR` comment, and at `v0.13.0` neither `csrc/CMakeLists.txt` nor `csrc/cmake/*`
  contains `FetchContent`; the single `ExternalProject_Add(extern_opbuild_gen_default)`
  (`csrc/CMakeLists.txt:444`) sits inside `if ((NOT BUILD_OPEN_PROJECT) AND ("${PRODUCT_SIDE}"
  STREQUAL "device"))`. Grep the exact fetch construct before naming a dependency.

Boundary: read from the tag-pinned trees; nothing was built here. Whether the pod lacked the
submodule (sdist/no `--recursive`) or had git but no route to gitcode.com is unstated — both fail at
the same statement.

## Start-up `OSError: Could not load this library: .../libtorchcodec_image.so` (`libnvrtc.so.13` missing)

Symptom class (`vllm-ascend#16074`, DeepSeek-V4-Flash W8A8, vllm-ascend `0.26.0rc1` on vLLM
`v0.26.0`): a text-only LLM serve dies before the engine starts. The traceback's cause is
`OSError: libnvrtc.so.13: cannot open shared object file`, raised from
`torchcodec/_core/ops.py load_image_library` -> `torch.ops.load_library(path)`, and the final line
names `.../site-packages/torchcodec/libtorchcodec_image.so`. Nothing in `vllm_ascend/` is on the
stack.

- The import is unconditional even for a text model at vLLM `v0.26.0`: `sampling_params.py` ->
  `v1/serial_utils.py` -> `multimodal/...` -> `vllm/multimodal/video.py:35
  from torchcodec.decoders import VideoDecoder`, inside `try: ... except (ImportError, RuntimeError)`
  at `v0.26.0` (`v0.25.0` catches `ImportError` alone). The library loads during `import torchcodec`,
  and the `OSError` is neither caught class, so it aborts the process; a *missing* torchcodec would
  merely become a `PlaceholderModule`.
- Where the package came from matters, and it is not the plugin: `vllm-ascend`'s own
  `requirements.txt`/`pyproject.toml` at `v0.26.0rc1` do not list it, vLLM's `[audio]` extra does not
  include it, and `VLLM_TARGET_DEVICE=empty` (what the plugin's Dockerfile uses) resolves
  `requirements/common.txt`, which carries only `opencv-python-headless` for video IO. vLLM's
  `requirements/cpu.txt:18-19` does list `torchcodec >= 0.14` "(required for the torchcodec video
  decoding backend)" — for a `cpu` target — and a plain `pip install torchcodec` can resolve a
  CUDA-13-linked wheel (the recorded failure names `libnvrtc.so.13`). So ask what added it
  (`pip show torchcodec`, the image's pip lines), and expect a `+cpu` variant to be the compatible
  one: `requirements/test/cpu.txt:1135` pins `torchcodec==0.14.0+cpu`.
- Version bounds: the module-level import and the `torchcodec >= 0.14` requirement both first appear
  at vLLM `v0.25.0` (absent at `v0.24.0`) and are gone by `v0.29.0`, where `3bb9c18f0c`
  ("[Multimodal] Reorganize video decoder backends", #49155) removed the import from `video.py`. The
  class therefore covers the paired 0.25-0.28 line; a v0.26 report pins to `video.py:35`, the line
  the recorded traceback names.
- Workarounds: (1) uninstall/omit `torchcodec` — the guarded import degrades to a placeholder and only
  video decoding is lost; (2) install the CPU build (`torchcodec==0.14.0+cpu` from the PyTorch CPU
  index) so the shared library has no CUDA dependency. Neither touches the model, tokenizer mode or
  DeepSeek flags.
- Thread precedent: the recorded reporter closed their own report as "非问题，应该是我们打镜像漏了组件有关"
  (not an issue; our image build was missing a component) with no maintainer diagnosis. Treat "our
  custom image installed a package the shipped image does not" as the first hypothesis — the shipped
  `Dockerfile`/`Dockerfile.a3` install vLLM from source with `--extra-index-url ${PYTORCH_INDEX_URL}`
  (`Dockerfile.a3:68` at `v0.26.0rc1`), which is what makes a `+cpu` variant resolvable; a hand-built
  image that installs it without that index can pick the CUDA wheel (candidate, not verified here).

## Non-root containers: device-node ownership, not `/etc/passwd` in the plugin

Symptom class (`[Usage]` questions): running the official image as `runAsUser: 1000` (or an
arbitrary UID) on Kubernetes/KServe; TP=1 is fine or untested, TP>1 fails during HCCL init with
`HcclGetRootInfo(&hcclID), error code is 19` and `ra init failed, return[19]
devicePhyId_[4], nicPosition[0]` (#9490, `v0.18.0rc1`, TP=2).

- Verified in-tree absence: `git grep -i -e getpwuid -e '/etc/passwd' -e HwHiAiUser -e runAsUser
  -e runAsNonRoot <tag> -- 'vllm_ascend/*' 'docs/*'` returns nothing at `v0.18.0rc1` and on `main`
  (2026-11-01; `main`'s only `securityContext`/`UID` hit is a PNG asset). `HcclGetRootInfo` exists
  in the plugin only as a ctypes binding
  (`vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:117-118`, called at `:219`), so
  the logged `(&hcclID)` / `ra init` text is the closed CANN/HCCL library's own output. No plugin
  code reads the passwd database: the requirement, if any, lives below the plugin and cannot be
  pinned from this repository.
- The decisive fact for arbitrary UIDs is device-node ownership, measured by the reporter on their
  nodes: `/dev/davinci*`, `/dev/davinci_manager`, `/dev/devmm_svm` and `/dev/hisi_hdc` are
  `HwHiAiUser:HwHiAiUser 0660`, and `HwHiAiUser` is UID/GID 1000. Their UID `65534` test failed at
  device discovery, UID 1000 progressed much further, and a derived image that adds a real passwd
  entry (`useradd --uid 1000 --gid 0 --home-dir /tmp ...`, `USER 1000:0`) then started TP=2 — a
  reporter-confirmed workaround in their environment, not a vendor requirement. Practical reading:
  run as UID 1000 with group 0/supplementary 1000, or as root; an arbitrary UID needs group 1000 for
  the device nodes at minimum.
- The image ships no non-root user of its own: `v0.18.0rc1:Dockerfile` is
  `FROM quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11` with no `USER`/`ENTRYPOINT` (only
  `CMD ["/bin/bash"]` at `:102`), and the `Dockerfile.310p`/`.a3`/openEuler variants add no
  `useradd`/`HwHiAiUser` either. Whatever identity exists comes from the CANN base image.
- Keep the competing candidate open: the same line reports `nicPosition[0]`/`devicePhyId_[4]`,
  i.e. HCCL's RA/NIC selection, and this deployment does not mount `/etc/hccn.conf` (the reporter
  says their Ascend runtime does not expose it) although the tag's own Kthena/KV-pool docs mount it.
  A missing hccn.conf/NIC mapping is equally consistent with `ra init failed`, so ask for the mount
  list and whether `/dev/davinci*` are readable as that UID before asserting "the passwd entry is
  required". `USER`/`LOGNAME`/`HOME` env vars are not documented as sufficient for anything.
- Documentation boundary: the thread reply says non-root guidance was added to the documentation,
  but no `non-root`/`runAsUser`/`HwHiAiUser` hit exists under `docs/` on `main` in this clone. Cite
  the thread for that guidance, not the tree, and say the tree was checked.
- Build-side non-root class in the same repository: `1d83968d7` (#20270, 2026-10-29, cherry-pick
  `a14a2571d`) fixes the wheel jobs after the CANN 9.2.1 manylinux base images shipped
  `.../ascendnpu-ir/bin/set_env.sh` unreadable by non-root — `docker run -u "$(id -u):$(id -g)"`
  then exits 1 in the entrypoint with `Permission denied` before `exec "$@"`, and the fix is
  `--entrypoint /bin/bash`. That is a base-image packaging regression, not a vllm-ascend defect, and
  it is the pattern to check when a non-root CI or build step fails before any Python runs.

## Wheel built against a newer glibc: `GLIBC_2.32 not found` and a missing `_C` op

Symptom class: a pip-installed `vllm-ascend` wheel on an older OS userspace logs
`Failed to import vllm_ascend_C:/usr/lib64/libc.so.6: version \`GLIBC_2.32' not found (required by
.../vllm_ascend_C.cpython-310-aarch64-linux-gnu.so)`, and start-up later dies during graph capture
with `AttributeError: '_OpNamespace' '_C' object has no attribute 'weak_ref_tensor'`. The title of
such a report usually quotes the GLIBC line, which is a warning, not the raising statement.

- The GLIBC line is emitted by `vllm_ascend/device_allocator/camem.py:56-66` at `v0.9.1rc1`: the
  `from vllm_ascend.vllm_ascend_C import ...` is wrapped in `except ImportError`, sets
  `camem_available = False` and logs "Sleep mode will be disabled". It costs sleep mode only.
- The fatal path is `vllm_ascend/compilation/piecewise_backend.py:205`
  `entry.output = weak_ref_tensors(output)` (import at `:32`, `from vllm.utils import
  weak_ref_tensors`). Upstream `vllm` `v0.9.1` `vllm/utils.py:1901-1908` implements it as
  `torch.ops._C.weak_ref_tensor(tensor)`, and `vllm/_custom_ops.py:20-22` logs its own
  `Failed to import from vllm._C with ImportError(...)` when the extension cannot load (the reported
  one names an undefined `_ZN3c108ListType3getE...`, i.e. a torch/c10 ABI mismatch).
- `_C` is a **shared namespace**: upstream `vllm` registers `weak_ref_tensor` with a `kCUDA` impl
  (`csrc/torch_bindings.cpp:19,31-32` at `v0.9.1`), and `vllm-ascend` registers the same schema with
  a `kPrivateUse1` impl (`csrc/torch_binding.cpp:290-294` at `v0.9.1rc1`,
  `TORCH_LIBRARY_EXPAND(_C, ops)`). On NPU the op therefore exists only if a library registered it;
  when both `.so`s fail to load, `torch.ops._C.weak_ref_tensor` is simply absent, which is exactly
  the reported AttributeError. Reading the GLIBC warning and the `vllm._C` warning as one cause is
  wrong: they are two independent load failures, and either one alone removes the op.
- Fix and boundary: `0d4bc0394` "Fix wheel glibc version incompatibility (#1582)", 2025-07-08,
  whose body says "Fixes #1533"; it changes only `.github/Dockerfile.buildwheel` and
  `.github/workflows/release_whl.yml` (the wheel build container), not plugin code. Backport
  `9bcf007d9` (#1808) is `git describe --contains` = `v0.9.1rc2~44`; `0d4bc0394` is
  `v0.9.2rc1~26` and also in `v0.10.0rc1`+. At `v0.9.1rc1` neither is an ancestor
  (`git merge-base --is-ancestor <sha> v0.9.1rc1` = no), which is the pin for a report on that build.
- The plugin later dropped its copy: `f335d75d8` "[Misc] remove weak_ref_tensor C code (#8726)"
  (2026-04-27) deletes only `csrc/ops.h` + `csrc/torch_binding.cpp`; its message says the call now
  uses `torch_npu._C._weak_ref_tensor`, which the tag's own Python wrapper confirms
  (`vllm_ascend/utils.py:1035-1042` at `v0.20.2rc1`, and `compilation/acl_graph.py:24` at that tag
  imports that wrapper rather than `vllm.utils`). `git describe --contains f335d75d8` =
  `v0.19.1rc1~48`, so this AttributeError belongs to the plugin's own C-op era (0.18 and earlier):
  do not port the diagnosis to a 0.19+ report without re-reading that tag's
  `vllm_ascend/utils.py`.
- Workaround for a stuck install: build/install `vllm-ascend` from source in the target image
  (thread-confirmed by the reporter) — that also re-registers the NPU impl; `--enforce-eager` avoids
  the piecewise backend altogether (code-path inference, not reporter-confirmed here). Upgrading the
  wheel to a tag containing the fix is the durable route.

## From-source reinstall loses `libvllm_ascend_kernels.so`

Symptom class: inside the shipped image the reporter does `pip uninstall vllm-ascend`,
`MAX_JOBS=4 pip install -e . --no-build-isolation` on their own checkout, and start-up logs
`ERROR camem: Failed to import vllm_ascend_C:libvllm_ascend_kernels.so: cannot open shared object
file: No such file or directory`, while the image's own tree has
`/vllm-workspace/vllm-ascend/vllm_ascend/lib/libvllm_ascend_kernels.so` and the fresh build has no
such file (issue #1114, `v0.8.5rc1` image). Read the line as the loader failing on a *dependency* of
the extension rather than on the extension: it is logged by
`vllm_ascend/device_allocator/camem.py:57-68`, whose `from vllm_ascend.vllm_ascend_C import ...` got
far enough to name a missing shared object, and `CMakeLists.txt:96` at that tag links the module with
`-Wl,-rpath,$ORIGIN:$ORIGIN/lib` — the kernels library is expected in `lib/` beside the module. The
consequence is bounded: `camem.py:140 assert camem_available` fires only when the camem (sleep-mode)
allocator is requested, so this ERROR alone does not explain a start-up crash.

- **Build gate at `v0.8.5rc1`.** `vllm_ascend/envs.py:35-36` reads `COMPILE_CUSTOM_KERNELS` with
  default `1` (`bool(int(...))`, so any `0` disables it); `setup.py:288-289` adds the
  `vllm_ascend.vllm_ascend_C` CMake extension only when it is truthy, and `setup.py:207-209
  build_extensions` returns immediately when it is not — the install then finishes silently with no
  cmake run and no `.so` at all. The default moved over time: the `v0.7.3rc2` notes say custom-op
  compilation is *disabled* by default (#371), the `v0.8.4rc1` notes say it is *enabled* by default
  (#466); `docs/source/installation.md:163` at `v0.8.5rc1` states the enabled default and `:167` still
  tells v0.7.3-dev builders to set `COMPILE_CUSTOM_KERNELS=1` manually. A checkout older than
  `v0.8.4rc1` is the first thing to check when a from-source build has no `.so` whatsoever.
- **Where the two libraries separate.** `CMakeLists.txt:98` installs both targets into
  `${VLLM_ASCEND_INSTALL_PATH}`, but the CANN `ascendc_library` macro adds its own
  `install(TARGETS ${target_name} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})` and writes the library
  to `${CMAKE_BINARY_DIR}/lib` (`ascendc_library`, `cmake/asc/legacy_modules/function.cmake`, read from
  the local `cann/asc-devkit` clone; the reporter's CANN 8.1.RC1 copy lives under
  `$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake`). The kernels library therefore installs to
  `<prefix>/lib/` while the pybind module installs to `<prefix>/`. In editable mode `setup.py:160-162`
  points `CMAKE_INSTALL_PREFIX` at `<checkout>/vllm_ascend` and `:255-265` copies every `.so` under the
  build temp into `<build_lib>/vllm_ascend` only when `get_command_obj("develop")` is a `develop`
  instance — so a build that produced both libraries can still leave one of them outside the tree.
- **310P and A5 builds legitimately have no kernels library.** From `f0ea80780` ("[Feature][310p]
  Add npu_causal_conv1d_310 AscendC Custom Op", #7798, 2026-03-30, first tag `v0.19.1rc1`) three
  coordinated `if(SOC_VERSION MATCHES "ascend310p.*|ascend950")` blocks skip the `ascendc_library`
  target, skip linking it, and install only `vllm_ascend_C` (`v0.23.0` `CMakeLists.txt:74-80`,
  `:183-196`, `:200-204`; the same shape on `main`). On such a build the file is absent by design and
  the loader error cannot come from it — do not chase the library on a 310P/A5 report; check the SOC
  the build baked in ([310p-atlas-300i-duo.md](310p-atlas-300i-duo.md)).
- **Thread.** #1114 has two comments: the maintainer links the installation docs and `Dockerfile#L21`
  (`ARG COMPILE_CUSTOM_KERNELS=1`, exported as `ENV COMPILE_CUSTOM_KERNELS`) and closes it `completed`.
  `COMPILE_CUSTOM_KERNELS=1` is thus the documented answer, not a diagnosed cause, and nothing in the
  thread separates "gate was off" from "cmake ran but the library did not land in `lib/`". The paste is
  mild evidence for the second: with the gate off there would be no `vllm_ascend_C` module to report a
  missing dependency.
- **Check, then fix.** `ls -l <site-packages|checkout>/vllm_ascend/*.so <...>/vllm_ascend/lib/` and
  `find <checkout> -name 'libvllm_ascend_kernels.so'`; print `COMPILE_CUSTOM_KERNELS` in the build
  shell; the tag's own `installation.md` lists the prerequisites (`gcc`, `g++`, `cmake`, `libnuma-dev`).
  Rebuild with `COMPILE_CUSTOM_KERNELS=1` rather than `pip uninstall`-ing the image's tree, or restore
  `vllm_ascend/lib/` from the image, then confirm with the import, not with log silence.

## Pip wheel missing a subpackage: `ModuleNotFoundError: No module named 'vllm_ascend.compilation'`

Symptom class: `pip install vllm-ascend==v0.9.0rc1`, then start-up dies during warmup inside
upstream frames with the plugin module missing:

```text
File "/vllm-workspace/vllm/vllm/compilation/backends.py", line 298, in call_module
    piecewise_backend = resolve_obj_by_qualname(
ModuleNotFoundError: No module named 'vllm_ascend.compilation'
```

- Both sides are verifiable. `vllm/compilation/backends.py:298` at vLLM `v0.9.0` is
  `resolve_obj_by_qualname(current_platform.get_piecewise_backend_cls())`, and the frame above it,
  `backends.py:498`, is the `PiecewiseCompileInterpreter(...)` call — two statements that pin the
  paired upstream release. The name comes from the plugin: `vllm_ascend/platform.py:262-267` at
  `v0.9.0rc1` returns the literal `"vllm_ascend.compilation.piecewise_backend.NPUPiecewiseBackend"`.
- The package is what is missing: `git ls-tree -r --name-only v0.9.0rc1 | grep vllm_ascend/compilation`
  lists only `piecewise_backend.py`, no `__init__.py`, and `setup.py:357` builds the wheel with
  `find_packages(exclude=("docs", "examples", "tests*", "csrc"))`, which skips a directory without
  `__init__.py`. So the subpackage never enters the wheel, while the same tree imports fine from a
  source checkout (PEP 420 namespace package) — which is why "works from source, fails from pip" is
  the tell.
- First release with the fix: `b75cb788d` "[Bugfix] add compilation/__init__.py to fix import error"
  (#1152) adds the file; `git tag --contains b75cb788d --sort=creatordate | grep '^v' | head -1` is
  `v0.9.0rc2`. The thread's own workaround is a source install at that branch
  (`git clone --depth 1 --branch v0.9.0rc2 ... && pip install -v -e .`). Say "upgrade to
  `>= v0.9.0rc2` or install from source", not a flag change.
- Generalize the shape: `ModuleNotFoundError: No module named 'vllm_ascend.<sub>'` from a *pip*
  install, where `git ls-tree` at that tag shows the directory, is a packaging gate. Check
  `__init__.py` at the tag and ask whether the reporter installed a wheel or a source tree before
  reading runtime code; `--enforce-eager` only avoids the piecewise backend by not compiling, so it
  hides rather than fixes this class.
- **A release-branch dev build can carry the same gap in several directories at once, and the fix
  can live only on that branch.** Issue #8686 reports `vllm-ascend 0.13.0rc4.dev3+gc9888eabd`
  (`origin/releases/v0.13.0`, built with `pip install .` in a Dockerfile) losing
  `vllm_ascend.models.layer` and `vllm_ascend.transformers_utils` from the wheel. The whole tree
  scan is one loop and reproduces the report exactly — three directories, the third
  (`_cann_ops_custom`) holding only `.gitkeep`:

  ```bash
  git ls-tree -r -d --name-only <sha> -- vllm_ascend | while read d; do
    git cat-file -e <sha>:"$d/__init__.py" 2>/dev/null || echo "$d"; done
  ```

  The fix is `58ea73f95` "[v0.13.0][Bugfix] Fix Missing __init__.py (#8729)" adding both files on
  `origin/releases/v0.13.0`; `git tag --contains 58ea73f95` is **empty** and it is not an ancestor
  of `main`. `main` got its own fix for one directory only, `471f0590b` (#9649, first tag
  `v0.20.2rc1`), and `vllm_ascend/transformers_utils/` no longer exists on `main` at all. So the
  answer is "branch tip after #8729, or a source install — not any released tag", and the reverse
  question ("is the shipped `v0.13.0` affected?") is **no**: the `v0.13.0` tag predates both
  directories, which entered the branch afterwards.
- **A branch-tip dev build is not the release it names.** Decide whether a release is affected from
  `git ls-tree <tag> -- <path>` on the tag itself, never from the branch that carries the fix; an
  empty listing means the path does not exist at that ref (do not read it as a truncated listing —
  check each ref separately rather than piping a multi-ref `ls-tree` through `head`).

## Start-up `!!!!!!! Segfault encountered !!!!!!!` / `Failed core proc(s): {'EngineCore': -11}` on a hand-built plugin

Symptom class: a from-source or re-built `vllm-ascend` (`...dev0+g<sha>`, `vllm <ver>+empty`)
segfaults during V1 EngineCore initialization even on a 0.5B dense model, no Python traceback from
the crashing process, `ldd` clean on `vllm_ascend_C*.so` and `libvllm_ascend_kernels.so`, and the
thread has nothing but a colleague ping. Nothing here identifies a mechanism; the value is the
order of checks, so the answer does not drift into model or kernel speculation.

- **Pin the build first.** `git describe --tags <sha>` can name the *previous* tag (issue #10136's
  `0.20.2rc2.dev0+g367b8e62d` is commit `367b8e62d`, which *is* `v0.20.2rc1`; there is no
  `v0.20.2rc2` tag at all — see [revision-pinning.md](revision-pinning.md)). Then the tag's own
  `Dockerfile` gives the intended matrix: `v0.20.2rc1` is
  `FROM quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11`, `ARG VLLM_TAG=v0.20.2`,
  `ARG SOC_VERSION="ascend910b1"`, so a CANN 9.0.0 / vLLM 0.20.2 report is *not* a version
  mismatch, and `vllm 0.20.2+empty` is just the `VLLM_TARGET_DEVICE=empty` build the tag's
  Dockerfile itself performs (vLLM `setup.py:1368-1369` appends `+empty`), not a defect by itself.
- **Their "fusion disabled" settings did not disable the plugin's fusion passes.** The recorded
  script put `fuse_norm_quant`, `fuse_act_quant`, `fuse_attn_quant`, `fuse_allreduce_rms`,
  `fuse_gemm_comms` and `enable_sp` under `compilation_config.pass_config`, while the plugin gates
  its fx passes on `additional_config["ascend_compilation_config"]` and reads only
  `pass_config.enable_sp` from upstream's `PassConfig`
  ([env-vars-and-additional-config.md](env-vars-and-additional-config.md)). Do not accept "turned
  fusion off and it still crashed" as excluding the fusion pass unless the
  `--additional-config '{"ascend_compilation_config": ...}'` form was used.
- **Diff the reporter's `pip list` against the tag's install lines; that is the only cheap
  deviation check.** `v0.20.2rc1` runs `pip uninstall -y triton triton-ascend` and then
  `pip install triton-ascend==3.2.1`, so a from-source environment listing **both** `triton 3.5.0`
  and `triton_ascend 3.2.1` already differs from every shipped image. Treat that as a candidate and
  ask for `python -c 'import triton, sys; print(triton.__file__)'` before reading kernels; it is
  not a proven cause of the segfault.
- **A hand-patched build script moves the custom-op install path.** Issue #10136's reporter patched
  `setup.py` (`CMAKE_PREFIX_PATH`/`Torch_DIR`) and `csrc/build_aclnn.sh` (`--noexec --target`, then
  `install.sh --install-path=...`); ask which directory the ops landed in and whether
  `vllm_ascend/lib/` carries them, because "the operator ran in a different tree" is a build
  question, not a model one.
- Without `gdb`, the substitute is the crashing rank's own stderr/plog (the parent prints only
  `Failed core proc(s)`), plus `ASCEND_LAUNCH_BLOCKING=1` and a single-device, `--enforce-eager`
  rerun. State that the mechanism is unknown rather than promoting a candidate.

## Pitfalls specific to these reports

- A loop such as `for t in tagA tagB; do git show $t:setup.py | grep -c '--cmakedir'; done`
  prints `0` for a **missing tag** as well: `git show` writes `fatal: invalid object name` to
  stderr and an empty stdout, and the pipeline's exit status is lost. Gate each iteration with
  `git rev-parse --verify -q "$t^{commit}"` before reading, or the "flag absent in tag X" table
  contains tags that do not exist in the clone.
- `git tag --contains <sha>` determines the release a build fix first appears in; the commit
  title does not. `96fa7ff63` and `d798125b1` above are found with `git log -S'"--cmakedir"'`,
  not by grepping commit messages for "pybind".
- Installed versions are not repository pins. `Requirement already satisfied: pybind11 ... (3.0.0)`
  in the reporter's log is the only runtime evidence; the repository merely leaves the pin open.
