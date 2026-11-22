# A missing `.so` in a launcher-managed deployment: read the launch command's env

Symptom class: `ImportError: <name>.so: cannot open shared object file` (or
`No module named …`) raised while a *connector module* is imported, i.e. before any
vllm-ascend logic runs, in a deployment started by a launcher — Kubernetes /
kserve / kthena `args:` or `command:`, a docker-compose `entrypoint`, a systemd
unit, an operator CR — rather than by `vllm serve` typed in the reporter's shell.

## The rule

**The environment that decides the load is the environment of the process that
runs `vllm serve`, and that is the launcher's, not the shell's.** Exports in
`~/.bashrc`, `/etc/profile.d/*` or an image profile script are not inherited when
the launcher execs the binary itself, and `collect_env.py` reports whatever shell
it happened to run in — so its `LD_LIBRARY_PATH` line can be complete while the
engine process has none of it. A pasted ENV dump is evidence about a shell, never
about the launched process; treat the launch spec as the primary artifact.

Reporter-confirmed on #7352 (v0.17.0rc1, Mooncake PD, 910B4, both pods under a
`kthena`-style launcher):

- The failure: `mooncake_connector.py:24` `from mooncake.engine import TransferEngine`
  → `ImportError: ascend_transport.so: cannot open shared object file: No such file
  or directory`, on the **prefill** pod at
  `ensure_kv_transfer_initialized` → `KVConnectorFactory.create_connector`, and on
  the decode pod too. `ascend_transport.so` is mooncake's Ascend transport plugin
  (`kvcache-ai/Mooncake` `doc/zh/ascend_transport.md`, linked from
  `examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md:10`),
  dlopen'd by `import mooncake.engine`; the name has no `lib` prefix, which is why
  a `find / -name 'libascend*transport*'` misses it.
- The env dump showed `LD_LIBRARY_PATH` **with** `/usr/local/lib`; the launched
  process still did not have it. The reporter's own conclusion (2026-03-19): the
  environment was not set correctly for the launch, and the fix was to put the
  export *in the boot command*:
  ```yaml
  args: |
      export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH && vllm serve ...
  ```
  Both pods needed it. Same image, same checkout, no rebuild.
- The trap in the other direction: an earlier round of this deployment (v0.14.0rc1,
  started differently) had printed `AscendDirectTransport register mem addr:…`
  lines, i.e. the plugin *was* resolvable there. That does **not** prove "the new
  image regressed"; it only proves the two pods were started with different
  effective environments. Compare launch commands before comparing images.

## Why `/usr/local/lib` is the directory being appended

The artifact is a target of the **source** build, and its install location explains the
reporter's one-line fix: in `kvcache-ai/Mooncake`
`mooncake-transfer-engine/src/transport/ascend_transport/CMakeLists.txt` the library is
`add_library(ascend_transport SHARED …)` (`:23` in the v0.3.8 export `823064aa`, the
pin behind #6949), `install(TARGETS ascend_transport DESTINATION lib)` (`:37`) with
`set_target_properties(… PREFIX "")` (`:46-48`) — so it lands at
`${CMAKE_INSTALL_PREFIX}/lib/ascend_transport.so`, i.e. `/usr/local/lib/…` under the
default prefix, and without a `lib` prefix. Quote that pair whenever "why that
directory" is the question, and give the remedies in the pin's own order: the
`LD_LIBRARY_PATH` export **inside the launch command** (step 4 below), Mooncake's
"copy the `.so` file to another path referenced by `$LD_LIBRARY_PATH`"
(`docs/source/design/transfer-engine/ascend_transport.md:48`), or the launcher-proof
one-liner in the image — `echo /usr/local/lib > /etc/ld.so.conf.d/mooncake.conf &&
ldconfig` — after which no per-process export exists for a launcher to drop.
Path drift to expect when grepping the tutorials: from v0.15.0rc1 onward they sit under
`docs/source/tutorials/features/` (`…/pd_disaggregation_mooncake_single_node.md:138`
at `3d43ed99`, #6949), not directly under `docs/source/tutorials/`.

## When the image recipe itself is the source

Ask for the `~/.bashrc` grep before blaming the reporter's launcher: vllm-ascend's own
`Dockerfile*` writes the jemalloc preload and the `/usr/local/lib` append **into
`~/.bashrc`**, in every variant present at the pin — `Dockerfile:81-82`,
`Dockerfile.a3:80-81`, `Dockerfile.openEuler:78-79`, `Dockerfile.310p:63-64`,
`Dockerfile.a3.openEuler:78-79`, `Dockerfile.310p.openEuler:59-60` (v0.16.0rc1
`a78a00e`; the same pair is at `:75-76`-ish in the v0.19.1rc2 tree of #12059, so this
spans versions). Consequences to state, each checkable in one grep:

- **The image's own default start never gets them.** `CMD ["/bin/bash"]`
  (`Dockerfile:84` and its siblings) without a TTY is non-interactive, so bash skips
  `.bashrc`; `docker run <image>`, k8s and CI are all in that state. The advertised
  default is broken by the same mechanism as `--entrypoint`.
- **The values are frozen at build time.** Both lines are `RUN echo "…$VAR…" >>
  ~/.bashrc`, executed by `/bin/sh -c`, so `$LD_LIBRARY_PATH` is whatever the base
  image's `ENV` held (`FROM quay.io/ascend/cann:…`, `Dockerfile:18`; there is no
  `ENV LD_*` in the recipe) and `$LD_PRELOAD` expands empty — the visible artifact is
  the trailing colon in `…libjemalloc.so.2:`. A later CANN switch or `set_env.sh`
  cannot be added by this mechanism.
- **The Dockerfile-native channel is `ENV`, not an entrypoint wrapper.** `ENV
  LD_PRELOAD=…` + `ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH` reaches
  `ENTRYPOINT`/`CMD`, k8s `command:`/`args:` and `python -m` alike; delete the
  `.bashrc` writes instead of keeping both (double preload). `Dockerfile.310p`
  carries an `entrypoint.sh` pattern as the belt-and-braces variant.
- **Verify from the running process**, never from `docker exec -it … echo $LD_PRELOAD`:
  `tr '\0' '\n' < /proc/1/environ | grep -E 'LD_PRELOAD|LD_LIBRARY_PATH'`.
- Do not convert this into a Mooncake claim. #12059's title ties the defect to
  Mooncake and the mechanism is plausible (`/usr/local/lib` for mooncake's `.so`), but
  the durable statement is *"the image's documented allocator/`.so` defaults do not
  reach non-interactive processes"*; a specific transfer failure needs a run without
  the workaround to show it.

## Procedure

1. **Ask for the launch artifact**: the pod spec / `args:` block / compose
   `command:` for **both** roles, and the `kubectl exec … -- cat
   /proc/1/environ | tr '\0' '\n' | grep -i ld_library` equivalent (or
   `python -c "import os;print(os.environ['LD_LIBRARY_PATH'])"` *inside* the
   running container). One of these turns the whole question into a diff.
2. **Find the file, then decide the class.**
   `find / -name 'ascend_transport*.so' 2>/dev/null` and
   `python -c "import mooncake, os; print(mooncake.__file__); print(os.listdir(os.path.dirname(mooncake.__file__)))"`.
   Present somewhere → path problem (step 3-4). Absent → image/build problem
   (step 6).
3. **Which copy is loaded** (image copies are routinely doubled): the image builds
   mooncake from source (`Dockerfile:38-39`, `Dockerfile.a3:39-40`,
   `Dockerfile.openEuler:39,45`, `Dockerfile.a3.openEuler:39,45` →
   `tools/mooncake_installer.sh -y`) *and* the CANN toolkit publishes its own copy
   under `<toolkit>/python/site-packages/mooncake`.
4. **Read the paths the tutorials prescribe, and quote them.** Both PD-Mooncake
   tutorials end their install section with the export and a note:
   `docs/source/tutorials/pd_disaggregation_mooncake_single_node.md:138,141` and
   `pd_disaggregation_mooncake_multi_node.md:217,220` —
   *"Ensure `/usr/local/lib` and `/usr/local/lib64` are in your `LD_LIBRARY_PATH`"*
   then `export LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH`;
   `DeepSeek-V3.1.md:300` and `Qwen3-235B-A22B.md:422` use the toolkit copy. Give
   the reporter the variant that matches the `find` hit, and put it in the launch
   command, not in a profile.
5. **Restart both roles and verify from inside** (`/proc/<pid>/environ`), because a
   profile-file export "fix" looks identical to no fix until the next `kubectl
   rollout`.
6. **Only if `find` returns nothing** does the image own it: check the tag is the
   platform variant for that SoC (A3 ≠ A2 ≠ 310P), and note that mooncake has been
   installed in the official images since v0.11.0rc3
   (`docs/source/user_guide/release_notes.md:261`, PR #4506), so an image that
   lacks it was rebuilt, is a plain `pip install mooncake-transfer-engine` (wheels
   carry no Ascend plugin), or is the wrong variant. The remedy is that image's
   own `tools/mooncake_installer.sh -y`, not a PyPI package.

## Do not

- Do not answer "your image is broken / the .so is absent" from the ENV dump alone.
  The dump is one shell's view; the launch command is the fact.
- Do not open RPATH / `$ORIGIN` / `LD_DEBUG` archaeology before steps 1-2 — the
  same trap list as `native-import-abort.md`, and it is the class of run that
  spends the budget and ships nothing.
- Do not treat "worked on the previous version" as an image regression when the
  previous run was launched by a different mechanism (interactive shell vs
  launcher `args:`); that is the exact inference #7352's triage ranked highest and
  the reporter later refuted.
