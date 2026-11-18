# NPU device visibility: the container cannot reach the cards it was given

## Scope

A report where the NPU "cannot be called" / the model cannot load, or the process lands on another
tenant's chips: the same command works on an idle host and fails while another container runs, or only
part of the ranks fail. The owner is the **device-visibility layer** — the container runtime
(`ASCEND_VISIBLE_DEVICES` at `docker run`), the HDK driver's device namespace, and vLLM's
rank → physical-device mapping (`ASCEND_RT_VISIBLE_DEVICES`) — not a `vllm_ascend/` compute path.
Distinct from a port collision (`../symptoms/startup-port-conflicts.md`), a device kernel fault
(`../symptoms/device-kernel-faults.md`) and a load-time quantisation failure
(`../symptoms/quantization-load.md`).

## Common rules

- **Two variables, two layers.** `ASCEND_VISIBLE_DEVICES` is read by the **container runtime** when
  the container is created (which `/dev/davinciN` nodes and driver state it gets); the `-e` must be on
  `docker run`. `ASCEND_RT_VISIBLE_DEVICES` is the **in-process** list that torch_npu/CANN and vLLM
  use. Setting only the second one, inside a container whose device namespace was built without the
  first, is not the same configuration — and it is what fails in the worked case below.
- **vLLM reaches the hardware through one variable only**: `NPUPlatform.device_control_env_var =
  "ASCEND_RT_VISIBLE_DEVICES"` (`vllm_ascend/platform.py:101`, same line at `v0.14.0rc1` and at
  `main`). Everything upstream then goes through it:
  `Platform.device_id_to_physical_device_id` (`vllm/platforms/interface.py:202-215` at vLLM `v0.14.0`)
  returns the rank id **unchanged** when the variable is unset or empty and `int(device_ids[device_id])`
  when it is set, so without it local ranks 0..N-1 are used as raw device ids in the container;
  `vllm/v1/engine/utils.py:177-215` (`set_device_control_env_var` / `get_device_indices`) rewrites the
  same variable for the engine subprocess.
- **vllm-ascend reads the same variable in two places and deliberately defers the runtime**:
  `vllm_ascend/cpu_binding.py:11` (`ASCEND_RT_VISIBLE_DEVICES = os.getenv(...)`, used at `:75-76` for
  CPU pinning) and `vllm_ascend/utils.py:246`, whose `enable_custom_op()` docstring says the lazy
  `vllm_ascend_C` import exists **"to avoid early initialization of CANN's RTS component. Ensure that
  `ASCEND_RT_VISIBLE_DEVICES` can be dynamically modified before `torch.npu.set_device()`"**. Exporting
  it after the NPU runtime is initialised is therefore too late for some paths.
- **`ASCEND_VISIBLE_DEVICES` is not in the code.** `git grep ASCEND_VISIBLE_DEVICES v0.14.0rc1` is empty
  (no `vllm_ascend/` file reads it at that ref). The single in-repo occurrence is a docker example at
  `main`, `docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md:38-39`, and it passes
  **both** variables with the same `${DEVICE_LIST}`:
  `-e ASCEND_VISIBLE_DEVICES=${DEVICE_LIST}` *and* `-e ASCEND_RT_VISIBLE_DEVICES=${DEVICE_LIST}`. That
  pair is the shape to copy; an answer that names only one of them reproduces the reporter's failure.

## Worked case: #6942 (NPU not callable, model cannot load, only while another container runs)

`vllm-ascend:v0.14.0rc1` image, Kylin V10 SP3, HDK 25.5, `docker run --device /dev/davinci6
--device /dev/davinci7 …` plus the shared `/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`;
`vllm serve /data/model/Qwen3-30B-A3B --tensor-parallel-size 2 --enable-expert-parallel`. The service
starts only when no other container is running on the host; the log was posted as screenshots, so no
error string is available.

- **Reporter-confirmed lever** (`marcushan84`, 2026-03-09): `-e ASCEND_VISIBLE_DEVICES=6,7` **at
  `docker run`** *and* `export ASCEND_RT_VISIBLE_DEVICES=6,7` inside the container. Each variable alone
  did not work — the maintainer's first suggestion (`handerason`, 2026-03-06) was the in-container
  `export` only. `reported`.
- **Mechanism** `hypothesis` with its falsifier, not a code defect: the plain `--device` list gives the
  container two davinci nodes while the driver inside it still enumerates through
  `davinci_manager`; with `ASCEND_RT_VISIBLE_DEVICES` unset, vLLM's rank→device mapping is the identity
  (rule above), so TP ranks 0/1 address device ids 0/1 — the other tenant's chips — and the load fails.
  Falsifier: `npu-smi info` and `torch.npu.device_count()` inside the container, with and without each
  variable; if the ids already resolve to 6/7 with one variable set, visibility is not the mechanism.
- **Answer shape**: workaround = the env pair, with the container-start half called out; root cause =
  the visibility layer with the verified anchors above and the falsifier; status = reporter-confirmed
  workaround, maintainer-closed as "resolved in the latest version" without a code change. This is an
  operator-owned failure: there is no `vllm_ascend/` line to hunt, and a run that spends its batch
  searching one produces no answer.

## Falsifying checks before reusing this route

- The failure follows the **model**, not the host or the co-tenant (also fails on an idle host) → a
  load/quantisation class (`../symptoms/quantization-load.md`).
- `ASCEND_RT_VISIBLE_DEVICES` is set correctly *and* the fatal ids still name other chips → read the
  launcher that spawns workers (`vllm/v1/engine/utils.py`, ray/V1 dp launchers), which may rewrite the
  variable for children.
- The report is on a `-310p` image with a `TransData`/op-store error → that is the SoC/image class, not
  visibility (`../symptoms/graph-capture-and-custom-ops.md` §B).

## Recheck conditions

The container runtime and HDK generation decide whether physical ids survive into the container, and
the reported values (6,7 with HDK 25.5) are the only ones verified here — do not generalise the
numbering. Re-grep `device_control_env_var` and `docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md`
at the reported tag, and re-read `Platform.device_id_to_physical_device_id` at the tag paired with that
release (it is still present at `main`, `vllm/platforms/interface.py:313-364`; `v0.14.0` is the tag the
line above was read at, not necessarily the reporter's); a release whose image entrypoint sets both
variables itself makes the pair unnecessary. Static reading only: no NPU here, so nothing in this page was reproduced.
