# HCCL env vars: what they select, and the support matrix that limits them

The launch script is where this class is decided. These are the vendor's documented semantics
(`cann/hccl` `docs/en/user_guide/hccl_env/` and `docs/zh/.../hccl_env/`, pinned at HEAD `f49377a5`,
`git describe` `v8.5.0-beta.1-1411-gf49377a5`). Re-read the page for the variable you quote — the
per-product lists grow — and say which product the answer is for: almost every value below is
supported on one family and not another.

## `HCCL_INTRA_PCIE_ENABLE` / `HCCL_INTRA_ROCE_ENABLE` — intra-server only

| `HCCL_INTRA_PCIE_ENABLE` | `HCCL_INTRA_ROCE_ENABLE` | intra-server link |
| --- | --- | --- |
| 1 | not set | PCIe |
| 1 | 0 | PCIe |
| 0 | 1 | RoCE |
| not set | 1 | RoCE |
| 0 | 0 | PCIe |
| not set | not set | PCIe |

- Both 1 is invalid; `PCIE=0` with `ROCE` not set is invalid; `ROCE=0` with `PCIE` not set is invalid
  (so `0 0` is legal but `unset 0` is not).
- **It selects the intra-server link and nothing else.** Cross-server (parameter-plane) communication
  still uses the device NIC. So "I exported `HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0`, why
  is it still doing RDMA init?" is answered by the flag's scope, not by a bug: a group that spans
  servers needs the NIC no matter what the intra-server flags say (`init-failures.md`, #1955).
- Product gate (`910b` tag in the doc): A2 **Box16 only** (plus Atlas 300T Pro for the `910` tag);
  **not** A3, not 950, not 310P.
- Box16 constraint (verbatim sense): Box16 has two modules, devices 0–7 and 8–15. In single-node use
  with PCIe intra-server links, if devices from both modules are used they must be used in matching
  numbers and the same plane — device 0 with 8, 1 with 9, and so on. Under RoCE intra-server this
  restriction does not apply. A topology question about a 16-device single node is answered here, not
  by counting the IPs in a plog.

## `HCCL_OP_EXPANSION_MODE` — where the collective is expanded

- A2 (`910b`) default `HOST`; also `HOST_TS`, `AI_CPU`, `AIV`. A3 default `AI_CPU`; 950 default
  `AICPU_TS`. An unsupported value falls back to the default instead of failing.
- `AIV` limits that decide real reports (identical in the A2 and A3 text unless marked):
  - **symmetric networking and inference features only**;
  - **does not support parallel communicators** — "multiple communicators cannot be simultaneously
    configured in AIV mode … otherwise, unexpected behavior may occur";
  - operator scope, which is the part that decides a multi-node run: on **A2**, `AlltoAllV`/`AlltoAllVC`
    are single-node-only in any mode and `AlltoAll`/`AllGather`/`ReduceScatter` are single-node-only
    **in graph mode** (`AllGatherV` only in single-operator mode); on **A3**, multi-node is allowed but
    only *within a SuperPoD* — cross-superpod is not supported in any of these operators;
  - A2 Box16: **cross-subrack communication is not supported**;
  - a failure shows as a kernel fault, `fault kernel_name=aiv_all_reduce_***`, and is then diagnosed
    by EI0002's task-exception path (`init-failures.md`);
  - the op timeout default differs: `HCCL_EXEC_TIMEOUT` is 1836 s under `HOST`/`HOST_TS` (A2) and
    1091 s under AIV, so "it hung for 20 minutes then timed out" is not the same wall under the two modes.
- `AI_CPU` under Ascend-IR graph: **at most 6 concurrent graphs per device**, else communication blocks
  because every AI CPU core is busy — relevant to a multi-graph / high-`max-num-seqs` capture run.
- What the plugin does with it (verified at `v0.26.0rc1`): `vllm_ascend/device/device_op.py:1140-1142`
  reads the variable only to map `CCU_SCHED` → `comm_mode="ccu"`; `AIV` is passed to HCCL untouched.
  One model patch **sets** it (`patch/platform/patch_minimax_m2_config.py:118-124`, "Set
  HCCL_OP_EXPANSION_MODE=AIV for MiniMax-M2 ACL graph capture"), and `patch/__init__.py:413` documents
  the same intent. So: the plugin does not own an AIV failure, but it can *choose* AIV for you.

## Timeouts, ports, logging

| var | default | use |
| --- | --- | --- |
| `HCCL_CONNECT_TIMEOUT` | 120 s (range 120–7200; the real wait adds 20 s for failure notification) | how long a link-establishment wait runs before EI0006. Raising it diagnoses ("is the peer late or absent?"), it does not fix |
| `HCCL_EXEC_TIMEOUT` | 1836 s (`HOST`/`HOST_TS` on A2, `AI_CPU`/`AICPU_CacheDisable` on A3), 1091 s (AIV) | collective execution wait; compare per-rank dispatch `timeStamp`s against it (EI0002) |
| `HCCL_NPU_SOCKET_PORT_RANGE` | NPU comm port 16666 | `auto` or a range when several processes share one NPU (EI0019 `listen … port[16666] fail`) |
| `HCCL_ENTRY_LOG_ENABLE` | off | `=1` logs every collective dispatch with `tag`, `count`, `dataType`, `streamId`, `comm` — the tool for "different op on different ranks" and for the same-comm/two-streams concurrency bug |

## The doc block versus the matrix: the reporter who already bisected (#18315)

Tutorials in the repo export the aggressive values. At `v0.27.1rc1`,
`docs/source/tutorials/models/GLM5.3.md:158` and `:169` are
`export HCCL_OP_EXPANSION_MODE="AIV"` and `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`,
inside the multi-node block for exactly "4 × Atlas 800 A2" (`:27`); same two lines in `GLM5.md:146,150`.
Reported case (`v0.26.0rc1`, GLM5.3-w8a8c8, 4 nodes × 4 devices, `--enable-expert-parallel`, title
quoting `v0.27.1RC1`): the engine dies in `aclnnMoeDistributeDispatchV4 failed, error code is 107020`
after a host-side `event wait timeout occurred during task execution … timeout=230s` /
`Notify record failed`, and **removing those two env vars lets it start** — the reporter's own A/B.

- When the reporter's A/B names the variable, that A/B *is* the workaround and it is
  reporter-confirmed: relay it, with the cost (loses AIV expansion / expandable segments).
- The mechanism is the matrix above, not a code hunt: AIV on A2/A3 requires symmetric inference
  networking, tolerates no parallel communicators, and in graph mode restricts AlltoAll/AllGather/
  ReduceScatter to a single node — a 4-node EP MoE run is outside that, and the failing op is the MoE
  all-to-all. Label it `hypothesis` with a falsifier: the same 4-node run with the env vars unset must
  start (already reported), and a single-node run with them set must *not* hit the same op failure.
- Also check the doc's own tested-image line before calling it a doc defect: at `v0.27.1rc1` GLM5.3 is
  stated as tested only on `v0.23.0`/`v0.23.0-a3` images in the multi-node co-located scenario, so a
  `v0.26.0rc1` run is outside the doc's own claim (`../symptoms/doc-guide-vs-capability.md`).

## Check before reuse

```bash
H=/workspace/repos/cann/hccl
git -C $H show HEAD:docs/en/user_guide/hccl_env/HCCL_INTRA_PCIE_ENABLE.md | sed -n '1,40p'
git -C $H show HEAD:docs/en/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md | sed -n '60,130p'
git -C $H show HEAD:docs/en/user_guide/hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md | sed -n '1,30p'
git -C $H show HEAD:docs/en/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md | sed -n '1,20p'
# the tutorial block the reporter copied (line numbers move between tags):
R=/workspace/repos/vllm-project/vllm-ascend
git -C $R ls-tree -r --name-only <tag> docs/source/tutorials/models | grep -i glm
git -C $R show <tag>:docs/source/tutorials/models/GLM5.3.md | grep -n 'HCCL_OP_EXPANSION_MODE\|PYTORCH_NPU_ALLOC_CONF'
```
