# A3 two-machine P/D with no supernode plane: which transport the rank table declares

Scope: a **pre-deployment plane question**, no log and no serve command —
"`v0.11.0rc0` Atlas 800T A3 双机 PD 分离部署，当前是否支持走参数面网络通信？", "两台机器只能
通过参数面 RoCE 通信，超平面不通，没有灵衢交换机". The deliverable is a capability verdict
plus the one command block that says which plane the P/D rank table actually
addresses. Not this page: a deployment that starts and cannot move KV
(`index.md`), a KV-pool store fault (`kv-pool-store-segments.md`), or a plane the
cluster has but a Pod cannot reach (`transport-topology.md`).

Worked case: #3959, vllm-ascend `v0.11.0rc0` (`00ba071`), vllm `v0.11.0rc3`
(`8ce5d31`). Inherit the rules in `SKILL.md` and `index.md`; rule 3 there
("not in the documented recipe" is not "unsupported") is the whole framing.

## 1. Four reads, no source walk

1. **What the tree validates.** `docs/source/tutorials/multi_node_pd_disaggregation.md:7`
   is the only PD tutorial at this ref and it is **A2**, three servers, 1P2D, at
   v0.10.1rc1; `docs/source/user_guide/support_matrix/supported_features.md:25` says
   "Prefill Decode Disaggregation 🟢 Functional … xPyD is supported" with no SoC
   qualifier. The A3 two-node tutorials that do exist are DP/TP, not PD
   (`multi_node_kimi.md:8`, `multi_node_qwen3vl.md:8`). Verdict line: the feature is
   supported, **no A3 two-machine-without-UB recipe is shipped at this ref**.
2. **Where A3 gets its `device_ip`.** `examples/disaggregated_prefill_v1/gen_ranktable.py:78-101`:
   A3 fills `device_ip` from `hccn_tool -i <id> -vnic -g` (`:80`) and adds
   `super_device_id` from `npu-smi info -t spod-info … SDID` (`:83`) and `super_pod_id`
   from `Super Pod ID` (`:85-86`); the non-A3 branch fills `device_ip` from
   `hccn_tool -i <id> -ip -g` (`:88-90`). `server_count` is `str(world_size)`
   (`:121-122`) and `gen_ranktable.sh` launches `torchrun --nproc_per_node 1
   --nnodes NNODES`, so it equals the number of **machines**. This branch is the
   highest-value finding of the case: on A2 the table carries the `-ip -g` value (the
   address HCCL is given as `device_ip` for the parameter plane, and the one the e2e
   PD script reads), on A3 it carries a `vnic` address instead, and nothing in the
   snapshot defines what plane `-vnic` reports (see §4).
3. **What the connector hands HCCL.** `vllm_ascend/distributed/llmdatadist_c_mgr_connector.py`:
   `:681 rank_table["server_count"] = "1" if is_same_server else "2"`; each device
   entry carries `("device_ip", …)` and `("super_device_id", …)` **filtered by
   `if v is not None`** (`:688-724`); the `super_pod_list` block exists only under
   `if self.soc_info == AscendSocVersion.A3` (`:731-754`); the finished table is
   logged (`:758 logger.info(f"rank table \n{rank_table}")`) and passed to
   `self.llm_datadist.link(...)` (`:762`). So two machines ⇒ `server_count "2"` plus
   `device_ip`, i.e. HCCL's own multi-server data path (`'device_ip' is not set
   correctly, must be set when multi Server or HCCL_INTRA_ROCE_ENABLE enabled`,
   `transport-topology.md` §2); the **only** place a supernode identity reaches HCCL
   is `super_pod_list` / `super_device_id`.
4. **Availability, before one more file.** `SOURCES.json` at this ref lists
   `cann/hccl`, `cann/hcomm`, `cann/hixl`, `cann/driver` as
   `no_verified_public_reference_by_opening` (only `cann/catlass` is exported), and
   `kvcache-ai/Mooncake` `v0.3.4.post2` contains **no** `ascend` string at all — its
   transports are `nvlink/rdma/tcp/nvmeof/cxl` — although vllm-ascend's own guide
   cites `doc/zh/ascend_transport.md`. Both are dead ends for this question; say so
   instead of walking them.

## 2. The trap a hand-edited A3 table walks into

`read_agent_metadata` takes `super_pod_id` with `.get("super_pod_id", None)`
(`:472`) and `super_device_id` the same way. Strip those keys from a table to "force
the parameter plane" and **both** sides become `None`, so
`is_same_pod = (remote == local)` at `:667` is `True` while `is_same_server` stays
False — and the connector emits **one** `super_pod_list` entry listing both machines
as a single super pod (`:734-744`). That is the opposite of the intent. Recorded
worker logs show the `None` shape often enough to recognise it:
`LLMDataDistCMgrAgentMetadata(super_pod_id=None, …, super_device_id=None)` (#2498:232,
#1680:341). If the table must be hand-written, give each machine a **distinct**
`super_pod_id` rather than deleting the fields.

## 3. Answer shape and ladder

Open with: PD over the parameter plane is not a knob — the connector declares two
servers with `device_ip` and lets HCCL choose the plane; on A3 it *additionally*
declares a supernode grouping. So "supported" is a mechanism statement and "validated
on A3 without a 灵衢 switch" is not available at this ref. Then:

1. **One block on each machine**, taken from the tree's own multi-node verification
   set (`docs/source/tutorials/multi_node.md:32-56`):
   `hccn_tool -i N -link -g`, `-net_health -g`, `-i N -ip -g | grep ipaddr`,
   `-i N -vnic -g` (A3), and `hccn_tool -i 0 -ping -g address <peer address>`.
   Whichever address answers the ping is the one that belongs in `device_ip`.
2. **Minimal pair**: `--tensor-parallel-size 1`, `--enforce-eager`, a small model, and
   the tutorial's `LLMDataDistCMgrConnector` with
   `DISAGGREGATED_PREFILL_RANK_TABLE_PATH` (`multi_node_pd_disaggregation.md:104-122`),
   table with `server_count` = 2 and `device_ip` from step 1.
3. **plog on both sides** with `ASCEND_GLOBAL_LOG_LEVEL=1`; grep
   `server_count|super_pod_id|device_ip|isUsedRdmaLevel0|HccsNum`. The connector's own
   `rank table` line (`:758`) is in the worker log *before* the plog, so the table
   HCCL received is already quotable.
4. **Do not recommend `tests/e2e/pd_disaggreate/` as the multi-node recipe.** At this
   ref `tests/e2e/run_disagg_pd.sh:32,35` takes both `device_ip`s from `hccn_tool`
   on **one** host (`-i 0` and `-i 1`), and the `AscendSimpleConnector` its
   `setup_pd.sh:34,85` configures has no implementation anywhere in `vllm_ascend/`
   (registered connectors are `LLMDataDistCMgrConnector`, `MooncakeConnectorV1`,
   `AscendStoreConnector`, `distributed/__init__.py:21-32`). It is stale, not
   validated.
5. **Precedent search is one grep, not a title sweep.** `grep -rl
   'super_device_id\|vnic\|参数面\|灵衢' /workspace/history` returns exactly #1680 and
   #2498 (both the `None` shape) and no plane-named issue: "nobody has filed this
   combination" is the finding. Broad Chinese tokens (`A3`, `双机`, `roce`) return
   hundreds of unrelated bodies and only spend the budget.

## 4. Evidence limits to state

- Which plane HCCL *selects* for a `server_count: "2"` table is not readable in this
  snapshot (`cann/hccl`, `cann/hcomm`, `cann/hixl` unavailable): source-verified is
  only *what the table declares*.
- `hccn_tool -vnic` semantics have no source here (`cann/driver` unavailable), so
  "the A3 branch points at the wrong plane" is a **hypothesis with a one-command
  test**, never a diagnosis. Keep the two claims separate.
- The Mooncake alternative carries the same gap: `mooncake_connector.py:801`
  (`use_ascend_direct`, default `False`) and `:843`
  (`self.engine.initialize(hostname, "P2PHANDSHAKE", "ascend", …)`) show the protocol
  string, not the plane; the Ascend transport implementation is not in the export.
- Version scope: vllm-ascend `v0.11.0rc0` (`00ba071`) + vllm `v0.11.0rc3` (`8ce5d31`);
  `A3` is `AscendSocVersion.A3` (`vllm_ascend/utils.py:554-557`, 910_93). The
  `-vnic` branch, the `server_count` expression and the `super_pod_list` block are
  all one-line reads that later trees may move — re-grep before quoting.

## 5. The recorded run (#3959)

The opening paragraphs were right — verdict plus the `hccn_tool` comparison — so the
cut was survivable, but the session then spent its whole 16 K output budget on four
expanding ranked causes and was cut inside cause 4, with no scope/send-back close and
no "what I could not verify". 26 assistant turns, **25 of them bare tool calls**
(2.3 K characters of prose in the whole session, all of it in the last message), 39
tool calls against a 16 K output cap. The detour was ~8 calls into `kvcache-ai/Mooncake` and `Ascend/mind-cluster`
(plane names, `super_device_id`, `vnic`): one look at the tree's availability list and
one `grep -rl` per third-party repo would have said "no source, ask for the plog".
For a no-log plane question the body is: verdict, the ping block, two ranked rows,
3-5 lines of limits.

**Replay (same paste, three turns / six calls).** `sources-summary.sh` plus this page,
then four greps — the `gen_ranktable.py:78-101` A3 branch, the connector's
`server_count`/`super_pod_list` lines, the support-matrix row and PD-tutorial scope, and
the one `grep -rl` precedent search — and the answer shipped whole: verdict, the
`hccn_tool`/`ping` block, the `super_pod_id` trap, the stale `tests/e2e` warning and the
limits. A page that already holds the case is the stopping condition, not a starting
point: the replay needed no Mooncake or mind-cluster walk.
