# RL weight sync (trl `vllm_serve` / ms-swift rollout): the singleton rank owns the fault

## When this page applies

A GRPO/PPO training job whose rollout is a separate `vllm serve` process, with both
sides' errors pasted at once:

```text
[rank0]: RuntimeError(… 'HCCL error: memory error')        # inside update_named_param
vllm_ascend/distributed/device_communicators/pyhccl.py:166 in broadcast
        → RuntimeError: HCCL error: not found specific resource
[rank1..6]: HcclAllgather, error code is 6
            EI0009 Transport init error … [Create][DestLink]Create Dest error! …
            createLink para:rank[6]… dst_rank[0]
```

`EI0009`/`DestLink` has its own page (`transport-destlink-ipc-import.md`) and it is
the **wrong lane here**: that page's hardware/driver ladder applies when the failed
edge set is a device or passthrough fault. In this shape the training side is
collateral.

## Step 1 — rank the errors by how many ranks report them

The fault belongs to the **singleton**: one process pastes a weight-update error
from a client RPC, six paste collective errors whose `dst_rank` is that same rank 0.
Rank 0 never reached `gather_object` because it was still inside
`update_named_param` (`_move_model_to_vllm`), so the others' allgather to it cannot
create a link. Say that in the first paragraph; it turns a six-rank "HCCL 故障" into
one channel to inspect. Ask for the plog only after this split, and expect the
in-window record on the singleton's device.

## Step 2 — identify what actually serves the channel at the pin

At vllm-ascend v0.10.2rc1 (`048bfd555319`):

- `vllm_ascend/distributed/device_communicators/pyhccl.py:32` `PyHcclCommunicator`.
- The tree does **not** alias it over upstream's `PyNcclCommunicator`: the only
  other mention is a TODO (`vllm_ascend/distributed/communicator.py:33`), and
  `rg 'pynccl|PyNccl'` finds nothing in the normal path. So a trl/ms-swift
  integration that lands in `pyhccl.py` does it through **its own patch** — the
  recorded traceback showed ms-swift's `swift/llm/infer/rollout.py`
  `patched_update_named_param`. Ask which patch supplies the class before ranking
  any trl-side behaviour as vllm-ascend's.
- The sentence is CANN's own: `HCCL_CHECK` formats with `HcclGetErrorString`
  (`pyhccl_wrapper.py:115-116`, `:203-209`), so `not found specific resource` =
  code 6 (`HCCL_E_NOT_FOUND`) and `memory error` = 3. Two different strings from the
  same call chain mean the client re-attempted a communicator already in a bad
  state — quote the code, not the sentence.

## Step 3 — use the init sequence to reject the easy stories

`pyhccl.py:94-122`: rank 0 calls `hcclGetUniqueId` under `torch.npu.device(device)`,
everyone else broadcasts it through the group (`group.broadcast_obj(src=0)` over the
`StatelessProcessGroup`), then `hcclCommInitRank(world_size, unique_id, rank)` — the
wrapper maps that to `HcclCommInitRootInfo` (`pyhccl_wrapper.py:216-219`, with a
root-info object rather than a raw id) — and finally a **warmup collective**:
`all_reduce(torch.zeros(1, device=device))` followed by `stream.synchronize()`.

Consequences worth stating instead of reading the connector:

- A group whose declared `world_size` exceeds the ranks that join blocks **in init**;
  it does not reach a first real broadcast. So a `NOT_FOUND` at
  `hcclBroadcast` is *not* "a peer never joined", whatever the trl `world_size`
  bookkeeping looks like.
- What remains is per-op: the device the communicator was built on versus the tensor
  and stream at the broadcast (the wrapper asserts the device and comments that a
  different device gives *illegal memory access*), the `src` rank against the group's
  own numbering, and whether that process already holds another HCCL communicator on
  the same card (trainer rank 0's world group). Check in that order; each is one
  printed line.
- A one-card server started with `ASCEND_RT_VISIBLE_DEVICES=<that card>` sees it as
  logical 0, so a hardcoded `device=0` in the server is *correct inside that
  process*; the interesting number is the pair (`world_size`, `rank`) the client
  passes to `init_communicator` versus how many ranks actually join.

## Step 4 — verify the channel by value, not by absence of exceptions

Reporter-side precedent (the corpus holds bodies only, no maintainer replies):
#459 asked this at 0.7.1 — *"推理节点和训练节点之间的参数通信是通过
PyNcclCommunicator.broadcast 实现的 … 在 NPU 部署，发现并未成功更新 vllm 的参数"* —
and #942's minimal two-process `StatelessProcessGroup` + `PyHcclCommunicator`
broadcast between `npu:0` and `npu:1` **received an all-zero tensor**: the op
returned success and moved nothing. So the first ladder rung is a two-process
broadcast of a known tensor with a value comparison (`#942`'s script shape), run on
the reporter's own visible-device pair. A clean exception there means the channel is
broken before any training configuration matters; silent zeros mean the same channel
"works" and the weight update is a no-op — a different report.

## Ladder, and the scope

1. Two-process broadcast of a known tensor with a value comparison (above).
2. Print on both processes: `ASCEND_RT_VISIBLE_DEVICES`, `torch.npu.device_count()`,
   current device, and the `world_size` / `rank` / `src` handed to
   `init_communicator`. They must describe one group of participants.
3. Ask which patch supplies `PyHcclCommunicator`, and whether the release has any
   validated weight-update path at all — grep `docs/` and `tests/` at the pin before
   proposing one (at v0.10.2rc1 there is no e2e test and no doc for RL weight sync).
4. Only then the plog of both processes (`~/ascend/log/debug/plog/plog-<pid>_*.log`).

Source-verified at vllm-ascend v0.10.2rc1 `048bfd555319`; the snapshot's `vllm` is
v0.9.1 while that image ships v0.10.2, so do not cite upstream line numbers across
that gap. The ownership split (singleton primary, training side cascade) is
source-and-log-verified; **why** the broadcast cannot resolve its resource is a
hypothesis until the standalone broadcast and the plog pair arrive.
