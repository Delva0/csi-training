#!/usr/bin/env python3
"""Reproduce the torch_npu `gather` emulation with no vLLM in the way.

Why this exists: vLLM's tensor-parallel logits gather ends in
`torch.distributed.gather`, and HCCL has no `gather`, so torch_npu emulates it with a
`dist.broadcast_object_list` of the per-rank sizes followed by one allgather
(`torch_npu/distributed/distributed_c10d.py` `_gather`: sizes at `:98-99`, object
broadcast at `:110` for the default group / `:126` for an explicit group, allgather at
`:132`). A 310P TP=8 `profile_run` died inside that handshake with
`EOFError: Ran out of input` (#1579). Running the primitive alone on the same host
decides whether the image's object collective owns it, before anyone re-reads the
serve command.

Usage, inside the reporter's image, on the node that failed:

    torchrun --nproc_per_node=2 <this script>                 # default group (:110)
    torchrun --nproc_per_node=2 <this script> --group         # explicit group (:126)
    torchrun --nproc_per_node=8 <this script> --group --tokens 8

Reading the result:
  * `GATHER OK` on every rank -> the emulation works at this world size; the logits
    tensor's shape/dtype/`storage_offset` path is back in play.
  * `EOFError: Ran out of input` (or a hang until the HCCL timeout) -> the image's
    torch_npu/CANN object collective is the owner; file there with the line number.
  * fails at 8 but passes at 2 -> world-size dependence, quote both runs.

The probe is fp16 by default (the 310P serve's dtype) and 2-D, matching the logits
shape (`tokens` rows x a sharded-vocab column count). It never calls vLLM.
"""

import argparse
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Standalone torch_npu gather-emulation probe (HCCL has no gather).")
    p.add_argument("--group", action="store_true",
                   help="pass an explicit new_group (the :126 branch of _gather)")
    p.add_argument("--tokens", type=int, default=8,
                   help="rows of the probe tensor (default 8)")
    p.add_argument("--cols", type=int, default=151936,
                   help="full column count, divided by world size into a per-rank "
                        "shard (default 151936 = Qwen3 vocab)")
    p.add_argument("--dtype", default="float16",
                   choices=("float16", "bfloat16", "float32"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        print(f"import failed ({type(exc).__name__}: {exc}); run this inside the "
              f"image that serves the model", file=sys.stderr)
        return 2

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local)

    dtype = getattr(torch, args.dtype)
    cols = max(1, args.cols // max(1, world))
    t = torch.full((args.tokens, cols), float(rank + 1), dtype=dtype,
                   device=f"npu:{local}")
    group = dist.new_group(list(range(world))) if args.group else None
    dst = 0
    gather_list = ([torch.empty_like(t) for _ in range(world)]
                   if rank == dst else None)

    try:
        dist.gather(t, gather_list, dst, group=group)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        print(f"GATHER FAILED rank {rank}/{world} group="
              f"{'explicit' if args.group else 'default'}: "
              f"{type(exc).__name__}: {exc}")
        return 1

    if rank == dst:
        sums = sorted({float(g.float().sum()) for g in gather_list})
        expect = [round(args.tokens * cols * float(r + 1)) for r in range(world)]
        ok = [round(s) for s in sums] == sorted(expect)
        print(f"GATHER OK rank {rank}/{world} group="
              f"{'explicit' if args.group else 'default'} "
              f"rows={args.tokens} cols_per_rank={cols} dtype={args.dtype} "
              f"values_ok={ok} per_rank_sums={[round(s) for s in sums]}")
    else:
        print(f"GATHER OK rank {rank}/{world} group="
              f"{'explicit' if args.group else 'default'} (non-destination)")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
