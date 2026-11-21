#!/usr/bin/env python3
"""Predict the `sum_page_size_bytes % page_size_bytes` assertion before a launch.

The Ascend DSA/sparse-C8 KV cache splits one raw int8 tensor into
k / v / indexer-k / indexer-scale slices using `sparse_kv_cache_ratio`, then
asserts the byte sum of the slices is a whole number of `page_size_bytes`
(`vllm_ascend/worker/model_runner_v1.py`, `_reshape_kv_cache_tensors`).
This script recomputes both sides from the checkpoint's head dims, so the
investigator knows whether the assertion is arithmetic (no ablation can help)
or an internal inconsistency between two specs.

Two layouts exist in the recorded snapshot:

  legacy  <= vllm-ascend v0.22.1rc1 (5f6faa0)
          spec = vllm_ascend/patch/platform/patch_kv_cache_interface.py
          page in real bytes, ratio in bf16-element virtual units
          -> the four floors are exact only when index_head_dim is even.
  v023    >= vllm-ascend v0.23.0 (5cb98ca)
          spec = vllm_ascend/core/kv_cache_interface.py
          page and ratio are both byte weighted and both carry
          sfa_dcp_replicated_indexer_size
          -> exact by construction; a firing assert means the sizing spec and
             the reshaping spec are not the same object.

Usage (A3, sparse LI-C8, DCP 16, DeepSeek-V3.2-like dims):

  uv run --no-project python kv-page-size-arithmetic.py \
      --layout v023 --mode li_c8 --soc A3 --dcp 16 \
      --k-head-dim 512 --v-head-dim 64 --index-head-dim 128

Exit code 0 when the identity holds for every simulated page count, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys


def dtype_size(kind: str, soc: str) -> int:
    if kind == "kv":
        # v0.23.0 A5 packs ckv as float8; A3 keeps bf16 k/v.
        return 1 if soc == "A5" else 2
    if kind == "c8_k":
        return 1  # int8 on A3, float8_e4m3fn on A5
    if kind == "c8_scale":
        return 4 if soc == "A5" else 2  # float32 on A5, float16 on A3
    raise ValueError(kind)


def legacy(kind: str, soc: str, dcp: int, k: int, v: int, idx: int) -> tuple[int, list[int], list[int]]:
    """v0.22.1rc1: page in real bytes, ratio in bf16-element virtual units."""
    del dcp  # the legacy spec has no DCP replication term
    num_heads_per_page = 1  # factored out in main()
    if kind != "li_c8":
        raise SystemExit("legacy layout only implemented for --mode li_c8")
    # A3: kv_lora + k_rope in self.dtype, indexer in c8 dtype, scale in fp16
    page = num_heads_per_page * ((k + v) * dtype_size("kv", soc) + idx * dtype_size("c8_k", soc) + dtype_size("c8_scale", soc))
    factor = dtype_size("kv", soc) // dtype_size("c8_k", soc)
    virtual = [k, v, idx // factor, 1]
    return page, virtual, virtual[:]


def v023(kind: str, soc: str, dcp: int, k: int, v: int, idx: int) -> tuple[int, list[int], list[int]]:
    """v0.23.0: page and ratio both byte weighted, both DCP-replication aware."""
    if kind == "li_c8":
        page = (k + v) * dtype_size("kv", soc) + idx * dcp * dtype_size("c8_k", soc) + dcp * dtype_size("c8_scale", soc)
        virtual = [
            k * dtype_size("kv", soc),
            v * dtype_size("kv", soc),
            idx * dcp * dtype_size("c8_k", soc),
            dcp * dtype_size("c8_scale", soc),
        ]
    elif kind == "sfa_c8":
        if soc != "A5":
            raise SystemExit("--mode sfa_c8 is the A5 merged-CKV layout (float8 ckv, float32 scale); "
                             "on A3 the spec asserts qk_rope_head_dim == 0 and the ckv dtype is int8")
        # ckv merged (kv_lora + rope packed), indexer in qli dtype, scale fp
        page = k * dtype_size("c8_k", soc) + idx * dcp * dtype_size("c8_k", soc) + dcp * dtype_size("c8_scale", soc)
        virtual = [
            k * dtype_size("kv", soc),
            idx * dcp * dtype_size("c8_k", soc),
            dcp * dtype_size("c8_scale", soc),
        ]
    else:
        raise SystemExit("--mode must be li_c8 or sfa_c8")
    return page, virtual, virtual[:]


def floor_splits(total: int, virtual: list[int], exact: bool) -> list[int]:
    """Mimic `int(kv_cache_tensor.size // (total_virtual / v_i))`."""
    tv = sum(virtual)
    if exact:
        return [total * v // tv for v in virtual]
    return [int(total // (tv / v)) for v in virtual]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=["legacy", "v023"], default="v023")
    ap.add_argument("--mode", choices=["li_c8", "sfa_c8"], default="li_c8")
    ap.add_argument("--soc", choices=["A3", "A5"], default="A3")
    ap.add_argument("--dcp", type=int, default=1)
    ap.add_argument("--k-head-dim", type=int, required=True, help="kv_lora_rank (or the k dim)")
    ap.add_argument("--v-head-dim", type=int, required=True, help="qk_rope_head_dim (or the v dim)")
    ap.add_argument("--index-head-dim", type=int, required=True, help="index_head_dim / indexer cache dim")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--num-kv-heads", type=int, default=1)
    ap.add_argument("--pages", type=int, default=64, help="simulated num_blocks")
    args = ap.parse_args(argv)

    build = legacy if args.layout == "legacy" else v023
    page_per_head, virtual, _ = build(args.mode, args.soc, args.dcp, args.k_head_dim, args.v_head_dim, args.index_head_dim)
    heads = args.block_size * args.num_kv_heads
    page = page_per_head * heads
    tv = sum(virtual)

    print(f"layout={args.layout} mode={args.mode} soc={args.soc} dcp={args.dcp} "
          f"block_size={args.block_size} num_kv_heads={args.num_kv_heads}")
    print(f"head dims: k={args.k_head_dim} v={args.v_head_dim} index={args.index_head_dim}")
    print(f"page_size_bytes      = {page}  (per head-page {page_per_head})")
    print(f"virtual terms        = {virtual}  total={tv}")
    print(f"2*total_virtual      = {2 * tv}   (legacy exactness needs page_per_head == 2*total_virtual)")

    bad = []
    for pages in (1, args.pages, args.pages + 1, 7):
        total = page * pages
        for exact in (True, False):
            splits = floor_splits(total, virtual, exact)
            s = sum(splits)
            ok = s % page == 0
            tag = "rational" if exact else "float-ratio"
            if not ok:
                bad.append((pages, tag, s, total, total - s))
            if pages == args.pages:
                print(f"  pages={pages:<6} {tag:<11} splits={splits} sum={s} "
                      f"tensor={total} deficit={total - s} sum%page={s % page} {'OK' if ok else 'ASSERT'}")
    print()
    if bad:
        print("VERDICT: the identity does NOT hold for this template -> `assert sum_page_size_bytes % "
              "page_size_bytes == 0` fires at engine start for this checkpoint's dims. Arithmetic, not "
              "memory/cards/HCCL; no flag ablation changes it (fix the spec arithmetic or the dims).")
        for pages, tag, s, total, deficit in bad[:4]:
            print(f"  pages={pages} {tag}: sum={s} tensor={total} deficit={deficit}")
        return 1
    print("VERDICT: the identity holds for this template -> the assertion is NOT arithmetic. Compare the "
          "spec used to size the tensor with the spec used in _reshape_kv_cache_tensors (two spec "
          "construction sites / c8 flag sets / sfa_dcp_replicated_indexer_size), then ablate one flag.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
