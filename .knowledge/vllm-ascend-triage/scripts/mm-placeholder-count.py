#!/usr/bin/env python3
"""Compare the two multimodal token counts in a VL input-assembly mismatch.

Placeholder side (prompt): area floor over each resized tile,
    (W*H) // (patch*patch*token_divisor),      divisor 16 for 16x, 4 for 4x
Feature side (vision tower + merge): per-stage per-dimension floor,
    (h//m1)*(w//m2) per stage, h = H//patch, w = W//patch

Origin: vllm v0.23.0 vllm/model_executor/models/minicpmv4_6.py:467-537
(placeholder formula) and :856-890 MiniCPMV4_6DownsampleMLP (per-stage merge,
merge_kernel_size=(2,2), `times` stages -> 16x uses 2 stages, 4x uses 1).
Recorded #14014 (196 placeholders vs 186 features); the page is
references/multimodal-inputs/placeholder-count-mismatch.md.

Run every Python invocation through uv:
  uv run --no-project python scripts/mm-placeholder-count.py --tile 392x392 --mode 16x
  uv run --no-project python scripts/mm-placeholder-count.py \
      --tile 434x434 --tile 434x434 --mode 16x --placeholders 116 --features 110

Tiles are the *resized* pixel size of each image or slice, i.e. the size the
image processor produced, not the original upload size. `--grid` is accepted as
an alternative when the paste prints the patch grid instead of pixels.
"""

from __future__ import annotations

import argparse
import sys


def parse_size(text: str) -> tuple[int, int]:
    sep = "x" if "x" in text else ","
    a, b = text.lower().split(sep, 1)
    return int(a), int(b)


def placeholder_count(h: int, w: int, patch: int, divisor: int) -> int:
    return (h * w) // (patch * patch * divisor)


def merge_count(h: int, w: int, stages: int, merge: int, patch: int) -> tuple[int, bool]:
    """Per-stage floors over the patch grid; exact == False means the reshape raises."""
    ch, cw = h // patch, w // patch
    exact = True
    for _ in range(stages):
        if ch % merge or cw % merge:
            exact = False
        ch, cw = ch // merge, cw // merge
    return ch * cw, exact


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", action="append", default=[],
                    metavar="HxW", help="resized pixel size of one image or slice (repeatable)")
    ap.add_argument("--grid", action="append", default=[],
                    metavar="HxW", help="patch grid instead of pixels (repeatable)")
    ap.add_argument("--patch", type=int, default=14, help="vision patch size (default 14)")
    ap.add_argument("--mode", choices=["16x", "4x"], default="16x",
                    help="downsample_mode / use_vit_merger: 16x = full merge, 4x = merger skipped")
    ap.add_argument("--merge", type=int, default=2, help="merge kernel per dimension per stage (default 2)")
    ap.add_argument("--stages", type=int, default=0,
                    help="override the stage count (default: 2 for 16x, 1 for 4x)")
    ap.add_argument("--placeholders", type=int, default=0,
                    help="M from 'to M placeholders'")
    ap.add_argument("--features", type=int, default=0,
                    help="N from 'Attempted to assign N = N multimodal tokens'")
    args = ap.parse_args(argv)

    stages = args.stages or (2 if args.mode == "16x" else 1)
    divisor = args.merge ** (2 * stages) if args.merge != 2 else (4 ** stages)

    tiles: list[tuple[int, int]] = [parse_size(t) for t in args.tile]
    grids: list[tuple[int, int]] = [parse_size(g) for g in args.grid]
    if not tiles and not grids:
        ap.error("give at least one --tile or --grid")

    print(f"patch={args.patch} merge={args.merge}x{args.merge} stages={stages} "
          f"mode={args.mode} area-divisor={divisor}")
    print(f"{'tile(px)':>11} {'grid':>9} {'placeholders':>12} {'features':>8}  exact")
    ph_total = ft_total = 0
    for h, w in tiles:
        ph = placeholder_count(h, w, args.patch, divisor)
        ft, exact = merge_count(h, w, stages, args.merge, args.patch)
        ph_total += ph
        ft_total += ft
        print(f"{h}x{w:>4} {h // args.patch}x{w // args.patch:>3} {ph:>12} {ft:>8}  "
              f"{'yes' if exact else 'NO -> reshape raises at merge'}")
    for gh, gw in grids:
        ph = (gh * gw) // divisor
        ft, exact = merge_count(gh * args.patch, gw * args.patch, stages, args.merge, args.patch)
        ph_total += ph
        ft_total += ft
        print(f"{gh * args.patch}x{gw * args.patch:<4} {gh}x{gw:>3} {ph:>12} {ft:>8}  "
              f"{'yes' if exact else 'NO -> reshape raises at merge'}")

    print(f"{'TOTAL':>11} {'':>9} {ph_total:>12} {ft_total:>8}")
    if len(tiles) + len(grids) > 1:
        print("(sum over tiles: compare with one request's M/N only if every tile belongs to it)")

    if args.placeholders or args.features:
        m = args.placeholders or ph_total
        n = args.features or ft_total
        print(f"\nreported: {m} placeholders, {n} features -> "
              f"{'M > N: prompt promised more placeholders than the vision path produced'
                 if m > n else 'M < N: features exceed placeholders (one image run lost)'
                 if m < n else 'equal: no count mismatch'}")
        print(f"computed: {ph_total} placeholders, {ft_total} features "
              f"(delta {m - ph_total:+d} / {n - ft_total:+d} vs formula)")
    print("\nlever checklist: per-request mm_processor_kwargs (max_slice_nums/"
          "downsample_mode/use_vit_merger reach the vision pass but not the "
          "placeholder call) | use_vit_merger=false is 4x features | placeholder "
          "text from a chat template or client, not the image processor")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
