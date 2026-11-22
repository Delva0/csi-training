#!/usr/bin/env python3
"""checkpoint-header-scan.py — one call: is every ``*.safetensors`` shard in a
model directory actually a safetensors *container*?

Why: ``safe_open`` parses the first 8 bytes of the file as a little-endian u64
header length and refuses an implausible one with
``safetensors_rust.SafetensorError: Error while deserializing header:
HeaderTooLarge`` (also ``HeaderTooSmall`` / ``invalid JSON``). That happens
before a single tensor is read, so the failure is a property of the FILE, not of
TP, the card, the image or the model code. The usual contents are an unresolved
Git-LFS pointer (``git clone`` without git-lfs), an HTML/JSON error page saved
under a ``.safetensors`` name, or a partially copied shard.

Usage:
  uv run --no-project python checkpoint-header-scan.py /model/Qwen2.5-VL-7B-Instruct
  uv run --no-project python checkpoint-header-scan.py '/model/*.safetensors'

Exit status: 0 = every shard parsed, 1 = at least one shard is not a safetensors
file (that file is the failure), 2 = nothing scanned. Read the report, not just
the status.

It also checks the **data region**: the header's own ``data_offsets`` say how many
weight bytes must follow the header, so a shard whose file ends before that
(interrupted copy, torn rsync, a partial re-download that kept the final name) is
``BAD`` even though its header parses. That is the "loads fine, then serves
garbage" family of a damaged local copy (recorded #6136: a re-downloaded
checkpoint fixed a `!!!!` answer) — see references/accuracy-benchmarks/
constant-token-degeneracy.md Step 3b.

Caveat: the 100 MB cap is the safetensors library's own ``MAX_HEADER_SIZE``; it
is not in any exported tree, so a shard this script passes is "looks like a
complete container", not "hashes match the hub" — ask for the download method and
compare each shard's byte size (or `sha256sum`) with the hub entry next.
"""

import glob
import json
import os
import struct
import sys

MAX_HEADER = 100 * 1024 * 1024
LFS_MAGIC = b"version https://example.invalid"


def classify(path):
    """Return (verdict, detail) for one file. verdict: 'ok' | 'bad'."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
    if len(raw) < 8:
        return "bad", f"file is {size} B, shorter than the 8-byte header length"
    (length,) = struct.unpack("<Q", raw)
    if raw.startswith(LFS_MAGIC[:8]) or (size < 4096 and raw.startswith(b"version ")):
        return "bad", ("Git-LFS pointer, not weights (git clone without git-lfs / "
                       "GIT_LFS_SKIP_SMUDGE=1): re-download with `hf download` / "
                       "modelscope")
    if raw.lstrip(b"\x00") == b"":
        return "bad", "first 8 bytes are zero (truncate/fallocate, or an empty copy)"
    if raw[:1] in (b"<", b"{") or raw.startswith(b"PK"):
        return "bad", f"starts with {raw[:6]!r}: HTML/JSON/zip body saved under a .safetensors name"
    if length == 0 or length > MAX_HEADER:
        return "bad", (f"header length {length} outside (0, {MAX_HEADER}]: "
                       "garbage prefix or an interrupted/overwritten copy")
    with open(path, "rb") as f:
        f.seek(8)
        blob = f.read(length)
    if len(blob) != length:
        return "bad", f"header says {length} B but only {len(blob)} B follow (truncated)"
    try:
        header = json.loads(blob.decode("utf-8", "replace"))
    except Exception as exc:                                    # noqa: BLE001
        return "bad", f"header is not JSON: {exc}"
    if not isinstance(header, dict) or not header:
        return "bad", "header JSON is not a tensor map"
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        return "bad", "header carries no tensors"
    # data_offsets are relative to the end of the header, so the largest end is
    # the number of weight bytes this shard promises; a shorter file is a short copy.
    extent = 0
    for name, info in tensors.items():
        offs = info.get("data_offsets") if isinstance(info, dict) else None
        if not (isinstance(offs, (list, tuple)) and len(offs) == 2):
            return "bad", f"tensor {name} has no data_offsets pair"
        extent = max(extent, int(offs[1]))
    expected = 8 + length + extent
    if size < expected:
        return "bad", (f"header declares {extent} B of tensor data ending at "
                       f"{expected} B but the file is {size} B: truncated / "
                       f"short copy")
    return "ok", f"{len(tensors)} tensors, data region {extent} B"


def main(argv):
    paths = []
    for arg in argv[1:]:
        hits = sorted(glob.glob(arg))
        paths.extend(hits if hits else [arg])
    if not paths:
        print("usage: checkpoint-header-scan.py <model-dir|*.safetensors> ...",
              file=sys.stderr)
        return 2

    bad = []
    scanned = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"{path}: absent — the shard the index names is not on disk")
            bad.append(path)
            continue
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
            if not files:
                print(f"{path}: no *.safetensors in this directory")
                continue
            index = os.path.join(path, "model.safetensors.index.json")
            referenced = set()
            if os.path.isfile(index):
                try:
                    with open(index) as f:
                        doc = json.load(f)
                    referenced = set(doc.get("weight_map", {}).values())
                except Exception as exc:                        # noqa: BLE001
                    print(f"{index}: unreadable ({exc})")
            on_disk = {os.path.basename(f) for f in files}
            for name in sorted(referenced - on_disk):
                print(f"index names {name}, not on disk")
                bad.append(name)
            for name in sorted(on_disk - referenced if referenced else set()):
                print(f"{name} on disk, not named by the index (renamed/copied in?)")
            paths_here = files
        else:
            paths_here = [path]

        for f in paths_here:
            scanned += 1
            verdict, detail = classify(f)
            size = os.path.getsize(f)
            print(f"{os.path.basename(f):<44} {size/2**30:8.3f} GiB  "
                  f"{'ok ' if verdict == 'ok' else 'BAD'}  {detail}")
            if verdict != "ok":
                bad.append(f)

    print()
    if not scanned:
        print("nothing scanned — point this at the model directory the serve "
              "command uses (the path in `Starting to load model <path>...`).")
        return 2
    if bad:
        print("VERDICT: at least one shard is damaged — not a container, or")
        print("shorter than the tensor data its own header declares. The engine")
        print("fails in the weight iterator (`safe_open`) before any device work")
        print("when the header is unusable, and reads a short copy as a broken")
        print("tensor later, so TP size, card, image and model support are not")
        print("involved: re-download or re-copy the named file(s). Ask which")
        print("tool produced the directory (`hf download`, `git clone`,")
        print("modelscope, rsync/netdisk) and compare each shard's byte size")
        print("with the hub.")
        print("See references/startup-failures/checkpoint-file-not-safetensors.md")
        print("and, for a serve that loaded but answers one repeated token,")
        print("references/accuracy-benchmarks/constant-token-degeneracy.md Step 3b.")
        return 1
    print("VERDICT: every shard parses as a container and holds the tensor data")
    print("its header declares, so a `HeaderTooLarge` / short-read report is NOT")
    print("explained by these files: ask for the full traceback and the exact")
    print("shard path from it (a second model directory, a stale copy, or")
    print("another rank's path can differ), then compare each shard's size or")
    print("`sha256sum` with the hub entry — this scan is structural, not a hash.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
