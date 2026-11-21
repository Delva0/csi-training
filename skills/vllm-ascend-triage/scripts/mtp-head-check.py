#!/usr/bin/env python3
"""Does this checkpoint actually carry the MTP (multi-token-predictor) head?

    uv run --no-project python mtp-head-check.py <model dir | config.json> [-q]

The MTP drafter is built from ``config.json`` alone: upstream
``vllm/config/speculative.py`` rewrites ``qwen3_5``/``qwen3_5_moe`` (and every
other ``MTPModelTypes`` family) into a ``*_mtp`` draft model and loads it from
the **target's own** checkpoint (``self.model = self.target_model_config.model``).
Nothing in that path looks at the shards, so a fine-tuned export that dropped the
head fails only later, in ``DefaultModelLoader.track_weights_loading``
(``ValueError: Following weights were not initialized from checkpoint``) -- or,
with ``--quantization`` set, not at all: it serves a randomly initialised head
(``Accepted: 0`` / garbage continuations).

Families differ in where the head lives in the shards. This script checks all the
shapes it knows and says which one matched:

  * ``mtp.*`` prefix          -- Qwen3.5 / Qwen3-Next style: ``Qwen3_5MTP.load_weights``
                                 remaps only keys starting ``mtp.`` to ``model.``
                                 (plus ``embed_tokens``/``lm_head``) and *skips*
                                 everything else.
  * layer index >= ``num_hidden_layers`` -- DeepSeek-family style, the extra
                                 layer is stored as the next trunk layer.
  * ``enorm``/``hnorm``/``pre_fc_norm*`` -- the MTP norms, whichever family.

Exit codes: 0 verdict printed (present or absent), 2 cannot decide.
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

MTP_CONFIG_KEYS = ("mtp_num_hidden_layers", "num_nextn_predict_layers", "n_predict")


def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 - report, do not raise
        return None


def first_nonempty_line(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    return line.strip()[:120]
    except OSError:
        pass
    return "?"


def shard_keys(model_dir):
    """Tensor names from the sharded index, else from each safetensors header."""
    keys, sources = [], []
    index = os.path.join(model_dir, "model.safetensors.index.json")
    doc = load_json(index)
    if doc and isinstance(doc.get("weight_map"), dict):
        keys = list(doc["weight_map"])
        sources.append(os.path.basename(index))
        return keys, sources
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))[:64]:
        try:
            with open(path, "rb") as fh:
                raw = fh.read(8)
                if len(raw) < 8:
                    continue
                (n,) = struct.unpack("<Q", raw)
                if n <= 0 or n > 200_000_000:
                    continue
                header = json.loads(fh.read(n))
        except Exception:  # noqa: BLE001 - a broken shard is another page's finding
            continue
        keys += [k for k in header if k != "__metadata__"]
        sources.append(os.path.basename(path))
    return keys, sources


def main(argv):
    quiet = "-q" in argv
    args = [a for a in argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__.strip().splitlines()[0])
        print("usage: mtp-head-check.py <model dir | config.json> [-q]", file=sys.stderr)
        return 2
    target = args[0]
    if target.endswith(".json"):
        model_dir, cfg_path = os.path.dirname(os.path.abspath(target)), target
    else:
        model_dir, cfg_path = target, os.path.join(target, "config.json")
    cfg = load_json(cfg_path)
    if not isinstance(cfg, dict):
        print(f"cannot decide: no readable config.json under {target}", file=sys.stderr)
        return 2
    # multimodal wrappers keep the text config nested
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg

    declared = {k: text.get(k, cfg.get(k)) for k in MTP_CONFIG_KEYS}
    declared = {k: v for k, v in declared.items() if v is not None}
    num_layers = text.get("num_hidden_layers", cfg.get("num_hidden_layers"))
    quant = cfg.get("quantization_config") or text.get("quantization_config")

    keys, sources = shard_keys(model_dir)
    if not keys:
        print(f"cannot decide: no safetensors index and no readable shard header in {model_dir}")
        print("  (a missing/odd shard set is its own finding: scripts/checkpoint-header-scan.py)")
        return 2

    layer_ids = set()
    for k in keys:
        parts = k.split(".")
        if len(parts) > 2 and parts[0] in ("model", "mtp") and parts[1] == "layers" and parts[2].isdigit():
            layer_ids.add(int(parts[2]))
    prefixed = [k for k in keys if k.startswith("mtp.")]
    norms = [k for k in keys if "enorm" in k or "hnorm" in k or "pre_fc_norm" in k]
    extra_layers = sorted(i for i in layer_ids if isinstance(num_layers, int) and i >= num_layers)

    present = bool(prefixed or norms or extra_layers)
    if not quiet:
        print(f"checkpoint: {os.path.abspath(model_dir)}  ({len(keys)} tensors from {', '.join(sources[:3])}"
              f"{' …' if len(sources) > 3 else ''})")
        print(f"  config: model_type={text.get('model_type', cfg.get('model_type'))!r} "
              f"num_hidden_layers={num_layers} "
              f"mtp layers={declared or 'none declared'} "
              f"quantization={'yes' if quant else 'no'}")
        print(f"  trunk layer ids: {min(layer_ids) if layer_ids else None}..{max(layer_ids) if layer_ids else None}")
        if prefixed:
            print(f"  mtp.* keys: {len(prefixed)}, e.g. {prefixed[:3]}")
        if extra_layers:
            print(f"  layer ids >= num_hidden_layers: {extra_layers[:6]}")
        if norms:
            print(f"  MTP norms: {norms[:4]}")
        if not present:
            print("  no mtp.* / extra-layer / MTP-norm key found: the head is NOT in this checkpoint")

    if present:
        print("VERDICT: MTP head PRESENT -- a *_mtp launch can load it from these shards")
        return 0

    print("VERDICT: MTP head ABSENT -- the drafter is still built from config.json, so")
    if quant:
        print("  with this quantized checkpoint the load-time check is OFF (default_loader.py")
        print("  enables it only when quantization is None): expect a silently random head,")
        print("  i.e. Accepted: 0 / acceptance 1.00, not a ValueError.")
    else:
        print("  an MTP launch raises ValueError: Following weights were not initialized")
        print("  from checkpoint: {...model.pre_fc_norm_*, model.layers.0.*, model.norm.*}")
        print("  at vllm/model_executor/model_loader/default_loader.py (track_weights_loading).")
    print("  Fix: drop --speculative-config, or re-export with the head. Deleting")
    print(f"  {list(MTP_CONFIG_KEYS)} from config.json is NOT a mitigation: the draft class")
    print("  defaults num_mtp_layers to 1 and still builds the drafter.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
