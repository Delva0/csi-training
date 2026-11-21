#!/usr/bin/env python3
"""Is the downstream copied forward written against the installed library body?

Run this in the reporter's image (or any interpreter that holds the image's
`transformers`), for an `AttributeError: '<Ascend…>' object has no attribute
'<per-layer attr>'` raised inside `vllm_ascend/ops/qwen2_decoder.py` — e.g.
`attention_type` at v0.22.1rc1, issue #11661/#11926/#11991:

    uv run --no-project python scripts/qwen2-body-drift-probe.py
    uv run --no-project python scripts/qwen2-body-drift-probe.py --config /data2/DeepSeek-OCR-2

It prints, without needing the model or an NPU:

1. the interpreter's `transformers.__version__` and the module file;
2. the checkpoint's `layer_types` / `sliding_window` when `--config` is given
   (a sliding layer makes a blanket `full_attention` default wrong, not neutral);
3. whether a `Qwen2DecoderLayer` instance exposes `attention_type`, and whether
   `Qwen2Model` has `has_sliding_layers`;
4. the installed `Qwen2Model.forward` and `Qwen2DecoderLayer.__init__` source,
   so it can be diffed against `vllm_ascend/ops/qwen2_decoder.py:169-247` at the
   report's pin — the copy there is what raised.

Read the flags, not the source, first: `INDEXES BY config.layer_types` plus
`mask mapping dict handled: yes` plus `layer.attention_type: False` means the
downstream copy is from another generation of the library and the repair is to
delete the copy (delegate to `super().forward()`), not to guard the attribute.
A guard only silences the raise; recorded #11991 started the service and then
misrecognised images. Quote the output as the check, not as the cause.
"""
import argparse
import importlib
import inspect
import json
import os
import sys

PIN = "vllm-ascend v0.22.1rc1 (5f6faa0c) / vllm v0.22.1 (0decac0d)"
UPSTREAM_COPY = "vllm_ascend/ops/qwen2_decoder.py:169-247"


def load_config(path):
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            top = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"config: unreadable ({exc})")
        return None
    for key in ("text_config", "llm_config", "language_config"):
        nested = top.get(key)
        if isinstance(nested, dict):
            return nested
    return top


def classify(forward_src):
    """What the installed body reads to pick each layer's mask."""
    if "causal_mask_mapping" not in forward_src and "mask_mapping" not in forward_src:
        return "no mask-mapping dict (body takes a plain 4D/tensor attention_mask)"
    if "decoder_layer.attention_type" in forward_src:
        return "INDEXES BY decoder_layer.attention_type"
    if "layer_types" in forward_src:
        return "INDEXES BY config.layer_types"
    return "mask-mapping dict handled, indexing form not matched — read the dump"


def dump(label, obj, limit):
    print(f"\n===== {label} =====")
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError) as exc:
        print(f"<source unavailable: {exc}>")
        return None
    lines = src.splitlines()
    for line in lines[:limit]:
        print(line)
    if len(lines) > limit:
        print(f"… {len(lines) - limit} more lines (--lines N)")
    return src


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", help="model dir or config.json of the failing checkpoint")
    ap.add_argument("--lines", type=int, default=160, help="source lines printed per object")
    args = ap.parse_args(argv[1:])

    print(f"pin the report was verified at: {PIN}")
    try:
        import transformers
    except Exception as exc:  # noqa: BLE001 - an import failure is a finding here
        print(f"transformers: import failed ({exc})")
        print("next: run this inside the image's python, not on the host")
        return 2
    print(f"transformers: {transformers.__version__}  ({transformers.__file__})")

    if args.config:
        cfg = load_config(args.config)
        if cfg is not None:
            print(f"config.layer_types: {cfg.get('layer_types')}")
            print(f"config.sliding_window: {cfg.get('sliding_window')}")
            print(f"config.use_sliding_window: {cfg.get('use_sliding_window')}")
    else:
        print("config.layer_types: (pass --config <model dir> to read the checkpoint's)")

    try:
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2DecoderLayer,
            Qwen2Model,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"transformers.models.qwen2: import failed ({exc})")
        print("next: this interpreter has no Qwen2 modeling module — report the version above")
        return 2

    try:
        cfg = Qwen2Config(hidden_size=896, num_hidden_layers=2)
        layer = Qwen2DecoderLayer(cfg, 0)
        attrs = [a for a in ("attention_type", "layer_type") if hasattr(layer, a)]
        print(f"layer.attention_type: {hasattr(layer, 'attention_type')}  (per-layer names present: {attrs or 'none'})")
    except Exception as exc:  # noqa: BLE001
        print(f"layer construction: failed ({exc})")

    has_sliding = getattr(Qwen2Model, "has_sliding_layers", None)
    print(f"Qwen2Model.has_sliding_layers: {'present' if has_sliding is not None else 'absent'}")

    fwd = dump("installed Qwen2Model.forward", Qwen2Model.forward, args.lines)
    if fwd:
        print("\nreading:", classify(fwd))
        print("mask mapping dict handled:",
              "yes" if "causal_mask_mapping" in fwd or "mask_mapping" in fwd else "no")
    dump("installed Qwen2DecoderLayer.__init__", Qwen2DecoderLayer.__init__, args.lines)

    print("\n===== next =====")
    print(f"1. diff this dump against {UPSTREAM_COPY} in the reporter's tree:")
    print("   diff <(uv run --no-project python -c 'import inspect;"
          " from transformers.models.qwen2.modeling_qwen2 import Qwen2Model;"
          " print(inspect.getsource(Qwen2Model.forward))') \\")
    print("        <(sed -n '195,247p' /vllm-workspace/vllm-ascend/vllm_ascend/ops/qwen2_decoder.py)")
    print("2. accept a repair only on a real request with output (an image OCR), never on")
    print("   'the service started' — a guard on the missing attribute does exactly that (#11991).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
