#!/usr/bin/env python3
"""checkpoint-key-prefix.py — report the tensor-name prefixes of a local HF
checkpoint, i.e. whether its tensors are root-level (``embed_tokens.weight``,
``layers.0...``, ``norm.weight``: a bare ``*Model`` export) or prefixed with
``model.`` (a ``*ForCausalLM`` export).

Why: vLLM's ``AutoWeightsLoader`` is prefix-strict. It groups checkpoint names
by their first dotted component and requires that component to be a child
module or direct parameter of the class the engine built
(``vllm/model_executor/models/utils.py``, ``_groupby_prefix`` ->
``_load_module`` -> ``There is no module or parameter named '<prefix>' in
<Class>``). A checkpoint whose names start at ``embed_tokens`` cannot load into
a ``*ForCausalLM`` class, which owns ``model`` and ``lm_head`` only. The
embedding/pooling wrapper does not have that problem: it strips ``model.`` and
delegates to the inner model loader, whose ``params_dict`` holds root-level
names — see references/pooling-embedding/embedding-checkpoint-as-generate.md.

Usage:
  uv run --no-project python checkpoint-key-prefix.py /model/Qwen3-Embedding-0.6B
  uv run --no-project python checkpoint-key-prefix.py model-00001-of-00002.safetensors
  uv run --no-project python checkpoint-key-prefix.py - < model.safetensors.index.json

Exit status: 0 = weights read, 1 = root-level (unprefixed) tensor names found,
2 = no safetensors weights could be read (ask the reporter for `ls` of the dir).
Read the report, not just the status.
"""

import glob
import json
import os
import struct
import sys

MAX_SAMPLES = 4


def read_headers(path):
    """Return the tensor names of one safetensors file (header-only read)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("file shorter than the 8-byte header length")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > 100 * 1024 * 1024:
            raise ValueError(f"implausible header length {length}")
        header = json.loads(f.read(length).decode("utf-8", "replace"))
    return [k for k in header if k != "__metadata__"]


def names_from_index(path_or_dash):
    if path_or_dash == "-":
        doc = json.load(sys.stdin)
    else:
        with open(path_or_dash) as f:
            doc = json.load(f)
    weight_map = doc.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("no 'weight_map' in the index json")
    return list(weight_map)


def collect(path):
    """Return (names, source description, list of files used)."""
    if path == "-":
        raise ValueError("stdin carries an index json, not a model directory")
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return read_headers(path), path, [path]
    if os.path.isfile(path) and path.endswith(".index.json"):
        names = names_from_index(path)
        return names, path, [path]
    if not os.path.isdir(path):
        raise ValueError(f"{path}: not a file or directory")
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        names = names_from_index(index)
        return names, index, [index]
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"no *.safetensors in {path}")
    names = []
    for f in files:
        names.extend(read_headers(f))
    return names, f"{len(files)} safetensors file(s)", files


def first_component(name):
    return name.split(".", 1)[0]


def load_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main(argv):
    if len(argv) != 2:
        print("usage: checkpoint-key-prefix.py "
              "<model-dir|*.safetensors|*.index.json|->", file=sys.stderr)
        return 2
    path = argv[1]

    try:
        names, source, _files = collect(path)
    except Exception as exc:                                    # noqa: BLE001
        print(f"could not read weights: {exc}")
        print("ask the reporter for `ls -l` of the model directory and the "
              "`Loading safetensors checkpoint shards: ... N/N` line, or a "
              "`config.json` paste — no verdict without the key list.")
        return 2

    # Distinct first components, in first-seen order, with counts.
    order, counts = [], {}
    for n in names:
        c = first_component(n)
        if c not in counts:
            order.append(c)
            counts[c] = 0
        counts[c] += 1

    print(f"tensor source: {source}")
    print(f"tensors: {len(names)}; distinct root components: {len(order)}")
    for c in order[:12]:
        sample = next(n for n in names if first_component(n) == c)
        print(f"  {c!r:24} {counts[c]:5d} tensors   e.g. {sample}")
    if len(order) > 12:
        print(f"  ... {len(order) - 12} more root components")

    model_prefixed = "model" in counts
    root_level = not model_prefixed and any(
        c in counts for c in ("embed_tokens", "layers", "norm", "tok_embeddings"))
    has_lm_head = any(n == "lm_head.weight" or n.startswith("lm_head.") for n in names)
    embed_keys = [n for n in names if n.endswith("embed_tokens.weight")]

    base = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else path
    cfg = load_json(os.path.join(base, "config.json"))
    st = load_json(os.path.join(base, "config_sentence_transformers.json"))

    print()
    print("verdict:")
    print(f"  root-level names (no 'model.' prefix): {'YES' if root_level else 'no'}")
    print(f"  'model.'-prefixed names:               {'yes' if model_prefixed else 'NO'}")
    print(f"  embed_tokens present:                  {embed_keys[0] if embed_keys else 'no'}")
    print(f"  lm_head in the checkpoint:             {'yes' if has_lm_head else 'no (pooler/embedding export)'}")
    if cfg is not None:
        print(f"  config.json architectures:             {cfg.get('architectures')}")
        print(f"  config.json model_type:                {cfg.get('model_type')}   "
              f"tie_word_embeddings={cfg.get('tie_word_embeddings')}")
    else:
        print("  config.json:                           not found next to the weights")
    if st is not None:
        keys = {k: st[k] for k in ("pooling_mode", "pooling_mode_lasttoken",
                                   "pooling_mode_mean_tokens", "matryoshka_dimensions")
                if k in st}
        print(f"  config_sentence_transformers.json:     {keys or list(st)[:6]}")
    else:
        print("  config_sentence_transformers.json:     absent (pooling type will be "
              "vllm's default / --override-pooler-config)")

    print()
    if root_level:
        print("A root-level export cannot load into a *ForCausalLM class: the")
        print("engine's AutoWeightsLoader only owns `model` and `lm_head` at the")
        print("root, so the first group ('embed_tokens' / 'layers' / 'norm') has no")
        print("owner and raises `There is no module or parameter named ...`.")
        print("Serve it as a pooling/embedding model (`--task embed`, or")
        print("`--runner pooling` on releases that still carry the flag): the")
        print("wrapper strips 'model.' and hands the names to the inner model")
        print("loader, whose params_dict holds root-level names.")
        print("See references/pooling-embedding/embedding-checkpoint-as-generate.md")
    elif model_prefixed:
        print("Names are 'model.'-prefixed, so the prefix is not this failure's")
        print("cause: re-read the traceback's Class name and the unmatched prefix")
        print("(it may be a real missing parameter, a quantization tensor, or a")
        print("stacking mismatch), not the served task.")
    else:
        print("Neither shape recognised — paste the first 10 keys and the")
        print("traceback line to the reporter before concluding anything.")
    return 1 if root_level else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
