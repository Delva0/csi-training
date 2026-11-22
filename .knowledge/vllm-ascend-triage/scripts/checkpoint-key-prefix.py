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
  uv run --no-project python checkpoint-key-prefix.py --scales /model/DeepSeek-V4-Flash
  uv run --no-project python checkpoint-key-prefix.py --dtypes /model/DeepSeek-V3-0324

``--scales`` switches to the quantisation census instead: which ``.scale`` /
``.weight_scale`` / ``.quant_bias`` tensors the checkpoint carries, grouped by module
family, plus whether the directory declares a quantisation at all, and the
``quant_model_description.json`` ``version`` / ``group_size`` pair that decides the
W4A8 MoE scale_bias load path (``version: "1.0.0"`` + ``group_size: 0`` predicts the
load-phase ``expanded size … must match the existing size`` crash; page
references/startup-failures/quant-aux-param-tp-shape-mismatch.md).  It answers the
mirror question of the prefix mode — a load-phase ``KeyError: '<param>.weight_scale'``
means the checkpoint owns a scale the engine built no parameter for, and the census
says whether the served directory is a quantised export (then the ``--quantization
ascend`` flag/method is the question) or a full-precision release whose few
FP8-stored projections ship a per-tensor scale (then the weight *variant* is the
question).  See references/startup-failures/weight-scale-keyerror-quant-variant.md.

``--dtypes`` switches to the *weight dtype* census instead: the histogram of the
header ``dtype`` field over every shard, the module families carrying FP8
(``F8_E4M3``/``F8_E5M2``) *weight* tensors (scale-like suffixes counted apart), and
the two quantisation declarations beside them.  It answers "is this checkpoint
executable on the NPU at all" — the upstream DeepSeek-V3/R1 release stores E4M3
weights, which the 910 does not run, and ``config.json`` cannot tell you: the pinned
conversion guide tells converters to delete those fields before a W8A8 export, so a
directory may hold FP8 tensors with no ``quantization_config`` at all and a bf16
directory may still carry a leftover one.  See
references/startup-failures/quantization-method-gate.md section E.

Exit status: 0 = weights read (prefix mode: no root-level names; ``--dtypes``: no
FP8 weight tensors), 1 = root-level (unprefixed) tensor names found, or FP8 weight
tensors found under ``--dtypes``, 2 = no safetensors weights could be read (ask the
reporter for `ls` of the dir). Read the report, not just the status.
"""

import glob
import json
import os
import struct
import sys

MAX_SAMPLES = 4


def read_header(path):
    """Return the safetensors header dict of one file (header-only read)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("file shorter than the 8-byte header length")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > 100 * 1024 * 1024:
            raise ValueError(f"implausible header length {length}")
        header = json.loads(f.read(length).decode("utf-8", "replace"))
    if not isinstance(header, dict):
        raise ValueError("header JSON is not a tensor map")
    return header


def read_headers(path):
    """Return the tensor names of one safetensors file (header-only read)."""
    return [k for k in read_header(path) if k != "__metadata__"]


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


QUANT_SUFFIXES = (
    ".weight_scale_inv",
    ".weight_scale",
    ".quant_bias",
    ".weight_offset",
    ".input_scale",
    ".scale",
)


def quant_suffix(name):
    for suf in QUANT_SUFFIXES:
        if name.endswith(suf):
            return suf
    return None


def module_family(base):
    """Strip the layer/expert indices so one line per module kind remains."""
    parts = base.split(".")
    keep, skip_next = [], False
    for tok in parts:
        if skip_next:
            skip_next = False
            continue
        if tok in ("layers", "experts", "mtp") and tok != parts[-1]:
            skip_next = True
            continue
        if tok in ("model",):
            continue
        keep.append(tok)
    return ".".join(keep) or base


def report_scales(names, source, path):
    base_dir = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else path
    cfg = load_json(os.path.join(base_dir, "config.json"))
    desc = load_json(os.path.join(base_dir, "quant_model_description.json"))

    kinds, families, examples = {}, {}, {}
    n_quant = 0
    for n in names:
        suf = quant_suffix(n)
        if suf is None:
            continue
        n_quant += 1
        kinds[suf] = kinds.get(suf, 0) + 1
        fam = module_family(n[: -len(suf)])
        families[fam] = families.get(fam, 0) + 1
        examples.setdefault(fam, n)

    print(f"tensor source: {source}")
    print(f"tensors: {len(names)}; quantisation tensors: {n_quant}")
    if kinds:
        print("  by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
        print(f"  by module family: {len(families)} distinct")
        for fam, cnt in sorted(families.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {fam:52} {cnt:5d}   e.g. {examples[fam]}")
        if len(families) > 15:
            print(f"    ... {len(families) - 15} more families")

    qcfg = None
    if cfg is not None:
        qcfg = cfg.get("quantization_config") or cfg.get("compression_config")
    print()
    print("quantisation declarations:")
    print(f"  quant_model_description.json:  {'present (modelslim/msmodelslim export)' if desc is not None else 'absent'}")
    if desc is not None:
        ver, gs = desc.get("version"), desc.get("group_size")
        gs_eff = 256 if gs is None else gs
        print(
            f"  quant_model_description.json version={ver!r} group_size={gs!r}"
            f"  -> new_quant_version={ver == '1.0.0'} is_per_channel_weight={gs_eff == 0}"
        )
        if ver == "1.0.0" and gs_eff == 0:
            print("    W4A8 MoE per-channel + new quant version: the scale_bias params are")
            print("    allocated with a parallel divisor (`16 // tp_size`) and stay tagged")
            print("    CHANNEL, so any parallel size > 1 dies at load with")
            print("    `RuntimeError: The expanded size of the tensor (16 // N) must match the")
            print("    existing size (16)` ->")
            print("    references/startup-failures/quant-aux-param-tp-shape-mismatch.md")
    print(f"  config.json quantization_config/compression_config: {qcfg if qcfg is not None else 'absent'}")
    if cfg is None:
        print("  config.json:                   not found next to the weights")

    print()
    print("verdict:")
    if n_quant == 0:
        print("  no scale-like tensors: this directory cannot produce a")
        print("  `KeyError: '<param>.weight_scale'`. The failing name came from another")
        print("  tree — re-check the served path, the bind mount and any cached copy")
        print("  (`vllm serve` on a mount root rather than on the checkpoint dir).")
    elif desc is not None or qcfg is not None or len(families) >= 4:
        print("  a quantised export: scales cover the main linear/MoE families and/or the")
        print("  directory declares a quantisation. The question is whether the engine")
        print("  selected the matching method (--quantization ascend, the flag, and the")
        print("  checkpoint's own description file) ->")
        print("  references/startup-failures/quantization-method-gate.md (section B).")
    else:
        print("  a full-precision release with a few FP8-stored projections: scales are")
        print("  confined to a handful of module families and nothing declares a")
        print("  quantisation, so the engine builds those linears unquantised and has no")
        print("  `weight_scale` parameter for the scale the loader meets -> compare the")
        print("  served directory against the tutorial's weight list and serve the")
        print("  documented quantised variant with `--quantization ascend`:")
        print("  references/startup-failures/weight-scale-keyerror-quant-variant.md")
    return 0


FP8_DTYPES = ("F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FNUZ")


def safetensors_files(path):
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return [path]
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.safetensors")))
    return []


def report_dtypes(path):
    """Weight-dtype census: is this checkpoint executable on the NPU at all?"""
    files = safetensors_files(path)
    if not files:
        print("no *.safetensors to read dtypes from: point this at the model "
              "directory (or one shard), not at an index json")
        return 2

    counts, weight_f8, scale_f8, examples, dtypes_seen = {}, {}, {}, {}, set()
    for f in files:
        try:
            header = read_header(f)
        except Exception as exc:                                # noqa: BLE001
            print(f"{os.path.basename(f)}: unreadable header ({exc}) — run "
                  "scripts/checkpoint-header-scan.py on this directory")
            return 2
        for name, info in header.items():
            if name == "__metadata__" or not isinstance(info, dict):
                continue
            dt = str(info.get("dtype", "?"))
            counts[dt] = counts.get(dt, 0) + 1
            dtypes_seen.add(dt)
            if dt not in FP8_DTYPES:
                continue
            if quant_suffix(name) is not None:
                scale_f8[name] = dt
                continue
            fam = module_family(name)
            weight_f8[fam] = weight_f8.get(fam, 0) + 1
            examples.setdefault(fam, name)

    base_dir = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else path
    cfg = load_json(os.path.join(base_dir, "config.json"))
    desc = load_json(os.path.join(base_dir, "quant_model_description.json"))
    qcfg = None
    if cfg is not None:
        qcfg = cfg.get("quantization_config") or cfg.get("compression_config")

    print(f"tensor source: {len(files)} safetensors file(s) under {base_dir}")
    print("tensors by dtype: " + ", ".join(
        f"{d} {c}" for d, c in sorted(counts.items(), key=lambda kv: -kv[1])))
    if weight_f8:
        print(f"  FP8 *weight* tensors: {sum(weight_f8.values())} in "
              f"{len(weight_f8)} module families")
        for fam, cnt in sorted(weight_f8.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {fam:52} {cnt:5d}   e.g. {examples[fam]}")
        if len(weight_f8) > 15:
            print(f"    ... {len(weight_f8) - 15} more families")
    if scale_f8:
        print(f"  FP8 *scale-like* tensors: {len(scale_f8)} "
              "(counted apart; a scale's dtype alone decides nothing)")

    print()
    print("quantisation declarations:")
    print(f"  quant_model_description.json:  "
          f"{'present (modelslim/msmodelslim export)' if desc is not None else 'absent'}")
    print(f"  config.json quantization_config/compression_config: "
          f"{qcfg if qcfg is not None else 'absent'}")

    print()
    print("verdict:")
    if weight_f8:
        print("  E4M3/E5M2 WEIGHT tensors are in this checkpoint: it is the upstream")
        print("  FP8 release (DeepSeek-V3/R1 and siblings), and the 910 executes none")
        print("  of it at the refs recorded here — no method name, config edit or")
        print("  graph flag changes that. Reporter-confirmed rule (#3534): \"check the")
        print("  weight dtype … if E4M3 weights are included, it can be considered")
        print("  that this is not supported on the NPU\". Routes: dequantize to bf16,")
        print("  or convert with msmodelslim to W8A8/W8A8_DYNAMIC and serve that with")
        print("  `--quantization ascend`.")
        print("  Scope: the support verdict is the *pinned release's* quant doc/support")
        print("  matrix, not this script — re-check it before applying the rule to a")
        print("  later release that adds an fp8 row.")
        print("  See references/startup-failures/quantization-method-gate.md section E.")
        return 1
    if "I8" in dtypes_seen and (desc is not None or qcfg is not None):
        print("  int8 weights plus a declared quantisation: a converted export. The")
        print("  question is whether the engine selected the matching method")
        print("  (`--quantization ascend` / the checkpoint's own quant_method) —")
        print("  sections A and B, and section C if the load dies on a lookup key.")
        return 0
    if "I8" in dtypes_seen:
        print("  int8 weights with NO declaration beside them: an export whose")
        print("  description/method is missing, so nothing selects it and the int8")
        print("  tensors are loaded into a dense model — section B (#2015).")
        return 0
    if qcfg is not None:
        print("  no FP8/INT8 weights but `config.json` declares a quantisation: the")
        print("  declaration is stale (the pin's guide has converters *delete* those")
        print("  fields before a W8A8 export), yet the engine still renames the method")
        print("  to `ascend` and hands that dict to the quant config — the section C")
        print("  KeyError, whose fix is the one config edit. Nothing about the weights")
        print("  is quantised.")
        return 0
    print("  full-precision weights and no declaration: this directory is not the")
    print("  source of any quantisation gate — look at the served path, not the file")
    print("  list (sections A–C).")
    return 0


def main(argv):
    flags = [a for a in argv[1:] if a.startswith("--")]
    positional = [a for a in argv[1:] if not a.startswith("--")]
    if flags not in ([], ["--scales"], ["--dtypes"]) or len(positional) != 1:
        print("usage: checkpoint-key-prefix.py [--scales|--dtypes] "
              "<model-dir|*.safetensors|*.index.json|->", file=sys.stderr)
        return 2
    path = positional[0]

    if "--dtypes" in flags:
        return report_dtypes(path)

    try:
        names, source, _files = collect(path)
    except Exception as exc:                                    # noqa: BLE001
        print(f"could not read weights: {exc}")
        print("ask the reporter for `ls -l` of the model directory and the "
              "`Loading safetensors checkpoint shards: ... N/N` line, or a "
              "`config.json` paste — no verdict without the key list.")
        return 2

    if "--scales" in flags:
        return report_scales(names, source, path)

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
