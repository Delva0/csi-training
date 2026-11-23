#!/usr/bin/env python3
"""Census a modelslim ``quant_model_description.json`` and price the decode ceiling.

Why: a report like ``Qwen2.5-7B-w8a8量化配置，没有性能收益（甚至有降低），精度同样降低``
(recorded #10604) carries two complaints and no evidence. The artifact's own
description decides the accuracy half — the value strings are the lookup keys into
vllm-ascend's ``ASCEND_QUANTIZATION_METHOD_MAP`` (``vllm_ascend/quantization/utils.py``
at v0.11.0 ``2f1aed98ccdb``: ``W8A8`` -> static per-tensor activation quant,
``W8A8_DYNAMIC`` -> per-token, ``FLOAT`` -> layer skipped back to unquantized) — and
the same file decides how much of the model never became int8, which caps the decode
speedup at ``2 / (1 + u)`` for ``u`` = the unquantized parameter share. Do the census
and the arithmetic before ranking any engine cause; see
``references/performance-throughput/quantized-dense-no-gain.md``.

Usage:
  uv run --no-project python scripts/quant-description-census.py /model/Qwen2.5-7B-w8a8
  uv run --no-project python scripts/quant-description-census.py /model/x --config /model/x/config.json
  uv run --no-project python scripts/quant-description-census.py - < quant_model_description.json
  uv run --no-project python scripts/quant-description-census.py --json <path> [--config <path>]

It prints:
  1. which description file was read (and the naming trap when the only file is
     ``quant_model_description_<scheme>.json``: the reader at v0.11.0 keeps names
     ending with ``quant_model_description.json``);
  2. the value histogram (scheme strings, plus any top-level ``fa_quant_type`` /
     ``kv_quant_type`` keys, which are attention/KV quantisation and not per-weight);
  3. the ``FLOAT`` (skipped) module families, and every family with its scheme(s);
  4. with a ``config.json``: per-family parameter counts, the unquantized share ``u``
     and the decode ceiling ``R = 2/(1+u)``.

The ceiling is arithmetic on the artifact and the config. It is not a measurement and
does not decide whether the reporter's numbers are plausible — quote it as the check.
Re-derive the scheme map at the reported release: the quantization package was
restructured more than once, and later refs read ``quantization/modelslim_config.py``.
"""
import json
import re
import sys
from pathlib import Path

NAME = "quant_model_description.json"
TOKEN_KEYS = ("fa_quant_type", "kv_quant_type", "kv_cache_type")


def find_description(target: Path):
    if str(target) == "-":
        return None, json.load(sys.stdin)
    if target.is_file():
        return target, json.loads(target.read_text())
    if not target.is_dir():
        raise SystemExit(f"not a file or directory: {target}")
    hits = sorted(target.glob("quant_model_description*.json"))
    if not hits:
        others = sorted(p.name for p in target.glob("*.json"))
        raise SystemExit(
            f"no quant_model_description*.json under {target}\n"
            f"  json files present: {', '.join(others) if others else '(none)'}\n"
            "  a full-precision directory served as a quantized one is the usual cause"
        )
    exact = [p for p in hits if p.name == NAME]
    if exact:
        return exact[0], json.loads(exact[0].read_text())
    print(f"WARNING: none of {[p.name for p in hits]} ends with '{NAME}'.")
    print("         the v0.11.0 reader only keeps that name "
          "(quant_config.py get_config_filenames); `cp` it to the short name.")
    return hits[0], json.loads(hits[0].read_text())


def family(key: str) -> str:
    k = re.sub(r"\.\d+\.", ".<n>.", key)
    k = re.sub(r"\.experts\.\d+\.", ".experts.<n>.", k)
    return k


def census(desc: dict):
    values, families, tokens = {}, {}, {}
    for key, val in desc.items():
        if key in TOKEN_KEYS or not key.endswith(".weight"):
            tokens[key] = val if not isinstance(val, dict) else val
            continue
        if isinstance(val, str):
            scheme = val
        elif isinstance(val, dict):
            scheme = val.get("quant_type") or val.get("quant_type_name") or json.dumps(
                val, sort_keys=True)
        else:
            scheme = json.dumps(val)
        values[scheme] = values.get(scheme, 0) + 1
        fam = families.setdefault(family(key), {})
        fam[scheme] = fam.get(scheme, 0) + 1
    return values, families, tokens


def read_config(path: Path):
    try:
        cfg = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        print(f"WARNING: could not read {path}: {exc}")
        return None
    for key in ("text_config", "llm_config"):
        if isinstance(cfg.get(key), dict):
            cfg = {**cfg, **cfg[key]}
    return cfg


def param_plan(cfg: dict):
    """Per-family parameter counts from a HF decoder config; None when a field is absent."""
    need = ("hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size")
    missing = [k for k in need if k not in cfg]
    if missing:
        print(f"WARNING: config.json lacks {missing}; skipping the ceiling.")
        return None
    hidden = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    heads = cfg.get("num_attention_heads") or 0
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim") or (hidden // heads if heads else 0)
    tied = bool(cfg.get("tie_word_embeddings", False))
    act = str(cfg.get("hidden_act", "silu")).lower()
    gates = 2 if act in ("gelu", "gelu_new", "gelu_pytorch_tanh", "relu", "quick_gelu") else 3

    kv = kv_heads * head_dim
    per_layer = {
        "q_proj": hidden * heads * head_dim,
        "k_proj": hidden * kv,
        "v_proj": hidden * kv,
        "o_proj": heads * head_dim * hidden,
        "mlp": gates * hidden * inter,
    }
    plan = {"embed_tokens": vocab * hidden}
    if not tied:
        plan["lm_head"] = vocab * hidden
    for name, count in per_layer.items():
        plan[name] = count * layers
    return plan, dict(hidden=hidden, layers=layers, inter=inter, vocab=vocab,
                      heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                      tied=tied, gates=gates)


FAMILY_MARKERS = (
    ("embed_tokens", ("embed_tokens",)),
    ("lm_head", ("lm_head",)),
    ("q_proj", ("q_proj", "qkv_proj", "fused_qkv", "wq")),
    ("k_proj", ("k_proj", "wk")),
    ("v_proj", ("v_proj", "wv")),
    ("o_proj", ("o_proj", "wo", "dense")),
    ("mlp", ("gate_up_proj", "up_proj", "down_proj", "gate_proj", "mlp", "experts",
             "w1", "w2", "w3")),
)


def covered_families(families: dict):
    """Plan families that carry at least one non-FLOAT entry (fused names included)."""
    out = set()
    fused = any("qkv" in fam or "fused_qkv" in fam for fam in families)
    for fam, schemes in families.items():
        if all(s == "FLOAT" for s in schemes):
            continue
        for plan_name, markers in FAMILY_MARKERS:
            if any(m in fam for m in markers):
                out.add(plan_name)
                if plan_name == "q_proj" and ("qkv" in fam or "fused_qkv" in fam):
                    out.update(("k_proj", "v_proj"))
    if fused and "q_proj" in out:
        out.update(("k_proj", "v_proj"))
    return out


def main(argv):
    args = [a for a in argv[1:]]
    config = None
    if "--config" in args:
        i = args.index("--config")
        config = Path(args[i + 1])
        del args[i:i + 2]
    if "--json" in args:
        i = args.index("--json")
        target = Path(args[i + 1])
        del args[i:i + 2]
    else:
        if not args:
            raise SystemExit(__doc__.split("Usage:")[1].split("It prints")[0])
        target = Path(args[0])
        del args[0]

    path, desc = find_description(target)
    if not isinstance(desc, dict):
        raise SystemExit("description is not a JSON object")
    print(f"== quant description: {path if path else '(stdin)'} ==")
    print(f"   entries: {len(desc)}")

    values, families, tokens = census(desc)
    print("\n-- 1. scheme histogram (per-weight entries) --")
    for scheme, count in sorted(values.items(), key=lambda kv: -kv[1]) or [("(none)", 0)]:
        print(f"  {count:6d}  {scheme}")
    if tokens:
        print("\n-- 2. top-level (non per-weight) keys --")
        for key, val in tokens.items():
            shown = val if isinstance(val, (str, int, float)) else type(val).__name__
            print(f"  {key}: {shown}")

    float_fams = sorted({f for f, s in families.items() if set(s) == {"FLOAT"}})
    print("\n-- 3. module families --")
    print(f"  FLOAT-only families ({len(float_fams)}):")
    for fam in float_fams:
        print(f"    {fam}")
    print("  every family with its scheme(s):")
    for fam in sorted(families):
        row = ", ".join(f"{s}x{n}" for s, n in sorted(families[fam].items()))
        print(f"    {fam}: {row}")

    cfg_path = config or (target / "config.json" if target.is_dir() else None)
    if not cfg_path or not Path(cfg_path).is_file():
        print("\n-- 4. ceiling: skipped (no config.json; pass --config) --")
        return 0
    cfg = read_config(Path(cfg_path))
    if cfg is None:
        return 0
    moe_keys = ("num_experts", "n_routed_experts", "num_local_experts",
                "moe_intermediate_size", "n_shared_experts")
    if any(k in cfg for k in moe_keys) or any("experts" in f for f in families):
        print("\n-- 4. ceiling: skipped (MoE checkpoint). --")
        print("  the per-layer expert budget is config- and EP-dependent, so 2/(1+u)")
        print("  from a dense-MLP formula would be wrong; use the scheme census above")
        print("  and price the expert weights separately.")
        return 0
    plan = param_plan(cfg)
    if plan is None:
        return 0
    counts, meta = plan
    quant = covered_families(families)
    print(f"\n-- 4. parameter budget from {cfg_path} --")
    print(f"  hidden={meta['hidden']} layers={meta['layers']} inter={meta['inter']} "
          f"vocab={meta['vocab']} heads/kv/head_dim={meta['heads']}/{meta['kv_heads']}/"
          f"{meta['head_dim']} tied_embed={meta['tied']} mlp_matrices={meta['gates']}")
    total = quant_params = 0.0
    print(f"  {'family':22s} {'params(B)':>10s}  {'int8?':5s}")
    for fam in ("embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj", "mlp"):
        if fam not in counts:
            continue
        n = counts[fam]
        is_q = fam in quant
        total += n
        if is_q:
            quant_params += n
        print(f"  {fam:22s} {n / 1e9:10.3f}  {'yes' if is_q else 'FLOAT'}")
    if total <= 0:
        return 0
    u = (total - quant_params) / total
    r = 2.0 / (1.0 + u)
    print(f"  unquantized share u = {u:.3f}  ->  decode ceiling R = 2/(1+u) = {r:.2f}x")
    print("  arithmetic only: a ceiling at zero overhead, not a prediction; per-step\n"
          "  launch cost and attention/KV bytes reduce it, and the recorded #1015\n"
          "  fp16-vs-w8a8 A/B came out slower than this ceiling allows.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
