#!/usr/bin/env python3
"""Which 310P rope implementation a checkpoint lands in, from its own config.json.

vllm-ascend replaces the rope layers on 310P only (`is_310p()` override block,
vllm_ascend/utils.py:781-809 at v0.23.0rc1): MRotaryEmbedding ->
AscendMRotaryEmbedding310 (per-forward global cos/sin slices built by the runner
when `uses_mrope`) and RotaryEmbedding -> AscendRotaryEmbedding310 (three
branches inside _rope_forward_oot). The branch is decided by the checkpoint
config, not by the model family, so read the config before ranking a rope
hypothesis.

    uv run --no-project python scripts/310p-rope-branch.py <config.json | model dir>

The script prints: multimodal or not, mrope or not, head_size / rotary_dim, the
predicted branch, whether the globals that branch reads are populated for a
multimodal config, and which recorded symptom that predicts (a profile_run crash
with `Check cos != nullptr failed`, or wrong text out of a serve that came up).
It reproduces a calculation; it does not decide the cause (SKILL.md rule 6).
"""
import json
import os
import sys

PINS = ("vllm-ascend v0.23.0rc1 (f4a08bdd) + vllm v0.23.0 (0fc695fc); "
        "first verified at v0.21.0rc1 (80610e4) + vllm v0.21.0 (ad7125a4)")

USAGE = (__doc__.strip().splitlines()[0]
         + "\nusage: 310p-rope-branch.py <config.json | model dir>")


def fail(msg: str) -> None:
    print(f"error: {msg}\n{USAGE}")
    raise SystemExit(2)


def load(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    except ValueError as exc:
        fail(f"{path} is not JSON: {exc}")
    return {}


def first_int(cfg: dict, *keys) -> int | None:
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        fail("exactly one argument (a model dir or config.json) is required")

    top = load(argv[1])
    text = top.get("text_config") if isinstance(top.get("text_config"), dict) else top
    nested = text is not top
    is_vl = (nested
             or "vision_config" in top
             or "thinker_config" in top
             or "audio_config" in top)

    rope = {}
    for src in (text, top):
        for key in ("rope_parameters", "rope_scaling"):
            val = src.get(key)
            if isinstance(val, dict) and val:
                rope = dict(val)
                break
        if rope:
            break

    partial = rope.get("partial_rotary_factor")
    if partial is None:
        partial = text.get("partial_rotary_factor", top.get("partial_rotary_factor"))
    try:
        partial = float(partial) if partial is not None else 1.0
    except (TypeError, ValueError):
        fail(f"unreadable partial_rotary_factor: {partial!r}")

    head_size = first_int(text, "head_dim", "head_size")
    hidden = first_int(text, "hidden_size", "n_embd")
    heads = first_int(text, "num_attention_heads", "n_head")
    if head_size is None and hidden and heads:
        head_size = hidden // heads
    if not head_size:
        fail("cannot derive head_size (need head_dim, or hidden_size and "
             "num_attention_heads, in the text config)")

    rotary_dim = max(1, int(head_size * partial))
    mrope = "mrope_section" in rope
    rope_type = rope.get("rope_type") or rope.get("type") or "default"

    print(f"pin assumed: {PINS}")
    print(f"model_type: {top.get('model_type')!r}   architectures: "
          f"{top.get('architectures')}")
    print(f"multimodal (nested text_config / vision_config): {is_vl}")
    print(f"rope_parameters: {rope or '{}'}   rope_type={rope_type}")
    print(f"head_size={head_size}  partial_rotary_factor={partial:g}  "
          f"rotary_dim={rotary_dim}")

    if mrope:
        print("\npredicted layer: AscendMRotaryEmbedding310 "
              "(mrope_section present -> uses_mrope)")
        print("  cos/sin: per-forward global slice, built by "
              "NPUModelRunner310._model_forward only when uses_mrope")
        print("  (_310p/model_runner_310p.py:657-668 at v0.23.0rc1 -> "
              "_310p/ops/rotary_embedding.py:68,239-252)")
        print("  next check: does the serve print anything about the mrope "
              "slice? a missing slice raises RuntimeError, not garbage "
              "(AscendMRotaryEmbedding310.forward_oot, :203 at v0.23.0rc1)")
        return 0

    print("\npredicted layer: AscendRotaryEmbedding310._rope_forward_oot "
          "(_310p/ops/rotary_embedding.py:136-190 at v0.21.0rc1, :121-186 "
          "at v0.23.0rc1)")
    if head_size == 128 and rotary_dim == 128:
        branch, needs_globals = "a: full rotary, head 128 -> " \
            "npu_apply_rotary_pos_emb(cos, sin)", True
    elif rotary_dim < head_size and rotary_dim == 64:
        branch, needs_globals = "b: partial rotary, rotary_dim 64 -> " \
            "npu_apply_rotary_pos_emb(cos, sin)", True
    else:
        branch, needs_globals = "c: " \
            "torch_npu._npu_rotary_embedding(positions, cos_sin_cache)", False
    print(f"  branch {branch}")
    if not needs_globals:
        print("  reads no globals: rope is not a 310P-specific suspect for this "
              "config; look at the vision tower path or the pipeline instead")
        return 0
    if is_vl:
        print("  globals: cos/sin are NOT allocated for a multimodal config "
              "(set_cos_and_sin skips VL -- `not is_vl_model(...)` at :78 -- and "
              "update_cos_sin returns early while _cos is None at :135-136)")
        print("  -> the op receives cos=None, sin=None, and the paste's symptom "
              "says which outcome you have:")
        print("     profile_run traceback, `aclnnApplyRotaryPosEmbV2 failed, error "
              "code is 161001` + `Check cos != nullptr failed` -> the op refused "
              "the None and the engine died: verdict, owner and repair are in "
              "soc-platform-paths/310p-vl-rope-cos-nullptr-startup.md (#13522).")
        print("     serve came up and answers wrong text -> the None was "
              "tolerated, every position is wrong: "
              "soc-platform-paths/310p-vl-ocr-garbage-output.md (#11936).")
        print("     optional confirmation on their box: "
              "torch_npu.npu_apply_rotary_pos_emb(q, k, None, None, "
              "rotary_mode='half') on two small tensors reproduces 161001.")
    else:
        print("  globals: allocated (non-VL config), so this branch is "
              "populated normally; rope is a weak suspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
