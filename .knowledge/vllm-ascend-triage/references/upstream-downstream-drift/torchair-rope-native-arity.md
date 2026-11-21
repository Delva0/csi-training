# `forward_native() takes from 3 to 4 positional arguments but 5 were given` at the torchair rope shim

## Scope

A per-rank `TypeError` raised on the **first dummy run** of a torchair-graph serve. The
downstream frame is `vllm_ascend/torchair/ops/torchair_rotary_embedding.py` `rope_forward_oot`
line 47 (`return self.forward_native(`), called from `rope_forward` (`:327`) through
upstream `custom_op.py` `forward`; above it the paste shows only upstream model frames
(`vllm/model_executor/models/deepseek.py:262 q, k = self.rotary_emb(positions, q, k)`).

Reported as a *deployment* failure — #3633's title is "Multi-Node-DP run failed" — while the
raise is inside `determine_available_memory` → `profile_run` → `_dummy_run`, before the first
collective (rule 5 below).

Neighbours: `impl-and-oot-signature-drift.md` (the `unexpected keyword argument` form of the
same rope contract), `patch-import-side-effect.md` (the shim never bound in that process),
`stale-module-binding.md`, and `../startup-failures/meta-kernel-missing.md` (the other
`_C::rotary_embedding` startup failure, a Meta registration instead of a signature).

## Rule 1 — the arity names the *upstream* class, and which rope class the checkpoint gets decides the crash

Verified at vllm-ascend `v0.11.0rc0` (`00ba071`) + vllm `v0.11.0rc3` (`8ce5d31`), the pairing
the same vllm-ascend tree names for itself (`docs/source/conf.py:68,76,80` — so it is not the
reporter's install that is unpaired):

| site | fact |
|---|---|
| `vllm_ascend/torchair/ops/torchair_rotary_embedding.py:44-51` | `if torchair_graph_config.enabled and not is_qwen_torchair: return self.forward_native(positions, query, key, offsets)` — four positional arguments after `self` |
| `vllm/model_executor/layers/rotary_embedding/base.py:82-87` | `RotaryEmbedding.forward_native(self, positions, query, key=None)` — **no `offsets`**, and the file has none: exactly "3 to 4 … but 5 were given" |
| `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:91-96` | `DeepseekScalingRotaryEmbedding.forward_native(..., key=None, offsets=None)` — this sibling **still** takes it |
| `vllm_ascend/torchair/torchair_model_runner.py:62` | `torchair_ops_patch()` runs only in the torchair runner, and it rebinds `AscendRotaryEmbedding.forward_oot = rope_forward` (`torchair/utils.py:217-221`) |
| `vllm_ascend/utils.py:521-524` | `REGISTERED_ASCEND_OPS["RotaryEmbedding"] = AscendRotaryEmbedding`, so *every* base rope in the model is the subclass whose `forward_oot` was just rebound |
| vllm `model_executor/layers/rotary_embedding/__init__.py:74-95` | `get_rope` returns plain `RotaryEmbedding` when `rope_scaling` is absent (and for `rope_type: default` without mrope), and `DeepseekScalingRotaryEmbedding` only for the DeepSeek yarn scaling |

So the failing set is **"unscaled rope + `torchair_graph_config` on"**, not "torchair is
broken": #3633 served `deepseek-ai/deepseek-moe-16b-base`, whose config carries no
`rope_scaling`, hence the base class. The tutorial the reporter followed serves
`vllm-ascend/DeepSeek-V3.1-W8A8` (`docs/source/tutorials/multi_node.md:121`), whose scaled rope
is routed through `native_rope_deepseek_forward` (`torchair_rotary_embedding.py:92-100`) and
survives the 4-argument call — that is why the documented path and CI never see it.

Do **not** generalise "qwen torchair models are exempt". `qwen2.py` passes
`is_qwen_torchair=True` only *inside* `if type(self.rotary_emb) is RotaryEmbedding:`
(`:133-136`), so the exemption is a property of that identity test, not of the model
family — and #3521 below falsifies it for a Qwen2-architecture model. At the same pin a
Qwen2-architecture checkpoint reached the *else* call `self.rotary_emb(positions, q, k)`
(`qwen2.py:136`, defaults `is_qwen_torchair=False`) and crashed, i.e. the rope instance
there is not the plain upstream class at that point. `qwen3_moe.py:240,260` passes
`is_qwen_torchair=True` explicitly and no recorded run exercises it. So the operative
statement is the *branch condition you can quote*: `torchair_graph_config.enabled and not
is_qwen_torchair`, evaluated on the instance that is actually installed.

**Reading the class name in the message:** Python prints the *qualname* of the function that
implements the method, so `RotaryEmbedding.forward_native()` appears even when the instance is
`AscendRotaryEmbedding` (it defines no override — `rg -n 'forward_native' vllm_ascend/` returns
only the sampler and this line 47). Never conclude "no Ascend subclass was involved" from it.

## Rule 2 — the exception's own arity settles the version question

"takes from 3 to 4 positional arguments but 5 were given" is a fingerprint of the *installed*
upstream class: its `forward_native` is `(self, positions, query, key=None)`. That refutes "your
image carries an older vllm whose rope still takes `offsets`" and "the image ships a vllm newer
than the plugin" in one line, without the reporter installing anything. Record the *class pair*
(base vs `DeepseekScalingRotaryEmbedding`) rather than a version number the report cannot show,
and mark that older releases did carry `offsets` on the base class (the v0.7.3 shim in
`impl-and-oot-signature-drift.md` declares it) as a recheck, not a cause: the crash is present
at the release pairing itself. The arithmetic that produces "3 to 4" (N and M count `self`)
and the two greps that read both signatures are `impl-and-oot-signature-drift.md` Rule 5;
this page owns the trigger set and the ladder.

## Rule 3 — the enabling line is usually the project's own tutorial

At the pin `docs/source/tutorials/multi_node.md:139,180` and `multi_node_kimi.md:96,136` pass
`--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true}}'`.
So "I followed the multi-node guide" and "torchair graph mode is on" are the same statement, and
the answer quotes the tutorial line beside the model it is written for. That contrast —
documented model has scaled rope, reported model does not — is the whole explanation of "why
does the official example work".

## Rule 4 — action ladder

1. **Drop the torchair key**, keep `{"ascend_scheduler_config":{"enabled":true}}`: the
   non-torchair shim (`vllm_ascend/ops/rotary_embedding.py:63-65`) accepts `offsets=None` and
   raises `NotImplementedError` only for a non-`None` value, and `torchair_ops_patch()` is
   unreachable outside the torchair runner, so nothing else in the serve changes.
2. If torchair is required: local patch of line 47 to `self.forward_native(positions, query, key)`.
   Behaviour-preserving **on this call path** because upstream models call
   `self.rotary_emb(positions, q, k)` (the paste's own `deepseek.py:262` frame proves it). Label
   it as a local patch, not the upstream repair, and compare numerics against the aclgraph run.
3. Switch to a scaled-rope DeepSeek checkpoint (V3/V3.1/R1) — routed through
   `native_rope_deepseek_forward`, never the base-class call.
4. A later rc/final is a plausible rung; say it is **unverified** when the snapshot holds only
   the reported tag, and give the one-grep recheck (`sed -n '40,52p'` of the torchair rope file
   at that tag).

## Rule 5 — the report's framing is not the phase

The traceback runs `engine/core.py _initialize_kv_caches` → `worker_v1.py determine_available_memory`
→ `profile_run` → `_dummy_run`, i.e. per-model startup. "Multi-Node-DP run failed" is therefore a
red herring: one node, `--data-parallel-size 1`, even TP1, fails identically. Put that
two-minute discriminator in the first answer instead of investigating HCCL, the rank table or the
`--data-parallel-*` group.

## What to send back

The full `--additional-config`, `python -c "import vllm, vllm_ascend; print(vllm.__version__, vllm_ascend.__version__)"`,
and the checkpoint's `config.json` `rope_scaling` block (its absence *is* the classification).

Recorded, two independent reports at the same pairing:

- #3633 — vllm-ascend v0.11.0rc0 (`00ba071`) + vllm v0.11.0rc3 (`8ce5d31`), 8-card DP×TP
  serve of `deepseek-moe-16b-base` (unscaled rope, DeepSeek model file).
- #3521 — the same two refs, image `vllm-ascend:v0.11.0rc0`, **two nodes / 16 NPUs**,
  `DeepSeek-R1-Distill-Qwen-7B`, whose traceback runs
  `torchair/models/qwen2.py:136` → `torchair_rotary_embedding.py:327` → `:47`. A
  Qwen2-architecture model is therefore **not** exempt: the qwen2 guard above took the
  unguarded branch. The reporter got no external follow-up, so the crash and its
  signature are source-plus-traceback facts while the trigger breadth stays a code-read
  claim.
