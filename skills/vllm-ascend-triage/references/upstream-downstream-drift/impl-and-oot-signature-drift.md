# `unexpected keyword argument` / missing attribute at an impl or `forward_oot` boundary

## Scope

Three messages, one contract: an upstream caller and its downstream replacement must
agree on the **call signature and the attribute names**. When they do not, the
failure is a `TypeError`/`AttributeError` raised *before* any kernel, usually at
layer construction or at the first attention call:

| message | boundary |
|---|---|
| `TypeError: <Ascend…Impl>.__init__() got an unexpected keyword argument '<kwarg>'` | upstream `Attention.__init__` → the backend impl's constructor (`q_lora_rank`, `kv_lora_rank`, `rotary_emb`, …) |
| `TypeError: rope_forward_oot() got an unexpected keyword argument '<kwarg>'` | a `forward_oot` CustomOp shim vs the *downstream* call site (`mla_v1.py`, `attention_v1.py`) |
| `AttributeError: 'RotaryEmbedding' object has no attribute 'cos_cached'` under `torch._dynamo` | an attribute name that the paired library never sets, read only on the traced/graph path or from a plugin-patched module |
| `TypeError: forward_native() takes from 3 to 4 positional arguments but 5 were given` | a **positional** over-supply: a downstream `forward_oot`/`forward` shim forwards an argument the upstream method's signature no longer has (Rule 5) |

This is the *signature* half of drift. Neighbours: `patch-copy-vs-upstream.md` (a
copied body lagging a state machine), `library-method-typo.md` (a misspelled library
method — no drift at all), `stale-module-binding.md` (the patch never reached the
consumer), and SKILL.md rule 7's unreachable-shim form (`forward_oot` bound but
never dispatched).

Recorded: #965, one report with four rows — vllm-ascend `v0.7.3` (`779eebb`) +
vllm `v0.7.3` (`ed6e907`) row 1 is source-verified here; the `0.8.5post1` image's
two rows are **outside the snapshot** and are answered as gaps with their checks.

## Rule 1 — for an impl-constructor `TypeError`, the routing is the finding, not the model

`impl_cls(...)` is called by upstream with whatever `**extra_impl_args` the model
supplies (vllm `v0.7.3` `vllm/attention/layer.py:114-118`), so the question is
**which impl class was selected**. At vllm-ascend `v0.7.3`:

- `vllm_ascend/platform.py:167-172` returns
  `vllm_ascend.attention.attention_v1.AscendAttentionBackend` for **any** `use_v1`
  model, *before* the `elif use_mla:` branch — so MLA never reaches the v0
  `AscendMLAAttentionBackend`.
- `vllm_ascend/attention/attention_v1.py:122-135` (`AscendAttentionBackendImpl.__init__`)
  declares the ten base arguments and **no `**kwargs`**, while the v0 MLA impl in
  `vllm_ascend/attention/attention.py:915,936-942` accepts `**extra_impl_args` and
  reads `extra_impl_args['q_lora_rank']` … . The attention package at that ref holds
  only `attention.py` and `attention_v1.py` — there is no `mla_v1.py`.
- The release states it itself: `docs/source/user_guide/release_notes.md:36`
  (under `## v0.7.3rc2`) — *"V1 engine is basic supported in this release. The full
  support will be done in 0.8.X release."*

So the verdict is **"this release has no MLA attention backend on the V1 engine"** —
not "your custom `q_lora_rank: null` config broke it". Any DeepSeek-V2/V3/R1 serve
fails identically there, and the reporter's own `V0 works / V1 dies at layer
construction` pair is the control that says so. Two one-line checks that transfer to
any image: `ls $(python -c 'import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))')/attention/`
(does an MLA impl module exist for this engine?) and the release-notes sentence.
Then the actionable rung is an engine/version choice — `VLLM_USE_V1=0` on 0.7.3, or a
release whose tree carries the MLA v1 module — not a patch to the model config.

## Rule 2 — a `forward_oot` kwarg mismatch is decided by two signatures in the *same* image

At `v0.7.3` the shim is
`vllm_ascend/ops/rotary_embedding.py:25-31`
(`rope_forward_oot(self, positions, query, key, offsets=None)`) and it reads
`self.cos_sin_cache` (`:34-37`). A caller in the same image that passes
`max_seq_len=` therefore disagrees with its own shim, which is a build/revision
inconsistency inside the image, not a pairing question — and only that call path
(the MLA prefill branch here) hits it.

The check is two reads in the *reporter's* tree, both one line:

```bash
sed -n '<caller>-3,<caller>+3p' $SITE/vllm_ascend/attention/<impl>.py   # what is passed
grep -n -A8 'def rope_forward_oot' $SITE/vllm_ascend/ops/rotary_embedding.py
```

If the caller passes a kwarg the shim lacks, say so and label the repair as a local
patch (accept and ignore the kwarg, or call with the shim's own signature) — it is
not a version-pairing answer. If the *shim* passes a kwarg upstream's `forward` no
longer takes, the direction is upstream's newer signature and the shim is stale: say
which side moved and cite both signatures.

## Rule 3 — an old attribute name shows up only where that code runs

`cos_cached`/`sin_cached` are pre-`cos_sin_cache` names; at vllm `v0.7.3`
`RotaryEmbedding` carries only `cos_sin_cache`
(`vllm/model_executor/layers/rotary_embedding.py`, `_set_cos_sin_cache`), so
`AttributeError: … no attribute 'cos_cached'` under dynamo's static attribute
resolution means some code that **only the traced path, or only a
plugin-patched module, executes** still reads the old name — graph mode traces every
attribute, while eager may never enter that branch. Candidate owners, one grep each
in the reporter's image: `vllm_ascend/patch/**`, the OOT rope shim, and any installed
accelerator plugin that patches vLLM model code (MindIE-turbo style adapters) — the
plugin is the one that survives an image change and the one a reporter does not
mention.

## Rule 4 — a report whose rows span releases/images gets one row per line, early

When the paste is a matrix (V0/V1 × 0.7.3/0.8.5 here), the answer is a table, not a
walk: `symptom | owner at that row | evidence class | the one check`, with rows whose
release is absent from the snapshot labelled **"not in snapshot"** and carrying the
`sed`/`grep`/`python -c` command that settles them on the reporter's own tree. The
rows are independent deliverables — the first one verified is worth shipping
immediately (`SKILL.md` rules 1 and 15), and sourcing a row whose tree you do not
have is the budget sink this page exists to prevent: #965 spent its whole 16 384-token
cap and emitted **no message text on any turn** while the 0.7.3 row had been
source-verified a third of the way in.

## Rule 5 — an arity message counts `self`, and the printed range names the callee class

`takes from 3 to 4 positional arguments but 5 were given` is a **positional**
over-supply, not a kwarg. N and M are the callee's required and total positional
parameters **including `self`**, and K counts the same way: under CPython,
`def forward_native(self, positions, query, key=None)` bound as a method and called
`f(positions, query, key, offsets)` prints exactly `from 3 to 4 … but 5 were given`
(two-line `uv run --no-project python` repro, no source needed). So the printed range
already names the callee shape, and the extra argument is in the shim:

| range | callee |
|---|---|
| 3 to 4 | base `RotaryEmbedding.forward_native(self, positions, query, key=None)` — no `offsets` |
| 4 to 5 | the DeepSeek-scaling / mRoPE subclass, whose `forward_native` still declares `offsets` |

Worked case #3521 (vllm-ascend `v0.11.0rc0` `00ba0710` ↔ vllm `v0.11.0rc3`
`8ce5d31`): upstream `vllm/model_executor/layers/rotary_embedding/base.py:82-86`
declares `forward_native(self, positions, query, key=None)` (3–4), while
`deepseek_scaling_rope.py:91-95` and `mrope.py:260-264` still declare `offsets`
(4–5) — upstream kept `offsets` inside those two subclasses only. Downstream,
`vllm_ascend/torchair/ops/torchair_rotary_embedding.py:36-52` (`rope_forward_oot`)
forwards four positionals to `self.forward_native(positions, query, key, offsets)`
whenever `torchair_graph_config.enabled and not is_qwen_torchair`, and
`torchair/utils.py:206-218` (`torchair_ops_patch`, called from
`torchair_model_runner.py:62`) binds that shim as `AscendRotaryEmbedding.forward_oot`
(`:218`); `CustomOp.dispatch_forward` (`vllm/model_executor/custom_op.py:81-106`)
selects `forward_oot` for the instance, so the shim is the live method. The Qwen2
torchair call site passes `is_qwen_torchair=True` only *inside*
`if type(self.rotary_emb) is RotaryEmbedding:` — the *upstream* class object
(`torchair/models/qwen2.py:133-136`) — so the exemption belongs to that guard, not to
the family: #3521 (a Qwen2-architecture checkpoint at the same pin) crashed through the
unguarded **else** call `qwen2.py:136`, i.e. the installed instance failed the identity
test there. `qwen3_moe.py:240,260` passes the flag explicitly and no recorded run
exercises it, so claim an exemption only for the branch you can quote. The trigger is
therefore **torchair graph mode on** plus a call site that never sets
`is_qwen_torchair`. Case detail — the model-side selector (`rope_scaling`), the tutorial
line and the reporter ladder — is `torchair-rope-native-arity.md`; this rule is the
arithmetic every arity message gets.

Two calls, then deliver: `rg -n -A6 'def forward_native' <upstream rope dir>` and
`rg -n 'forward_native\(|<shim name>\(' vllm_ascend/`. Repair: do not forward
`offsets` on the base path (its semantics live in the subclass bodies) — branch on
`isinstance(self, …)`. Same-day rung for the reporter: torchair graph mode off
(`--additional-config '{"torchair_graph_config": {"enabled": false}}'`) or the
vllm-ascend release whose shim matches the installed vllm. Recheck both signatures at
the image's pins; the range in the paste cannot go stale, the file lines can.

Delivery: #3521 had the mechanism inside a message on its 20th turn ("… no longer
accepts the 4th `offsets`, while `torchair_rotary_embedding.py:47` still passes it")
and then re-read registration and escape hatches until the 16 384-token cap, shipping
no verdict, no evidence table and no repair. An arity message is a two-grep answer:
the ranked diagnosis, the patch direction and the reporter's workaround belong in the
message that first names the mismatch.

## Rule 6 — "X's interface is different from vllm's" with no traceback is a *design* question

Reported #6549 as a Bug whose whole body is
`0.13.0 Release vllm-ascend MLA interface is different form 0.13.0 vllm` plus two
`<!-- Failed to upload … -->` images: no log, no frames, no version dump. That is not
a signature-drift failure yet — it is a question about *which class runs*, and the
answer is two call sites plus the registration that decides between them. Do not open
with a crash theory the report cannot support, and do not read "different" as "broken":
the MLA pair is **deliberately** divergent, because the Ascend impl fuses the q/kv
down-projection, kv-norm and RoPE into CANN ops inside `forward`.

The worked pair (verified at vllm-ascend `v0.14.0rc1` `52d4acfa` ↔ vllm `v0.14.1`
`d7de043`; the report's own 0.13.0 tag is **not** in the snapshot, so state the
generation and give the recheck `sed`):

| Surface | Upstream vLLM | vllm-ascend |
|---|---|---|
| declared contract | `vllm/v1/attention/backend.py:693 class MLAAttentionImpl`; `forward(self, layer, hidden_states_or_cq, kv_c_normed, k_pe, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None)` (`:720-731`) | `vllm_ascend/attention/mla_v1.py:655 class AscendMLAImpl(MLAAttentionImpl)`; `forward(self, layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv=False, output=None)` (`:1448-1456`) |
| caller | `vllm/attention/layer.py:468 class MLAAttention`; `self.impl.forward(self, q, kv_c_normed, k_pe, kv_cache, attn_metadata)` (`:848`) | `vllm_ascend/ops/mla.py:133 AscendMultiHeadLatentAttention.forward` → `torch.ops.vllm.mla_forward(...)` (`:145`) → `self.mla_attn.impl.forward(self.mla_attn.layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output)` (`:164-166`) |
| why the upstream caller never runs | — | OOT class map `vllm_ascend/utils.py:715` `"MultiHeadLatentAttentionWrapper": AscendMultiHeadLatentAttention`; backend selector `vllm_ascend/platform.py:467` `(True, False) → vllm_ascend.attention.mla_v1.AscendMLABackend`, so no `vllm/v1/attention/backends/mla/*` impl is ever constructed on NPU |

Three facts that make the diff concrete, and that a patch author needs:

1. `layer` → `layer_name`: upstream hands the impl the `MLAAttention` **module**, Ascend
   hands it the prefix **string**, in the same first positional slot — copied upstream
   bodies therefore fail or silently build the wrong metadata.
2. no `kv_c_normed` / `k_pe` pair: the prologue is fused, which is also why
   `MLACommonImpl` (upstream `vllm/v1/attention/backends/mla/common.py`) is not
   inherited — precedent `#868`: *"Why not `AscendMLAImpl` just inherit
   `MLACommonImpl` for common functions?"*, asked at v0.8.5rc1 and still the design.
   The **builder** side is the opposite: `AscendMLAMetadataBuilder`
   (`mla_v1.py:202`) does subclass upstream's `MLACommonMetadataBuilder`, so "does it
   reuse upstream?" has two different answers inside one backend.
3. `AscendMLABackend.get_name()` returns `"ASCEND_MLA"`, or `"FLASH_ATTN"` when
   `VLLM_USE_V2_MODEL_RUNNER` is set (`mla_v1.py:67-71`, an explicit `HACK` to dodge
   upstream's KV-init assertion) — so code branching on the backend name behaves
   differently on NPU, and even changes name with the runner. `get_impl_cls()`
   switches to `AscendMlaCPImpl` under context parallel (`:82-89`), so "the signature"
   is not one class either.

Verdict shape: *yes, deliberately different; here is what each side declares and what
actually runs; here is the surface to target if you are extending it* (the Ascend impl
/ `AscendMLAMetadataBuilder`, or a plugin backend at the selector), and *the abstract
`MLAAttentionImpl` contract is not honoured verbatim — that is a docs gap, not a
hardware bug*. Then separate the two forms before ranking anything: the **crash** form
of this subject is Rule 1 and `#3138` (`TypeError: get_attn_backend() got multiple
values for argument 'use_mla'`, vllm-ascend's call site lagging upstream's keyword),
which needs a traceback; the **question** form above needs none. With two uploads that
failed, the first rung is still "paste the text of those screenshots".

## What to send back

The image's `vllm_ascend/attention/` listing and `vllm_ascend/__init__.py`
`__version__`, `pip show vllm`, the two signatures above, and — for the
`cos_cached` row — `grep -rn 'cos_cached' $SITE/vllm $SITE/vllm_ascend
$SITE/mindie_turbo`. For an arity message, the two signatures themselves:
`python -c "import inspect, vllm.model_executor.layers.rotary_embedding as r; print(inspect.signature(r.RotaryEmbedding.forward_native))"`
plus the matching `vllm_ascend` shim line.
