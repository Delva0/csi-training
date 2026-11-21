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

## What to send back

The image's `vllm_ascend/attention/` listing and `vllm_ascend/__init__.py`
`__version__`, `pip show vllm`, the two signatures above, and — for the
`cos_cached` row — `grep -rn 'cos_cached' $SITE/vllm $SITE/vllm_ascend
$SITE/mindie_turbo`.
