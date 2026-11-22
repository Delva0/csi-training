# `KeyError: '.weight'` / `'lm_head.weight'` — the quant-method lookup asked for a
key no checkpoint description can carry

Lane B, every rank, at **model construction** (after `Starting to load model`, before
any shard progress), on a quantised dense checkpoint whose model file builds its
embedding without a `prefix`:

```text
(Worker_TP2 pid=3201) ... vllm_ascend/ops/vocab_parallel_embedding.py:81, in __init__
    quant_method = quant_config.get_quant_method(self, prefix=prefix)
  ... vllm_ascend/quantization/quant_config.py, in is_layer_skipped_ascend
    is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"
KeyError: '.weight'
```

The tell is the **key itself**: it *begins with a dot*. `prefix` was the empty string,
so the lookup asked for `'.weight'`. (Frame numbers are the reporter's image: in the
pinned tree the same call is `vllm_ascend/ops/vocab_parallel_embedding.py:85` with
`prefix: str = ""` at `:52`, and the hard index is `quant_config.py:176`. Cite the tree
you grepped and note the offset — do not call a 4-to-6-line difference a different
build.) Recorded #5564 (vllm-ascend v0.13.0rc1
`1b5d5abf` + vllm v0.13.0 `72506c98`, DeepSeek-R1-Distill-Llama-70B int8,
`--tensor-parallel-size 4`, workers `Worker_TP0..3`, then
`EngineCore_DP0 … WorkerProc initialization failed`), where the reporter had already
found the fix and it worked.

**This is not `quantization-method-gate.md` §C**, which owns a *named* layer key
(`KeyError: 'model.layers.0.self_attn.q_proj.weight'`) raised from the same function
on a foreign format renamed onto `ascend` (or, when the directory is the
Ascend-native conversion and the reporter already runs it under MindIE, §C-ii's
description-source question — same key, different owner). Same frame, opposite fact: there the
checkpoint's description does not match the keys the engine asks for; here the engine
asks for a key nobody could write. Route on the quoted key: a bare `.weight` (a dot
at the front) is the empty-prefix case below; a key whose module path is one of the
model's own `hf_to_vllm_mapper` `orig_to_new_prefix` origins (`lm_head.`,
`model.visual.`, `model.language_model.`) is the **renamed-prefix** case further
down, also this page; any other key with a real transformer-layer path is §C.

## The mechanism: an upstream model file omits one kwarg

Verified at those two pins (one-commit snapshots):

1. `vllm/model_executor/models/llama.py:386-390` builds
   `VocabParallelEmbedding(self.vocab_size, config.hidden_size, quant_config=quant_config)`
   — **no `prefix=`** — so the constructor's default (`prefix: str = ""`, threaded
   through vllm-ascend's `AscendVocabParallelEmbedding.__init__`, whose default is the
   same) reaches the quant-method lookup as `""`.
2. The sibling models in the same tree do pass it:
   `qwen2.py:383-388` and `deepseek_v2.py:1259-1264` both use
   `prefix=f"{prefix}.embed_tokens"`, and `llama.py`'s own `lm_head` uses
   `prefix=maybe_prefix(prefix, "lm_head")` (`llama.py:573-578`). So `llama.py` is the
   odd one out, and `embed_tokens` is its only un-prefixed module.
3. Upstream tolerates the empty prefix; the downstream lookup does not. Upstream
   `is_layer_skipped` (`vllm/model_executor/layers/quantization/utils/quant_utils.py:290+`)
   compares a prefix against an `ignored_layers` **list** and never subscripts a dict
   with it, while vllm-ascend's `is_layer_skipped_ascend` subscripts
   `quant_description[prefix + '.weight']` directly (`vllm_ascend/quantization/quant_config.py:176` at that pin; the reporter's image prints the same statement at
   `:170`, and their `get_quant_method` at `:129` vs the pin's `:116` — a small file
   offset, same code). Two facts, one crash: an upstream omission amplified by a hard
   index.
4. Consequence for classification: the trigger is a **pair** of facts in the model
   file — `quant_config=` **present** and `prefix=` **absent** — not a model family.
   A model that passes no `quant_config` at all is safe (`gemma2.py:266`,
   `internlm2.py:264`, `baichuan.py:293`), and so is a family that threads a prefix
   (`qwen2.py`, `deepseek_v2.py`). A full grep at vllm v0.13.0 `72506c98` finds
   **12** `VocabParallelEmbedding` sites with the pair (`apertus.py:336`,
   `arcee.py:196`, `exaone.py:314`, `exaone4.py:307`, `granite.py:269`,
   `granitemoeshared.py:164`, `grok1.py:314`, `hunyuan_v1.py:609`, `jais2.py:322`,
   `llama.py:386`, `molmo.py:808`, `nemotron_nas.py:243`) plus **5** un-prefixed
   `ParallelLMHead` sites that reach the same lookup (`adapters.py:405`, `:463`,
   `afmoe.py:619`, `minimax_m2.py:501`, `qwen3_omni_moe_thinker.py:670`). Llama is
   only the recorded one — do not narrow the rule to it, and do not carry the list to
   another pin without re-running the grep.

## Procedure — three calls before recommending a patch

1. **Look at the reporter's own model file, not the pin.** The reported family first:
   `rg -n 'VocabParallelEmbedding\(|ParallelLMHead\(' -A 5 <vllm>/model_executor/models/<model_family>.py`
   — `quant_config=` without `prefix=` in that block is the cause. Then, if the block
   *does* carry a prefix, run the same pair over the whole directory to see whether the
   crash comes from another module (MTP/drafter, a VL tower) rather than the main
   embedding: the condition is the pair, and the deepest `vllm_ascend` frame only says
   which layer asked, not which file decided. A block that *does* carry a prefix is not
   automatically right: a literal `prefix="lm_head"` inside a VL wrapper is renamed by
   that class's `hf_to_vllm_mapper` — see the renamed-prefix section below.
2. **Check the description has the key the fix will look up.** The file is
   `quant_model_description.json` in the served model directory (the name
   `AscendQuantConfig.get_config_filenames` declares,
   `vllm_ascend/quantization/quant_config.py:92` at that pin):
   `uv run --no-project python -c "import json,sys;d=json.load(open(sys.argv[1]));print({k:v for k,v in d.items() if 'embed_tokens' in k})" <model dir>/quant_model_description.json`.
   `model.embed_tokens.weight: "FLOAT"` → the one-line prefix fix lands in the
   `UnquantizedEmbeddingMethod` branch and is correct. **Key absent** → the prefix fix
   converts this crash into `KeyError: 'model.embed_tokens.weight'`, which is a
   description/export question (their conversion did not record the embedding), not a
   code bug. Say that before they patch, or they will report the second error as a
   regression. That follow-on error arriving without a prior empty-prefix crash is
   `quantization-method-gate.md` §C-iii: the key is then correctly prefixed, and
   `config.json`'s marker dict (which upstream returns before any glob) is the other
   source the same census has to rule out.
3. **Confirm which layer still lacks it.** After patching `embed_tokens`, the next
   un-prefixed module would crash the same way; check `lm_head` in the same file. In
   the recorded case `lm_head` already passes `maybe_prefix(...)`, which is why one line
   sufficed — read it, do not assume it.

## The renamed-prefix variant: a literal `prefix=` that `hf_to_vllm_mapper` rewrote (recorded #8367)

Same lane (model construction, every rank, before any shard bar), same function, but the
quoted key **has** a module path — and that path is exactly one the model's own weight
mapper renames:

```text
(Worker_TP0 pid=2029) ERROR ... [multiproc_executor.py:852]   File ".../vllm_ascend/quantization/modelslim_config.py", line 292, in get_linear_quant_type
(Worker_TP0 pid=2029) ERROR ...     quant_type = quant_description[prefix + ".weight"]
(Worker_TP0 pid=2029) ERROR ... KeyError: 'lm_head.weight'
```

Recorded #8367 (vllm-ascend v0.18.0rc1 `99e1ea0f` + vllm v0.18.0 `bcf2be96`;
`vllm serve /data/Qwen3-VL-32B-Instruct-w8a8-new --quantization ascend
--tensor-parallel-size 4 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", …}'
--mm_processor_cache_type shm --async-scheduling`). Verified at those two pins:

1. The model file hard-codes the prefix: `vllm/model_executor/models/qwen3_vl.py:1327-1332`
   builds `ParallelLMHead(…, prefix="lm_head")` literally, while the same class's parent
   passes `prefix=maybe_prefix(prefix, "language_model")` (`:1434`, class default
   `prefix: str = "model"` at `:1391`) and other families in the tree use
   `maybe_prefix(prefix, "lm_head")`. The lookup key is therefore `lm_head.weight`.
2. The same class's mapper renames that prefix: `hf_to_vllm_mapper = WeightsMapper(
   orig_to_new_prefix={"model.visual.": "visual.", "lm_head.":
   "language_model.lm_head.", "model.language_model.": "language_model.model."})`
   (`:1374-1381`), and `configure_quant_config` applies it to the description **before**
   the model is built (`vllm/model_executor/model_loader/utils.py:49` →
   `apply_vllm_mapper` → `hf_to_vllm_mapper.apply_dict`, `models/utils.py:109-114`, the
   prefix rule in `_map_name` at `:77-82`).
3. No export spelling can satisfy that lookup: `lm_head.weight` is renamed to
   `language_model.lm_head.weight` by the mapper, while a description that already says
   `language_model.lm_head.weight` is left alone — the engine still asks for
   `lm_head.weight`. **Editing the checkpoint's description cannot fix this**; the file to
   change is the model file (or the release).
4. The downstream side is only the hard index: `modelslim_config.py:292` is the `else`
   branch, taken because a `proj_name` of `lm_head` is not in
   `packed_modules_model_mapping`, and `quant_prefix_mapper` (`:432-434`) returns the
   prefix unchanged. Same pair as the empty-prefix case one indent later — an upstream
   model file's `prefix` kwarg plus a hard dict index.

What to answer, and the traps:

- The fix is to thread the prefix — `prefix=maybe_prefix(prefix, "lm_head")` at the
  reported class, which with the parent's `language_model` prefix yields
  `language_model.lm_head`, the key the mapped description carries. Grep the reporter's
  own family first (`rg -n 'prefix="lm_head"|"lm_head\.":' <vllm>/vllm/model_executor/models/`):
  this is a per-model-file property, not a per-release one, and VL wrappers are where it
  appears.
- Not a `.get(prefix + ".weight", "FLOAT")` default — `get_quant_method` decides every
  linear, so the service would start and the arithmetic would be wrong (same reasoning as
  the empty-prefix fix above).
- Rungs that cannot move it: `--enforce-eager`, `--compilation-config`
  (`cudagraph_mode`, capture sizes), `--mm_processor_cache_type`, `--async-scheduling`,
  `--max-model-len`, TP size. Construction precedes the first shard read and does no
  device work, so the oddest flag in the command is not a lead here (SKILL.md rule 11).
- Acceptance is one real request compared against the same weights served unquantised
  (or a reference run), not a healthy `/v1/models`: the change decides which quant method
  the head uses.
- Version scope: verified only at the two pins above. Whether a later release threads the
  prefix is not in the snapshot — say "cannot be checked from here" and hand over the grep
  that settles it at the reporter's ref.

## What to answer

- The one-line change **is** the fix, and it is the reporter's own finding
  (reporter-confirmed in #5564): in the model file's embedding construction, add
  `prefix=f"{prefix}.embed_tokens"`, matching `qwen2.py`. Do **not** use
  `maybe_prefix(prefix, "embed_tokens")` here — with an empty parent prefix it yields
  `embed_tokens`, not `model.embed_tokens`.
- The durable fix is an **upstream vLLM** change making `llama.py` consistent with its
  siblings; a downstream image cannot fix it for the next release otherwise. A
  vllm-ascend override for llama is not present at that pin (`vllm_ascend/` has no
  `models/` directory, `patch/` carries a deepseek worker patch, and its only `llama`
  mention is a comment about `llama_4_scaling`), so a hand edit in the container or a
  source build is the only local route. Without a code change the alternatives are the
  non-quantised (bf16) weights of the same model.
- **Do not "fix" it with `self.quant_description.get(prefix + '.weight', "FLOAT")`.**
  That function serves `LinearBase`, `FusedMoE` and `VocabParallelEmbedding`
  (`get_quant_method`'s branches), so a silent `"FLOAT"` default would run genuinely
  quantised linears as unquantised — the service starts and the output is wrong. If a
  guard is wanted downstream, scope it to the embedding branch (or raise an error that
  names the empty prefix).
- **Acceptance is not "the service started".** The change picks *which quant method the
  embedding uses*, so ask for one real request compared against the same weights served
  in bf16 (or a GPU/reference run), not just a healthy `/v1/models`.
- Version scope: the reporter saw the same `KeyError` on **0.11.0rc3 and 0.13.0**
  images, so this is not a 0.13 regression and an rc upgrade/downgrade is not a rung —
  the `prefix=` grep in their own `llama.py` is. The pinned support matrix states the
  family is supported (`docs/source/user_guide/support_matrix/supported_models.md:14`
  `DeepSeek Distill (Qwen/Llama) ✅`, `:24` `Llama2/3/3.1 ✅` at v0.13.0rc1) while
  `tests/` ships no llama case at that pin — a coverage gap, not a support verdict, and
  worth naming as the reason the path shipped.

## Traps this page exists for

- **Do not read the key as a checkpoint-format finding.** `KeyError: '.weight'`
  says the *engine* asked with an empty prefix; `KeyError: 'lm_head.weight'` on a VL
  model says it asked with a literal prefix its own mapper renamed. A key with a real
  transformer-layer path and no mapper origin belongs to the quant-description-mismatch
  lane (`quantization-method-gate.md` §C), and an aux `*_scale`/`*_offset` key to
  `weight-scale-keyerror-quant-variant.md`.
- **The frames below the loader are the model's, and the deepest `vllm_ascend` frame is
  not the owner.** `vocab_parallel_embedding.py` only forwards `prefix=`; the value was
  decided in the upstream model file. Answer with both anchors (`llama.py:386-390` +
  `quant_config.py:176`) instead of patching the embedding class.
- **Construction-time means every launch flag is irrelevant** — `--enforce-eager`,
  `--max-model-len`, TP size, graph mode, HCCL: the failure precedes the first shard
  read and produces no device work.

## Scope and limits

Verified at vllm-ascend v0.13.0rc1 (`1b5d5abf`) + vllm v0.13.0 (`72506c98`) for the
empty-prefix case, and at vllm-ascend v0.18.0rc1 (`99e1ea0f`) + vllm v0.18.0
(`bcf2be96`) for the renamed-prefix case: the
constructor defaults, the `quant_config=`-without-`prefix=` condition and the 17 call
sites listed above, the hard dict index, the `quant_model_description.json` file name
and the support-matrix rows. That site list is a property of **that** vllm pin — re-run
the two-line grep rather than quoting the families for another release. Not verified:
whether a later rc already passes a prefix (the snapshot carries one vllm-ascend pin),
and the reporter's image file content — their line numbers (`:129`/`:170`) differ from
the pin's (`:116`/`:176`), so always re-grep in the tree being quoted. The 0.11.0rc3
claim is the reporter's, not checked against a 0.11 pin.
