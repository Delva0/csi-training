# An embedding checkpoint started as a *generation* service

Worked case: vllm-ascend **#5549** — image
`quay.io/ascend/vllm-ascend:v0.10.0rc1-310p`, Ascend 310P3, vllm **v0.10.0**
(`6d8d0a2`) + vllm-ascend **v0.10.0rc1** (`4604882`) — `vllm serve
/model/Qwen3-Embedding-0.6B --tensor-parallel-size 1 --trust-remote-code --dtype
float16 --enforce-eager --gpu-memory-utilization 0.08` (the reporter's later
comment on the same issue, 2025-12-31, carries this command verbatim, adding only
`--port 8011` and `--served-model-name qwen3_embedding`; **no `--task`, no
`--runner`**). Two flags in it are inert and must not become leads:
`--trust-remote-code` never instantiates the model directory's own code, because
`Qwen3ForCausalLM` is a registered architecture that
`vllm_ascend/models/__init__.py:57` replaces with
`vllm_ascend.models.qwen3:CustomQwen3ForCausalLM` — recheck with
`rg -n 'Qwen3ForCausalLM' vllm_ascend/models/__init__.py` at the reported ref — so
the class named in the error is downstream's replacement, not the checkpoint's own
remote code; and `--enforce-eager` changes no routing. The one *missing* flag is
the answer.

## Signature

The engine never becomes ready; the failure is at the **first weight group**:

```
INFO  Starting to load model /model/Qwen3-Embedding-0.6B...
Loading safetensors checkpoint shards:   0% Completed | 0/1
ERROR EngineCore failed to start.
  default_loader.py:259  loaded_weights = model.load_weights(
  vllm_ascend/models/qwen3.py:156  return loader.load_weights(weights)
  vllm/model_executor/models/utils.py:291  autoloaded_weights = set(self._load_module("", self.module, weights))
  vllm/model_executor/models/utils.py:277  raise ValueError(msg)
ValueError: There is no module or parameter named 'embed_tokens' in CustomQwen3ForCausalLM
```

and the engine-init line above it says `pooler_config=None`.

A second record of the same ValueError, **#2556**, puts it on a different
platform with a dirtier paste: an Ascend 910B (A2) host, the same vllm v0.10.0 +
vllm-ascend v0.10.0rc1, `torch 2.7.1+cpu` / `torch_npu 2.7.1.dev20250724`,
`npu-smi 24.1.rc1`, a repacked directory (`…_master_20250606_02`) served with
`--max-model-len 32768 -tp 1 --enable-prefix-caching --enable-chunked-prefill`
and **no `--task`**. Its log came only as a screenshot: the OCR keeps the frame
*names* — `…/vllm_ascend/models/q3.py` (i.e. `qwen3.py`) →
`return loader.load_weights(weights)` and `model_executor/models/utils.py line
277 raise ValueError(msg)` — the same `AutoWeightsLoader` path as #5549 — but it
loses the class name after `in` and mangles the init dump, so here the issue
*text*'s `in CustomQwen3ForCausalLM` is the decisive token and the OCR only
corroborates the loader frame. Classification is unchanged and
platform-independent: 310P and 910B fail identically, before any device op.

Two ways this is *not* answered: "the 310P can't do Qwen3-Embedding" and "the
checkpoint is corrupt". The 310P gates (NZ formats, eager-only, fp16) are
`../soc-platform-paths/atlas-300i-310p.md`; nothing in this traceback reached
them.

## Read the message before reading any source

The two variable parts of that `ValueError` are computed, not typed:

- `prefix` is the **qualified** name (`utils.py:139` `_get_qualname` =
  `base_prefix + rest`). `'embed_tokens'` with no leading `model.` therefore
  means the unmatched group sits at the **root** of the checkpoint, not under
  `model.`.
- the class after `in` is `type(self.module).__name__` — the top module the
  engine actually built (`utils.py:275-277`). A **pooling** build is renamed by
  `vllm/model_executor/models/adapters.py:132`
  (`_get_pooling_model_name`), so it reads `…ForEmbedding` /
  `…ForSequenceClassification`, never `…ForCausalLM`. Seeing the decoder class
  name proves the launch asked for a generation service, and
  `pooler_config=None` in the init line says the same thing a second time.

Both readings together also settle the checkpoint shape: the class owns `model`
and `lm_head` only, so a root-level group can only come from a checkpoint whose
tensors are written as `embed_tokens.weight`, `layers.N…`, `norm.weight` (a bare
`*Model` export) rather than `model.embed_tokens.weight` (**#5549 did not paste
the key list** — the root-level shape is inferred from the traceback plus the
pinned loader; one command confirms it).

## Confirming checks (two, both cheap)

1. **The key prefixes, without a GPU and without loading weights** — one call:
   `uv run --no-project python <skill_root>/scripts/checkpoint-key-prefix.py
   <model dir>`. It reads the safetensors header (or
   `model.safetensors.index.json`), prints the distinct root components, whether
   `lm_head` exists, `config.json`'s `architectures`, and whether a
   `config_sentence_transformers.json` is present; exit 1 means root-level names.
2. **The launch line**: is `--task embed` (or `--runner pooling`) in it? If the
   only "embedding" thing about the deployment is the model path, that is the
   finding.

Do not open `AutoWeightsLoader` before these: the answer is a launch-flag plus a
checkpoint-shape question, not a loader defect.

## Why `--task embed` is the fix (source-verified at vllm v0.10.0)

- `config.py:935-985` `_resolve_runner`: with `task=auto`, the suffix table maps
  `ForCausalLM` → `generate`, so an embedding checkpoint published under a
  `*ForCausalLM` architecture is served as a decoder even though this release
  *can* run it as a pooler. `_RUNNER_TASKS` (`config.py:104-108`) lists the
  pooling tasks: `encode`, `embed`, `classify`, `reward`; `--task embed` is the
  CLI spelling (`--task embedding` is only a deprecated alias, `config.py:570-575`).
- There is no registry gate to fight:
  `vllm/model_executor/models/registry.py:305` sets
  `is_pooling_model=True,  # Can convert any model into a pooling model` in
  `_ModelInfo.from_model_cls`, so `_get_supported_pooling_tasks`
  (`config.py:889-910`) offers `embed` for **any** registered architecture,
  including the `Qwen3ForCausalLM` entry vllm-ascend overrides.
- With a pooling task, `pooler_config` becomes non-`None`
  (`config.py:793-817`; it also picks up `config_sentence_transformers.json` via
  `get_pooling_config`, and honours `--override-pooler-config`), which is exactly
  the value printed in the init line — so that log field is the flag's shadow.
- The wrapper fixes the prefix problem by construction
  (`adapters.py:40-99`): `as_embedding_model` (`:102`) deletes `lm_head` /
  `logits_processor`, and `ModelForPooling.load_weights` then sees `self.model`
  as the only child with parameters (`:79-86`), applies
  `WeightsMapper(orig_to_new_prefix={"model.": ""})`, and calls
  `self.model.load_weights(weights)` — the inner loader's `params_dict` holds
  root-level names, so a root-level export and a `model.`-prefixed one both land.
- vllm-ascend v0.10.0rc1 carries the runner side: `worker/model_runner_v1.py:202`
  `self.is_pooling_model = self.model_config.pooler_config is not None`,
  `profile_run` → `_dummy_pooler_run` (`:2013`),
  `get_supported_pooling_tasks` (`:2787`) surfaced by `worker/worker_v1.py:351`.

## Not downstream drift — say so before the reporter files a bug

`vllm_ascend/models/qwen3.py:149-156` is a **verbatim copy** of upstream
`Qwen3ForCausalLM.load_weights` (`vllm/model_executor/models/qwen3.py:314-321`,
same `AutoWeightsLoader` and the same `skip_prefixes=["lm_head."]` rule under
`tie_word_embeddings`). Any vLLM build fails this checkpoint the same way, so the
ownership claim is "the launch asked for the wrong service", not "the Ascend
port broke the loader". Pooling/embedding models are a supported, validated part
of this project (#644 v0.7.3 and #3927 v0.11.0rc1 both list pooling models in
their release checklists; #1960 is the *recognition* gate for an architecture
vLLM does not know at all).

## Ladder an answer can be shipped from

1. **Same tree, one added flag**: `vllm serve <dir> --task embed --dtype float16
   --enforce-eager` (+ 310P env: `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`,
   `../soc-platform-paths/atlas-300i-310p.md`), then `POST /v1/embeddings`.
   Nothing is converted and no weights change. **Carry the reporter's own
   `--gpu-memory-utilization` through unchanged** (here `0.08`); do not raise it in
   the line you hand back. It is a fraction of *total* device HBM, and this paste's
   `npu-smi` already shows `41146` MB of the chip's `44280` MB held by an unrelated
   `java` process, so `0.9` asks the profiler for ~40 GB on ~1.8 GB of free HBM and
   the next stop is `No available memory for the cache blocks` rather than a
   pooling result. Put the free-chip step in the same paragraph whenever the paste
   shows an occupied device — `ASCEND_RT_VISIBLE_DEVICES=<idle id>` or stop the
   other process.
2. **If the vectors look wrong afterwards**: the pooling type. `Qwen3-Embedding`
   pools the last token; if `config_sentence_transformers.json` is absent the
   default comes from vLLM, so compare one vector against a reference before
   ranking numerics — `--override-pooler-config '{"pooling_type":"LAST"}'` is the
   knob (`config.py:795-806`). Wrong-vector lanes are
   `index.md` class 3 / `chunked-prefill-vector-corruption.md`.
3. **If it now dies on memory** (this reporter ran `--gpu-memory-utilization
   0.08` and their own `npu-smi` shows device 0 at `42467/44280` MB with a
   `java` process holding `41146` MB): the budget is 0.08 × 44.28 GB ≈ 3.5 GB
   while the free HBM on that device is ~1.8 GB, so expect the next stop at
   `determine_available_memory` / KV init — `../runtime-memory/kv-init-and-format-cast-alloc.md`,
   or free the device / pin another one.
4. **Only if the key list comes back `model.`-prefixed** with the class already a
   pooling one: the prefix is not the cause; hand the exact unmatched prefix and
   the class name to the loader/stacking lanes
   (`../upstream-downstream-drift/index.md`, `../startup-failures/index.md`).

## What this page does not establish

- The reporter's real key list and `config.json` (not pasted in #5549); step 1 of
  the checks is what turns the inference into evidence. The launch line itself is
  reporter-confirmed by the later comment, so the "no `--task`" half of the
  diagnosis no longer rests on inference — only the root-level key shape does.
  #2556 did not paste the key list either, and a repacked directory name is a hint
  about the export, not a defect.
- That `--task embed` actually fixed #2556: the reporter's only later reply on
  that issue was a mention of a maintainer, with no log.
- That a 310P host completes an embedding serve at this pin — no reporter
  confirmation exists in the trace, and step 3's memory arithmetic is unproven
  for their host.
- Which release first *documents* `--task embed` for Qwen3-Embedding; the
  recorded pooling cases (#5525 `--task embed` on v0.11.0, #17797 `--runner
  pooling` on v0.23.0) show the flag in use later, so re-grep `_RUNNER_TASKS` and
  `is_pooling_model` before quoting this page's line numbers at another pin.
