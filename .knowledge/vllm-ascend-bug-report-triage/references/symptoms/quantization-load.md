# Load-time failures: dense load, two `KeyError` families and weight-file integrity

Four classes share the load stage: the checkpoint was loaded **dense** (bf16 materialisation, OOM),
the loader **indexed a description entry that does not exist** (`KeyError` on `quant_description`),
a **model's own `load_weights` indexed a parameter the built module does not have**
(`KeyError` on `params_dict`: a `*.weight_scale` for an FP8 artefact, a `*.deq_scale` for a W8A8
one), or the **shard on disk is not a valid file** (`SafetensorError … incomplete metadata, file not fully covered`). The discriminators are the
logged args, the dict being indexed and the deepest *callee* — a frame whose callee lives in
`site-packages` means data, not code — not the model family.

A variant of §2 shares its frame but not its cause: a **native FP8 / compressed-tensors checkpoint**
routed into the Ascend description reader, where the "description" is the HF `quantization_config`
dict and the first missing key has a `visual.` prefix (§2b).
A third route to the same frame skips the reader entirely: at `v0.7.3rc2` and earlier the quantiser
declares **no** description filename, so the file is never opened and the dict indexed is whatever the HF
config supplied (§2c).

A fifth class is the **weight layout the quantiser chose, rejected by the operator that consumes it**
(§5: int4 weights converted to FRACTAL_NZ with `group_size 256` — the operator only accepts 64/128).
A sixth is not a failure at all: a projection of a "W8A8" model runs `aten::matmul` because the
checkpoint's own description says `"FLOAT"` for it (§6).

A seventh is an artefact question, not a code path: the checkpoint's quantisation scheme is one the
release's own tutorial never lists for that model — a community w8a8 conversion of a model documented with
bf16/w4a8 only — and the symptom is accuracy or instruction-following loss that disappears with the
documented weights (§7).

An eighth is a **shard-format** mismatch, not a quantisation one: a pre-sharded artefact
(`model-rank-<r>-part-*.safetensors`) handed to the default loader dies on
`assert loaded_weight.shape[output_dim] == self.org_vocab_size` inside vLLM's vocab-parallel weight loader,
and the fix is the doc's own `--load_format sharded_state` (§8, #11406).

A different failure again belongs to another page: a **device kernel** faulting during the startup
profile after the weights loaded (`QuantBatchMatmulV3…`, DFX `runtime error 507015`) is a CANN operator
fault, not a load defect — see `device-kernel-faults.md`.

## 1. Quantised checkpoint loaded dense → startup OOM

Worked case: vllm-ascend #1127 — "DeepSeek-R1-W8A8 NPU out of memory on 2×8×910B2 (64GB) with the
0.7.3.post1 image", TP=8, PP=2, `--max-model-len 1024`, `--gpu-memory-utilization 0.95`.

### Signature in the log

- OOM during model *construction* (weight allocation), before profiling or any request.
- The traceback runs `vllm/model_executor/layers/linear.py` → `RowParallelLinear` →
  `UnquantizedLinearMethod.create_weights`, and the engine config prints `quantization=None`.
- `UnquantizedLinearMethod` plus `quantization=None` together mean vLLM decided the checkpoint is
  dense. A 671B model then asks for bf16-sized weights (≈1.3 TB / 16 ranks ≈ 84 GB per device) on
  64 GB cards — OOM regardless of `--gpu-memory-utilization`.

### Why it happens (verified code path, vLLM v0.7.3 + vllm-ascend v0.7.3.post1)

- `ModelConfig._get_quantization_config` (`vllm/config.py`, ~line 3213 at v0.7.3) calls
  `get_quant_config` only when `model_config.quantization is not None`.
- `ModelConfig._verify_quantization` derives that value from `hf_config.quantization_config` (or
  `compression_config`). The Ascend override, `AscendQuantConfig.override_quantization_method`,
  returns `"ascend"` whenever an NPU is present — but it is only consulted *inside* the
  `quant_cfg is not None` branch.
- Therefore a msmodelslim-style checkpoint whose `config.json` has no `quantization_config` block
  (quantisation described only in `quant_model_description.json`) resolves to dense bf16. The
  description reader is `vllm/model_executor/model_loader/weight_utils.py:132`.
- At `v0.7.3.post1` the Ascend quant methods are thin wrappers over `AscendQuantizer`, which imports
  `mindie_turbo`; the shipped FAQ says w8a8 needs `vllm-ascend[mindie-turbo]` at 0.7.3, is native
  from **v0.8.4rc2** (where `quantization/w8a8.py` and `w8a8_dynamic.py` first appear), and that
  "w8a8 DeepSeek is working in process … please run DeepSeek with BF16 now".

### Workaround candidates (pick per version)

1. Make the quantisation unambiguous: pass `--quantization ascend` explicitly (vllm-ascend registers
   `AscendQuantConfig`, whose `override_quantization_method` returns `ascend` on NPU), and on ≤0.8.3
   install the mindie-turbo extra. This is only valid when the directory really carries a
   quantisation description; otherwise the load fails as class 2 below.
2. If w8a8 of that model is not supported at the reported version, serve the BF16 checkpoint, or use
   more nodes / PP so the dense footprint fits (upgrade to ≥ v0.8.4rc2 for native w8a8).
3. Reducing `--gpu-memory-utilization` or `--max-model-len` does **not** fix this class: the
   allocation precedes KV-cache sizing.

### Check before reuse

`grep -n quantization_config <model>/config.json`; read `_verify_quantization` and
`_get_quantization_config` at the upstream tag paired with the reported vllm-ascend release. A pure
PP/TP misconfiguration can also OOM, but then `quantization` is set correctly in the log — use that
as the discriminator. Established by reading the tags' code and FAQ, not by a run: say so.

### The other way a quantised checkpoint serves dense: the fp8 → bf16 fallback

A log line seen at `v0.17.0rc1` on A3 while loading an fp8 MiniMax-M2 checkpoint, before the profile run:

```text
Detected fp8 MiniMax-M2 checkpoint on NPU. Disabling fp8 quantization and loading dequantized bf16 weights instead.
```

`[reported]` — it is the reporter's own banner, not a line this harness has read in a clone. It changes what
the rest of the log means: the weights are materialised in bf16, so per-card memory, load time and any
"quantisation seemed to do nothing" observation are all explained by the fallback rather than by the
quantiser. Recheck before quoting it at another ref with `git grep -n 'Detected fp8\|dequantized bf16'
<ref> -- vllm_ascend`; where the message no longer exists, the family may have gained a real fp8 path
(`quantization-load.md` §2b discusses the native-FP8/compressed-tensors route, which is a different one).

## 2. Quant description KeyError (`KeyError: '<module>.weight'`)

Worked case: vllm-ascend #3494 (Qwen2.5-72B-Instruct-w8a8, TP4, prose `vllm-ascend 0.11.0`,
`torch-npu 2.7.1`, `--quantization ascend`), closed as stale with a weight-provenance answer.

### Symptom

`load_model` fails in every worker:

```
... vllm_ascend/ops/vocab_parallel_embedding.py ... quant_config.get_quant_method(self, prefix=prefix)
... vllm_ascend/quantization/quant_config.py, in is_layer_skipped_ascend
    is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"
KeyError: 'model.embed_tokens.weight'
```

then the wrapper errors `WorkerProc failed to start` / `EngineCore failed to start` — wrappers, not
causes. Deduplicate the repeated per-rank tracebacks and keep the deepest `vllm_ascend/` frame.

### Code at the reported ref

`vllm_ascend/quantization/quant_config.py`, `is_layer_skipped_ascend`, indexes `quant_description`
directly (the fused-shard and `experts` branches above it do the same). `get_quant_method` routes
`LinearBase`, `FusedMoE` and `VocabParallelEmbedding` through `is_layer_skipped_ascend`, so a
description missing *any* participating module raises before a weight is read.

Line-anchor caveat from #3494: the traceback names `:147`, which matches `v0.10.2rc1` (the log
banner agrees — `vLLM API server version 0.10.2`), while the tag the prose claims, `v0.11.0`, has
the same statement at `:170`. Use the statement, not the line, and record which ref the line matched
(`../workflow/version-pinning.md`, `scripts/ref_line_of.sh`).

### Mechanism

The checkpoint's `quant_model_description.json` is the contract: it must list every quantised *and*
unquantised module (`"FLOAT"` marks unquantised). Descriptions produced by an older or different
quantiser omit unquantised modules such as `model.embed_tokens`; direct indexing then raises. This is
a checkpoint/description defect exposed, not created, by the loader's strict indexing. At HEAD the
quantiser lives under `vllm_ascend/quantization/configs/modelslim_config.py`, where some lookups
already use `.get` and others still index, so the answer is version-specific.

### Workaround (relay-level, `reported`)

Obtain a description-complete W8A8 checkpoint for the same model from the supported conversion path
(a modelslim / vllm-ascend-converted release, e.g. a ModelScope `…-w8a8-vllm-ascend` repository)
and keep `--quantization ascend`. Issue #3494 was closed as stale with exactly that answer: use the
published w8a8 weights rather than a self-quantised directory. A load-time failure with an `ascend`
quantised model should be checked against the supported conversion path
(`vllm_ascend/quantization/…`, `modelslim_config.py`, `_310p/quantization/modelslim_config.py`)
rather than against kernels.

### Hypothesis worth stating as such

Adding `"model.embed_tokens.weight": "FLOAT"` to the description should clear this specific
KeyError, but it is unverified: later modules may be missing too, and a hand-edited description can
disagree with the tensor layout. Do not present it as the fix without evidence.

### The same lookup's positive branch: a layer the checkpoint marks `FLOAT` runs `aten::matmul`

A layer whose description entry says `"FLOAT"` does not raise; it is served unquantised (plain
`aten::matmul`), which is correct behaviour and not a dispatch gap. Full treatment, the per-release file
move and the answer shape: §6 (#979).

### When the missing key carries a wrapper prefix, read §2d

A `KeyError` whose key is `llm.model.embed_tokens.weight` (or any other wrapper-prefixed module path) is §2's
statement on a key the multimodal wrapper built. Its own section — the prefix rule, the one-command
discriminator against §2, the levers and the `HEAD` behaviour — is §2d below; do not re-derive it here.

### Class recurrence (`reported` via issue search)

`repo:vllm-project/vllm-ascend is_layer_skipped_ascend` returns 29 issues, including #7350 (K2.5
w4a8, `KeyError: 'embed_tokens.weight'`) and #5794 (GLM-4.7,
`KeyError: 'model.layers.92.head.weight'`, "only 0-91 in quant_model_description.json"). Same
signature, different missing module — the class, not the model, is the reusable part. #3867 (MiniCPM-V-4_5
int4 on `0.11.0`) is the wrapper-prefix variant above: `KeyError: 'llm.model.embed_tokens.weight'`.

### Check before reuse

`git grep -n 'quant_description\[prefix' <ref>` and whether the branch still uses dict indexing or
has moved to `.get(...)`. Confirm the reported ref actually contains the strict indexing.

## 2b. An FP8 / compressed-tensors checkpoint read as an Ascend description (vision prefix first)

Same `KeyError` surface as §2, different cause and a different answer, and the two are easy to confuse
because both name a missing `"<module>.weight"`: here the checkpoint declares its own quant method and no
per-module description file exists at all. Worked case: vllm-ascend #4313 — `allenai/olmOCR-2-7B-1025-FP8`
(Qwen2.5-VL architecture, a **native FP8** compressed-tensors checkpoint,
`quantization_config.quant_method: fp8`), 910B, `vllm-ascend 0.11.0rc1`, no `--quantization` flag, single
node:

```text
vllm_ascend/ops/linear.py:84 quant_config.get_quant_method(self, ...)
vllm_ascend/quantization/quant_config.py:110  is_layer_skipped_ascend(prefix, ...)
vllm_ascend/quantization/quant_config.py:161  is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"
KeyError: 'visual.blocks.0.attn.qkv.weight'
```

**Verified chain at that pair (vllm-ascend `v0.11.0rc1` + vLLM `v0.11.0`):**

1. `platform.py:56` advertises `supported_quantization = [ASCEND_QUANTIZATION_METHOD]` — `ascend` is the
   only quant method this build offers.
2. `AscendQuantConfig.override_quantization_method` (`quant_config.py:93-97`) returns `"ascend"`
   whenever `torch.npu.is_available()`, without looking at `hf_quant_cfg`, so the checkpoint's
   `{"quant_method": "fp8", …}` is routed into the Ascend class instead of being rejected.
3. `get_quant_config` takes the HF `quantization_config` **before** any `quant_model_description.json`
   lookup (vLLM `v0.11.0` `model_executor/model_loader/weight_utils.py:238-250`), and
   `AscendQuantConfig.__init__` assigns `self.quant_description = quant_config` (`:58`). The
   "description" is therefore the FP8 config dict, whose keys are
   `quant_method`/`fmt`/`weight_block_size`/`ignored_layers` — **no per-module `"<module>.weight"`
   entries at all**, because that form only comes from a ModelSlim `quant_model_description.json`.
4. `get_quant_method` sends every `LinearBase` through `is_layer_skipped_ascend`, which at `:161` does
   `self.quant_description[prefix + '.weight'] == "FLOAT"` (and `:150` for fused shards). The first
   linear built raises.
5. The failure is at **module construction**, before any weight tensor is read, and the key it names is
   the first mapped module the checkpoint never described. The first linear built is the **vision
   tower** (Qwen2.5-VL constructs `self.visual` before the language model), which is why the missing key
   has a `visual.` prefix — a `visual.`/`vision.` name in a load-time description KeyError is the
   fingerprint of this variant rather than of §2's incomplete description.

### Workaround ladder

1. **Re-quantise to W8A8** with ModelSlim and serve `--quantization ascend` (the thread's own answer:
   `hhd52859` 「用modelslim量化得到w8a8权重」, `Ascend/msit` `msmodelslim`; its
   `quant_model_description.json` carries per-module keys). Costs a quantisation run; the supported path
   at every release.
2. **Move to a release that has an `fp8` method**, if the checkpoint is block-wise FP8. Generic HF `fp8`
   is re-registered to `AscendFp8Config` from **`v0.21.0rc1`** (`fcb30bebe`, #9757); from
   **`v0.27.1rc1`** (`061a92696`, #14852) that class serves only block-wise scales and refuses
   per-tensor/per-channel checkpoints with a `NotImplementedError` naming the two supported routes
   (re-quantise with ModelSlim, or an llm-compressor checkpoint with `--quantization
   compressed-tensors`). Verified at `main`: `quantization/configs/fp8_config.py:46`
   (`weight_block_size is None`) and `:55` (`activation_scheme != "dynamic"`).
3. **`--quantization compressed-tensors`** only from `v0.12.0rc1` on (`bdc66972d`, #4036, which also
   narrows the override to `hf_quant_cfg.get("quant_method") is None`; the release tag `v0.12.0` does
   **not** contain that commit — `merge-base --is-ancestor` returns 1 — so check the tag, not the line),
   and only for W8A8 static/dynamic compressed-tensors. It is *not* a route for a plain `fp8` checkpoint.
4. **BF16 base model** if neither path is acceptable: ~2× weight bytes, no quantisation work.

Do not hand-add `.weight` entries to the FP8 dict: the values would have no scheme behind them (§2's
hypothesis caveat applies twice over).

### Discriminator against §2

§2 is "the description exists but omits a module"; this is "there is no description and the method was
never supported". One command each at the reported ref:

```bash
git -C /workspace/repos/vllm-project/vllm-ascend grep -n "supported_quantization" <ref> -- vllm_ascend/platform.py
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:vllm_ascend/quantization/quant_config.py | grep -n -A6 "def override_quantization_method"
ls /path/to/model | grep -E 'quant_model_description.json|config.json'
```

A `quantization_config` in `config.json` with no `quant_model_description.json` is this section; the
opposite (a per-module description present) is §2. And before claiming the release "should have loaded
it", `git grep -n 'supported_quantization' <ref> -- vllm_ascend/platform.py` shows what the release
actually accepts — `platform.py:140` lists `fp8`/`compressed-tensors` only at `v0.23.0rc1`.

### Status and limits

At `v0.11.0rc1` this is an unsupported-quant-method failure wearing a misleading `KeyError` (the honest
issue title is "an unsupported `quantization_config` should be rejected at startup with a clear message,
not indexed as a description"). Fixed later in two steps (override narrowing at `v0.12.0rc1`, a real
`fp8` config from `v0.21.0rc1`, block-wise-only gate at `v0.27.1rc1`); the intermediate
`v0.12.0rc1` … `v0.20.x` behavior for a plain `fp8` checkpoint (no ascend implementation registered,
vLLM's own `Fp8Config` selected) was **not** verified here. No hardware: nothing was loaded.

## 2c. `KeyError` on the first projection because no description file was read (`≤ v0.7.3rc2`)

Same frame as §2, different owner: the loader never opened `quant_model_description.json`, so the
missing key says nothing about the checkpoint's completeness. Worked report #569 — Qwen2.5-32B-Instruct
W8A8 produced with the MindIE/modelslim `examples/models/qwen/convert_quant_weight.sh` (the reporter had
first proved the same directory serves under `mindieservice_daemon`), `vLLM Version: 0.7.3`,
`vLLM Ascend Version: 0.7.3rc2`, `torch-npu 2.5.1.dev20250320`, CANN `8.0.0`, `vllm serve … -tp 1`.

### Verified at the reported ref

- The effect is the same frame with an *empty* description dict instead of an incomplete one: the loader
  asks the quant class for its filename list, gets none, and never touches the model directory.
- The traceback's line is the tag's: at `v0.7.3rc2` `is_layer_skipped_ascend` starts at
  `quantization/quant_config.py:110` and `:124` is the **fused-shard** index
  `is_shard_skipped = self.quant_description[shard_prefix + '.weight']` (`qkv_proj` →
  `q_proj`/`k_proj`/`v_proj`); `:135` is the unfused form. The fused mapping, not the model, chose the
  reported key.
- `AscendQuantConfig.get_config_filenames()` (`:74-76` at that ref) returns `[]`, in every tag up to and
  including `v0.7.3rc2` (`v0.7.1rc1`, `v0.7.3rc1`, `v0.7.3rc2`). Upstream vLLM `v0.7.3`
  `model_executor/model_loader/weight_utils.py:132 get_quant_config` asks for that list at `:180`,
  returns `quant_cls()` when it is empty (`:183-184`), and globs `*.json` only after `:186` — so the
  description file on disk is never opened.
- Which dict is then indexed is decided one branch earlier: a non-None HF `quantization_config` wins at
  `:141-153` (`return quant_cls.from_config(hf_quant_config)`). `AscendQuantConfig.__init__(self,
  quant_config)` has **no default** at that ref, so `quant_cls()` at `:184` cannot build an empty
  description — it raises `TypeError: __init__() missing 1 required positional argument`. A `KeyError`
  on a `<module>.weight` key therefore also proves the HF-`quantization_config` branch supplied a dict
  with no per-module entries.
- Fix: `d491957e0` *"new modelslim quantization model support v0.7.3"* (#743, 2025-04-30) changes exactly
  that one line, `[]` → `["quant_model_description.json"]`. `git tag --contains d491957e0` =
  `v0.7.3`, `v0.7.3.post1` — absent from `v0.7.1rc1`, `v0.7.3rc1` and `v0.7.3rc2`.
- The class moves, so locate it before trusting a line: `git grep -n 'def get_config_filenames' <ref> --
  vllm_ascend/quantization` (`quant_config.py` → `modelslim_config.py` → `configs/modelslim_config.py`),
  and later refs list the file in more than one place.

### Answer shape

- **Workaround:** the release, not a flag — vllm-ascend ≥ `v0.7.3` (`v0.8.5rc1`+ is also reported
  working). Maintainer `wangxiyuan`, 2025-05-14: "0.7.3 and 0.8.5rc1 works with w8a8 now. Please try
  again"; the thread's earlier answer (Yikun, 2025-04-18/19) is that w8a8 is still WIP (PRs #511/#580).
  No `--quantization` / `--additional-config` value makes an rc2 build read the file. Cost: a wheel
  bump, and the matching vLLM tag with it.
- **Check the HF block before promising that the upgrade alone fixes it.** Because `:141-153` outranks
  the file list, a checkpoint whose `config.json` carries a `quantization_config` is still indexed from
  that dict at `v0.7.3`:
  `uv run --no-project python -c "import json;print(json.load(open('<model>/config.json')).get('quantization_config'))"`.
  Block absent → the upgrade is the fix; block present without `<module>.weight` keys → re-convert, or
  drop the block so the modelslim description is read (proposed, not reported).
- **Discriminator against §2:** the ref plus which module raised. A `KeyError` at ≤ `v0.7.3rc2` on
  `model.layers.0.self_attn.q_proj.weight` (Qwen2) or `…q_a_proj.weight` (DeepSeek) is this class; the
  same KeyError at ≥ `v0.7.3` means the file was read and is incomplete (§2).

### Status and limits

The thread carries no code-level diagnosis and no reporter confirmation of a rerun, so the upgrade is
maintainer-attributed. The MindIE/modelslim converter's output (`config.json` shape, and hence which
branch of `get_quant_config` fired for the reporter's directory) is not inspectable here — those sources
are not cloned — so the branch statement above rests on the reported `KeyError` plus the tag's code, not
on their files.

## 2d. The key is under a wrapper prefix the loader does not strip (`llm.model.embed_tokens.weight`)

`MiniCPM-V-4_5-int4` on `v0.11.0` with `--quantization ascend`, TP=1: the engine loads the wrapper,
reaches the **inner** LLM's embedding, and dies at `is_layer_skipped_ascend` with
`KeyError: 'llm.model.embed_tokens.weight'` — §2's statement, on a key that carries a submodule prefix.
Verified at `v0.11.0`:

- `vllm_ascend/ops/vocab_parallel_embedding.py:81` sends the embedding through
  `quant_config.get_quant_method(self, prefix=prefix)` while `VocabParallelEmbedding` is constructed, so the
  lookup key is the **module path** the wrapper built, not a name the checkpoint chose. The deepest frame is
  `vllm_ascend/quantization/quant_config.py:152 in is_layer_skipped_ascend`; engine death is the wrapper pair
  (`EngineCore failed to start` / `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`).
- `vllm_ascend/quantization/quant_config.py:107-108` strips exactly one wrapper prefix:
  `if prefix.startswith("language_model"): prefix = prefix.split('.', 1)[-1]`. The else-branch lookup is
  `is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"` at `:170` at that tag (the report's
  traceback names `:152`, the same statement in a neighbouring build — quote the statement and the ref, as
  in §2's line-anchor caveat).
- The `llm.` prefix is upstream's: vLLM `v0.11.0` `model_executor/models/minicpmv.py:1005` builds the
  language model with `prefix=maybe_prefix(prefix, "llm")` (the traceback's `minicpmv.py:1004`/`:1656`
  agree), so the Qwen3 submodule's modules are `llm.model.*`. A description emitted by quantising the
  *inner* model — or keyed on the HF names `model.embed_tokens.weight` — carries no `llm.` form.

One command on the model directory separates this from §2 (run it on the reporter's box, or ask them for
its output; the checkpoint is in no clone):

```bash
uv run --no-project python -c "import json,sys;d=json.load(open(sys.argv[1]+'/quant_model_description.json'));print([k for k in d if 'embed_tokens' in k][:5]);print(len(d))" <model-dir>
```

- **no `embed_tokens` key at all** → §2: the description is incomplete (add the row, or re-quantise);
- **`model.embed_tokens.weight` (or another prefix) present** → this section: the description is keyed on
  the inner model's names, and the fix belongs to the checkpoint or to the loader's prefix mapping.

**`llm.` is not §2b's `visual.` fingerprint.** A `visual.`/`vision.` key with no `quant_model_description.json`
in the directory means a native FP8/compressed-tensors checkpoint was routed into the Ascend reader (§2b); an
`llm.`/`language_model.`-prefixed key on a *quantised Ascend* checkpoint is the wrapper/description prefix
disagreement here. Fingerprint the prefix before choosing the section. Recheck at the reported tag with
`git grep -n 'startswith(' <ref> -- vllm_ascend/quantization/quant_config.py` and
`git grep -n 'quant_description\[prefix' <ref> -- vllm_ascend`: later releases moved the class into
`quantization/configs/modelslim_config.py`, so an empty grep at `main` means "moved", never "fixed" — a
recorded run burned two messages on two empty `git grep … main` calls.

Levers, cheapest first: (1) edit `<model>/quant_model_description.json` — add
`"llm.model.embed_tokens.weight": "FLOAT"` and every other missing entry — cheap and reversible, but one
key at a time is one round trip per module, so diff the description's keys against the wrapper's parameter
names and fix them in one pass; (2) re-quantise the bf16 checkpoint of the same wrapper with
ModelSlim (W8A8), which emits per-module keys for the wrapper's own layout (§2's first rung); (3) serve the
bf16 `MiniCPM-V-4.5` checkpoint without `--quantization ascend` — correct, but loses the int4 footprint;
(4) the thread's only lever is 「use the latest release or rc version」 (maintainer, `reported`, never
confirmed) — relay it attributed; it is not a mechanism.

Status and limits: the failing line, the one-prefix strip and the upstream `llm` prefix are `verified` at
the tag pair; *which* description shape this checkpoint has is `hypothesis` and the command above is the
falsifier. `HEAD` no longer crashes here — the skip logic moved to
`vllm_ascend/quantization/configs/modelslim_config.py`, where `_find_quant_description_key` (`:320-327`)
returns `None` instead of indexing and `get_quant_method` then returns `UnquantizedEmbeddingMethod()` with
only a `debug` log (`:885-889`), while `quant_prefix_mapper` (`:676-738`) adds per-`model_type` alias
tables. Two consequences: a missing entry at `HEAD` becomes a **silently unquantised** module (look for an
accuracy/throughput anomaly, not a `KeyError`), and those alias tables carry no `minicpm` entry, so do not
claim the nested-prefix case is solved there without checking the model type.

## 3. Quantised checkpoint, unquantised model (`KeyError: '...weight_scale'` / `'...deq_scale'`)

Worked case: vllm-ascend #12675 — "use v0.23.0rc1 Deployment deepseek-v4-flash error", image
`quay.io/ascend/vllm-ascend:v0.23.0rc1`, `deepseek-ai/DeepSeek-V4-Flash` from ModelScope (original
**FP8** weights), `--dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 8192
--tensor-parallel-size 8 --enable-expert-parallel --tokenizer-mode deepseek_v4 --block-size 128
--async-scheduling`, and **no `--quantization` flag**; open, labels `bug`, `triaged`, `llm-model`,
`deepseek`, `deepseek-v4`.

### Signature in the log

Every worker dies during load; wrappers are `WorkerProc failed to start` /
`EngineCore failed to start`, and the deepest project frame is the **model's own** loader, not the
quant-config helper:

```
File ".../vllm-ascend/vllm_ascend/models/deepseek_v4.py", line 1535, in load_weights
    param = params_dict[name]
KeyError: 'model.layers.2.self_attn.indexer.wq_b.weight_scale'
```

(also seen: `model.layers.N.self_attn.wkv.weight_scale`). Different ranks fail on different layers,
so dedupe. At `v0.23.0rc1` `deepseek_v4.py:1535` is exactly that statement, so the traceback
fingerprints the ref rather than leaving it inferred.

### Mechanism

`params_dict = dict(self.named_parameters())` is built from the **constructed** module, and the
`weight_scale`/`weight_offset` entries come from the quantisation scheme (for example the 310P
`get_perchannel_param`, or the w8a8 scheme's `get_dynamic_quant_param`), which is only attached when
a quantisation config is built. With no `--quantization` and no `quantization_config` in
`config.json`, vLLM builds an unquantised model while the checkpoint still ships the FP8 `*_scale`
tensors; the loader iterates the checkpoint's names, so the first scale entry raises. This is the
**opposite direction** to §2: there the model expects a description entry the checkpoint lacks; here
the checkpoint carries tensors the model never allocated.

### Discriminator

| | §2 description KeyError | §3 scale KeyError |
| --- | --- | --- |
| missing key | a module **weight** (`model.embed_tokens.weight`) | a **scale** (`...wq_b.weight_scale`) |
| dict indexed | `self.quant_description` in `is_layer_skipped_ascend` | `params_dict = dict(self.named_parameters())` |
| frame | `quantization/quant_config.py` | the model file's `load_weights` |
| logged args | `--quantization ascend` present | no `--quantization` |
| cause | description incomplete | model built unquantised for a quantised checkpoint |

### Code at the reported ref (v0.23.0rc1, verified)

- `vllm_ascend/models/deepseek_v4.py:1310` `load_weights`; `:1341` `params_dict =
  dict(self.named_parameters())`; the non-expert branch guards only `.bias`
  (`if name.endswith(".bias") and name not in params_dict: continue`), then the KV-scale remap and
  `is_pp_missing_parameter`, then indexes `params_dict[name]` unguarded at `:1535`.
- `:1382` rewrites **any** checkpoint name ending in `.scale` to `.weight_scale`
  (`if name.endswith(".scale"): name = name.replace(".scale", ".weight_scale")`). A native DeepSeek
  FP8 checkpoint that stores per-module `.scale` tensors therefore reaches the loader under the
  `weight_scale` name, even when the module itself was built dense.
- The same file already shows where a module is deliberately unquantised: `Compressor` (`:596`)
  builds `wkv`/`wgate` (`:622-635`) with `quant_config=None if get_ascend_device_type() in
  {AscendDeviceType.A5} else quant_config` (A5 compressor kernels need it), while `Indexer.wq_b`
  (`:550-555`) takes `quant_config` unconditionally. So "which modules have a `weight_scale`
  parameter" is hardware- and config-dependent at this tag.
- vllm-ascend v0.23.0rc1 **does** support FP8: `platform.py:140` lists `ascend`,
  `compressed-tensors`, `fp8` and `deepseek_v4_fp8` in `supported_quantization`, and
  `quantization/fp8_config.py:137` registers `deepseek_v4_fp8` onto `AscendFp8Config` (same class as
  `fp8`), which logs `"Using the vLLM Ascend fp8 Quantization now!"` via `logger.info_once`. Absence
  of that line in the reporter's log is the cheapest falsifier for "the quant config was never
  applied".

### Workaround (maintainer-attributed, #12675)

kunpengW-code (COLLABORATOR, 2026-07-24): "The original weights are in FP8 format and cannot be
directly run on Ascend A3 or A2. You need to use Modelslim to re-quantize the weights", pointing at
`docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html`. That is the
workaround to relay: serve a modelslim-converted W8A8/W4A8 directory (with `--quantization ascend`),
not the upstream FP8 release — it is the supported conversion route, not a code-level fix, and no
vllm-ascend flag makes an FP8 checkpoint loadable on A2/A3. An automated triage bot had already
labelled it "a DSv4 Flash weight loading issue", which is a routing note, not a mechanism.

### Status and limits

Which quant config was active in the report is **not proven**; keep it a hypothesis with the
log-line check above. No code fix was found after the report in the clone
(`git log -S'weight_scale' --since=2026-07-01 -- vllm_ascend/models/deepseek_v4.py` returns only
refactors); at HEAD the file moved into the `vllm_ascend/models/deepseek_v4/` package. Do not present
a hand-patched checkpoint rename as the fix.

### Check before reuse

`grep -n quantization_config <model>/config.json` and the start command's `--quantization`;
`git show <tag>:vllm_ascend/models/<Model>.py | grep -n 'params_dict\[name\]\|\.scale'` for the
model in the report; and
`git grep -n 'supported_quantization\|register_quantization_config' <tag> --
vllm_ascend/platform.py vllm_ascend/quantization/` for what the release actually supports. If the
model *was* built with a quant config and the scale key is still missing, the checkpoint's
quantisation format differs (w4a8/fp8 vs w8a8) — a different defect, so state which of the two you
verified. A missing `weight_scale` parameter is a model-construction question; the checkpoint's own
scale naming is a conversion question.

### The `deq_scale` variant: the missing name identifies the method, and a `--quantization` that never reached the process (worked case #5813)

The W8A8 artefact produces the same shape as the `weight_scale` cases above with a **different missing
parameter**:

```
File ".../vllm/model_executor/models/qwen2.py", line 496, in load_weights
    param = params_dict[name]
KeyError: 'layers.0.mlp.gate_up_proj.deq_scale'
```

reached through `models/qwen3.py`'s `load_weights` → `AutoWeightsLoader.load_weights`. Report: vllm-ascend
#5813, the *shipped* `vllm-ascend/Qwen3-8B-W8A8` ModelScope directory, reported on `0.11.0` and
`0.13.0rc1` alike with `--dtype auto` on A3; a second reporter hit the same key on Qwen3.5
(`gate_gate_up_proj.deq_scale`). `deq_scale` is the W8A8 per-channel dequant scale, so a `deq_scale` key
with no parameter means the Ascend method never wrapped that `LinearBase` — the model was built dense (the
§3 mechanism), not that the checkpoint is unusual.

**Every Ascend linear registers exactly what its method's `create_weights` asks for, so the parameter name
in the `KeyError` names the method the module was built with** (`quantization/quant_config.py:322-372` at
`v0.13.0rc1` calls `get_weight` + `get_pertensor_param` + `get_perchannel_param` of the selected method;
`get_perchannel_param` is `quantization/w8a8.py:65-80`):

| method selected | parameters registered | a `deq_scale` key from the checkpoint gives |
| --- | --- | --- |
| `W8A8` static, `params_dtype` bf16/fp16 (`deq_scale` at `:72`/`:75`, float32 on bf16, int64 on fp16) | weight, input_scale/offset, quant_bias, **deq_scale**, weight_scale/offset | loads |
| `W8A8` static but `params_dtype` neither bf16 nor fp16 (same function) | everything above **minus** `deq_scale` | `KeyError: ...deq_scale` |
| `W8A8_DYNAMIC` (`quantization/w8a8_dynamic.py:55-67`) | weight, weight_scale, weight_offset only | `KeyError: ...deq_scale` |
| `AscendUnquantizedLinearMethod` (description says `FLOAT`, or no quant config at all) | weight (+bias) | `KeyError: ...deq_scale` |

Which method is selected is decided by `quant_model_description.json`, never by the checkpoint's tensor
names: `AscendQuantConfig.get_config_filenames()` (`quant_config.py:91`) returns that one filename,
`get_quant_method` reads `quant_description["<prefix>.weight"]` and maps the value through
`ASCEND_QUANTIZATION_METHOD_MAP` (`quantization/utils.py:get_linear_quant_type` /
`get_quant_method_modelslim`). `gate_up_proj` is looked up as `gate_proj` + `up_proj` **only** when the
packed mapping is available — vLLM installs it from the model class
(`vllm/model_executor/model_loader/utils.py:271-292` at vLLM `v0.13.0`), and `packed_modules_model_mapping`
in `quant_config.py` has no dense `qwen3` entry, so a dense Qwen3 uses the model class's own mapping.

Applied to the shipped artefact (fetched from the hub, 2026-11-17): `config.json` has **no
`quantization_config`** (model_type `qwen3`, bf16) and the directory holds one
`quant_model_weight_w8a8.safetensors` plus a 98 KB `quant_model_description.json` whose metadata is
`model_quant_type: "W8A8"`, `is_dynamic: false`, `fa_quant_type: null`, and whose per-parameter keys are all
`W8A8` except `model.embed_tokens.weight: "FLOAT"` (entries use `gate_proj`/`up_proj`, never
`gate_up_proj`). Therefore:

- Auto-detection cannot fire: `AscendQuantConfig.override_quantization_method` returns `None` whenever the
  HF quant cfg is absent (`quant_config.py:99-107`), and this directory has none. **`--quantization ascend`
  is mandatory here** — `ASCEND_QUANTIZATION_METHOD = "ascend"` (`vllm_ascend/utils.py:44`) is what makes
  vLLM glob and load `quant_model_description.json` (`vllm/model_executor/model_loader/weight_utils.py`
  `get_quant_config`, ~`:300-315` at vLLM `v0.13.0`), after which the static-W8A8 method registers
  `deq_scale` and the checkpoint loads.
- **Do not read `online_quantization.py:173 patched_model_load_weights` as evidence of an online
  quantiser.** At vLLM `v0.13.0` that function is the decorator `support_quantized_model_reload_from_hp_weights`
  applied to `AutoWeightsLoader.load_weights` (`vllm/model_executor/models/utils.py:321`; body at
  `…/model_loader/online_quantization.py:147-173`), so it sits in the stack of **every** load and passes
  straight through unless `model.weight_metadata_and_attr_saved` says a reload is in progress.
- **A misspelled flag cannot produce this traceback.** At vLLM `v0.13.0` an unresolvable value raises
  `ValueError: Unknown quantization method: …` in `ModelConfig._verify_quantization`
  (`vllm/config/model.py:943-950`), and `--quantization ascend` with no `quant_model_description.json`
  raises `ValueError: Cannot find the config file for ascend` (same `get_quant_config`); both fire **before**
  any weight is read. So the run behind this traceback had no effective quant method at all — the flag
  absent (or dropped by the shell), not the reporter's `--quantization ascenId` typo. A description that
  *lacks* a key raises `KeyError: '<prefix>.weight'` at model construction (§2), not on `params_dict` at
  load.
- Where the flag goes missing silently: the report's own `docker-compose.yaml` passes the command as
  `command: >`, a YAML **block scalar**. Inside a block scalar `#` is content, not a YAML comment, and `>`
  folds the lines into one long line — so the first `#` starts a *shell* comment and everything after it
  (`--device ascend`, `--quantization ascenId`, `--trust-remote-code`, …) is dropped without an error. Read
  the running process before reading the loader: `ps -ef | grep vllm` or
  `docker inspect <container> --format '{{.Config.Cmd}}'`.

Workaround, cheapest first: put `--quantization ascend` on the effective command line, or add
`"quantization_config": {"quant_method": "ascend"}` to the checkpoint's `config.json` so selection no longer
needs the flag. Cost: none; the quantised method is what the checkpoint was exported for. Confirm with the
absence of the `deq_scale` `KeyError`, or with `Using the vLLM Ascend modelslim Quantization now!`
(`logger.info_once` in `get_quant_method_modelslim`) in the load log — not with throughput.

Status and limits: two reporters, **no maintainer answer**; bf16 weights start on the same deployment, which
is the control that rules out device/CANN. That the *reporter's* final command really omitted the flag cannot
be proven from the pasted compose file (their comment elides it as `...`), so the mechanism class is
`verified` and that report's instance is `hypothesis`; the falsifier is the missing
`Using the vLLM Ascend modelslim Quantization now!` line (the Ascend quant config was never applied) or a
different first fetch failure in their build log.

Check before reuse: fetch the artefact's own two files rather than asking the reporter —
`curl -sS 'https://modelscope.cn/api/v1/models/<owner>/<model>/repo/files?Revision=master&Recursive=true'`
lists the directory, and
`curl -sSL 'https://modelscope.cn/models/<owner>/<model>/resolve/master/quant_model_description.json'`
returns the description (Hugging Face: `…/resolve/main/<file>`); then compare `model_quant_type`/
`is_dynamic` against the scale names in the shard (`deq_scale` ⇒ static W8A8). Downloading a model card or a
JSON is prompt-free evidence; only the model weights are out of scope.

## 4. Weight-file integrity: the deepest frame's callee is a third-party file reader

Worked case: vllm-ascend #15836 — "Feedback for 使用DeepSeek-V4-Flash-w8a8-mtp部署报错",
`quay.io/ascend/vllm-ascend:v0.23.0`, 1×Atlas 800 A2 (8×64 GB, 910B3), weights from the ModelScope
cache, `--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'`,
`--quantization ascend`, MTP + `FULL_DECODE_ONLY` cudagraph.

### Signature in the log

- `safetensors._safetensors_rust.SafetensorError: Error while deserializing header: incomplete
  metadata, file not fully covered`, raised under
  `vllm/model_executor/model_loader/weight_utils.py:1038 in _load_file` →
  `:1054 in multi_thread_safetensors_weights_iterator` →
  `vllm_ascend/models/deepseek_v4.py:1351 in load_weights`.
- Only some ranks die (`Worker_TP6_EP6`, while TP1/2/4/5/7 log `Using OOT custom backend for
  compilation`): each rank opens the shards independently, so the *reporting* rank is arbitrary.
- No `vllm_ascend/` frame owns the exception: the last project frame is a `for name, loaded_weight in
  weights:` generator loop, and the raise comes from `safetensors.safe_open`'s header parser.

### Verified at the reporter's pair

`.github/vllm-release-tag.commit` at `v0.23.0rc1`/`v0.23.0` names vLLM `v0.23.0`, where
`vllm/model_executor/model_loader/weight_utils.py` line 1038 is exactly
`result = load_file(st_file, device="cpu")` and line 1054 is `state_dict = future.result()`. So the
traceback is consistent with the image and there is no version ambiguity to resolve.
`enable_multithread_load` is read by truthiness (`default_loader.py:245`/`:267`,
`extra_config.get("enable_multithread_load")`): the v0.23.0 tutorial's string `"true"` and the
reporter's boolean `true` both enable it — and a string `"false"` would also enable it (a
generic `--model-loader-extra-config` trap, see `../vllm-ascend/config-flag-lifetimes.md` Trap D).

### Meaning (library behaviour, `reported` outside this repo)

The header's tensor offsets do not cover the file's payload: the shard is shorter than the header
describes, i.e. a truncated or partially written download (`modelscope download`/`huggingface-cli`
interrupted, disk full, `cp` of a growing file, or a `.tmp`/`*._____` partial still in the cache).
The same message is reported for truncated LoRA and model files in unrelated projects, so it is not
specific to this stack.

### Discriminator, then workaround

1. Read the deepest frame's **callee**, not the frame's module: `load_file`/`safe_open` (safetensors),
   `torch.load`, `json.load` on `config.json` — the defect is the file, not vllm-ascend.
2. Hand the reporter two commands and no theory: `ls -l <cache>/<shard>.safetensors` against the
   repository's file listing/size, and `sha256sum` against the ModelScope/HF file hash; also look for
   partial files next to it.
3. Workaround: delete/re-download the failing shard (`modelscope download --force`, or re-pull the
   single file), then start again. **Do not offer `enable_multithread_load: false` as the workaround**
   — it only changes which rank reports the error first.

### Status and limits

Not a vllm-ascend code defect; as an issue it is "your weights are incomplete". The actionable part of
#15836 is documentation: a maintainer (COLLABORATOR, 2026-09-07) said the
`docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html` page is the **main
branch** doc, "not recommended to use it on the 0.23.0 image", and that after switching to the 0.23.0
doc "new errors have occurred, and this needs to be reviewed" — so the doc/version mismatch is open,
the safetensors error is the reporter's environment. Do not claim the tutorial ships broken flags
without reading the doc at the tag (`git show <tag>:docs/source/tutorials/models/<Model>.md`).

### Check before reuse

`git grep -n 'multi_thread_safetensors_weights_iterator\|def _load_file' <ref> -- vllm/model_executor
/model_loader/weight_utils.py` in the vLLM clone to confirm the frames, and
`git show <ref>:.github/vllm-release-tag.commit` for the pair. If the deepest callee is a project
function (e.g. a checkpoint-format converter in `vllm_ascend/quantization/`), this is §2/§3, not §4.

## 5. int4 + FRACTAL_NZ with `group_size 256`: the operator accepts only 64/128

Worked case: vllm-ascend #5603 — image `vllm-ascend:0.13.0`, single NPU, Qwen3-8B **W4A8**
quantised by the vendor guide with `--group_size 256`, torch_npu `2.8.0`, CANN `8.3.RC2`, aarch64,
openEuler 24.03; thread answered by `kunpengW-code` (2026-06-05) and still open.

### Signature in the log

```text
Loading model weights took 8.2463 GB
... EngineCore failed to start.  ->  _initialize_kv_caches  ->  determine_available_memory()
RuntimeError: npu_weight_quant_batchmatmul:... NPU function error:
  call aclnnWeightQuantBatchMatmulV2 failed, error code is 161002
AclNN_Parameter_Error(EZ1001): when weight's dtype is [int4], weight's format is [FRACTAL_NZ],
  antiquantGroupSize should be 128 or 64, but actual antiquantGroupSize is [256].
```

Two markers make the class unmistakable: `EZ1001`/`AclNN_Parameter_Error` (an argument validation, not a
device fault), and the stage — `determine_available_memory()` means the **profile run**, so the server
dies before the KV cache exists and nothing about `--max-model-len` or `--gpu-memory-utilization`
matters.

### Mechanism at `v0.13.0` (verified)

- `utils.py:133` `maybe_trans_nz(weight)` decides **only** by dtype: `torch.float` stays ND (`:134-137`),
  bf16/fp16 convert only when `VLLM_ASCEND_ENABLE_NZ == 2` (`:139-145`), and **everything else** —
  including the packed int4 weight — is `torch_npu.npu_format_cast(weight,
  ACL_FORMAT_FRACTAL_NZ)` (`:148`). `group_size` is never consulted.
- W4A8 calls it in `process_weights_after_loading` (`quantization/w4a8_dynamic.py:172` linear,
  `:479-480` MoE w13/w2) and then passes `antiquant_group_size=self.group_size` to
  `npu_weight_quant_batchmatmul` (`:167`), where `group_size` is
  `vllm_config.quant_config.quant_description.get("group_size", 256)` (`:38-39`). So the checkpoint's
  own description sets 256, the layout conversion ignores it, and the operator — which supports NZ for
  int4 only at 64/128 — rejects the pair at the first matmul.
- The same shape survives the `main` refactor: `utils.py:392` `maybe_trans_nz` now delegates to
  `_should_trans_nz`, which still decides by dtype / `weight_nz_mode` / hardware policy and not by
  `group_size`, so this remains a "your quantisation settings and this layout disagree" class rather
  than an old-tag bug. Re-read `_should_trans_nz` at the tag before claiming it was fixed.

### Workaround (maintainer-attributed, both walked through by the reporter's thread)

1. `export VLLM_ASCEND_ENABLE_NZ=0` — keeps the weights in ND, so the same 256-group checkpoint loads;
   costs the NZ-layout matmul (some throughput), no requantisation.
2. Re-quantise with `--group_size 128` (or 64) and keep NZ on — costs a full quantisation run, best
   accuracy/performance.
   The ModelScope model `vllm-ascend/Qwen3-8B-W4A8` is **third-party** (`huangdong2022`: "not published
   by vllm-ascend organization, we have no right to delete") and still ships 256, so prefer the
   reporter's own requantisation over that mirror.

### Status and limits

An operator support boundary plus a doc gap (the guide offers `group_size 256`, the NZ path needs 128),
not a defect introduced by the release. The honest follow-up issue is "`maybe_trans_nz` should skip NZ
when `group_size not in (64, 128)` at the W4A8 call sites, and the quantisation guide should say so".
Not verified: whether a later release auto-degrades to ND; no NPU here.

## 6. A projection of a "W8A8" model runs `aten::matmul` — the description says `"FLOAT"`

Worked case: #979 (`vLLM 0.8.5.post1` / vllm-ascend `0.8.5rc1`). A profiler trace of
`vllm-ascend/Qwen2.5-0.5B-Instruct-w8a8` and `vllm-ascend/Qwen3-8B-w8a8` shows the MLP's
`down_proj` (`RowParallelLinear`) executing `aten::matmul` instead of a quantised matmul. Output is
correct and nothing errors — this is a **description/artefact** question, not a missing code path.

### Verified at the reported ref (`v0.8.5rc1` = `ec27af346`)

- `vllm_ascend/quantization/quant_config.py:83` `AscendQuantConfig.get_quant_method()` handles every
  `LinearBase` (which includes `RowParallelLinear`) by first asking `is_layer_skipped_ascend(prefix,
  packed_modules_mapping)` (`:96-131`), whose only test is
  `self.quant_description[prefix + '.weight'] == "FLOAT"` (`:119` non-fused, `:129` fused) and which
  returns `UnquantizedLinearMethod()` (`:88-90`).
- The quantised path is `AscendLinearMethod`, whose `apply` special-cases `RowParallelLinear` to pass
  the tensor-parallel rank into the quant method (`:202-205`). It is simply never constructed for a
  module the description marks `FLOAT`, so the float layer's forward is plain `aten::matmul`.
- The docs state the same contract at `main`: `docs/source/user_guide/feature_guide/quantization.md:119`
  — "Layers the checkpoint left unquantized … are listed in `ignored_layers` or
  `modules_to_not_convert` and are served unquantized."

### The statement moves between releases (grep the statement, not the path)

| ref | location of the `== "FLOAT"` skip |
| --- | --- |
| `v0.8.5rc1` | `vllm_ascend/quantization/quant_config.py:119`/`:129` |
| `v0.14.0rc1` | `vllm_ascend/quantization/modelslim_config.py:455`/`:465` |
| `v0.20.2rc1` | `vllm_ascend/quantization/modelslim_config.py:587`/`:599` |
| `main` | `vllm_ascend/quantization/configs/modelslim_config.py` — same semantics, expressed as `return quant_type if quant_type != "FLOAT" else None`, with the docstring `If the quantization type is "FLOAT", it will return None to indicate no quantization` |

### Answer shape

1. Ask for the checkpoint's `quant_model_description.json` and read the module:
   `jq '.["model.layers.0.mlp.down_proj.weight"]' …` (`"FLOAT"` vs `"W8A8"`/`"INT8"`). Weight-file
   inspection by a maintainer (`kunpengW-code`, 2026-06-04) found `"FLOAT"` for `down_proj` in both
   models of #979 — `reported`, and the cheapest check the reporter can repeat.
2. If it is `"FLOAT"`: nothing to fix. No flag makes that layer use `quant_matmul` — the scales and the
   int8 weight are not in the checkpoint, and hand-editing the description to `"W8A8"` would fail on the
   missing `weight_scale` (the §3 class). If the user needs the kernel, re-quantise with ModelSlim
   including that projection (cost: accuracy — a small model's `down_proj` is often left in high
   precision exactly for that reason) or use an artefact that quantises it.
3. If it is **not** `"FLOAT"`: the skip is not the answer; check the key actually looked up
   (`is_layer_skipped_ascend`'s prefix and the fused `packed_modules_mapping` rewrite), and that the
   loaded method is the ascend linear method rather than an upstream one.

### Status and limits

The thread carries two incompatible statements: `leo-pony` (2025-11-11) "Bug has been fixed, now you can
try with v0.9.1 or v0.11.0rc1" and the later weight-file finding above. The code read supports the
second; the first is not evidence of a fix to this layer, and the reporter never confirmed either.
Whether the two published artefacts are still `FLOAT` today must be re-checked against the files, not
this page.

## 7. An accuracy report on a weight variant the release's own tutorial never lists (#7507)

Before reading a quantization kernel for an accuracy report, read the model page's **Model Weight list at
the reported ref**: it is that release's own statement of which quantised artefacts were validated, and it
costs one `git show`.

Worked case: 4-machine PD disaggregation on the `v0.17.0rc1` line, `1×2P + 1×2D`, a **community w8a8
conversion** of GLM-5 (`modelers.cn/models/Eco-Tech/GLM-5-w8a8`), reported as an instruction-following
failure — the prompt demands exactly 20 result items and the model emits far more
(「明确指出严格输出20条结果，实际上推理出了远远超过20条的结果」). Two controls in the report narrow it: the same
test script on an H20 GPU returns the expected 20, and 「把权重换成w4a8正常」 — the same Ascend deployment
with `w4a8` weights answers normally. So the prompt, the script and PD itself are not the trigger; the
quantised artefact is.

- **First read is the model's tutorial at their tag, one command**:
  `git show v0.17.0rc1:docs/source/tutorials/models/GLM5.md | sed -n '1,40p'`. Verified at that tag: the
  *Model Weight* block (`:17-22`) lists `GLM-5` (BF16, `ZhipuAI/GLM-5`), `GLM-5-w4a8`, `GLM-5-w4a8` with
  MTP — **no w8a8 artefact** — plus `msmodelslim` to re-quantise; `:28` says "vLLM and vLLM-ascend only
  support GLM-5 on our main branches. you can use our glm5 docker images for inference", and the deployment
  block pins the `glm5-a3` image (`:45`). The `GLM-5-w8a8` / `GLM-5.1-w8a8` entries first appear at
  `v0.18.0rc1`: `git show <tag>:docs/source/tutorials/models/GLM5.md | grep -c -i w8a8` gives 0 at
  `v0.16.0rc1` and `v0.17.0rc1`, 12–13 from `v0.18.0rc1` on. The page is not backported to every release,
  so confirm it exists at the reporter's ref
  (`git ls-tree -r --name-only <ref> docs/source/tutorials/models | grep -i <model>`).
- **What that licenses**: the answer is "this checkpoint's quantisation scheme is not one the release
  documents or validates", with the workaround being an artefact the page lists (w4a8, or bf16) — relayed
  as reporter-confirmed, or `>= v0.18.0rc1` where the variant is listed. It does *not* license a
  `path:line` in `quantization/methods/` for the accuracy loss: no clone shows what the community export
  wrote.
- **Do not spend the batch inside the quantiser.** At that ref `quantization/methods/` has
  `w8a8_dynamic.py`, `w8a8_static.py`, `w8a8_mxfp8.py` and `w8a8_pdmix.py` (scheme `W8A8_MIX`, "static
  W8A8 for KV consumer/decode, dynamic for prefill"), and which one runs is decided by the checkpoint's
  own quant description through `modelslim_config.py` / `quant_parser.py`. Reading those files cannot
  tell you what the mirror's `quant_model_description.json` says — ask the reporter for it (the §6 check)
  instead.
- **Leads if the mechanism must be named** — all after that tag, so none is in the reporter's build:
  the range's fusion-rounding fixes (`e205cc961` `[BugFix][Perf] Fix MoE MulsAdd fusion matching and
  preserve intermediate rounding (#19778)`, `0bbf9248a` "match bias-less norm+quant (#19756)") are
  accuracy-shaped but read as **perf** fixes with `fuse_muls_add` semantics; the GLM-5 + PD fix in the
  same range is `2f285ddcc` "Preserve GLM shared slots with layerwise KV transfer (#19880)". Nothing here
  was read as a hunk in the recorded run, and reading `main`'s quantization code for a variant the
  reported ref never validated is the month-ahead-clone trap (`../workflow/version-pinning.md`). State the
  mechanism as `hypothesis` with the falsifier "the same w8a8 weight on a later release, or on a single
  node without PD, reproduces the degradation" — which also separates the export from the deployment.
- **Falsifier of the whole section's route**: an issue where the tutorial *does* list the same
  quantisation, or a maintainer comment naming a code defect for that scheme, moves the answer into §1–§6
  instead. Absence from the ref's weight list is a *support statement*, not a missing feature: quote the
  list and the "main branches / dedicated image" sentence, name the release that first ships the variant,
  and hand back the A/B that isolates the quantised artefact (w8a8 vs w4a8 vs bf16, same prompt and
  sampling).

## 8. A pre-sharded checkpoint loaded with `load_format=auto` → `AssertionError` in the vocab loader (#11406)

Signature: 310P (Atlas 300I Duo), a modelslim `w8a8sc` artefact served out of a **`TP<n>/…-tp<n>`
directory**, `--quantization ascend`, and the engine dies during weight load, before any graph, profile or
request: `get_model` → `base_loader.py:64 load_model` → `vllm/model_executor/layers/
vocab_parallel_embedding.py:463 in weight_loader` → `assert loaded_weight.shape[output_dim] ==
self.org_vocab_size` → `AssertionError`, then `WorkerProc failed to start` on every TP rank. Two banner
lines are the evidence, not decoration: the engine-config dump says `load_format=auto`, and the loader prints
`Loading safetensors checkpoint shards: 0/N`.

Verified at vLLM `v0.21.0` (`vocab_parallel_embedding.py:463`): `weight_loader` asserts that the incoming
tensor's vocab dimension equals the **full** `org_vocab_size`; only a parameter whose `packed_dim ==
output_dim` takes the other branch (`:457`). A checkpoint written per rank
(`model-rank-<r>-part-*.safetensors`, the `ShardedStateLoader` layout) stores `embed_tokens.weight` already
narrowed to its own slice of the vocabulary, so under the default loader the assert fires on the first file
it reads. `--tensor-parallel-size` does not enter the comparison: the two numbers are "rows in this file"
versus "vocabulary of the whole model".

The lever is in the release's own 310P block: at `v0.21.0rc1`
`docs/source/tutorials/hardwares/310p.md:127`, `:148` and `:171` all three Qwen3-W8A8SC serve commands end in
`--load_format sharded_state` (offline examples pass `load_format="sharded_state"`, `:235/:300/:368`).
At vLLM `v0.21.0` `ShardedStateLoader.load_weights` never calls the model's `load_weights`: it globs
`model-rank-<rank>-part-*.safetensors`, narrows the parameter view to the tensor's own shape and
`param_data.copy_(tensor)` (`sharded_state_loader.py:154`), so the `org_vocab_size` assert is never reached.
vllm-ascend's `_310p/sharded_state_loader_310p.py` only overrides `save_model` and the
`parameters_type_map.json` writer, so the load path is upstream's.

- **Workaround (`proposed`; the reporter never retested): add `--load_format sharded_state`** — the
  documented flag the reporter's container argv omits (gpustack's argument list *and* the `non-default args`
  dump lack it, while `load_format=auto` is what ran). Cost: it requires the per-rank files, so it answers
  for these Eco-Tech directories, not for an ordinary HF checkpoint — the other direction fails loudly
  instead (`Could not find checkpoint files …, only pre-sharded checkpoints are currently supported!`,
  `sharded_state_loader.py:134`).
- **Do not conclude "a TP4 directory must not be served with TP4".** The same page pins
  `TP4/Qwen3-32B-w8a8sc-310-vllm-tp4` with `--tensor-parallel-size 4` (32B) and `TP1/…-tp1` with
  `--tensor-parallel-size 1` (8B/14B), so a directory name matching the TP size is the *intended* layout, and
  the 8B/14B blocks carry the same `--load_format`. A "pre-split checkpoint, drop TP" story is wrong and it
  reads plausible enough to consume the whole reply.
- **The diff is mechanical, and it comes first.** Line the reporter's argv up against the serve block at
  their ref (`git show <tag>:docs/source/tutorials/hardwares/310p.md`), flag by flag — in this report the
  difference is exactly one flag, and finding it costs one `git show` rather than a reading of the loader,
  the checkpoint layout and the `cudagraph_mode` in the same argv.
- **Status and limits:** #11406 is open (6 comments) and the maintainer's only reply asks for the start
  script (「启动脚本发一下，感觉没有完全按照脚本部署，因为我们内部有持续的每日冒烟测试」) — i.e. the argv diff *is*
  the maintainer's next step. That `sharded_state` suffices is `hypothesis`; falsifier: the same assert with
  `--load_format sharded_state` in the banner, which would move the owner into the `w8a8sc` quantiser or the
  `parameters_type_map.json` it reads.

## 9. The scheme no longer exists: a declared sunset answered as a regression (#15247)

Signature: a body that compares two refs by **line count and class name** — "[Bug] main (0.26 and later) no
longer supports **dense** W4A8_DYNAMIC for qwen3.6-27B, but v0.23.0 does", with the file's lengths
(`w4a8.py` 778 vs 421 lines) and no traceback at all. Both halves of that premise need checking: the module
**moved** and the class **was** removed, and the removal is a deprecation announced one release earlier. The
end state is a registry miss at load time, not a numerics defect, and the thread closed `wontfix`. Do not go
looking for a load-path defect, and do not take the body's numbers as evidence (778 lines at `v0.23.0` is
right; the `main` figure counts a path that no longer exists).

Verified at the clone (`git grep -n 'register_scheme("W4A8_DYNAMIC"' <tag> -- vllm_ascend/quantization`):

| tag | `W4A8_DYNAMIC` registrations |
| --- | --- |
| `v0.23.0`, `v0.24.0rc1` | `linear` (`methods/w4a8.py:38`) + `moe` (`:231`) |
| `v0.26.0rc1`, `v0.26.0rc2` | `linear` (`methods/w4a8.py:39`) + `moe` (`:236`), class `AscendW4A8DynamicLinearMethod` |
| `v0.27.1rc1` … `v0.29.0.rc1`, `main` | **`moe` only** (`methods/w4a8/w4a8.py:36`; `:44` from `v0.28.0.rc1`) |

The removal is `efda0446d` "[Refactor][quantization] Remove deprecated W4A8Linear/W8A8PDMixMoE schemes and
sync W4A8 MoE refactor (#13713)", 2026-08-08, first tag **`v0.27.1rc1`** (`merge-base --is-ancestor` yes at
`v0.27.1rc1`, no at `v0.26.0rc2`). The announcement is one release earlier, in the release notes rather than
in code: `v0.26.0rc1:docs/source/user_guide/release_notes.md:75` — "Added deprecation warnings for W4A8
linear, W4A8 MoE per-group, and W8A8 PDMix MoE quantization. These quantization paths will be removed in a
future release. [#13850]".

**The failure the reporter sees (verified at `v0.27.1rc1`).** `get_scheme_class(quant_type, layer_type)`
returns `None` (`quantization/methods/registry.py:52`, a `dict[(quant_type, layer_type)]` lookup), and
`quantization/configs/modelslim_config.py` turns that into

```text
NotImplementedError: Currently, vLLM Ascend doesn't support quant_type=W4A8_DYNAMIC for layer_type=linear.
Please use a supported quantization format or load the model with its original float weights.
```

(`:559-569` at `v0.27.1rc1`, the raise at `:564`; the same literal is at `quantization/modelslim_config.py:500`
at `v0.26.0rc2` and `:464` at `main`), so a dense checkpoint fails on its first linear layer at load time.
Quote that line — the reporter's "找不到对应的 linear scheme" is a paraphrase of it, and the message names
the **pair** `(quant_type, layer_type)`, so it distinguishes a dense run from a MoE one.

Thread (#15247, closed `wontfix`): maintainer `kunpengW-code` names the reasons — the W4A8 **Linear** path
never quantised activations (it effectively ran as W4A16), ModelSlim upstream does not maintain or support
the Linear-W4A8 workflow, and the mode was only ever used for Qwen3-32B — and the replacement: **W4A4**,
via FlatQuant or LAOS (`W4A4_LAOS_DYNAMIC`, #5143; the `w4a4` module at that ref already registers both).

- **Workaround ladder (relay in this order):** pin `<= v0.26.0rc2` for a dense W4A8 checkpoint (the last tag
  with the linear scheme; cost = the whole 0.27+ line), re-quantise to **W4A4** (cost = a re-quantisation
  pass), or carry the deleted class locally (cost = a patch against a refactor that keeps moving). None is
  reporter-confirmed: the thread is a maintainer statement, not a repair.
- **The file move is not the regression.** The module became a package between those tags
  (`quantization/methods/w4a8.py` → `methods/w4a8/{w4a8.py,w4a8_mxfp4.py}`), so a report's line counts
  compare two different paths. The per-tag fact is `git grep register_scheme`, never `wc -l`.
- **What is *not* the mechanism:** a broken checkpoint, `--quantization` not reaching the process (§3), an
  NZ-layout/`group_size` operator refusal (§5), or a missing description file (§2). Those fail earlier,
  inside the quantiser, with an operator or `KeyError` signature.
- **Read the release notes' deprecation sections before calling a removal a bug**:
  `docs/source/user_guide/release_notes.md` carries `Deprecation and Configuration Changes`,
  `Ready to Deprecate` and `Deprecation & Breaking Changes`; grep it, never print it
  (`../vllm-ascend/repo-map.md`).
- **The general rule.** A `[Bug]` whose whole content is "X works in tag A, not in main/0.26+" is answered
  by `git grep -n '<symbol>' <each tag> -- vllm_ascend` across the range plus the release notes'
  `Deprecation`/`Other` bullets, before any code reading: a per-`(method, layer_type)` registry answers a
  removed scheme with `None`, so the symptom is always "unsupported quant_type … for layer_type …", never a
  numerical error. Say which tag last carried it, which commit removed it, and what replaces it.
- **Status line**: "declared deprecation, closed `wontfix`" is not "fixed-in". Name the tag that removed it,
  the tag that announced it and the replacement — never a regression, a bug, or something a future release
  will restore.
