# Quantization start-up failures: KeyErrors, device-op parameter errors, retired schemes

Quantization start-up failures that arrive as "this checkpoint no longer loads". The raising
statement decides the class:

- two distinct statements produce a `KeyError` while loading weights — the quant-config
  layer-skip lookup against `quant_model_description.json` and a model's own
  `params_dict[name]` lookup after checkpoint name rewriting;
- argparse rejecting `--quantization/-q ascend` before any checkpoint is opened (the parser-hook
  gap, recorded after the W8A8 part);
- a device-operator `Parameter_Error` from the W4A8 int4/NZ layout and antiquant group size,
  which kills the engine core with no Python `KeyError` at all;
- a device-operator dtype-contract rejection of the plugin's *deliberate* two-step quantized MLP
  (`aclnnQuantMatmulWeightNz ... When scale dtype is bfloat16, out dtype should also be bfloat16
  but it is DT_INT32`), which points at the installed CANN/torch-npu rather than a plugin dtype
  bug;
- a deliberately retired scheme, which raises `NotImplementedError` naming the
  `quant_type`/`layer_type` pair.

The mirror of the first class also arrives here because it reads like one: a `"FLOAT"` entry in
`quant_model_description.json` legitimately leaves that layer unquantized, and the report says
"the plugin does not quantize layer X" instead of showing a `KeyError`.

Read the raising line first; the mechanisms and per-tag boundaries are below.

## W8A8 quant-config lookups: `KeyError: '<prefix>.weight'`

### Symptom

`vllm serve --quantization ascend` (ModelSlim / W8A8 checkpoint) aborts while loading
weights. The traceback runs through the model, `get_quant_method`, then the quant config's
layer-skip lookup and ends with `KeyError: '<module prefix>.weight'`, e.g.
`model.embed_tokens.weight` or `model.layers.0.self_attn.q_a_proj.weight`.

Three causes produce this KeyError; decide which one before proposing a checkpoint edit:
(A) a ModelSlim `quant_model_description.json` that omits a module, (B) the Ascend config being
selected for a checkpoint that is *not* ModelSlim (GPTQ/AWQ/...), so `quant_description` is that
checkpoint's HF `quantization_config` and contains no per-layer keys at all, and (C) the description
and the runtime module prefix being in different namespaces. See "Why `ascend` is selected for a
non-ModelSlim checkpoint", "Cause C" and the Qwen3-VL `lm_head` instance below.

### Mechanism (code-verified)

The skip lookup reads the checkpoint's `quant_model_description.json`
(`AscendQuantConfig.get_config_filenames()` at v0.11.0; `MODELSLIM_CONFIG_FILENAME` at
v0.20.2rc1) and indexes it directly with `prefix + '.weight'`. A module the JSON does not list raises `KeyError`
instead of being treated as unquantized (`"FLOAT"`). For cause A the failure is therefore about
the description file's key set, not about the model architecture; for cause B no such file is read
at all.

Verified locations:

- `v0.7.3rc2:vllm_ascend/quantization/quant_config.py` — the description file is not read **at
  all** at this rc: `get_config_filenames()` returns `[]` (`:75-76`), so `quant_description` is only
  the HF `quantization_config` dict and *every* `--quantization ascend` load dies at the first
  linear layer (`get_quant_method` `:93`, direct index `:135`). From `v0.7.3` (final) the same class
  returns `["quant_model_description.json"]` (`:72-73`) and the statement pair moves to `:90`/`:132`
  (`v0.8.5rc1` `:69-70`, index `:129`). Use that to date a pasted traceback: a `:90`/`:132` pair is
  the post-rc2 layout, a `:93`/`:135` pair is the rc2 layout — decide with
  `git show <tag>:vllm_ascend/quantization/quant_config.py | grep -n -A2 'def get_config_filenames'`,
  not from the version string alone.
- `v0.9.0rc1:vllm_ascend/quantization/quant_config.py:134` —
  `is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"` in the else branch
  of `is_layer_skipped_ascend` (defined at line 109). Same direct index for fused shards
  at line 123.
- `v0.11.0:vllm_ascend/quantization/quant_config.py:170` — same statement, same function.
  Issue #3494's log reports this statement at line **147**, i.e. a close but older
  revision: use the statement, not the tag's line number.
- `v0.20.2rc1:vllm_ascend/quantization/modelslim_config.py` — layout moved; the same
  direct index survives in `get_linear_quant_type` (`quant_description[prefix + ".weight"]`,
  around line 330) and for packed shards (line ~317). Attention key lookups there already
  use `.get`.
- `main:vllm_ascend/quantization/configs/modelslim_config.py` — tolerant lookup:
  `_find_quant_description_key` (line ~320) returns `None` when neither `<prefix>.weight`
  nor `<prefix>` exists, and `get_quant_type_for_layer` then returns `None` (unquantized).
  The fused/packed path still raises `KeyError` deliberately when a description mixes
  expanded and fused shard entries. The `.get`-based lookup entered with commit
  `33b77e4f7` (`git log -S'quant_type = quant_description.get(quant_key)'`) — a locator,
  not proof that this change was the fix for these reports. At `main` the old flat
  `quantization/quant_config.py` path is gone (split into `quantization/configs/` and
  `quantization/methods/`), so `git show main:vllm_ascend/quantization/quant_config.py` printing
  "does not exist" means a moved file, not a removed check.

### Cause C instance: `KeyError: 'lm_head.weight'` on a dense Qwen3-VL W8A8 load — the plugin remapped the description, not the prefix

This is cause C (description namespace != runtime module prefix) with a known fix tag, not a
missing description entry. Recorded in vllm-project/vllm-ascend#8367 (open; its only comment
pings a maintainer, no mechanism): `v0.18.0rc1`, Qwen3-VL-32B-W8A8, start-up dies with
`KeyError: 'lm_head.weight'` from
`v0.18.0rc1:vllm_ascend/quantization/modelslim_config.py:292`
(`quant_type = quant_description[prefix + ".weight"]` in `get_linear_quant_type`).

- vLLM hardcodes the module prefix and renames the checkpoint tensor:
  `v0.18.0:vllm/model_executor/models/qwen3_vl.py:1331` `ParallelLMHead(..., prefix="lm_head")`,
  and `:1374-1378` `hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={... "lm_head.":
  "language_model.lm_head." ...})`. The same pair exists at `v0.17.0:1317`/`:1359-1362` and
  `v0.19.1:1408`/`:1452-1455`, and still `main:1919`/`:1972` — the prefix/tensor split is
  unchanged upstream through `main`, so the plugin is where the guard lives.
- The plugin applies that mapper to the *description* and not to the layer prefix:
  `v0.18.0rc1:modelslim_config.py:428` `self.quant_description =
  hf_to_vllm_mapper.apply_dict(self.quant_description)` turns the description's `lm_head.weight`
  into `language_model.lm_head.weight`, while `get_quant_method` keeps receiving `lm_head`
  (`quant_prefix_mapper` at that tag returns the prefix unchanged, `:432-434`). The checkpoint is
  complete — this is a namespace split, so adding a `lm_head.weight: "FLOAT"` entry hides it
  rather than fixing it.
- Fixed in the plugin: `8e5b20b85` (#7869, 2026-04-02) adds
  `if model_type in ["qwen3_vl"] and prefix == "lm_head": prefix = "language_model.lm_head"`
  (`v0.19.1rc1:modelslim_config.py:487-489`, still at `v0.20.2rc1:525-527`; commit text "TODO:
  remove it when vllm fixes the WeightsMapper bug of qwen3-vl"). `git tag --contains 8e5b20b85`
  starts at `v0.19.1rc1`, so `v0.18.0rc1`/`v0.18.0` are affected and any >=0.19.1 image is the
  upgrade path — not a checkpoint edit.
- `main` generalised the shim off the model name: in
  `vllm_ascend/quantization/configs/modelslim_config.py` `quant_prefix_mapper` remaps `lm_head`
  only when `"lm_head.weight"` is absent *and* `"language_model.lm_head.weight"` is present
  (`:681-687`), and the GLM5-next branch also tries the reverse `language_model.lm_head` ->
  `lm_head` candidate (`:705-706`).
- Reporter-stated workaround on `v0.17.0rc1` (not analysed in the thread): rename the
  `QUANT_MODEL_PREFIX_MAPPINGS` key `"qwen3_vl_text"` to `"qwen3_vl"` in `modelslim_config.py`.
  Code that supports it: `v0.17.0rc1:modelslim_config.py:50-62` holds only `"qwen3_vl_moe"` and
  `"qwen3_vl_text"`, and `quant_prefix_mapper` (`:478-484`) maps the vLLM-side prefix onto the
  checkpoint-side name (`visual.` -> `model.visual.`, `language_model.lm_head.` -> `lm_head.`,
  `language_model.model.` -> `model.language_model.`), so a dense `qwen3_vl` model gets no mapping
  at all. That table is gone at `v0.18.0rc1` (the rewrite moved into `apply_dict` above), so the
  rename cannot be carried forward as a 0.18 workaround.

### Why `ascend` is selected for a non-ModelSlim checkpoint (cause B)

The plugin forces its own quant config on NPU in old releases, and upstream vLLM then hands it the
checkpoint's HF quantization dict:

- `v0.9.0rc2` and `v0.11.0` `AscendQuantConfig.override_quantization_method` returns the ascend
  method name (`ASCEND_QUATIZATION_METHOD` at v0.9.0rc2, `ASCEND_QUANTIZATION_METHOD` at v0.11.0)
  whenever `torch.npu.is_available()`, ignoring the declared
  `quant_method` and the user's `--quantization` value. Upstream vLLM v0.9.0
  (`vllm/config.py:_verify_quantization`) iterates every supported method's override and takes the
  first non-`None`, so `ascend` wins even though `config.json` says `gptq`.
- `vllm/model_executor/model_loader/weight_utils.py:get_quant_config` at vLLM v0.9.0 returns
  `quant_cls.from_config(hf_quant_config)` as soon as the HF config has `quantization_config`
  (or `compression_config`); it never reaches `get_config_filenames()`. `AscendQuantConfig.__init__`
  simply stores that dict as `self.quant_description`, so `is_layer_skipped_ascend` indexes GPTQ
  hyper-parameter keys with `<layer>.weight` and raises `KeyError`.
- From `v0.13.0rc1` the override only fires when `hf_quant_cfg` has **no** `quant_method`
  (`if not quant_method and torch.npu.is_available()`); same body at v0.16.0rc1, v0.18.0rc1 and
  v0.20.2rc1. By v0.16.0rc1 the method lives in
  `vllm_ascend/quantization/modelslim_config.py` (`AscendModelSlimConfig`), so a `git show
  <tag>:vllm_ascend/quantization/quant_config.py | sed -n '/override_quantization_method/,/return/p'`
  that prints nothing means "file moved", not "override removed" — grep the whole `vllm_ascend/` tree.
- At `main` the registered Ascend quant configs are only ModelSlim, CompressedTensors, FP8 and
  ModelOpt-MXFP8 (`vllm_ascend/quantization/__init__.py`); no GPTQ config exists there.

So for cause B the fix is a supported checkpoint format, not extra description keys.

**Cause B on a vision-language model names a vision prefix.** The plugin's VL classes forward the
process-wide `vllm_config.quant_config` into the vision tower, so the forced Ascend config is asked
about `visual.*` modules too: at `v0.11.0rc1`
`AscendQwen2_5_VLForConditionalGeneration.__init__` (`vllm_ascend/models/qwen2_5_vl.py:521-530`)
passes `quant_config` to `AscendQwen2_5_VisionTransformer`, and
`Qwen2_5_VisionAttention`'s `QKVParallelLinear` then reaches
`quant_config.get_quant_method` → `is_layer_skipped_ascend`. Recorded instance #4313
(vllm-ascend 0.11.0rc1, `allenai/olmOCR-2-7B-1025-FP8`, Qwen2.5-VL):
`KeyError: 'visual.blocks.0.attn.qkv.weight'` at `quant_config.py:110` -> `:161`, and both
lines match **`v0.11.0rc1` exactly** (`:93-97` unconditional override, `:110` the skip call, `:161`
the direct index). The whole `vllm_ascend/quantization/` tree at that tag holds only
`quant_config.py`, `utils.py`, `w4a4_flatquant_dynamic.py`, `w4a8_dynamic.py`, `w8a8.py`,
`w8a8_dynamic.py` — no FP8 scheme. FP8 lands with `fcb30bebe` ("DeepSeek-V4 for Ascend950",
#9757), **first tag `v0.21.0rc1`** (`git tag --contains fcb30bebe --sort=creatordate | head -1`). Thread
statements agree: "目前的昇腾卡不支持FP8吧" and, from a maintainer, "The quant method is not support
on ascend now", with the suggested replacement being a msmodelslim/ModelSlim W8A8 checkpoint. So an
FP8 checkpoint's `quantization_config` (`{"quant_method": "fp8", ...}`) is exactly cause B: pick a
ModelSlim W8A8 build or a release that ships the FP8 method; do not look for a missing description
entry.

**Cause C: the description's key namespace is not the runtime module prefix.** A `KeyError` whose
key carries a wrapper segment (`llm.model.embed_tokens.weight`, `language_model.model.layers.0...`)
is not the same finding as an omitted module: the description can list that layer under the
converter's own namespace. The plugin rewrites prefixes per model type before the lookup, and the
table is what to read:

- `v0.11.0rc0`, `v0.11.0` and `v0.13.0` `vllm_ascend/quantization/quant_config.py` strip only a
  leading `language_model` (`v0.11.0rc0:98-99`; `v0.13.0:135` applies
  `quant_prefix_mapper(model_type, prefix)` and then `:137-138` the same strip).
- `main:vllm_ascend/quantization/configs/modelslim_config.py` holds
  `QUANT_MODEL_PREFIX_MAPPINGS` / `QUANT_MODEL_SUBSTR_MAPPINGS`, keyed by model type
  (`deepseek_v4`, `deepseek_v41` only at this clone), mapping runtime prefixes back to the
  checkpoint namespace (`language_model.model.` -> `model.`, `model.embed_tokens.` -> `embed.`, ...),
  applied by `quant_prefix_mapper` (`:677`).
- There is no `llm.`/MiniCPM entry in that table at any tag checked and no MiniCPM model file in
  `vllm_ascend/` on `main`, so a VL wrapper that nests its inner LLM under `llm` gets no rewrite.
  When the traceback key carries a wrapper prefix the plugin does not map, name the mismatch and
  say which of C/A/B the evidence supports; do not turn it into a kernel or attention fix.
- From `v0.18.0rc1` the plugin also rewrites the *description* through the model's vLLM
  `hf_to_vllm_mapper` (`AscendModelSlimConfig.apply_vllm_mapper` -> `apply_dict`), so a model that
  hardcodes a module prefix and maps the checkpoint tensor elsewhere splits the two namespaces in
  the other direction — the Qwen3-VL dense `lm_head` case above has the statements and the fix tag.

### Recorded reports and their boundaries

- **#3494** (`https://github.com/vllm-project/vllm-ascend/issues/3494`, vllm-ascend 0.11.0,
  Qwen2.5-72B W8A8, TP=4): `KeyError: 'model.embed_tokens.weight'`. Attachment log shows
  `VocabParallelEmbedding` → `get_quant_method` → `is_layer_skipped_ascend`. Maintainer
  reply: it is a model-weights problem — deploy weights produced for vllm-ascend
  (`RR2025/Qwen2.5-72B-Instruct-w8a8-vllm-ascend` on ModelScope) and keep the
  `--quantization ascend` flag. Thread closed as stale with no reporter confirmation, so
  the resolution is a maintainer statement plus matching code evidence, not a verified fix.
- **#2960** (`https://github.com/vllm-project/vllm-ascend/issues/2960`, image
  `quay.io/ascend/vllm-ascend:v0.9.0rc2`, vLLM 0.9.0, Qwen2.5-32B-Instruct-GPTQ-Int8, TP=1,
  `quantization=ascend`): `KeyError: 'model.layers.0.self_attn.q_proj.weight'` raised at
  `vllm_ascend/quantization/quant_config.py:92 get_quant_method` ->
  `:123 is_layer_skipped_ascend` (`self.quant_description[shard_prefix + '.weight']`). Both
  statements are at exactly those lines at tag `v0.9.0rc2`, so the pin is proven by statement and
  line. Cause B: a GPTQ checkpoint, no `quant_model_description.json` involved. Maintainer
  statements: 22dimensions 2025-10-15 "vllm-ascend doesn't support gptq quantization alg[o]rithm
  right now", earthmanylf 2026-06-15 "we don't support gptq for qwen, and has no plan for gptq
  quantization for qwen"; the issue is still open, so this is a maintainer limitation statement
  supported by the code above, not a fix. The report is internally inconsistent
  (`served_model_name=Qwen2-72B-Instruct-GPTQ-Int8` for a Qwen2.5-32B path) — note it, but the
  failure is format-level and unaffected.
- **#1127** follow-up (`https://github.com/vllm-project/vllm-ascend/issues/1127`,
  v0.9.0rc1, DeepSeek-R1 W8A8): with `--quantization ascend`, `KeyError:
  'model.layers.0.self_attn.q_a_proj.weight'`. The same reporter saw TP16 load and serve
  while TP8/PP2 failed inside `fused_experts_with_all2all` — a collective/topology
  failure, separate from the KeyError; do not fuse the two conclusions.
- **#2015** (`https://github.com/vllm-project/vllm-ascend/issues/2015`, vllm-ascend `v0.9.2rc1`,
  `DeepSeek-R1-bf16-hfd-w8a8`, TP16 on one 16x910B2C node): the body stops at
  `Engine core initialization failed. See root cause above` / `WorkerProc initialization failed`,
  and the worker paste is in the reporter's own later comment:
  `vllm_ascend/quantization/quant_config.py:137 is_layer_skipped_ascend`,
  `is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"` ->
  `KeyError: 'model.layers.0.self_attn.q_a_proj.weight'`, called from `:92 get_quant_method`.
  Both lines match **`v0.9.2rc1` exactly**. This is cause A, and the reporter's confirmed fix is a
  checkpoint-config merge rather than a plugin change: move every field of that build's
  `quant_model_description_w8a8_dynamic.json` into the `quantization_config` section of
  `config.json` ("This solution resolved the issue completely"). `q_a_proj` is a plain
  `ReplicatedLinear`, so the fused-shard branch cannot cover it; in the reporter's account the
  "hfd" mixed-precision build shipped an incomplete description. Issue closed 2026-05-27 on that
  reporter finding, with no code change in the thread. The shape is worth remembering: a start-up
  that hides the worker traceback behind the engine's fixed message still gets an answer from the
  thread's later comments.
- **#569** (`https://github.com/vllm-project/vllm-ascend/issues/569`, vLLM 0.7.3 + vllm-ascend
  **0.7.3rc2**, Qwen2.5-32B-Instruct W8A8, 910B3, `vllm serve ... -tp 1`): the same
  `KeyError: '...self_attn.q_a_proj.weight'` family on weights converted by **MindIE**
  (`examples/models/qwen/convert_quant_weight.sh -type qwen_w8a8`), not ModelSlim. At 0.7.3rc2 no
  description file is read at all (first verified location above), so this is the pre-release WIP
  state, not a checkpoint key omission — maintainers said so twice (Yikun 2025-04-18 "w8a8 support
  is still working in progress", wangxiyuan 2025-04-24 "w8a8 support is still need more work to
  do") and closed it 2025-05-14 with "0.7.3 and 0.8.5rc1 works with w8a8 now. Please try again".
  A commenter's DeepSeek-R1-W8A8 stack in the same thread (vLLM 0.8.4, source install,
  `get_quant_method` `:90` -> `is_layer_skipped_ascend` `:132`) is the post-rc2 layout and therefore
  a genuine cause A. Do not merge the two diagnoses: same exception text, different eras.
- **v0.7.3.post1** already registers the `ascend` quant config
  (`vllm_ascend/quantization/quant_config.py:44-45`, `@register_quantization_config("ascend")`),
  so `--quantization ascend` exists in that release. `mindie_turbo` is imported separately
  (`quantization/quantizer.py:40`, worker start-up) as an accelerator; on that image a
  stale mindie-turbo surfaced as `TypeError: wrapper_weights_iterator...` during weight
  loading. Reading only the (0-byte) `quantization/__init__.py` hides all of this.

- **#3867** (`https://github.com/vllm-project/vllm-ascend/issues/3867`, titled vllm-ascend
  **0.11.0**, `MiniCPM-V-4_5-int4` on one 910B, `python -m vllm.entrypoints.openai.api_server ...
  --quantization ascend`): `KeyError: 'llm.model.embed_tokens.weight'` raised through
  `vllm_ascend/ops/vocab_parallel_embedding.py:81` ->
  `vllm_ascend/quantization/quant_config.py:120 get_quant_method` ->
  `:152 is_layer_skipped_ascend`. Those two statements sit at exactly `:120` (the
  `VocabParallelEmbedding` branch's skip call) and `:152` at **`v0.11.0rc0`**, while `v0.11.0` moved
  them to `:129`/`:170` — the installed build is the rc0 tree, so pin rc0 rather than the title.
  `llm` is vLLM's own MiniCPM-V wrapper prefix
  (`v0.11.0:vllm/model_executor/models/minicpmv.py:1004-1005`,
  `prefix=maybe_prefix(prefix, "llm")`; the vision tower is `vpm`), so the failing lookup is the
  inner Qwen3 LLM's embedding and cause C applies (no `llm` rewrite at any tag). Causes A and B stay
  open too: the thread never shows `quant_model_description.json` or `config.json`, so whether the
  description keys the embedding differently, omits it, or the checkpoint is a non-ModelSlim int4
  build forced through `--quantization ascend` is undecided. Maintainer reply was only "use the
  latest release or rc version"; closed stale. Ask for those two files, do not edit the plugin.

### The mirror case: a `"FLOAT"` entry is the designed unquantized path, not a missing quantization

A report whose evidence is a profiler dump ("`mlp.down_proj` / `RowParallelLinear` still runs
`aten::matmul` instead of `quant_matmul`") is answered from the checkpoint's description, not from
`linear.py`. `is_layer_skipped_ascend` returns True exactly when the description value is `"FLOAT"`
(`v0.8.5rc1:vllm_ascend/quantization/quant_config.py:129`, fused-shard branch just above), and
`get_quant_method` then returns `UnquantizedLinearMethod()` (`:83` method, `:89` the return), i.e.
plain `aten::matmul` — the observed behaviour is the plugin correctly honouring the checkpoint.

Verified against the two published examples named in the report (fetched 2026-11-01 from
`https://modelscope.cn/models/vllm-ascend/<model>/resolve/master/quant_model_description.json`,
`model_quant_type: "W8A8"`, static per-tensor keys `weight/weight_scale/weight_offset/input_scale/
input_offset/quant_bias/deq_scale`):

- `Qwen2.5-0.5B-Instruct-w8a8` (24 layers) and `Qwen3-8B-w8a8` (36 layers): every
  `model.layers.N.self_attn.{q,k,v,o}_proj.weight` and `mlp.{gate,up}_proj.weight` is `"W8A8"`,
  while **every `model.layers.N.mlp.down_proj.weight` is `"FLOAT"`** (24/24 and 36/36), as are the
  norms, `embed_tokens`, `lm_head` and Qwen3's `q_norm`/`k_norm`.
- So the reporter's profile is faithful: the only MLP `RowParallelLinear` is unquantized *by the
  checkpoint's own recipe*. Whether msmodelslim excludes `down_proj` by design, and whether a
  W8A8-DYNAMIC build would include it, is **not established** — check the values before claiming a
  generator gap.

### Diagnostic checks

0. Read `config.json` -> `quantization_config.quant_method` (or `compression_config`) first. If it
   is not a ModelSlim/ascend format, this is cause B: stop editing the description and pick a
   supported checkpoint instead.
   From `v0.9.1` on, the tag's own `docs/source/user_guide/feature_guide/quantization.md` also
   answers this exact KeyError in FAQ #1 ("How to solve the KeyError
   \"xxx.layers.0.self_attn.q_proj.weight\"?"): make sure `ascend` is the quantization method, and
   check the checkpoint was converted by the ModelSlim branch the doc names
   (`br_release_MindStudio_8.1.RC2_TR5_20260624` at `v0.11.0`); otherwise "Maybe some new models
   need to be adapted". The FAQ is absent at `v0.8.5rc1` and present at `v0.9.1`, `v0.11.0`,
   `v0.13.0rc1` (`git grep -l 'q_proj.weight' <tag> -- docs`), and the named ModelSlim release
   changes between tags — quote it with the tag in hand.
1. From the traceback, collect the module prefixes that were looked up.
2. Compare them with the keys of the model directory's `quant_model_description.json`
   (keys are `<prefix>.weight`; values are quant types or `"FLOAT"`). Fetch it directly when the
   report links a published checkpoint:
   `curl -sL https://modelscope.cn/models/<org>/<model>/resolve/master/quant_model_description.json`.
   For a "layer X is not quantized" report (no `KeyError`), the *value* at that prefix is the whole
   answer: `"FLOAT"` = intentionally unquantized, absent = `KeyError` class above.
3. Classify the miss before editing anything: **(C)** the module exists under another wrapper
   prefix (the key carries `llm.` / `language_model.` / `visual.`), **(A)** the description omits an
   unquantized layer, or **(B)** there is no ModelSlim description at all. Compare the traceback key
   with the description's own keys and with the tag's `quant_prefix_mapper` table; printing the
   first few description keys settles C and A at once.

### Workaround candidates (code-backed, not hardware-verified)

- Use a checkpoint in a format vllm-ascend actually implements — ModelSlim W8A8 produced for
  vllm-ascend, or the compressed-tensors / fp8 paths registered at `main`. GPTQ (and AWQ) have no
  Ascend scheme, so no description-file edit makes such a checkpoint run (cause B, #2960).
- For cause A, use a W8A8 checkpoint whose description covers every module of the model (matches
  the maintainer reply in #3494).
- Add the missing `<prefix>.weight: "FLOAT"` entries to the description so the direct
  index resolves to "FLOAT" and the layer is treated as unquantized.
- Move to a release with the tolerant lookup (main-era code above) so an absent entry
  means unquantized rather than fatal.

None of these was executed on NPU in the review environment; they follow from the read
code and should be labelled as candidates, not validated fixes.

## Earlier failure: `argument --quantization/-q: invalid choice: 'ascend'`

Not a KeyError at all, but it arrives as the same class of report ("quantization is not
supported") and is decided before any checkpoint is opened. Report template
(`https://github.com/vllm-project/vllm-ascend/issues/902`, image
`quay.io/ascend/vllm-ascend:v0.8.5rc1`, QwQ-32B-W8A8, `-tp 4`):

```text
vllm serve: error: argument --quantization/-q: invalid choice: 'ascend' (choose from 'aqlm', ...)
```

- The `--quantization` choice list is upstream's static `QuantizationMethods` `Literal`
  (`vllm/model_executor/layers/quantization/__init__.py:8`, with
  `QUANTIZATION_METHODS = list(get_args(QuantizationMethods))` at `:38` at vLLM `v0.9.0`), not the
  runtime registry that `@register_quantization_config` writes to — hence the plugin must both
  register its config *and* extend the parser's choice list. Upstream gives it the hook for exactly
  this: `vllm/engine/arg_utils.py:1557-1571` at v0.9.0 calls
  `current_platform.pre_register_and_update(parser)` **after** the CLI arguments are added, with the
  comment "the plugin may adding a new kind of quantization method to --quantization argument".
  The hook is old: upstream **vLLM `v0.7.3`** (`ed6e9075d`) already builds
  `--quantization` with `choices=[*QUANTIZATION_METHODS, None]` (`vllm/engine/arg_utils.py:561-565`,
  and `--speculative-model-quantization` at `:763`) and calls
  `current_platform.pre_register_and_update(parser)` at `:1414`, with
  `Platform.supported_quantization: list[str] = []` at `vllm/platforms/interface.py:112`. So for
  every recorded case the missing half is plugin-side; no upstream upgrade is part of the fix.
  At `v0.8.5rc1` the plugin only does the registration:
  `vllm_ascend/platform.py:61` declares `supported_quantization = ["ascend"]` and
  `pre_register_and_update` (`:66-75`) merely imports `AscendQuantConfig`, whose
  `@register_quantization_config("ascend")` is at `vllm_ascend/quantization/quant_config.py:41`.
  Nothing touches `parser._option_string_actions['--quantization'].choices`.
- Fix: commit `00e0243561720ad3ba66ba43e84031dd0e91814b` (#877, "enable online serving
  quantization") adds exactly that append (guarded by `hasattr(quant_action, 'choices')`), plus the
  `ASCEND_QUATIZATION_METHOD` constant. Ancestry in the clone: **absent from `v0.8.5rc1`**, present
  from `v0.9.0rc1` onward (`git describe --contains` on the commit; `v0.9.0rc1` also has the extra
  `and quant_action.choices` guard, and by `v0.10.2rc1` the constant is spelled
  `ASCEND_QUANTIZATION_METHOD`). That matches the maintainer reply on #902: "0.8.5 doesn't work",
  cherry-pick the commit by hand, or use v0.9.1 / v0.11.0rc1.
- Do not read this as a checkpoint-format problem: no `quant_model_description.json` key set is
  consulted, and editing the description cannot make argparse accept the value. Also do not
  conclude "the plugin does not support `--quantization ascend` in 0.8.x" — the config is
  registered and only the CLI surface is missing.
- Triage check: an `invalid choice` line is this class — it is raised while building the argparse
  parser, so it precedes engine/device start-up and any weight load; a
  `KeyError: '<prefix>.weight'` while loading weights is the W8A8 description class above. Which
  release appends the choice is readable from the tag
  (`grep -n 'quant_action' vllm_ascend/platform.py` at the reported ref) without hardware.
- **`v0.7.3` / `v0.7.3.post1` is the same gap, one release line earlier** (recorded #2819, vLLM
  v0.7.3 + vllm-ascend v0.7.3.post1, `--quantization ascend` on a QwQ/Qwen-72B W8A8 path). At
  `v0.7.3.post1` `vllm_ascend/platform.py:63-70` `pre_register_and_update` only imports
  `AscendQuantConfig` (plus the `patch_*` modules), `supported_quantization` is `["ascend"]`, and
  the line never received `00e0243`, so the first release with the appended choice is still
  `v0.9.0rc1`. No release in the 0.7/0.8 lines makes `--quantization ascend` parse — and the tag's
  own docs say what to use instead: `docs/source/faqs.md:80-86` ("w8a8 quantization is already
  supported … originally on v0.8.4rc2 or heigher, If you're using vllm 0.7.3 version, w8a8
  quantization is supporeted with the integration of vllm-ascend and mindie-turbo, please use
  `pip install vllm-ascend[mindie-turbo]`"), `suppoted_features.md:30` ("Quantization 🟡 Planned …
  Will support in v0.8.x") and `tutorials/multi_node.md:67` (remove `quantization_config` from
  `config.json` for DeepSeek V3/R1). The `--quantization` flag itself appears nowhere in that
  tree's docs (`git grep -e '--quantization' -e 'quantization ascend' v0.7.3.post1 -- docs
  README.md` prints nothing), and there is no `multi_npu_quantization.md` before `v0.8.5rc1`. So
  quote the FAQ's own route rather than calling quantization unsupported: reading
  `vllm_ascend/quantization/__init__.py` (0 bytes at `v0.7.3.post1`) or `quantizer.py` shows the
  config exists; the parser (and, on 0.7.3, the mindie-turbo extra) is what gates it.

## Model `params_dict` lookups: `KeyError: '<prefix>.weight_scale'` (DeepSeek-V4)

Recorded in vllm-project/vllm-ascend#12675: v0.23.0rc1, DeepSeek-V4-Flash, all TP ranks
fail during `load_model` with keys such as
`model.layers.0.self_attn.wkv.weight_scale` and
`model.layers.4.self_attn.indexer.wq_b.weight_scale`. The raising frame is the model, not
the quant config: `vllm_ascend/models/deepseek_v4.py` `load_weights`, statement
`param = params_dict[name]` (line 1535 at `v0.23.0rc1`), called from upstream
`vllm/model_executor/model_loader/default_loader.py:394`.

### Mechanism (code-verified at `v0.23.0rc1`)

- Before lookup the model renames checkpoint tensors; lines 1381-1382 rewrite any name
  ending `.scale` to `.weight_scale`. A checkpoint that carries `.scale` tensors therefore
  asks for `weight_scale` parameters.
- Only a quantized linear registers one: `vllm_ascend/quantization/methods/fp8.py:53`
  creates `weight_scale`, and `quantization/method_adapters.py:138` lists `weight_scale`
  among the per-channel quant params. An unquantized `ReplicatedLinear` has `weight` only,
  so the direct index raises.
- Two ways the layer ends up unquantized: the process runs without `--quantization ascend`
  (every linear unquantized), or that field is built with `quant_config=None`.
  `Compressor.wkv` and `Compressor.wgate` do exactly that when
  `get_ascend_device_type() == AscendDeviceType.A5` (lines 626 and 634), and
  `Indexer.weights_proj` always does (line 563). `DeepseekV4Attention.wkv`,
  `Indexer.wq_b` and the compressor's quantized path pass `quant_config`, so their failure
  is a quantized/unquantized mismatch between checkpoint and model, not a missing module.
- Compare with the `.weight` class above: there the description file lacks an entry the
  quant config indexes; here the checkpoint has a scale the model never allocated.

### Recorded report and its boundary

- The #12675 body contained only the traceback — no `collect_env.py`, no serve line, no
  hardware. The maintainer's first reply is "Where are the weights obtained from? Can the
  startup command be provided?".
- The reporter then posted a serve line that mounts and serves the **base**
  `deepseek-ai/DeepSeek-V4-Flash` ModelScope repo with `--dtype bfloat16` and **no
  `--quantization ascend`**, while the tag's tutorial requires the
  `vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp` checkpoint plus `--quantization ascend`
  (see [model-tutorial-recipes.md](model-tutorial-recipes.md)). That delta is verified;
  which of the two unquantized paths produced this traceback is not (hardware and
  checkpoint contents were never supplied). No maintainer root cause exists in the thread.

### Diagnostic order for this class

Read the checkpoint's tensor names for the failing prefix (`.scale` present?), confirm
whether the reported command passes `--quantization ascend`,
then check whether the failing module is one that receives `quant_config=None` on the
reported device type. The candidate repair is to run the documented quantized checkpoint
with its documented flags — a candidate, not a reproduced fix.

### Workaround candidate (code-backed, not hardware-verified)

Run the documented quantized checkpoint with its documented flags — the
`vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp` build plus `--quantization ascend` at v0.23.0rc1
(see [model-tutorial-recipes.md](model-tutorial-recipes.md)) — so every linear the model
indexes has a `weight_scale` allocated. Not reproduced on NPU, and the reporter's command was
never retested.

## The same lookup under a fused shard name: `KeyError: 'layers.N.mlp.gate_up_proj.deq_scale'`

Report (#5813, image `quay.io/ascend/vllm-ascend:v0.13.0rc1`, `vLLM Version: 0.13.0` from
`collect_env.py`, the documented `vllm-ascend/Qwen3-8B-W8A8` checkpoint, single rank
(`--tensor-parallel-size 1`), `python -m vllm.entrypoints.api_server`; the reporter adds that
`0.11.0` fails identically). The innermost frame is the model loader, not the quant config:
`vllm/model_executor/models/qwen2.py:496 param = params_dict[name]` ->
`KeyError: 'layers.0.mlp.gate_up_proj.deq_scale'`, reached through
`qwen3.py:331 -> models/utils.py:335 -> :288 -> :261 -> online_quantization.py:173`.

Two of those frames are pass-throughs at vLLM `v0.13.0` and must not be read as evidence:

- `qwen3.py:331` is `Qwen3ForCausalLM.load_weights` building `AutoWeightsLoader(self)`, whose
  inner `Qwen2Model.load_weights` is the frame that raises.
- `models/utils.py:321` decorates `AutoWeightsLoader.load_weights` with
  `@support_quantized_model_reload_from_hp_weights`; its first branch is
  `online_quantization.py:172-173 return original_load_weights(...)` for the offline / first-run
  case. Every model load at this tag passes that line, so `patched_model_load_weights` in a
  traceback says nothing about torchao online quantization (its `assert ... == "torchao"` sits
  after the branch).

Mechanism, verified at `v0.13.0`:

- `params_dict = dict(self.named_parameters(...))` belongs to the inner model, which is why the
  reported key has no `model.` prefix.
- `stacked_params_mapping` rewrites `gate_proj`/`up_proj` -> `gate_up_proj` **before** the
  unguarded `param = params_dict[name]` (the `else` branch below does have
  `if name not in params_dict: continue`). Any shard-suffix tensor — `gate_proj.deq_scale`,
  `up_proj.weight_offset` — therefore surfaces as `gate_up_proj.<suffix>` (and the attention shards
  `q_proj`/`k_proj`/`v_proj` as `qkv_proj.<suffix>`), a name that exists in no checkpoint. Look the
  key up as the shard, not as the reported fused name.
- `.deq_scale` names the **static** W8A8 scheme: only
  `AscendW8A8LinearMethod.get_perchannel_param` registers it
  (`vllm_ascend/quantization/w8a8.py:70-75`, float32 for bf16 params, int64 for fp16), while
  `AscendW8A8DynamicLinearMethod` registers `weight_scale` only. `ASCEND_QUANTIZATION_METHOD_MAP`
  (`quantization/utils.py:32-38`) maps a description value `W8A8` to the static method and
  `W8A8_DYNAMIC` to the dynamic one. So a `.deq_scale` miss says the fused linear was created
  unquantized or with a non-static scheme, whereas a `.weight`/`.weight_scale` miss points at the
  description file.
- For the fused MLP the skip lookup reads the description's *shard* entries
  (`is_layer_skipped_ascend`, `quantization/quant_config.py:163-196`, maps `gate_up_proj` ->
  `gate_proj`/`up_proj` and indexes `<shard>.weight`; `get_linear_quant_type` reads the same
  shards). A description that fuses the MLP name, or carries different values for `gate_proj` and
  `up_proj`, is the first thing to compare.

Boundary: the recorded trace was truncated before any answer, and the pasted launch block cannot be
the run that produced the traceback — it is a compose template (`--served-model-name Qwen3-0.6B`)
and it spells the flag `--quantization ascenId`, which vLLM v0.13.0 rejects in
`ModelConfig._verify_quantization` (`config/model.py:944-948`,
`ValueError: Unknown quantization method: ascenId`) long before weight loading.
No real serve line, no `config.json`, no `quant_model_description.json` for the failing checkpoint,
so the cause stays open between: description shards marked `FLOAT` (skip), description static-vs
dynamic mismatch for gate/up, and a load without `--quantization ascend`. Ask for those two files
and the serve line first. Candidate checks (not run here): serve the same directory with
`--quantization ascend`, then confirm the startup line "Using the vLLM Ascend modelslim
Quantization now!" (emitted by `get_quant_method_modelslim`) and that the fused MLP got
`AscendW8A8LinearMethod`.

Reading trap for the description file: on a 36-layer, layer-major file,
`grep -n 'gate_up_proj\|down_proj' desc.json | head -20` printed only `down_proj` entries and
reads as "no fused/gate key", because layer 0's shards come after layer 0's `down_proj`. Count each
family separately (`grep -c '<pat>'`) before concluding a key is absent.

## Upstream model lookup: `KeyError: '...experts.w2_weight_offset'` (Qwen3-VL MoE W8A8)

Recorded in vllm-project/vllm-ascend#7744: `v0.14.0rc1`, `Qwen3-VL-235B-A22B-Instruct-W8A8`
on one Atlas 800I A2 (TP=8, `--enable-expert-parallel`), every worker dies during load and the
API server reports only the wrapper

```text
Exception: WorkerProc initialization failed due to an exception in a background process.
...
RuntimeError: Engine core initialization failed. See root cause above.
```

The worker traceback ends in upstream vLLM, not the plugin:
`vllm/model_executor/models/qwen3_vl_moe.py:279 in load_weights`,
`param = params_dict[name_mapped]`, `KeyError: 'layers.37.mlp.experts.w2_weight_offset'`,
reached through `online_quantization.py:173 patched_model_load_weights` and
`qwen3_moe.py:749`. Read `WorkerProc failed to start` as a wrapper line: the class is decided by
the innermost worker traceback, and here it is a checkpoint tensor the rank's MoE module never
registered — the same lookup shape as the DeepSeek-V4 `weight_scale` case above, with an
asymmetric-quant `weight_offset` name under a fused-expert mapping.

Boundary:

- The reporter states the same quantized model and the same launcher start fine on
  `v0.17.0rc1`. The maintainer answered "this is a known issue around quantized weights loading
  of Visual Encoder", suggested `v0.17.0` (or the commercial `v0.13.0`), and the issue closed
  `wontfix`. That reporter A/B is the usable workaround: move to `v0.17.0`+ and re-test the
  identical checkpoint.
- Not reproduced here, and the "Visual Encoder" wording does not match the failing prefix —
  `layers.37.mlp` is the text-model MoE. Which parameter set is missing and why stays open.
  Candidate direction only: upstream builds `expert_params_mapping` from
  `self.get_expert_mapping()` (`qwen3_vl_moe.py:187-198` at `v0.14.0`), while the plugin's MoE
  W8A8 method does register `w13_weight_offset`/`w2_weight_offset` in `get_weight`
  (`vllm_ascend/quantization/methods/w8a8_dynamic.py`), so check which quant config that layer
  actually received at the reported tag before filing it as a checkpoint defect.

## W4A8 int4 weights in FRACTAL_NZ with `antiquantGroupSize 256`

### Symptom

The engine core dies during load/profiling, so the API server only reports
`RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`.
The cause sits in the worker log:

```text
RuntimeError: npu_weight_quant_batchmatmul:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:99
NPU function error: call aclnnWeightQuantBatchMatmulV2 failed, error code is 161002
AclNN_Parameter_Error(EZ1001): when weight's dtype is [int4], weight's format is [FRACTAL_NZ],
antiquantGroupSize should be 128 or 64, but actual antiquantGroupSize is [256].
```

Recorded report: vllm-project/vllm-ascend#5603, image vLLM Ascend 0.13.0, single NPU, Qwen3-8B
quantized following the tag's own W4A8 tutorial, with `ERR00100 PTA call acl api failed` and
`error code is 161002` in the worker log.
There is no `KeyError`: the traceback ends in `torch.ops.npu.npu_weight_quant_batchmatmul`
inside the compiled graph.

### Mechanism (code-verified at the tags named)

- The W4A8 dynamic **linear** scheme feeds the checkpoint's group size straight into the device
  op: `quantization/w4a8_dynamic.py:39-40`
  (`group_size = quant_description.get("group_size", 256)`) and `:163-167`
  (`antiquant_group_size=self.group_size`) at `v0.13.0rc1`. The same statements live in
  `quantization/methods/w4a8.py:42-43` and `:158` at `v0.14.0rc1`, and `:95` / `:190` at
  `v0.23.0` — so a description without the key also means 256.
- The layout decision ignores that group size. `vllm_ascend/utils.py:127-142` at
  `v0.13.0rc1`: `maybe_trans_nz` returns early only for fp32 and (unless the value is 2) for
  bf16/fp16, and its final branch casts **every other dtype** — the int4 quant weight emitted by
  `w4a8_dynamic.py:172 layer.weight.data = maybe_trans_nz(layer.weight.data)` — to
  `ACL_FORMAT_FRACTAL_NZ`. The switch is on by default: `VLLM_ASCEND_ENABLE_NZ` defaults to 1
  (`envs.py:120-121`). So the documented recipe produces an int4+NZ weight, and torch_npu's
  `npu_weight_quant_batchmatmul` accepts int4+NZ only at group 64/128.
- The same rule survives later as the `weight_nz_mode` policy, still without any group-size
  test: `_should_trans_nz` on `main` excludes only fp32, meta tensors and k==1/n==1 matrices,
  forces NZ for the 310P `FORCE_NZ` profile, and converts "quantized or other supported dtypes"
  by default. The env→`--additional-config` migration and per-tag spelling are in
  [env-vars-and-additional-config.md](env-vars-and-additional-config.md).
- The MoE scheme NZ-casts its experts the same way (`w4a8_dynamic.py:485-486`); a group-size
  limit for the grouped-matmul op was not established.

### Documented-path contradiction

At every tag checked, the tag's own Qwen3-8B-W4A8 tutorial prescribes `--group_size 256` and
never mentions NZ or the 64/128 acceptance: `docs/source/tutorials/Qwen3-8B-W4A8.md:65` at
`v0.13.0rc1`, `:67` at `v0.14.0rc1`, `docs/source/tutorials/models/Qwen3-8B-W4A8.md:67` at
`v0.17.0rc1` and `v0.20.2rc1`. In the `v0.13.0rc1` docs it is also the only `group_size` value at
all (`git grep -n 'group_size' <tag> -- 'docs/*'`, locale aside). The tutorial disappears by
`v0.23.0`, together with the scheme below. So "the reporter misconfigured it" is wrong here:
the failing artifact is the documented one.

### Workarounds (maintainer-stated and mechanically consistent; not executed here)

- Re-quantize with `--group_size 128` (or 64).
- Or keep 256 and stop the NZ cast: `VLLM_ASCEND_ENABLE_NZ=0` in the launch environment; on
  later tags the equivalent is `--additional-config '{"weight_nz_mode": 0}'`.

Thread boundary: menogrey ("torch_npu API limit"), then kunpengW-code with exactly these two
solutions plus "this part of code is to be reconstructed"; the issue was later closed as stale,
so **no release is confirmed to fix it**. Do not cite a fix commit for this class without
finding one. A reporter note that the published ModelScope `vllm-ascend/Qwen3-8B-W4A8` checkpoint
is also 256 got the answer that the model is not published by the vllm-ascend organisation — so
"just use the official checkpoint" is neither a workaround nor an accepted explanation.

## `aclnnQuantMatmulWeightNz ... When scale dtype is bfloat16, out dtype should also be bfloat16 but it is DT_INT32`

### Symptom

Engine start-up dies in `profile_run` (so the front end only reports
`EngineCore failed to start` / `Leave wait_for_load`), and the worker log carries

```text
RuntimeError: ... call aclnnQuantMatmulWeightNz failed, detail:EZ1001:
When scale dtype is bfloat16, out dtype should also be bfloat16 but it is DT_INT32
```

Recorded #2925 (vllm-ascend 0.9.0rc2, `torch-npu == 2.5.1.dev20250320`, `torch == 2.5.1`,
`CANN = 8.0.0`, 910B, DeepSeek-V3-0324 W8A8 served through xinference; the plugin frame is
`vllm_ascend/models/deepseek_v2.py:116 in forward -> torch_npu.npu_quant_matmul`).

### Mechanism (code-verified at `v0.9.0rc2`)

The int32 output is deliberate, part of a two-step quantized MLP — do not "fix" it as a dtype bug:

- `CustomDeepseekV2MLP.forward` (`vllm_ascend/models/deepseek_v2.py:113-141`) takes the fast path
  when `gate_up_proj.quant_method.quant_method` is `AscendW8A8DynamicLinearMethod` (`:107-111`).
  It then calls `npu_quant_matmul(x, self.gate_up_proj.weight, self.gate_up_proj.weight_scale,
  output_dtype=torch.int32)` at **`:116`** (exactly the reported line/statement), feeds the int32
  result to `npu_dequant_swiglu_quant` with `weight_scale_fp32`, and finishes with a bf16-output
  `npu_quant_matmul` for `down_proj` (`:132-138`).
- `AscendW8A8DynamicLinearMethod.get_perchannel_param` creates `weight_scale` with
  `dtype=params_dtype` (bf16) (`quantization/w8a8_dynamic.py:533-535`) and
  `process_weights_after_loading` NZ-casts the int8 weight (`npu_format_cast(..., 29)`, `:564`) —
  hence the `...WeightNz` operator name. So the rejected combination is the plugin's own design:
  bf16 scale + int32 out.
- The design survives: at `v0.9.1` the same intent moved into a per-layer attribute
  (`gate_up_proj._ascend_quant_config = {"output_dtype": torch.int32, "pertoken_scale": False,
  "return_scale": True}` at `:230-234`, consumed by `w8a8_dynamic.py:796-816`), unchanged at
  `v0.10.0rc1` (`:269-273`) and `v0.10.2rc1` (`:277-281`). A plugin-side dtype mistake would have
  been fixed in that rewrite; it was not.

### What the boundary actually is

The remaining explanation is the **installed stack**, not the plugin: aclnn enforces a dtype
contract that the pinned runtime satisfies and the reported one does not. Compare the reporter's
versions with the tag's own declarations: `v0.9.0rc2:docs/source/installation.md:14-16` requires
`CANN >= 8.1.RC1` and `torch-npu >= 2.5.1`, and `requirements.txt` pins `torch-npu==2.5.1` — while
the report shows `CANN = 8.0.0` (below the documented minimum) and a **`2.5.1.dev20250320` preview
build** predating the release. Treat that mismatch as the primary lead; which CANN/torch-npu release
first accepts int32 output with a bf16 scale is **not established** here, and the DeepSeek W8A8
dynamic path had no CI coverage at `v0.9.0rc2` (`tests/long_term/test_deepseek_v2_lite_tp2_accuracy.py`
runs the float `deepseek-ai/DeepSeek-V2-Lite`), so "CI passes" is not available as a refutation.

Triage: read the raising statement before anything else. If it is this two-step MLP, report the
dependency delta (`installation.md` of the reported tag vs the pasted `collect_env.py`) and ask for
the pinned CANN/torch-npu rerun; do not propose description edits, `--quantization` changes or an
upgrade inside 0.9.x as a fix — none of those change the dtype the plugin passes.

## A retired scheme: `NotImplementedError ... doesn't support quant_type=... for layer_type=...`

A report of the form "dense/linear W4A8 worked in v0.23.0 and is gone on main" is a planned
removal, not a regression:

- `quantization/methods/w4a8.py` registered both halves: `@register_scheme("W4A8_DYNAMIC",
  "linear")` at `:38` and `("moe")` at `:231` at `v0.23.0` (778 lines); `:39`/`:236` at
  `v0.26.0rc1` (834 lines). `v0.26.0rc2` still has the flat file.
- The `v0.26.0rc1` release notes announce it under **Ready to Deprecate**: "Added deprecation
  warnings for W4A8 linear, W4A8 MoE per-group, and W8A8 PDMix MoE quantization. These
  quantization paths will be removed in a future release" (#13850).
- The removal is `efda0446d` — #13713, "Remove deprecated W4A8Linear/W8A8PDMixMoE schemes and
  sync W4A8 MoE refactor", 2026-08-08 — **first tag `v0.27.1rc1`**, absent from `v0.26.0rc1`.
  The same commit reorganizes the module into
  `quantization/methods/w4a8/{__init__,w4a8,w4a8_mxfp4}.py`, where only
  `@register_scheme("W4A8_DYNAMIC", "moe")` survives (`w4a8/w4a8.py:36`).
- Loading such a checkpoint then raises, from `create_scheme_for_layer`:
  `NotImplementedError: Currently, vLLM Ascend doesn't support quant_type=W4A8_DYNAMIC for
  layer_type=linear. Please use a supported quantization format or load the model with its
  original float weights.` Locators: `quantization/configs/modelslim_config.py:563-569` at
  `v0.27.1rc1`, `:468-474` on `main`. It is a clear error, **not** a silent unquantized
  fallback — do not write up garbage output for this path without a log showing the fallback.
- Maintainer's reasons and replacement (#15247, closed `not_planned`): the W4A8 linear path does
  not actually quantize activations (effectively W4A16), performs poorly, has no upstream
  ModelSlim support, and was only used by Qwen3-32B; migrate to W4A4 (FlatQuant or LAOS).

Triage: prove it with the release-note line plus `git tag --contains efda0446d --sort=creatordate | head -1`
and answer with the replacement scheme. Do not inherit the reporter's file-level premise — on
`main` `quantization/methods/w4a8.py` does not exist at all, so
`git show main:<path> | wc -l` prints `0`, not "421 lines with only the MoE class". Run
`git ls-tree -r --name-only <ref> | grep -i w4a8` before quoting a path, a class list or a line
count. The two locators above are **block spans**, not the `raise` line: the message string greps at
`:469` on `main` and `:564` at `v0.27.1rc1`, the `raise NotImplementedError` at `:474` / `:569`. A
fresh grep landing a line or two inside a recorded range is the same site — do not report it as
line drift or spend a round reconciling it.
