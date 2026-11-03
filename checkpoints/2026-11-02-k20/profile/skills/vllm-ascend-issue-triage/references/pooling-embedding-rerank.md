# Pooling / embedding / rerank reports (BERT-family `bge-*`, Qwen3-Embedding)

Scope: a report serving an encoder-only pooling model — `bge-base-zh-v1.5`,
`bge-base-en-v1.5`, `bge-m3`, `bge-reranker-v2-m3`, XLM-RoBERTa, `Qwen3-Embedding`,
`Qwen3-Reranker` — on vLLM Ascend. Four terminal shapes: three raised before the model forward
runs, one a runtime hang on the response path; none of them is an attention or kernel question, so
do not open the attention/quant paths.
Accuracy-only divergence after a healthy first request is a different class
([prefix-cache-and-chunked-prefill.md](prefix-cache-and-chunked-prefill.md)).

Verified by reading tagged source in `/repos/vllm-project/{vllm,vllm-ascend}` at the tags named
below; no NPU run was made, so "works" below means "no longer gated at that tag", not "measured".

## 1. `NotImplementedError: VLLM_USE_V1=1 is not supported with ['BertModel']`

Every element is upstream vLLM; the plugin contributes nothing to this traceback.

| Step | Location (v0.9.2) | Content |
|---|---|---|
| raise | `vllm/engine/arg_utils.py:1397-1400` (raise at `:1398`) | `if not model_config.is_v1_compatible: _raise_or_fallback(feature_name=model_config.architectures, recommend_to_remove=False)` |
| property | `vllm/config.py:1439-1441` | `is_v1_compatible` → `ModelRegistry.is_v1_compatible(architectures)` |
| registry | `vllm/model_executor/models/registry.py:573-578` | `return not model_cls.supports_v0_only`; `:135` maps `"BertModel": ("bert", "BertEmbeddingModel")` |
| flag | `vllm/model_executor/models/interfaces.py:659-662` | `SupportsV0Only.supports_v0_only: ClassVar[Literal[True]] = True` |
| carrier | `vllm/model_executor/models/bert.py:389`, `:447` | `BertEmbeddingModel(nn.Module, SupportsV0Only, SupportsQuant)`, same for `BertForSequenceClassification` |

The quoted architecture (`BertModel`) is the HF config's `architectures` entry, not the vLLM
class that carries the flag (`BertEmbeddingModel`) — that is why grepping `BertModel` in
`vllm_ascend/` returns nothing. Do not read the empty grep as "plugin-specific bug".

**The explicit env var is the whole defect.** `_raise_or_fallback`
(`arg_utils.py:1726-1737`) raises only when `envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1`;
otherwise it logs `['BertModel'] is not supported by the V1 Engine. Falling back to V0.` and
returns `False`. `envs.is_set` is `name in os.environ` (`vllm/envs.py:973-977`) and upstream's
default is `VLLM_USE_V1=1` (`envs.py:637-638`), so dropping the `export VLLM_USE_V1=1` line — or
setting `VLLM_USE_V1=0` — starts the same model on V0, while `=1` converts the graceful fallback
into a start-up failure. vllm-ascend v0.9.2rc1 only reads the variable
(`git grep -n 'VLLM_USE_V1' v0.9.2rc1 -- vllm_ascend`), and its release notes already say V1 is
the default and that this line is the last V0-supporting release.

Upgrade boundary: upstream 1cd6eaba54 `Support encoder-only models without KV-Cache (#21270)`
(plus 39052dbca8, token_type_ids) is first contained in `v0.10.1`; `SupportsV0Only` is still on
`BertEmbeddingModel` at `v0.10.0` (`bert.py:450`). Upgrading upstream alone is not an answer for
an Ascend report — the plugin's own pooling path has separate boundaries (next section, table).

Thread state for the recorded #2169: a single collaborator reply — "This issue should be created
for upstream vLLM" — and a `not_planned` close. So there is no maintainer workaround to quote;
the workaround above is source-derived and should be labelled as such.

## 2. `IndexError: tuple index out of range` at `vllm_ascend/core/scheduler.py:209`

Report shape: bge-m3 (any pooling model) on vllm-ascend **0.11.0rc0** + vLLM 0.11.0rc3 starts,
registers `/v1/embeddings`, then dies on the first request; the client sees `EngineDeadError` /
HTTP 400. Three log markers identify it before any code read: `Disabling chunked prefill for
model without KVCache`, `Only "last" pooling supports chunked prefill and prefix caching;
disabling both.`, and `vllm cache_config_info with initialization after num_gpu_blocks is: 1`.

- `vllm_ascend/platform.py:139-140` (v0.11.0rc0): `if (model_config is not None and not
  model_config.use_mla and not scheduler_config.async_scheduling):` — no pooling check — and the
  inner gate at `:145-150` (`is_multimodal_model`, `structured_outputs_config.backend == "auto"`,
  `scheduler_delay_factor`, `send_delta_data`, `policy == "fcfs"`) then sets
  `ascend_scheduler_config.enabled = True`. An embedding model is non-MLA and sync-scheduled, so
  it is routed to `AscendScheduler`.
- `vllm_ascend/core/scheduler.py:209`: `blocks = new_computed_blocks.blocks[0]`, in the ordinary
  (non-P/D) schedule path. `KVCacheBlocks.blocks` is `tuple[list[KVCacheBlock], ...]` per KV-cache
  group (`vllm/v1/core/kv_cache_manager.py:19-33` at v0.11.0rc3); a model with no KV cache has
  zero groups, so the tuple is empty and the first scheduled request raises before model execution.
- Upstream does not index it: `git grep -n 'new_computed_blocks.blocks\[0\]' v0.11.0rc3 -- vllm`
  finds nothing. This indexing exists only in the Ascend copy, so "upstream has the same bug" is
  false.

Fix boundary: 02c26dcfc `[Feat] Supports Aclgraph for bge-m3 (#3171)` adds
`and model_config.runner_type != "pooling"` to the platform condition; it is an ancestor of
`v0.11.0rc1` but **not** of `v0.11.0rc0`. The `blocks[0]` line still exists at `v0.11.0` because
the guard now keeps pooling models out of `AscendScheduler` — presence of the line is not evidence
of an unfixed bug. Workaround: use vllm-ascend ≥ `v0.11.0rc1` — the code guard is there even
though the matching release note ("Pooling models, such as bge-m3, are supported now. #3171") is
backfilled only into later changelogs, under the `v0.11.0rc1` heading from `v0.12.0rc1` on; the
`v0.11.0` tag's own `release_notes.md` has no such line. On rc0, the force-enable is skipped by
`--async-scheduling`, a non-`fcfs` `--scheduler-policy`, or a non-default structured-output backend;
those are code-read candidates, not tested fixes — offer the upgrade first and label the rest.

Path note: `vllm_ascend/core/scheduler.py` exists through v0.11.x and is gone by `v0.12.0rc1`
(Ascend scheduler removed — first-removal tag and rationale in
[env-vars-and-additional-config.md](env-vars-and-additional-config.md)). `git show
origin/main:vllm_ascend/core/scheduler.py` fails with `fatal: path ... does not exist`; run
`git ls-tree -r --name-only <ref> | grep 'core/sched'` before any cross-release comparison.

## 3. `vllm serve: error: argument --task: invalid choice: 'rerank'`

`--task` is an argparse choice over upstream **vLLM**'s `TaskOption`, so the plugin cannot add a
value and the failure happens before any model loading. TaskOption at `v0.9.1` is
`vllm/config.py:82-83` (`auto, generate, embedding, embed, classify, score, reward,
transcription`), moving to `vllm/config/model.py:64-65` by `v0.11.0`; neither has `rerank`.

`_RUNNER_TASKS["pooling"] = ["embed", "classify", "score", "reward"]` (`v0.9.1`
`vllm/config.py:90-95`, pooling at `:92`). Reranking is therefore `--task score` and embedding is
`--task embed` (the documented pooling serve line at v0.12.0rc1 is
`vllm serve Qwen/Qwen3-Embedding-8B --task embed`).
The HTTP route is `/rerank` (`/v1/rerank`, `/v2/rerank` appear in a 0.11.0rc3 server log), which
is exactly why users copy `rerank` into `--task`. Fix the spelling, then re-answer the real
question ("is this architecture supported on this engine") from the table below.

## 4. Client hangs on `/v1/embeddings` while the server access log says `200 OK`

Report shape: plugin `v0.11.0` (final, so shape 2's `AscendScheduler` gate is already fixed),
BERT-family pooling model (`bge-m3`), 910B, CANN 8.3.RC2, k8s; `curl`/`requests` never returns and
the server prints `POST /v1/embeddings HTTP/1.1" 200 OK` beside a *worker-side* line
`('Warning: torch.save with "_use_new_zipfile_serialization = False" is not recommended for npu
tensor, ...')`. No exception, no traceback, `/dev/shm` already excluded by the reporter.

- **That warning has no caller in either tagged tree — do not grep the two repos for it.**
  `git grep -n 'torch\.save' v0.11.0 -- vllm` returns one real call, `safetensors.torch.save_file`
  in `distributed/kv_transfer/.../shared_storage_connector.py`, plus a `torch-saved` docstring line
  in `entrypoints/renderer.py` (the unescaped dot matches that hyphen — do not read it as a second
  caller); `_use_new_zipfile_serialization` matches nothing in vLLM `v0.11.0` nor in `vllm_ascend/`
  at `v0.11.0rc0`/`v0.11.0`. The emitter is the torch/torch_npu seam:
  Torch's storage `__reduce__` (`torch/storage.py`, `_StorageBase.__reduce__` and
  `UntypedStorage.__reduce__`) serialises with `torch.save(self, b,
  _use_new_zipfile_serialization=False)`, and `torch_npu/utils/serialization.py`
  (`_add_serialization_methods`) wraps `torch.serialization._legacy_save` to print exactly that
  text. So the line means "an NPU tensor/storage was pickled" — a process-boundary marker, not a
  plugin frame. Grep those two files, not the repos.
- Which boundary produced it is the open question, and the report does not answer it: a uvicorn
  `200 OK` access line is written when the server has finished the response, so the warning is at
  best consistent with a tensor crossing engine-core → API-server (the `/v1/embeddings` return path
  the thread comment also names) — keep that as a **candidate** and ask whether the engine even ran
  the request.
- Thread for the recorded `vllm-ascend` #6424 (open, no maintainer reply): two comments, both
  `NONE` association — "建议的解决方式：升级 torch_npu版本 … 该问题属于 vllm-ascend 在 embedding API
  返回阶段与 torch_npu 的兼容性问题" plus an AscendHub image link. Reporter-side, not maintainer.
- What would settle it: whether the engine ran the request (`Running: N reqs` and a completed
  `execute_model` in the API-server log vs nothing), a `py-spy dump` of the ApiServer process
  (blocked in `recv`/unpickling vs in the response writer), and the same request against a newer
  image / torch_npu. If the API server is past the body and the client still waits, the next layer
  is the transport (k8s service/proxy), not the plugin.

## Per-release pooling support, engine included

| Ref | Evidence | What it establishes |
|---|---|---|
| vllm-ascend v0.7.3rc1 | release notes, #229 | initial pooling support; `bge-base-en-v1.5` / `bge-reranker-v2-m3` "work now" (V0 era) |
| v0.9.1 | `vllm_ascend/worker/worker.py:112-113` | `runner_type == "pooling"` → `NPUPoolingModelRunner`; V1 pooling runner absent |
| vLLM v0.9.1 / v0.9.2 | `bert.py:389`, `:447` | BERT pooling classes are `SupportsV0Only` → `--task embed\|score` only on V0 |
| vllm-ascend v0.9.2rc1 | release notes; `support_matrix/supported_features.md` | "Pooling model works with V1 engine now … Qwen3 embedding model (#1359)"; matrix row: Pooling V0 🟢 Functional / V1 🟡 Planned, "V1 support rely on vLLM support" |
| vLLM v0.10.1 | 1cd6eaba54 (#21270) first tag | encoder-only models without KV cache allowed; flag removed from `BertEmbeddingModel` |
| vllm-ascend v0.11.0rc1 | 02c26dcfc (#3171) first tag | bge-m3 kept off `AscendScheduler`; the "bge-m3 supported now" note exists only in ≥v0.12.0rc1 changelogs |
| vllm-ascend v0.12.0rc1 | e32014ac1 (#3122) first tag; release notes | "Pooling model, such as bge, reranker, etc. are supported now"; V1 pooling runner and `tests/e2e/pull_request/one_card/pooling/test_scoring.py` arrive |
| vllm-ascend v0.20.2rc1 | e029f683d (#8846) first tag | 310P (Atlas 300I DUO) pooling: non-causal attention mask + flash-attention forward for embedding/classification/scoring, with 310P e2e tests. Nothing earlier on 310P |

Doc-reading caveats for "is bge supported" answers:

- `release_notes.md` is one cumulative changelog, written ahead of and patched after its tags: the
  `v0.9.1` tag's file already contains a `## v0.9.2rc1` section, and the `v0.11.0rc1` section's
  bge line only appears from `v0.12.0rc1` on. Match a note to a release by the commit/tag that
  implements it, not by the heading alone, and read the same file at a later tag when a release's
  own file is silent.
- No bge tutorial exists at these tags — only `tutorials/single_npu_qwen3_embedding.md`
  (0.9.2rc1-0.11.x), `tutorials/Qwen3_embedding.md` (0.12) and `tutorials/models/Qwen3-*.md` on
  `main`. A bge serve line cannot be diffed against a tutorial.
- The per-release `support_matrix/supported_models.md` "Pooling Models" table lags: at v0.9.2rc1
  it lists only XLM-RoBERTa-based and Molmo while the same tree's release notes already claim
  Qwen3-Embedding on V1. Quote release notes for the model axis and the support matrix for the
  engine axis; call the table stale instead of reading omission as "unsupported".
- "Pooling model supported" in a release note never covers every pooling architecture:
  decoder-based embedders worked from v0.9.2rc1, while BERT encoders needed upstream #21270 and
  plugin #3171. Name the release *and* the engine (V0/V1) in every answer.
- **310P is a third axis.** `e029f683d` "[Feature] Support pooling models on 310P platform" (#8846,
  2026-05-27) adds the non-causal mask and forward path and is first contained in `v0.20.2rc1`;
  before it, a Qwen3-Embedding serve on Atlas 300I DUO dies while *loading* weights, not in
  attention. #1991 (`v0.9.2rc1-310p`, Qwen3-Embedding-4B) is that boundary: the maintainer
  reproduced on 310P3, answered "Qwen3-Embedding is not supported on 310I series" with
  `ValueError: There is no module or parameter named 'layers' in CustomQwen3ForCausalLM` from a
  *main source checkout* (`/home/mnj/code/vllm-ascend/vllm_ascend/models/qwen3.py:156`; that file
  does not exist at `v0.9.2rc1`), and later pointed at #8846 / "use `main` or wait the next
  version". The eager-only advice that precedes it is the 310P platform note
  ([310p-atlas-300i-duo.md](310p-atlas-300i-duo.md)). The reporter's own pasted error was a different
  layer entirely — `AclSetCompileopt ... 500001` / `init runtime platform info failed, SocVersion =
  Ascend310P3` at `set_device` — which is the container device-access class
  ([image-and-build-reports.md](image-and-build-reports.md)) and was never diagnosed in that thread;
  answer both layers instead of picking one.

## One-call verification

```bash
# shape 1, at the reporter's vLLM tag; ranges below are v0.9.2's (see the line-shift note)
cd /repos/vllm-project/vllm && R=v0.9.2 && \
 git show $R:vllm/engine/arg_utils.py | sed -n '1397,1400p;1726,1737p' && \
 git show $R:vllm/config.py | sed -n '1439,1441p' && \
 git show $R:vllm/model_executor/models/registry.py | sed -n '135p;573,578p' && \
 git show $R:vllm/model_executor/models/interfaces.py | sed -n '659,662p' && \
 git show $R:vllm/model_executor/models/bert.py | grep -n '^class BertEmbeddingModel\|SupportsV0Only'

# shape 2, at the reported plugin tag
cd /repos/vllm-project/vllm-ascend && R=v0.11.0rc0 && \
 git show $R:vllm_ascend/platform.py | sed -n '139,151p' && \
 git show $R:vllm_ascend/core/scheduler.py | sed -n '203,211p' && \
 git tag --contains 02c26dcfc | grep -E '^v0\.' | sort -V | head -3
```

Quote `file:line` from these tags only; line numbers move between releases (the same raise is
`:1398` at v0.9.2, `:1371` at v0.10.0 and `:1424` at v0.10.1; TaskOption moves file at v0.11.0).

## Boundaries

- Nothing here was executed on hardware. The V0-fallback workaround, the rc1 upgrade boundary and
  the rc0 flag candidates are source- and release-note-derived; only the reporter's own log lines
  are observed runtime behaviour.
- The reporter's version string is what they typed. A `0.1.dev1+gf5f88f78b` style build resolves
  to a vllm-ascend commit (f5f88f78b) and can sit between tags; pin it before quoting line numbers.
- A `not_planned` close plus "file it upstream" does not mean the report is wrong, and a merged
  plugin PR does not prove the reporter's build contains it — check `git tag --contains`.
