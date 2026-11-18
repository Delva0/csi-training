# Pooling/embedding models that never serve: two start-up classes and how to tell them apart

## Scope

Reports for `--task embed` / `--runner pooling` / sentence-transformers checkpoints (`bge-*`,
`Qwen3-Embedding-*`, rerankers) whose service **starts or dies without ever returning a vector**, where the
pooling attention route of `pooling-and-embedding.md` (wrong *vectors* on a healthy service) is not the
owner. Two verified members, each with its own page:

- **A.** `NotImplementedError: VLLM_USE_V1=1 is not supported with ['BertModel']` — upstream vLLM's
  engine-selection oracle, raised only because the env var was *set* → `v1-engine-selector.md` (#2169).
- **B.** First request dies inside the fork's scheduler on a model with **no KV cache at all**:
  `IndexError: tuple index out of range` at `vllm_ascend/core/scheduler.py:209` → `ascend-scheduler.md`
  (#3462).

Not this page: an embedding that differs from the GPU reference while the service is healthy and fast
(`pooling-and-embedding.md`); a `transformers` `model type <X>` floor (`model-type-support-floor.md`); a
KV-budget refusal or a stall after weight load (`startup-and-stall.md` §3b, §3).

## Common rules

- **Two facts decide both members, and both are in the log**: which engine actually ran (`V1 LLM engine`
  in the banner, `Supported_tasks: ['encode', 'embed']`) and whether the model has any KV-cache layers.
  The attention-free fingerprint is three lines that appear together before the traceback:
  `Only "last" pooling supports chunked prefill and prefix caching; disabling both.`,
  `Disabling chunked prefill for model without KVCache`, and `num_gpu_blocks is: 1`.
- **The architecture string in the traceback is a registry key, not a directory name.** `BertModel`
  resolves to `bert:BertEmbeddingModel`; a checkpoint served as `bge-base-zh-v1.5` never appears in the
  Python text. Quote the key when looking the class up
  (`<vllm-ref>:vllm/model_executor/models/registry.py`).
- **Both members are decided by a launch choice (an env var, an `additional_config` key) plus a version,
  never by a patch you write.** Answer with that choice and say which release stops needing it; the
  mechanism is what lets you say *why* the workaround works.
- Keep the two engines apart: an answer that says "vllm-ascend cannot serve embedding models" is wrong at
  `v0.9.2rc1` and later — that tag ships `vllm_ascend/worker/model_runner.py` (V0),
  `worker/model_runner_v1.py`, `worker/pooling_model_runner.py` and a `single_npu_qwen3_embedding`
  tutorial (`git ls-tree -r --name-only v0.9.2rc1 -- vllm_ascend/worker`).

## Discriminating the two members

- `NotImplementedError … VLLM_USE_V1=1 is not supported with [...]`, deepest frame in
  `vllm/engine/arg_utils.py`, no worker line: §A. The env var being **set** is the trigger — unset it (or
  `=0`) and the oracle warns and falls back to V0; the owning repository is upstream vLLM, so do not read
  `vllm_ascend/` code for it (`v1-engine-selector.md`).
- `IndexError: tuple index out of range` inside `schedule()` at `vllm_ascend/core/scheduler.py`, with
  `Using configured V1 scheduler class …AscendScheduler` above it: §B. Confirm the two "no KVCache" log
  lines before asserting no-KV, print `len(kv_cache_groups)` from the config rather than assuming, and
  remember the scheduler is **opt-in** (`ascend_scheduler_config.enabled`, default false) and gone at
  `v0.12.0rc1` — lever: drop the block, override `scheduler_cls` inside it, or upgrade
  (`ascend-scheduler.md`).
- A wrong **vector** on a healthy service, a first-vs-second-request divergence, or an accuracy delta →
  `pooling-and-embedding.md`. Neither member here produces wrong output.

## Reported cases

- **#2169** — `vllm 0.9.2` + vllm-ascend `0.9.2RC1`, `bge-base-zh-v1.5`, `export VLLM_USE_V1=1` then
  `vllm serve …`; one comment (`yiz-liu`, 2026-05-27) "This issue should be created for upstream vLLM."
  No reporter confirmation of the unset-var workaround in the recorded segment; `--enforce-eager` is
  irrelevant (the raise happens in `create_engine_config`, before any device work).
- **#3462** — `v0.11.0rc0`, openEuler 24.03 aarch64 / Kunpeng 920, bge-m3, `/v1/embeddings`; thread has one
  comment (`Tflowers-0129`, 2026-05-28) 「请问在新版本中重新尝试过吗」, state `closed`, no mechanism. The
  mechanism and both levers are read from `v0.11.0rc0` and `v0.12.0rc1`, not from the thread.

## Check before reuse

```bash
# A: the oracle, its caller and the class attribute at the vLLM ref
git -C /workspace/repos/vllm-project/vllm show <vllm-ref>:vllm/engine/arg_utils.py \
  | sed -n '/def _raise_or_fallback/,+14p'
git -C /workspace/repos/vllm-project/vllm show <vllm-ref>:vllm/model_executor/models/registry.py \
  | grep -n '"<Architecture>":\|supports_v0_only'
# B: the index, the empty-tuple constructor, and the no-KV config branch
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:vllm_ascend/core/scheduler.py \
  | grep -n 'new_computed_blocks.blocks'
git -C /workspace/repos/vllm-project/vllm show <vllm-ref>:vllm/v1/core/kv_cache_manager.py \
  | sed -n '/def create_empty_block_list/,+4p'
```

Both classes are version-scoped: `supports_v0_only` and the whole `_is_v1_supported_oracle` disappear once
upstream rewrote pooling onto `runner_type` (gone from vLLM `v0.11.1`), and `AscendScheduler` exists only on
the 0.10/0.11 lines. Re-read the class page's own version section before quoting either line.
