# `NotImplementedError: VLLM_USE_V1=1 is not supported with …`: upstream's engine selector, not a plugin bug

## Scope

Startup dies while the engine config is built — before any worker, weight load or NPU call — with
`NotImplementedError: VLLM_USE_V1=1 is not supported with ['<Arch>'].` and the deepest frame in
`vllm/engine/arg_utils.py`. The same raise carries a feature name instead of an architecture list
(`--enable-prompt-embeds`, `--load-format sharded_state`, `--preemption-mode`,
`--logits-processor-pattern`). The message is upstream vLLM's **V1 oracle**: no `vllm_ascend/` line
raised it, so the fix is never a plugin change. Worked report: #2169 (`bge-base-zh-v1.5`, architecture
`BertModel`, vllm 0.9.2 / vllm-ascend 0.9.2RC1; one comment, "This issue should be created for upstream
vLLM.").

Not this page: a pooling model that **serves** and returns wrong vectors (`pooling-and-embedding.md` —
a recorded run loaded that page for this traceback and chased `--task embed` support instead), a
`transformers` model-type floor (`model-type-support-floor.md`), a field the fork never plumbed
(`missing-downstream-feature.md`), or a graph-mode refusal (`aclgraph-mode-and-capture.md`).

## Three-value semantics of `VLLM_USE_V1` (vLLM ≤ `v0.11.0`)

Verified at vLLM `v0.9.2`: `engine/arg_utils.py:1073-1079`
(`try_v1 = envs.VLLM_USE_V1 or not envs.is_set("VLLM_USE_V1")`), `:1397` (`if not
model_config.is_v1_compatible:`), `:1726-1738` (`_raise_or_fallback`), `envs.py:637-638`
(default `1`), `:973-977` (`is_set` = the name is in `os.environ`).

| `VLLM_USE_V1` | what a V0-only architecture/feature gets |
| --- | --- |
| **unset** | V1 is tried, the oracle logs `… is not supported by the V1 Engine. Falling back to V0.` and the service **runs on V0** |
| **`1` exported** | `NotImplementedError` — this report |
| **`0`** | V1 is never tried |

**The export, not the value, decides whether it raises.** A report whose start command opens with
`export VLLM_USE_V1=1` therefore carries its own workaround in its own first line, and the honest status
is "self-inflicted by an explicit opt-in on an architecture the release does not V1-support".

## Gate chain at the reported ref

1. `arg_utils.py:1397-1398` `if not model_config.is_v1_compatible:` →
   `_raise_or_fallback(feature_name=model_config.architectures)` (the reporter's frame names `:1398`);
   with the env var set it raises the message verbatim, the list printing as `['BertModel']`.
2. `vllm/config.py:1439` `is_v1_compatible` → `ModelRegistry.is_v1_compatible(architectures)`.
3. `vllm/model_executor/models/registry.py:573-578` returns `not model_cls.supports_v0_only`.
4. `registry.py:135` maps `"BertModel": ("bert", "BertEmbeddingModel")`, and `bert.py:389` declares
   `class BertEmbeddingModel(nn.Module, SupportsV0Only, SupportsQuant)` — the **architecture**, not the
   task (`--task embed`), is what the oracle tests, so grepping the request flags finds nothing.

One-line check at any vllm ref:
`git -C /workspace/repos/vllm-project/vllm grep -n '"<Arch>": (' <ref> -- vllm/model_executor/models/registry.py`
then `git show <ref>:vllm/model_executor/models/<module>.py | grep -n 'SupportsV0Only'`.

## Workaround (at vllm-ascend `v0.9.2rc1`, proposed — the reporter never confirmed)

- **Drop the export** (`unset VLLM_USE_V1`, or `VLLM_USE_V1=0`): the same command falls back to V0, which
  is where this architecture is supported. The plugin's V0 path exists at that ref and is selected by the
  same variable — `vllm_ascend/platform.py:182-193` picks `vllm_ascend.worker.worker.NPUWorker` when
  `envs.VLLM_USE_V1` is false, with `worker/pooling_model_runner.py` for pooling. Cost: V1-only
  behaviour (e.g. `custom_ops=["all"]` is set only under V1, `platform.py:205-207`) and V0 is the
  deprecated engine.
- **Or serve a pooling architecture that is not V0-only** — Qwen3-Embedding with `--task embed`,
  documented at vllm-ascend `v0.9.2rc1` (`docs/source/tutorials/single_npu_qwen3_embedding.md`, "only
  0.9.2rc1 and higher support the model"). A different model, not a drop-in for bge: say so.
- **Root-cause fix is an upgrade:** `SupportsV0Only` was dropped from `BertEmbeddingModel` by
  `1cd6eaba54` (#21270, "Support encoder-only models without KV-Cache", 2025-07-26), first tag
  **`v0.10.1rc1`** — `git merge-base --is-ancestor 1cd6eaba54 v0.10.0` fails, `… v0.10.1rc1` succeeds.
  The vllm-ascend half must move with it (pair `v0.9.2rc1` → vLLM `v0.9.2`,
  `../vllm-ascend/version-pairing.md`), so the answer is "vllm-ascend ≥ v0.10.1rc1", not a wheel swap.

## Version scope: this class does not exist on later refs

`_is_v1_supported_oracle`, `is_v1_compatible` and `SupportsV0Only` are all present through vLLM
`v0.11.0` and all gone from `v0.11.1` (`c3ee80a01a`, "[V0 deprecation] clean up
is_v1_supported_oracle" #28116), and absent at `v0.14.0`/`v0.23.0`. Never transplant this message onto a
later report: there the same complaint is an unknown-knob or architecture-support question.

## Ownership and status

- The raise is upstream vLLM's, at the vllm tag the plugin pairs with; file the code half there and quote
  **both** versions (a vllm-ascend tag alone cannot place this frame). An `ERR99999 UNKNOWN applicaiton
  exception` tail in the log is the CANN logger reacting to that uncaught Python exception, not a second
  fault.
- #2169: one maintainer comment pointing upstream, no fix link, no reporter confirmation → status
  "moved upstream / open", never "fixed".

## Falsifier and recheck

Falsifier: the same command with `VLLM_USE_V1` unset starts, and the log shows the `Falling back to V0`
warning — then the explicit export was the whole defect. If it still dies with the identical message
after the variable is removed, the build has no V0 path at all: the lever is an upgrade, not a flag.
