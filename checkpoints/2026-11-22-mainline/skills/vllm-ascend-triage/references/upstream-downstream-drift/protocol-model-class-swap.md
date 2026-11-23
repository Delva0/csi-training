# A patch that replaces a public pydantic protocol model, not a function

The fourth dispatch failure beside `patch-import-side-effect.md` (hook never ran),
`oot-class-registration-drift.md` (replacement keyed by class name) and
`stale-module-binding.md` (consumer cached the pre-patch name): here the patch
**succeeds globally** and the *other* consumers of the same class break, because it
did not wrap an upstream function — it defined a **subclass of an upstream pydantic
model and rebound the class identity**, then rewrote the field annotations of models
the consumer was already building.

Recorded #8201: image `vllm-omni:v0.18.0` (built `FROM
quay.io/ascend/vllm-ascend:v0.18.0rc1`), model `qwen-image-edit`, vllm-omni
image-to-image. vllm-ascend `v0.18.0rc1` (`99e1ea0f`) + vllm `v0.18.0` (`bcf2be96`).

## Symptom

The vllm-omni request path raises a pydantic `ValidationError` whose text names a
vllm protocol model, not a vllm-ascend class: `Input should be a valid dictionary or
instance of UsageInfo [type=model_type, input_value=UsageInfo(...),
input_type=UsageInfo]`, i.e. the input **is** a `UsageInfo` and is still rejected. The
reporter has usually localised it already, citing
`vllm_ascend/patch/platform/patch_minimax_usage_accounting.py` and the block that
assigns `engine_protocol.UsageInfo = UsageInfo` / `chat_protocol.UsageInfo = …`. Say
at once that the file exists at the pin and that it is **unconditional** — the
report's own premise ("此 Patch 可能是一个特定版本的临时修改") is right about the
version and wrong about the scope.

## What the patch actually does (verified at the pin)

| site | line | what it establishes |
|---|---|---|
| `vllm_ascend/patch/platform/patch_minimax_usage_accounting.py` | 1-249 | a backport of upstream PR #37955 (cited in the patch's own registration header, `patch/__init__.py:194-212`): MiniMax-M2 `</think>` reasoning-token counting |
| same, `:40-51` | | `class UsageInfo(engine_protocol.UsageInfo)` with one extra field (`completion_tokens_details`), `__module__` of both new classes set to `vllm.entrypoints.openai.engine.protocol`, then `engine_protocol.UsageInfo`, `chat_protocol.UsageInfo`, `chat_serving.UsageInfo`, `chat_serving.CompletionTokenUsageInfo` rebound |
| same, `:54-62` | | `_rebuild_model_field` writes `ChatCompletionResponse.usage`, `ChatCompletionStreamResponse.usage` and `RequestResponseMetadata.final_usage_info` into `__annotations__` **and** `model_fields[...].annotation`, then `model_rebuild(force=True)` — a global mutation of three pydantic classes |
| same, `:202-249` | | the other half: `OpenAIServingChat.chat_completion_full_generator` is replaced wholesale under `_ascend_original_…` |
| `vllm_ascend/patch/platform/__init__.py:32` | | imported **unconditionally** — no `vllm_version_is` gate, no env switch, unlike `patch_multiproc_executor` (`DYNAMIC_EPLB`) and `patch_balance_schedule` (`VLLM_ASCEND_BALANCE_SCHEDULING`) in the same `if` block |
| `vllm_ascend/platform.py:138` | | `AscendPlatform.pre_register_and_update()` → `adapt_patch(is_global_patch=True)` → the patch package. It lands at **platform registration**, i.e. after the API server's own imports and after any OOT package (vllm-omni) imported `vllm.…protocol` |
| upstream `vllm/entrypoints/openai/engine/protocol.py:104-108` | | vllm v0.18.0 `UsageInfo` has only `prompt_tokens/total_tokens/completion_tokens/prompt_tokens_details`; `CompletionTokenUsageInfo` and `completion_tokens_details` **do not exist** anywhere in that tree — so the patch changes the class's shape, not just its behaviour |
| upstream, `:26-28` | | `OpenAIBaseModel` is `ConfigDict(extra="allow")` — the extra detail could ride the existing class, so the subclass is a choice, not a necessity |

The same rebinding is why a **chat** `usage` body on this stack can carry
`completion_tokens_details` although upstream's chat schema has none, and why a Qwen
serve reads `null`/`0` there even with `--reasoning-parser qwen3`: the schema half is
process-wide, only the token *count* is MiniMax-gated (`:95-100` and `_make_usage_info`
`:131-146` at v0.22.1rc1; the same file at v0.18.0rc1 is `:40-62`). So a
`reasoning_tokens` value is not evidence that a parser ran — the triage consequence and
the field to read instead are in `../accuracy-benchmarks/reasoning-output-routing.md`
(§*inverse shape*).

## Mechanism — pydantic v2 refuses a base instance where a subclass is annotated

Two forms, one cause; both are silent until an instance crosses the boundary:

1. **The consumer holds the pre-patch class.** `from vllm.entrypoints.openai.engine.protocol
   import UsageInfo` executed at module import binds the *original* class. The patch
   then rebinds the module attribute and rewrites `ChatCompletionResponse.usage` to the
   *subclass*. pydantic v2 validates model instances by exact class (or subclass),
   never by coercion downwards, so constructing/validating a response with the stale
   instance raises `model_type`.
2. **The consumer's own schema cached the old class.** A pydantic model or
   `TypeAdapter` in the consumer that embedded the class before the patch keeps its
   core schema; `model_rebuild(force=True)` rebuilds only the *patched* models, never
   the consumer's.

The reporter's own workaround is the discriminator: **passing a dict instead of the
instance makes the error go away**, because a dict is re-validated into the new class.
That is evidence for the mechanism, so quote it rather than treating it as a
coincidence.

## Cheap checks (no NPU, one script inside the reporter's image)

```python
from vllm.entrypoints.openai.engine import protocol as ep
Old = ep.UsageInfo
from vllm_ascend.utils import adapt_patch
adapt_patch(is_global_patch=True)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
print(ChatCompletionResponse.model_fields["usage"].annotation is Old)   # False => swap live
ChatCompletionResponse.model_validate({"id": "x", "object": "chat.completion",
    "created": 0, "model": "m", "choices": [],
    "usage": Old(prompt_tokens=1, total_tokens=2, completion_tokens=1)})  # reproduces model_type
```

- The three greps that bound the blast radius, in one call:
  `rg -n 'UsageInfo|request_metadata|ChatCompletionResponse\(' <consumer-tree>` and
  `rg -n 'UsageInfo' <vllm-ascend>/vllm_ascend/patch/`.
- The reporter's `ValidationError` body settles form 1 vs form 2: `input_type` equal
  to the old `UsageInfo` class is form 1; a `loc` that runs through the consumer's own
  model path is form 2. Ask for the full block (`loc`, `type`, `input_type`), the
  image's `vllm.__version__` / `vllm_omni.__version__`, and the probe output. Do not
  assert which form it is from public text alone.

## Ownership, and the repair split (matches the public outcome)

The patch is the defect. Its own header says so: *"Remove this patch once the upstream
MiniMax usage-accounting fix is in the runtime vLLM version used by vllm-ascend."*
The maintainer's reply on #8201 (gcanlin, 2026-04-14) — "It may be a bug in
vllm-ascend. Could you please submit a PR to vllm-omni as a workaround?" — and the
reporter's "done. https://github.com/vllm-project/vllm-omni/issues/2777" fix the
ownership for future reports: **root cause downstream (vllm-ascend), shipped
workaround in the consumer (vllm-omni)**. Do not re-litigate ownership, and do not
frame the consumer's dict-passing as the fix.

- **Consumer side (what actually unblocks the reporter).** Hand the protocol models
  plain dicts, or resolve the class at call time from the module
  (`from vllm.entrypoints.openai.engine import protocol as ep;
  ep.UsageInfo(...)`) instead of importing the name at module scope. Both survive the
  swap in either direction.
- **vllm-ascend side (the root fix).** Do not change a public class's identity: with
  `extra="allow"` (`engine/protocol.py:26-28`) the detail can be attached to the
  existing `UsageInfo`; alternatively gate the whole patch to the MiniMax-M2 path or
  behind an env flag, like its siblings. An `except ValidationError` / `getattr`
  guard is not a repair — it starts the service with the accounting still wrong.
- **Acceptance is two requests, not one.** A non-MiniMax chat request's `usage` must
  stay the v0.18.0 baseline (no `completion_tokens_details` key); a MiniMax-M2
  reasoning request must carry `completion_tokens_details.reasoning_tokens` with
  `0 ≤ reasoning_tokens ≤ completion_tokens`. "不再报错" is not the test.

## Also true inside vllm, and worth telling the reporter

The same patch is inconsistent with its own tree: the non-chat endpoints keep the
**old** class and assign into the rebuilt `final_usage_info`
(`vllm/entrypoints/openai/completion/serving.py:420` built, `:445`;
`vllm/entrypoints/openai/speech_to_text/speech_to_text.py:714`;
`vllm/entrypoints/serve/disagg/serving.py:194` built, `:203`; the `pooling/*`
`usage=` sites). It
does not raise only because `RequestResponseMetadata` is a plain `BaseModel` with no
`validate_assignment`; a `model_validate` on that path, or `validate_assignment=True`
added later, turns the same latent mismatch into a second `model_type` error.

## Not this page

- A `ValidationError for ModelConfig` / `VllmConfig` (`Unexpected keyword argument`,
  unsupported quantization, unregistered model type) is config validation:
  `../startup-failures/model-type-not-recognized.md`,
  `../startup-failures/quantization-method-gate.md`.
- A response *field* that vanishes or stays `0` on a MiniMax/GLM serve is the
  accounting lane, not the class swap: `../accuracy-benchmarks/reasoning-output-routing.md`
  (its `patch_glm_reasoning_usage_accounting.py` section carries the sibling patch).
- A consumer that never sees the Ascend class at all, with no pydantic error, is one
  of the three dispatch pages listed at the top.

## Limits / recheck

- vllm-omni's source is not in the task workspace (`SOURCES.json` carries no
  vllm-omni entry); every vllm-omni statement above is the reporter's and the
  maintainer's, not a read of that tree. Say so.
- The reporter notes the file is absent from `main`. The snapshot pins v0.18.0rc1
  only, so `main` cannot be checked here — record it as the reporter's observation,
  consistent with the patch's Future Plan.
- Recheck at another ref: `rg -n 'patch_minimax_usage_accounting|patch_glm_reasoning_usage_accounting'
  <vllm-ascend>/vllm_ascend/patch/platform/__init__.py` and whether
  `_rebuild_model_field` still exists; the class-swap block is the whole hazard, so a
  version that adds a gate or drops the rebinds invalidates the ranking above.
