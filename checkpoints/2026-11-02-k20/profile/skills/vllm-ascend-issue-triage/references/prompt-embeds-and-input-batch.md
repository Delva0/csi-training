# `prompt_embeds` and other upstream input-path features the plugin's InputBatch lacks

Applies when a request carries `prompt_embeds` instead of token ids (offline
`LLM.generate(prompt_embeds=...)`, the OpenAI completions API with embeddings, or any harness
that pre-computes inputs), and the engine dies inside the plugin's copied input state. The
general shape — upstream v1 gains a request-input feature, the plugin's fork of
`InputBatch`/`CachedRequestState`/model runner does not carry it — repeats for any new
request field, so the method below is the reusable part.

## Signature

```text
File ".../vllm_ascend/worker/model_runner_v1.py", line 650, in _update_states
    self.requests[req_id] = CachedRequestState(
File ".../vllm_ascend/worker/npu_input_batch.py", line 70, in __post_init__
    self.num_prompt_tokens = len(self.prompt_token_ids)
TypeError: object of type 'NoneType' has no len()
```

Verified at `v0.11.0rc2` (`/repos/vllm-project/vllm-ascend`, issue #4505): the plugin's
`CachedRequestState` dataclass (`npu_input_batch.py:46`) declares only
`prompt_token_ids: list[int]` — no `prompt_embeds` field — and `__post_init__` (`:69-70`) does
`len(self.prompt_token_ids)`. `model_runner_v1.py:648-650` then passes
`prompt_token_ids=new_req_data.prompt_token_ids`, which is `None` for an embeds-only request.
Line numbers shift by ~2 between the rc and the final of a line; match the statement.

## The upstream contract it diverges from

At vLLM `v0.11.0`, `vllm/v1/worker/gpu_input_batch.py` carries
`prompt_embeds: Optional[torch.Tensor]` (`:46`) and computes
`num_prompt_tokens = length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds)`
(`:49-50`, helper in `vllm/utils/__init__.py`), with `get_token_id` raising an explicit
"provided via prompt_embeds, and its ID is unknown" error (`:67-72`). The runner gates its
paths on `model_config.enable_prompt_embeds` (`gpu_model_runner.py:212`) and copies the field
into `CachedRequestState` at `:593`.

Prompt-embeds support for the v1 engine is upstream `9a4600e4dc` "[CORE] Prompt Embeddings
Support for v1 Engine (#24278)" (2025-09-18), first tag `v0.11.0rc1`.

## Fix and workaround

- Plugin side: `216fc0e8e` "[feature] Prompt Embeddings Support for v1 Engine (#3026)",
  5 files / 447 insertions — `vllm_ascend/worker/model_runner_v1.py` (+115/-17),
  `vllm_ascend/worker/npu_input_batch.py` (+54), `examples/prompt_embed_inference.py`, an e2e
  test and a workflow entry. First tag `v0.12.0rc1`; **not** in `v0.11.0`, `v0.11.0rc2` or
  `v0.11.0rc3` (`git merge-base --is-ancestor` both ways).
- Therefore the workaround for a 0.11.x report is an upgrade to `>= v0.12.0rc1` with the paired
  vLLM, not a local edit. A one-line patch of `__post_init__` only moves the failure: the
  fix also touches `InputBatch.add_request`-side accounting and the runner's
  `_prepare_inputs`/forward kwargs (that is what the 115-line runner hunk is). Say this when a
  reporter proposes the one-liner.
- Reproduce the fork state in one call each:
  `git grep -c 'prompt_embeds' <tag> -- 'vllm_ascend/*'` (0 before the fix) and
  `git show <tag>:vllm_ascend/worker/npu_input_batch.py | grep -n 'num_prompt_tokens\|prompt_embeds'`.

## Reusable check for any new request-input field

1. Grep the plugin tree at the reported tag for the feature's symbol; zero hits means the fork
   predates it, not that the feature is "unsupported by design".
2. Diff the copied dataclass against the paired upstream file (`gpu_input_batch.py` /
   `gpu_model_runner.py` at the vLLM tag of the same version number) and name the missing field.
3. Bound the fix with `git tag --contains <fix-sha>` and state whether the reported build is
   before or after it; never quote `main`'s support as the reported build's support.

Evidence boundary: read from the two clones at `v0.11.0rc2` / `v0.11.0`. The recorded pass that
found this ended `output_tokens` before writing an answer, so the "upgrade to
`v0.12.0rc1`" recommendation is inferred from the commit's file set and containment, not from a
reporter-confirmed upgrade.
