# A VLM whose answer ignores the image: the request key, the placeholder token, the encoder budget

## Scope

An image/audio/video request reaches a **healthy** engine (weights loaded, KV sized, requests served,
no `vllm_ascend/` frame anywhere) and the model answers as if the image were not there: a caption of
the wrong picture, a generic answer to "what is in this image", or symbol soup immediately after the
`<image>` marker (`Generated text: ca, \n  a A . γ`). Nothing failed and nothing stalled — the
modality never entered the model.

The three members, in the order they are worth checking:

1. **The request dict's key.** vLLM reads exactly one spelling, `multi_modal_data`; a client key that
   differs by a letter is dropped silently (below).
2. **The placeholder.** The prompt must carry the checkpoint's own placeholder token
   (`<image>`, `<|vision_start|>`, `<image_1>`, …) and the model's processor must expand it to the
   image's feature count. A prompt written for another template leaves a literal token that decodes
   from its embedding row.
3. **The budget.** `--limit-mm-per-prompt` and the encoder cache
   (`Encoder cache will be initialized with a budget of N tokens, and profiled with M image items of
   the maximum feature size.`) clip the second image or truncate the features; those log lines are the
   evidence, not a code read.

Not this page: a startup death before the tokenizer/renderer is built (`model-type-support-floor.md`
for the transformers floor, `tokenizer-and-chat-template.md` for the chat-template guard); a quantised
checkpoint scoring below the reference (`quantization-load.md` §7); garbage that only appears in graph
mode or only with a parallel feature (`aclgraph-mode-and-capture.md` §D/§G,
`flashcomm1-and-moe-reduce.md` §B); an embedding/pooling disagreement (`pooling-and-embedding.md`);
"the model is just weak at this task" (`benchmark-accuracy.md`).

## Common rules

- **`Prompt: '…<image>…'` in the output proves nothing.** `LLM.generate` echoes the prompt string it
  was given. Count instead: the response's `prompt_tokens`/`usage.prompt_tokens` against the prompt's
  text length (an image adds the processor's expansion — 576 tokens for `llava-1.5-7b-hf`), the
  engine's encoder-cache line, and the multimodal-input count at `VLLM_LOGGING_LEVEL=DEBUG`. The
  cheapest falsifier is `len(prompt_token_ids)` against the image's placeholder count: a text-only run
  is short by the whole image.
- **The image is harvested from one key and the rest of the dict is never validated.** At vLLM
  `v0.9.2` the dict path is `vllm/inputs/parse.py:118-133 parse_singleton_prompt` — the `"prompt"`
  branch returns the **whole dict** as `parsed_content` (a `TypedDict` is a static hint; nothing checks
  keys at runtime) — and the only consumer is
  `vllm/inputs/preprocess.py:418 _process_text` (text prompts) or `:357 _process_tokens`:
  `if multi_modal_data := parsed_content.get("multi_modal_data"):` → `_process_multimodal(...)`,
  `else:` → `_tokenize_prompt(...)` + `token_inputs(...)`. So `multi_model_data`, `multi_modal`,
  `mm_data`, `images` all select the **text-only** branch, and the model decodes the placeholder token
  from its embedding row. The V0 beam path repeats the same key test at `vllm/entrypoints/llm.py:625-626`.
  `limit_mm_per_prompt={"image": 1}` on the `LLM(...)` call does not help: it sizes the encoder cache, it
  does not inspect requests.
- **Read the reporter's own script before any code window.** The misspelling is visible in the body and
  the check is free: `grep -n 'multi_modal\|"prompt"' <the pasted script>`. Chat clients are the mirror
  case — there the field is a content part (`{"type": "image_url", "image_url": {"url": …}}`) parsed by
  upstream's `parse_chat_messages`, and a base64 string pasted into `content` of `/v1/completions` has
  no path at all. Do **not** then hunt upstream for the function that "swallowed" the key: the prompt is
  a `TypedDict`, so no runtime validation exists to blame and the file/line does not change the fix.
  Recorded wrong turn: a run had that thread in batch 1 and spent five more batches on the exact drop
  site, then shipped no reply.
- **The rule survives refactors, the line does not.** The key is still `multi_modal_data` at
  `v0.11.0`/`v0.13.0` (`vllm/inputs/preprocess.py`), and by `v0.19.1` it is read in `vllm/inputs/llm.py`
  + `vllm/renderers/inputs/preprocess.py`, where the data is assembled by the V1 renderer /
  `ProcessorInputs`. Before quoting, `git grep -n 'multi_modal_data' <ref> -- vllm/inputs vllm/renderers
  vllm/multimodal`, and treat the reporter's `path:line` as the fingerprint of their build rather than
  reusing `:418`.
- **Symbol soup right after the marker is the same defect, not a device fault.** A single placeholder
  token with no expansion carries no vision context; a `. γ`-style continuation is the language model
  free-running from one meaningless embedding. Do not route it to `device-kernel-faults.md`.

## Worked report: #2040 (`0.9.2rc1`, Llava-1.5-7b-hf, V0 offline, A2 910B)

- Environment: vLLM `0.9.2`, `vLLM Ascend 0.9.2rc2.dev0+gb5b7e0e`, torch-npu `2.5.1.post1`, CANN
  `8.1.RC1`, transformers `4.52.4`, 16×910B2C; run as `VLLM_USE_V1=0 python3 llava_test.py`; the
  `LLM(...)` call sets `max_model_len=4096, limit_mm_per_prompt={"image": 1}`; the log carries
  `WARNING: Failed to register custom ops, all custom ops will be disabled` (not the cause).
- Symptom: `Prompt: 'USER: <image>\nWrite a long descriptive caption for this image in a formal tone.
  ASSISTANT:', Generated text: ca, \n  a A . γ`.
- The body's script: `inputs = {"prompt": prompt, "multi_model_data": {"image": image}}` — one letter
  away from the key vLLM reads.
- Thread: closed; **one** comment, `wm901115nwpu` (CONTRIBUTOR, 2026-03-07) 「测试脚本有问题」 with a
  corrected script (`"multi_modal_data"`) whose log produces the correct caption
  (`The image captures a woman playing tennis…`). That run is on vLLM `v0.14.1` (V1 path), i.e. the
  lever is confirmed on a later stack than the reporter's — say so.
- Mechanism (`verified` at vLLM `v0.9.2`): the misspelled key never reaches
  `_process_multimodal`, so `preprocess.py:418` tokenizes the text and the `<image>` token
  (`image_token_index=32000` in the checkpoint's `config.json`, which is also its pad id) is fed as an ordinary token.
  `limit_mm_per_prompt` only sized the encoder cache, so nothing warned.
- Workaround (**contributor-confirmed**, same weights): spell the key `multi_modal_data`. Cost: none.
  When the caller cannot change the field, go through the chat endpoint / `LLM.chat()` with an
  `image_url` part — the dict is then built upstream-side.
- Falsifier and check for the reply: after the fix `usage.prompt_tokens` grows by the placeholder
  expansion (576 for llava-1.5) and the encoder-cache line becomes non-zero; if tokens grow and the
  caption is still wrong, the key was not the defect and the class is member 3 (budget) or a
  template/processor mismatch.
- Status and limits: no vllm-ascend code is involved and nothing was fixed; the dict handling is
  unchanged in later releases, so the class recurs with every new client. Not verified: whether the
  reporter's `0.9.2rc2.dev0` build would have raised on an unknown key — at `v0.9.2` it does not.

## Answer shape

symptom → which member (key / placeholder / budget) and the line that decides it → workaround for
*their* call site with its cost → root cause `path:line` at the reported vLLM ref plus the mechanism
("the only reader is `_process_text`'s `multi_modal_data` lookup, and the rest of the dict is
unvalidated") → the check that proves it (prompt-token growth, encoder-cache line) → status (usually
"not a plugin defect, no code change"). Quote the reporter's own script line as the evidence — it is
in the body and settles it without a clone.
