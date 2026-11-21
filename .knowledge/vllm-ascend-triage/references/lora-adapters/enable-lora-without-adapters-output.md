# `--enable-lora` changes the base model's answers with no adapter loaded

## Scope

The deployment starts with `--enable-lora` and **no `--lora-modules`** (or with one
probe adapter), the engine serves, and the *answers* are wrong while everything works
with the flag removed: wrong stop token (`stop_reason: 151643` = `<|endoftext|>` where
Qwen3 should stop on `151645` = `<|im_end|>`), `reasoning_content: null` with raw
`<think>` left inside `message.content`, word/phrase repetition (`重要重要`,
`性质的性质`), and premature truncation in both streaming and non-streaming mode.
Recorded #8863 (vllm-ascend v0.14.0rc1 @`52d4acfa`, vllm v0.14.1 @`d7de043d`,
Qwen3-32B, 2×910B3, `--enforce-eager`).

This is the **output-correctness** face of the LoRA track. The crash faces are in the
parent `index.md` (capture descriptor, fusion `AttributeError`, punica compile guard,
MoE gates); the slow face is `decode-overhead-vs-base-model.md`.

## Rules

1. **The reporter's control is the whole attribution: adapter data is not implicated.**
   Same command minus `--enable-lora` answers correctly, and the failure reproduces with
   the flag and **zero** adapters, so what is under test is the code path the flag
   installs at model load, for every request. Do not ask for adapter files, `r`,
   `target_modules` or a merge export.
2. **Collapse the symptoms into one, and say which kind of stream is wrong.** A
   `stop_reason` id is a **sampled token id**: `151643` means the model emitted eos, not
   that the frontend mis-picked a stop string; `reasoning_content: null` with `<think>`
   in `content` means the stream never carried the closing think token the parser splits
   on; repetition and truncation are the same degeneration. So the report is *one*
   token-stream defect, not four frontend bugs — and the discriminator is token-level.
3. **Verified at the pin: what the flag installs, and why the adapter *math* cannot be
   the corruptor when there are no adapters.**
   - `vllm/lora/layers/logits_processor.py:180-185` calls
     `punica_wrapper.add_lora_logits(...)` on **every** `_get_logits`, and every target
     linear/embedding was replaced in place at load (`vllm/lora/models.py:474-502`).
     `vllm_ascend` does not override `can_update_inplace`, so upstream's `True`
     (`vllm/platforms/interface.py:501`) applies and the in-place mutation is the result.
   - `vllm_ascend/lora/punica_npu.py:23` re-registers six Ascend `*WithLoRA` classes
     (`vllm_ascend/lora/utils.py:98-104`), and `:24-27` selects the torch fallback
     **only** for 310P or `max_lora_rank >= 128` — a default-rank 910B serve is on the
     AscendC path (`vllm_ascend/lora/lora_ops.py`).
   - The AscendC expand **always accumulates**: `csrc/kernels/bgmv_expand.cpp:225`
     `Add(yLocal, yLocal, yInLocalFP32, …)`, with `yIn`/`yOut` the same tensor
     (`csrc/torch_binding.cpp:455`). With no adapter the stacked `A`/`B` are zeros, so
     the add is `+0` and the amplitudes are unchanged — **the wrapper arithmetic is
     inert for this report**, which is the fact that redirects the investigation.
   - The prefill halves are skipped outright with no adapter
     (`punica_npu.py:60,87,116` `if self.no_lora: return`), while `add_lora_logits`
     (`:318-357`) and the decode halves are not guarded — so on a prefill-only step the
     LoRA path that actually runs is the logits one.
4. **Two candidates are already refuted; do not spend budget on them.**
   - *The dropped `add_inputs`.* `lora_ops.py:35-50` accepts `add_inputs` and never
     forwards it (the registered op is
     `bgmv_expand(x, weight, indices, y, slice_offset, slice_size)`,
     `csrc/torch_binding.cpp:439-441`). It reads like an accumulate-vs-overwrite
     divergence, but rule 3's `Add` makes the kernel accumulate unconditionally, and
     every in-tree caller wants accumulate (`add_lora_logits` passes `add_inputs=True`,
     `punica_npu.py:355`). Inert here — check callers and the kernel body before ranking
     a parameter drop.
   - *The `None` return / meta stub.* `add_lora_logits` returns `None`, which matters
     only when `can_update_inplace()` is `False`; and
     `csrc/torch_binding_meta.cpp:72-75` returns `at::empty_like(y)` where the real op
     returns the mutated `y`, so a FakeTensor trace would see uninitialised logits.
     Both are unreachable on Ascend under `True`.
5. **What is left, in rank order.** (a) The module *replacement* itself: the base math
   now runs through `*WithLoRA.forward` → `base_layer`, so a fused/quantized linear
   path, a `quant_method` or a dtype cast that the unwrapped class had can be lost —
   diff one replaced module's forward against its non-LoRA twin (`rg -n 'base_layer'
   vllm_ascend/` decides whether any downstream class is LoRA-aware). (b) The logits
   bookkeeping: `_get_logits` slices the padded vocab at
   `logits_processor.py:188`, whose padding comes from the platform hook
   (`vllm/platforms/interface.py:508`, not overridden downstream — one `rg
   'get_lora_vocab_padding_size'` closes or opens that lane), and both special tokens in
   this report sit at the very end of the vocab where a sharded/padded gather misaligns
   first. (c) A post-pin fix, invisible in the snapshot.
6. **The discriminator is a token comparison, not a text one.** One prompt served twice,
   identical except the flag, with `"temperature": 0` and `"return_token_ids": true`:
   the first N ids settle it. Same ids and different text → the frontend/detokenizer
   owns it (stop-token set, reasoning parser, streaming deltas:
   `../serving-frontend-faults/index.md`, `../tool-call-streaming/index.md`). Different
   ids → the forward/logits lanes of rule 5.
7. **Keep the graph and the flag apart.** `--enforce-eager` in the recorded repro
   removes capture from the question; a report without it first needs the `has_lora`
   capture-dispatch row (`graph-capture-descriptor-mismatch.md`), which is a *crash*, not
   a corruption.
8. **Version scope.** The pin is the release the reporter ran, and a fix landing after it
   is invisible here. If a later rc/final exists at triage time the one-line re-run of
   the same body on it is rung 1, with the mechanism below it left open; if the pin *is*
   the current release, say the defect has to be answered from it.

## Answer shape

Verdict first: `the flag, not the adapter, is the trigger — enabling LoRA replaces the
target modules and installs the logits LoRA path for every request, while the adapter
arithmetic itself is provably a no-op with no adapter loaded`. Then the one token-level
check of rule 6 as the cheapest action, the refuted candidates of rule 4 so the reporter
does not re-walk them, the two remaining owner lanes, and the version rung. Nothing here
justifies shipping a patch before that token comparison: a wrapper-level change that
makes the text look right is not a fix.
