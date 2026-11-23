# LoRA on a VL model: the ViT adapter is dropped upstream, and the quoted shape is not the OOM

## Scope

A LoRA / multi-LoRA serve of a **multimodal** model (`SupportsMultiModal` +
`SupportsLoRA`: Qwen3-VL dense/MoE, Qwen2.5-VL, Qwen2-VL) where the adapter is
said to cover the vision tower — "[Bug]: Qwen3VL 8B LoRA（with ViT）Error", a
verl / ASI RL rollout whose `target_modules` name `visual.*` — and the symptom
is one of: an OOM during multimodal prefill with "请求显存明显超出预估", answers
that are worse than the training rollout, or a hand-picked source line the
reporter says blows up. Not this topic: a text-only LoRA capture crash
(`graph-capture-descriptor-mismatch.md`) or slow LoRA decode
(`decode-overhead-vs-base-model.md`); a VL request rejected on placeholder /
feature counts (`../multimodal-inputs/index.md`).

## Rules

1. **First check is upstream's multimodal-LoRA filter, not the punica kernel.**
   `vllm/lora/models.py:613-624` `_filter_unsupported_mm_module` returns True for
   every module whose name starts with `get_mm_mapping().connector +
   .tower_model`, and the manager then `continue`s at `:489-497` with
   `Regarding multimodal models, vLLM currently only supports adding LoRA to
   language model, %s will be ignored.` For Qwen3-VL the tower prefix is
   `model.visual.` (`vllm/model_executor/models/qwen3_vl.py:1605-1612`). Three
   consequences, and they are the answer: (a) the vision-tower adapter is
   **never applied** at vllm v0.11.0 on any platform, so it cannot crash, OOM or
   alter a token — any mechanism that charges the OOM to ViT LoRA arithmetic is
   refuted before the memory math; (b) the defect it does leave is a
   **training/inference divergence**: an RL policy that trains `visual.*` keeps
   those weights while rollout silently ignores them (accuracy lane, not this
   page; the reporter must be told the adapter is partly inert); (c) the server's
   own log carries the proof, one grep — the warning names the module
   (`… model.visual.blocks.0.attn.qkv will be ignored`). Re-grep the filter at
   the pin: the filter is older than the rc, but the prefix set is model-side.
2. **A tensor shape quoted from a debugger is a missing artefact, not the error.**
   `变量 x.shape [5040, 1, 1152]` (#7419) is not a traceback. 5040 is the patch
   count (`num_patches = grid_t*grid_h*grid_w`, computed from `smart_resize`
   under `vllm/model_executor/models/qwen3_vl.py:615-632`), 1152 is the vision
   hidden default (`:101`), and the middle `1` is the batch axis the Ascend tower
   adds at `vllm_ascend/models/qwen2_5_vl_without_padding.py:452`
   (`hidden_states = hidden_states.unsqueeze(1)`) because its attention feeds
   `torch_npu._npu_flash_attention_unpad` `s b h d` tensors (`:126-140`). Qwen3-VL
   dense and MoE are registered to that file unconditionally
   (`vllm_ascend/models/__init__.py:11-19`), so the report's serve ran it. Two
   uses: an upstream-vs-Ascend shape diff showing `[L, H]` vs `[L, 1, H]` is the
   deliberate pangu axis, **not** a defect; and 5040×1152 bf16 is ~11 MB, so it
   cannot be the OOM owner — the reporter's own "即使这行跳过，后面代码还是会OOM"
   says the device was already full. The producing line ends this lane: do not
   read the attention block or the merger afterwards.
   A `Conv3d` frame under `patch_embed` instead means the **upstream** tower ran:
   the Ascend patch embed replaces the conv with a dense matmul
   (`:349-355`), so no conv3d exists on the Ascend path.
3. **Then the deliverable is the memory budget, decided by numbers.** Ask for the
   allocator's own sentence (`Tried to allocate … GiB` / `NPU out of memory`,
   with its frame), `npu-smi info` sampled **during** the rollout, the rollout's
   `gpu_memory_utilization` / `max_model_len` / `max-num-seqs`, `max_loras` /
   `max_lora_rank` from the adapter config, and whether verl colocates engines
   with sleep mode. Two real levers, neither a mechanism claim: cap image/video
   pixels through the processor's `size["longest_edge"]` (`qwen3_vl.py:624` is
   where it enters `smart_resize`), and lower `gpu_memory_utilization` one step.
   Which of KV pool / weights / adapters / vision activations owns the last
   allocation is what those numbers settle, not local source.

## Delivery

Both decisive facts are one grep each, and the recorded run held them by its 14th
and 19th tool turns and typed **no message text at all** — 20 assistant turns,
45.7 K characters of reasoning, cut at the 16 384-token cap with the answer never
written (#7419). The message "the ViT adapter is ignored at this ref; the shape
is the Ascend tower's batch axis at ~11 MB; here are the four numbers I need" was
writable on turn 4 and owed at G1. Treat the shape as **missing evidence**: state
the gap and ask for the allocator line instead of deriving the mechanism.
