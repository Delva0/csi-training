# DeepSeek-V4 accuracy below the published score: the thinking-mode and effort default

## Scope

A benchmark or eval run scores **below the vendor's published number** on a reasoning-bound dataset
(GPQA-Diamond, AIME, MMLU-Pro) for a DeepSeek-V4 / V4.1 / V4-Flash checkpoint served through the OpenAI
chat endpoint: the service is healthy, single requests answer fluently, tokens/s is normal, no traceback,
and the same engine on a different prompt class looks fine. The usual trigger is an eval harness whose
model config sets `chat_template_kwargs={"thinking": False}` (AISBench and similar) or passes no reasoning
kwarg at all, and the tail of the report is a score comparison (e.g. "GPQA 57–63, official 71"). The
owning artifact is the frontend patch `vllm_ascend/patch/platform/patch_deepseek_v4_thinking.py`, which
rewrites the request's `chat_template_kwargs` and the tokenizer's `apply_chat_template`, i.e. it decides
`thinking_mode` and `reasoning_effort` for every request.

Not this page: an accuracy drop that appears **only** with a parallel/KV feature
(`spec-decode-and-parallelism.md`, `flashcomm1-and-moe-reduce.md`), a quantisation-scheme accuracy
question (`quantization-load.md` §6), a chat-template `ValueError` at render time
(`tokenizer-and-chat-template.md`), a non-DeepSeek model whose thinking is missing or arrives in another
field (`reasoning-parser-output.md`), a malformed answer body (`tool-call-streaming.md`), or a
documented `Known Issues` accuracy delta in the release notes.

## Common rules

- **Read the eval config before any code.** `chat_template_kwargs`, `reasoning_effort`, `max_out_len` and
  the postprocessor (`extract_non_reasoning_content`) are in the report; they decide the mode, and the
  mode decides the score. Kernel, quant and PD questions come later, if at all.
- **The patch is applied whether or not the reporter exported anything.** At `v0.20.2rc1`
  `vllm_ascend/patch/platform/__init__.py:37` imports it unconditionally;
  `VLLM_ASCEND_APPLY_DSV4_PATCH` (`envs.py:116`, default `0`) gates *other* DSV4 patches (`:46`
  `patch_kv_cache_coordinator`, `patch_speculative_config`), not this one — the same "a switch is not
  evidence" rule as `../vllm-ascend/config-flag-lifetimes.md`. From `v0.25.x` the file also installs
  itself only `if not hasattr(deepseek_v4_encoding, "REASONING_EFFORT_PROMPTS")`, so on a newer vLLM the
  patch may be inert — read that condition at the ref rather than assuming either way.
- **Two entry points, either can leave thinking off.** `_patched_build_chat_params` (`:48` at
  `v0.20.2rc1`) injects `enable_thinking` **only** when the request set `reasoning_effort` *and* the
  caller did not already pass `enable_thinking`; the tokenizer wrapper (`:76`) then reads `thinking` /
  `enable_thinking` with `False` defaults. A client that sends neither key, or sends
  `{"thinking": False}`, renders `thinking_mode="chat"`.
- **The patch moves.** Read the numbered `Why`/`How` entry in `vllm_ascend/patch/__init__.py` at the tag
  (`12a` at `v0.20.2rc1`) and the file itself; its line numbers and even its mechanism change between
  releases. `git log --oneline --all -- vllm_ascend/patch/platform/patch_deepseek_v4_thinking.py`
  locates it.
- **The score signature is the discriminator.** A reasoning-bound set falling (GPQA) while a knowledge
  set matches official (MMLU-Pro) is a mode/effort difference, not a quantisation or collective defect —
  a broken kernel or a duplicated all-reduce moves both. Say which set and which delta you compare.
- **The "official" number is a claim too**: it is measured with the checkpoint's documented effort. Quote
  the reporter's own differential if they ran one (they often do: "with thinking high it is right").

## Verified at `v0.20.2rc1` (tag `367b8e6`)

`vllm_ascend/patch/platform/patch_deepseek_v4_thinking.py`:

```text
 :48  _patched_build_chat_params: if self.reasoning_effort is None or "enable_thinking" in user_kwargs: return params
      else chat_template_kwargs["enable_thinking"] = self.reasoning_effort != "none"
 :90  thinking = kwargs.get("thinking", False)
 :91  enable_thinking = kwargs.get("enable_thinking", False)
 :92  thinking = thinking or enable_thinking
 :93  thinking_mode = "thinking" if thinking else "chat"
:101  reasoning_effort = kwargs.get("reasoning_effort")
      "none" -> thinking_mode = "chat", effort None
      "max" | "xhigh" -> "max"
:110  any other string -> "high"          (no validation: "low"/"medium"/a typo all become "high")
```

So **neither key present means chat mode**, and `_patched_build_chat_params` only ever sets
`enable_thinking` from `reasoning_effort`. Upstream vLLM `v0.20.2` `vllm/tokenizers/deepseek_v4.py:34-44`
carries the same `thinking`-defaults-`False` shape and accepts only `"max"`/`"high"` (anything else
becomes `None`), so at this tag the *off-by-default* is shared and the ascend patch adds the
`enable_thinking` OR and the `none`/`xhigh` compression.

## The maintainers later made thinking the default (verified lineage)

| commit | PR | first tag | hunk |
| --- | --- | --- | --- |
| `f22a49096` | #13519 "Align DeepSeek V4 reasoning effort with vLLM" | `v0.25.1rc1` | tokenizer side: `if "thinking" not in kwargs and "enable_thinking" not in kwargs: thinking_enabled = True`; effort `low/minimal/medium → low`, `max → max`, else `high`; UT `test_deepseek_v4_defaults_to_thinking_with_high_effort` asserts the prompt starts `…Reasoning Effort: Absolute maximum` and ends `<｜Assistant｜><think>` |
| `b62ca55a0` | #14317 "Align DeepSeek V4 parser thinking default" | `v0.25.1rc1` | parser side: `DeepSeekV4Parser.__init__` sets `chat_template_kwargs["enable_thinking"] = True` when the request carries neither key |
| `99d2ade34` | #14035 "Align DeepSeek V4 system tool rendering" | `v0.25.1rc1` | tool-rendering half |
| `3fd0e217f` | #13993 "Support DeepSeek V4 0731 reasoning effort" | `v0.26.0rc1` | the 0731 effort values |
| `85625bd77` | #14952 "Select DeepSeek V4 effort mapping by checkpoint config" | `v0.26.0rc1` | effort mapping selected by the checkpoint config |
| `f7f63a922` | #14624 "Align DeepSeek V4 frontend behavior" | `v0.26.0rc1` | frontend alignment |
| `a4440739a` | #10612 | later | deletes the patch as redundant |

`git tag --contains <sha>` is the check. The direction of these commits is the evidence that the old
default was wrong: a bare request (no `thinking`, no `reasoning_effort`) must run thinking with a high
effort, and every older ref runs it in chat mode instead. The whole file is rewritten across that series
(`render_message`, `REASONING_EFFORT_PROMPTS`, `DeepSeekV4Parser` hooks replace the `build_chat_params` +
tokenizer pair), so quote line numbers only beside the tag they were read at.

## A reporter who passes `thinking: False` is not hitting a bug

Their own config disables reasoning; the score delta then follows. Use the arithmetic as the
discriminator: same weights, same engine, a **reasoning-bound** benchmark drops 8–14 points in chat mode
while a knowledge benchmark (MMLU-Pro) stays within ~0.5 of official. The #10413 thread is exactly that
shape — GPQA 57–63 vs 71 official, MMLU-Pro 85.91 vs 86.2, and the reporter's own last comment "I test
<think> HIGH in this GPQA dataset, the accuracy seems right". The deliverable is then label + workaround,
with the missing default as the fileable half — split the two, do not present the mode difference as a
kernel bug.

## Worked case: #10413 — GPQA 57–63 against an official 71

`v0.20.2rc1`, DeepSeek-V4-Flash-w8a8-mtp, A2 4P1D (16 NPU per node, decode `dp_size 32`),
`--max-model-len 135000 --max-num-batched-tokens 4096`, `--enforce-eager`, MTP 1, Mooncake hybrid PD,
`--tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4`, `VLLM_ASCEND_APPLY_DSV4_PATCH=1`. The
failing harness block is AISBench `gpqa_gen_0_shot_str` with `generation_kwargs.chat_template_kwargs =
{"thinking": False}`, `max_out_len=65536`, `pred_postprocessor=extract_non_reasoning_content`.

- **Reporter's own differential (thread, `[reported]`):** `thinking:max` on MMLU-Pro gives 85.91 against
  86.2 official; GPQA stays low in chat mode; "I test <think> HIGH in this GPQA dataset, the accuracy
  seems right". That settles the lever before any code is read.
- **Conclusion:** for this reporter it is not a platform defect — the eval disabled thinking.
- **Falsifier:** with thinking forced on, a still-low GPQA while MMLU-Pro matches moves suspicion to the
  answer postprocessor, then to the W8A8/W4A8 quant path — not to the frontend. A second cheap check:
  send the same prompt twice, once with `chat_template_kwargs={"thinking": true}` and once without, and
  compare — thinking produces a reasoning block before the answer and a visibly longer completion; if
  both are chat mode (or both thinking), the score gap is another class.
- **Status:** open at the time of reading, no maintainer diagnosis in the thread; the class-level defect
  worth filing is the *no-flag* default (a harness that sends no thinking keys silently runs chat mode on
  every tag before `v0.25.1rc1`), which is what `b62ca55a0` changed.

## Workaround ladder

1. **In the eval client, not the server**: `chat_template_kwargs={"thinking": true}` or
   `reasoning_effort: "high"` (equivalently `"max"`). Reporter-confirmed in #10413. Keep
   `extract_non_reasoning_content` postprocessing — it strips the reasoning block and keeps the answer,
   so the extraction stays valid.
2. **Cost of (1)**: outputs get much longer, so `max_out_len` (65536 in the report), `--max-model-len`,
   `--max-num-seqs`/`--max-num-batched-tokens` and the context budget bite; the run becomes decode-bound
   and the score is only comparable to numbers taken in the same mode.
3. **Server-side default**: `>= v0.25.1rc1` (or the release that contains `f22a49096`) makes an omitted
   kwarg mean thinking/high, so harnesses that pass nothing stop silently scoring chat mode. Check the
   tag the reporter can install; do not answer "upgrade" without it.

## Check order

1. Pull `chat_template_kwargs` / `reasoning_effort` / the postprocessor and `max_out_len` out of the
   report's harness block; note whether the reporter sends `thinking: False` explicitly or nothing.
2. Read the patch at the *reported* ref:
   `git show <ref>:vllm_ascend/patch/platform/patch_deepseek_v4_thinking.py | sed -n '40,120p'` plus
   `git grep -n 'patch_deepseek_v4_thinking' <ref> -- vllm_ascend/patch/platform/__init__.py` for the
   import condition.
3. `git log --oneline --all -- vllm_ascend/patch/platform/patch_deepseek_v4_thinking.py` → the alignment
   series above; then `git tag --contains <sha>` for the first tag that fixes the reporter's case
   (`f22a49096`/`b62ca55a0` → `v0.25.1rc1`).
4. Answer with the mode lever, the score-signature discriminator, the harness change (explicit flag, not
   a silent default) and the first tag that aligns the default if the reporter sent no flag.

## Recheck conditions

The file's shape and its import condition change every few releases; the effort strings are
checkpoint-specific from `85625bd77` on, so an `"xhigh"`→`"max"` compression read at `v0.20.2rc1` may not
hold later. Re-grep `thinking = kwargs.get`, `REASONING_EFFORT_PROMPTS` and the parser hook at the
reported tag. No NPU is needed for any of this, so it applies to the reporter's hardware unchanged.
Nothing here was run: no AISBench dataset was executed, and every accuracy statement is the reporter's.
