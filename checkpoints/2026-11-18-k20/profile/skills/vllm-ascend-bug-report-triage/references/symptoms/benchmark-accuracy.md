# Benchmark accuracy: below the model's own number, or two eval tools disagree

## Scope

The service is healthy — no crash, no stall, no traceback — and the **score** is wrong: an lm-eval /
evalscope / ais-bench run (gsm8k, mmlu-pro, GPQA, AIME) lands 5–90 points away from the number the
reporter expects. Two shapes, and the evidence order differs:

- **tool vs tool** — the same model and dataset score X on one harness and X−6 on another (#10068).
  Usually a measurement difference: prompt rendering, default sampling, extraction metric.
- **one harness, one number, a config delta** — the reporter's own control (tp=4 vs tp=2+dp=2) separates
  collapse from fine. Then the question is which parallel/config key actually ran, and whether the MoE
  collective under it is DP-aware (#1791 → #1289).
- **an offline script with image/audio input that answers garbage** — every prompt is wrong, no crash and
  no metric to compare, and the delta tracks the *request* rather than the model (§4, #2040); the same
  split applies to a served endpoint whose client drops the content part.

Not this page: fluent garbage or repetition that a parallel/KV feature produces
(`spec-decode-and-parallelism.md` §2, `flashcomm1-and-moe-reduce.md` §B), a DeepSeek-V4 that lost its
thinking mode (`deepseek-v4-thinking-and-effort.md`), a quantised variant the ref's tutorial never lists
(`quantization-load.md` §7), an OOM or refusal after copying a serve block (`startup-and-stall.md` §3b),
a PD/multi-node freeze (`pd-collective-hang.md`) or a documented known-issue regression (`release_notes.md`
at the ref).

## Common rules

- **The comparison that decides anything is harness vs the model's official number, not harness vs
  harness.** A maintainer relayed the ais-bench team's answer verbatim (#10068): 「it is meaningless to
  compare the accuracy of tools with each other」 — align each tool with the model card, using the
  dataset's own alignment recipe (ais-bench keeps one at
  `ais-bench-benchmark-rf.readthedocs.io/en/latest/best_practices/replicate_llm_datasets_accuracy.html`,
  evalscope's per-benchmark config at `evalscope/benchmarks/_meta/<benchmark>.json`). So the answer is
  never "tool A is broken", and not a platform defect either: it is the delta that makes the two numbers
  commensurable, plus the request path where the difference lives.
- **The three named causes of a tool-vs-tool gap, in order**: (1) `/v1/chat/completions` vs
  `/v1/completions` — chat template and default system prompt included; (2) default sampling parameters;
  (3) the evaluation metric / answer extraction. Every recorded report of this shape is one of those three
  or a `max_tokens` truncation that hides the answer line. A fourth, when both tools use the chat path: the
  few-shot rendering (`--apply_chat_template` below).
- **A sampling parameter the request omits is not greedy.** With `--generation-config auto` (vLLM's
  default: `ModelConfig.generation_config: str = "auto"`, `vllm/config/model.py:267`; `get_diff_sampling_param`
  returns `{}` only for `"vllm"` at `:1391`/`:1408`, at `v0.19.1`) the server takes the checkpoint's own
  `generation_config.json` values for every sampler field the request leaves unset:
  `get_diff_sampling_param()` becomes `self.default_sampling_params` in each OpenAI entrypoint
  (`entrypoints/openai/chat_completion/serving.py:141`, `completion/serving.py:74`) and is consulted per
  field in `chat_completion/protocol.py:408-449`. A harness that pins `temperature=0` and one that sends
  nothing therefore run **different samplers against the same server**; on a 10-way multiple-choice set
  that alone is a multi-point gap. Read `generation_config.json` in the served directory, and remove the
  variable with `--generation-config vllm` or explicit values in both tools.
- **The harness's own meta file is the prompt contract — read it at the harness version they ran.**
  evalscope's `evalscope/benchmarks/_meta/mmlu_pro.json` (main, fetched 2026-11-17) sets `few_shot_num: 5`,
  `eval_split: test`, an **empty** `system_prompt`, and a `prompt_template` that requires the last line
  `ANSWER: [LETTER]` ("Think step by step before answering"), aggregated by mean over 14 subjects. A
  `max_tokens` / `max_out_len` that truncates the chain of thought deletes that terminal line, and a
  correct model then scores as wrong — the classic constant 5–6 point deficit. The cheapest read of a
  third-party file is `github` `/repos/<owner>/<repo>/contents/<path>` with
  `accept: application/vnd.github.raw+json` (`../workflow/evidence-sources.md`); neither harness is
  cloned here, so `web_fetch` on `raw.githubusercontent.com` (a recorded 120 s timeout) is the wrong tool.
- **Ask for the same prompt's raw generation from both harnesses, not another benchmark run.** Five raw
  outputs (`--log_samples` in lm-eval, the prediction field in evalscope) split the two causes a score
  cannot: fluent-but-unparsed text is the harness/prompt; repetition (`the the the`, `7777…`, `, and, and`)
  is a real reduction fault. One curl per tool shows whether the difference is the rendered prompt, the
  sampling, or the extraction.
- **A collapse to 0.15–0.17 on five-shot gsm8k is not noise.** Five-shot gsm8k with a working model is
  ≈0.86–0.89 in these very threads; a tenfold drop means the few-shot turns never reached the model's chat
  template, or the model emitted garbage. Route on that split before reading code.
- **`--apply_chat_template` + `--fewshot_as_multiturn` change what is sent.** On Qwen3 these two lm-eval
  flags render the few-shot examples as chat turns and the thinking-mode template applies; removing both
  restored gsm8k in #1791 (`leo-pony`, reporter-confirmed on `v0.11.0rc0`, `reported`). Quote the lever and
  state the confound: a TP4/DP1 run **with the same two flags** scored 0.884 in that thread, so the flags
  do not explain a TP2/DP2-only failure.
- **A config key in the failing command is not a key that ran** (`../vllm-ascend/config-flag-lifetimes.md`,
  Trap D). `platform.py` rewrites `parallel_config` arithmetic before the worker sees it — §2 is the
  measured example, and the same reading applies to any `additional_config` parallelism key.
- **Do not build a code-level mechanism from a harness delta.** No clone contains the reporter's harness.
  One ref-pinned fact (the gate, the default, the group algebra) plus the checklist and the differential is
  the deliverable; the rest is `hypothesis` with its falsifier.

## 1. Two harnesses disagree: evalscope 5–6 points below ais-bench (#10068)

Report: vLLM `0.19.1` + vllm-ascend `0.19.1rc1`, `transformers 5.5.3`, `torch_npu 2.9.0`, CANN `8.5.1`,
8×910B3 on a Kunpeng-920 host, MMLU-Pro; the same image scores 5–6 points lower on evalscope than on
ais-bench, and ais-bench matches the model's official number (screenshots only — the report quotes no
score table). Closed.

Thread, and the whole answer: `MengqingCao` 2026-06-05 「It seems not an issue of vllm-ascend?」;
`linfeng-yuan` 2026-06-08 relays the ais-bench developers plus the three causes above, the alignment-doc
link and a pointer to evalscope's `mmlu_pro.json`. No commit, no code window, nothing to look up in
`vllm_ascend/`.

- **Deliverable**: the alignment doc's recipe for the dataset, the two curl probes, and the request-shape
  checklist in §3 — labelled *reported* (maintainer + the upstream tool's team), with the "which of the
  three it is in your run" left as a measurement the reporter makes.
- **Do not hunt a plugin defect for this shape.** Recorded wrong turns: reasoning-parser field names,
  `--reasoning-parser` moving thinking out of `content`, Evalscope reading `message.content`. All real
  classes, none localisable from a score delta. `git grep -n enable_thinking <ascend-ref> -- vllm_ascend`
  returning nothing (it does at `v0.19.1rc1`) is not evidence either way.
- **No clone of either harness exists in the workspace** — `ModelEngine-Group/` holds only
  `unified-cache-management`, and there is no `evalscope` checkout. Read a foreign harness file with the
  `github` tool (`/repos/modelscope/evalscope/contents/…`, `accept: application/vnd.github.raw+json`),
  not `web_fetch` on `raw.githubusercontent.com`.

## 2. One harness, a collapse that tracks a parallel dimension (#1791 → #1289)

Report: vLLM `0.9.2` + vllm-ascend `v0.9.2rc1`, Qwen3-30B-A3B, lm-eval `gsm8k` 5-shot with
`--apply_chat_template --fewshot_as_multiturn`, `tensor_parallel_size 2`, `data_parallel_size 2`,
`enable_expert_parallel`, `additional_config.expert_tensor_parallel_size 2`, `enforce_eager`:
`exact_match` 0.1653 (flexible) / 0.1547 (strict). Closed as `duplicated` of **#1289** (「Qwen3-30B-A3B
Shows Precision Issues in DP2+TP2 Parallel Mode」, whose fingerprint is DP rank 0 emitting `the the the
the …` while rank 1 is coherent).

### The key the reporter set is not the key that ran (verified, `v0.9.2rc1`)

At `v0.9.2rc1` — the last tag that parses the key — `additional_config.expert_tensor_parallel_size` is read
into `AscendConfig` (`vllm_ascend/ascend_config.py:48-49`) and then overridden by the platform in
`check_and_update_config`:

```python
# vllm_ascend/platform.py at v0.9.2rc1
parallel_config.expert_tensor_parallel_size = parallel_config.tensor_parallel_size   # :135
if parallel_config.enable_expert_parallel:
    parallel_config.expert_tensor_parallel_size = 1                                   # :140
elif ascend_config.expert_tensor_parallel_size > 0:                                   # :143
    parallel_config.expert_tensor_parallel_size = ascend_config.expert_tensor_parallel_size
parallel_config.expert_parallel_size = (
    parallel_config.world_size_across_dp // parallel_config.expert_tensor_parallel_size)  # :147-149
```

So with `--enable-expert-parallel` **on** — which both the failing and the "fixed" commands have — the
`additional_config` value is never read, and `ep_size = world_size_across_dp`.
`vllm_ascend/distributed/parallel_state.py:28-65` at the same tag then builds the EP group as
`range(i, world_size, expert_tensor_parallel_size)` over the **whole world** (DP replicas included) and the
ETP group as contiguous blocks of `expert_tensor_parallel_size` ranks. Consequence: at tp=2/dp=2 the single
EP group spans both DP replicas, and the routed-expert reduction is only correct if the MoE code carries
DP-aware logic. That is exactly what `ttanzhiqiang` says in #1289: 「the all_gather solution of qwen3's moe
model must follow the dp logic of `fused_moe.py`, but it currently follows the `common_fused_moe.py` logic.
There is no dp logic here, so the lack of dp communication leads to incorrect accuracy」 — a maintainer-side
diagnosis of the class (`common_fused_moe.py:42` `forward_oot` has no DP logic at that tag; `fused_moe.py`
is the file with it), still `reported`, not a verified hunk. The key was removed outright by `8cfd25799`
"[Dist][EP] Remove ETP/EP maintained in vllm-ascend" (#1681), first tag `v0.10.0rc1`, so on later tags it
is an unknown `additional_config` entry that is never read
(`../vllm-ascend/config-flag-lifetimes.md`, Trap D).

### What the thread confirms, and what it does not

- `zhangxinyuehfad` 2025-07-14: **tp=4, no dp** → 0.8840 / 0.8704. `ApsarasX`: refer to #1289.
- `leo-pony` 2025-10-10: 「The issue lies with the parameters `--apply_chat_template` and
  `--fewshot_as_multiturn`. Removing these two parameters resolves the problem in vllm-ascend
  v0.11.0rc0」 → tp=4 **dp=2**, etp=4, no chat-template flags: 0.8590 / 0.8969, and asked to close.
- **Two variables moved at once** (release `0.9.2rc1` → `0.11.0rc0`, and the lm_eval prompt flags), and
  the tp=2+dp=2 configuration was never re-run without the chat-template flags. So the thread's confirmed
  delta is the *lm_eval* prompt shape — a chat-template `apply_chat_template` few-shot rendering — not a
  proof that DP+EP was fine.
- Fix candidates named in #1289 are unverified: PR #1441 「试了没有解决」 (tried, did not fix), #1678
  suggested but unmapped to #1791, #1381 is the aclgraph dp=1 half. No commit was ancestry-checked against
  a tag for this report. Status: closed, no code fix named.

### Differential to ask for (and the falsifier)

- Same lm_eval invocation with `--apply_chat_template --fewshot_as_multiturn` **removed** (that is the one
  reporter-side lever with a measured number behind it), then
- tp=2/dp=1 vs tp=2/dp=2 with everything else equal: if only the dp=2 run collapses, the class is the DP
  dimension; if both collapse, it is the prompt/sampling path and §3 applies.
- Print the effective algebra once: `expert_tensor_parallel_size` and `expert_parallel_size` after
  `check_and_update_config` — at this tag they are `1` and `tp × dp`, whatever the `additional_config`
  says.
- What ships is the flag lever (`reported`), the DP>1 MoE all_gather statement as a maintainer diagnosis
  (`hypothesis` with the DP=2-only differential as its test), and the ETP fact (`verified` at the tag).

## 3. Request-side checklist before calling it an engine defect

| what to check | why it changes the score | where |
| --- | --- | --- |
| chat template applied or raw completion | a system prompt, a thinking-mode wrapper, or a different few-shot rendering changes the answer distribution; lm-eval's `--apply_chat_template`/`--fewshot_as_multiturn` are exactly this | the harness's model adapter; the model's `chat_template.jinja` at the tag |
| sampling the client actually sends | an omitted `temperature`/`top_p` is filled from the checkpoint's `generation_config.json` under `--generation-config auto` — so greedy in one tool and `temperature 0.6/top_p 0.95` in the other is a **default**, not a bug (rule and line references above) | the paired vLLM tag, not the fork |
| `max_tokens` vs the answer format | a truncated thinking/reasoning block never reaches the answer line, so extraction scores it wrong | the harness's per-benchmark config (`_meta/<bench>.json`) |
| extraction / metric | `flexible-extract` vs `strict-match`, a regex for 「answer is (X)」 vs a letter logprob | the harness's metric code; two filters disagree by several points on the same generations |
| parallelism axes | which `tp`/`dp`/`ep`/`etp` values ran, after `platform.py` rewrote them (§2) | the log's own banner + `platform.py` at the tag |
| multimodal data on the prompt | the dict key must be spelled `multi_modal_data`; an unknown key is dropped without validation, so the image never reaches the model and the placeholder token embeds as text (§4) | the paired vLLM tag's `parse.py`/`preprocess.py`; the caller's own script |

`../workflow/answer-and-claims.md` ("Wording that keeps you honest") governs the label: a
maintainer's 「not an issue of vllm-ascend」 plus the harness team's alignment doc is *reported*; the
reporter's next run is what settles it.

## 4. Image input answers garbage: the multimodal data never reached the model (#2040)

An **offline script with image/audio input** (or a served endpoint whose client drops the content part)
answers every prompt wrongly, with no crash and no metric to compare — so the delta tracks the *request*,
not the model, and the three members are the caller's dict key, the placeholder token and the encoder
budget. That is its own class page, `multimodal-input-not-applied.md`; this section only fixes the place
in the accuracy order.

- Worked report: vLLM `0.9.2` + vllm-ascend `0.9.2rc1` (`VLLM_USE_V1=0`), Llava-1.5-7b-hf, offline
  `LLM.generate` with one image, sample output `ca,  a A . γ`. The reporter's key was
  `multi_model_data` and the thread's own 测试脚本有问题 (with a corrected script) is the lever.
- The one ref-pinned fact: vLLM reads only `multi_modal_data` (`vllm/inputs/preprocess.py:418
  _process_text`, `:357 _process_tokens`, at `v0.9.2`), the rest of the prompt dict is never validated,
  so a misspelled key is dropped silently and the placeholder token embeds as text.
- **Keep the confound in the reply**: the contributor's control ran on vLLM `0.14.1` with `V1` +
  `PIECEWISE` in its banner, not on the reporter's `0.9.2`/V0, so it confirms the *input shape*, not a
  version delta.
- Do not spend batches on the exact drop site — the class page carries the mechanism and the check
  (`usage.prompt_tokens` growth, `len(prompt_token_ids)`, the encoder-cache line).

## Order of work

1. Read the last comment and the report's tail for a lever (a flag pair, "use tool X", "compare to
   official").
2. Read the harness meta/prompt file at the version the reporter ran, then ask for ~5 raw generations.
3. Line up the request path: endpoint, template flags, explicit sampler values, `max_tokens` /
   `max_out_len`, answer extraction, few-shot count.
4. Only then the platform: DP>1 MoE (`fused_moe` vs `common_fused_moe`, the ETP branch), a quantised
   variant the tutorial does not list, or the model family's thinking default.
5. Reply: lever (attributed) + commensurability checklist + the one ref-pinned fact + `status and limits`
   naming what was not verified.

## Recheck conditions

- `get_diff_sampling_param` / `generation_config`: re-read at the paired vLLM tag — the field and the merge
  site move (`vllm/config/model.py`, `entrypoints/openai/*/serving.py`).
- The ETP branch exists only through `v0.9.2rc1`; from `v0.10.0rc1` the parallel config comes from vLLM's
  own `ParallelConfig`, so read there instead of `vllm_ascend/platform.py`. `platform.py`'s ETP/EP algebra,
  `init_ascend_model_parallel` and the MC2/ALLGATHER choice all move between releases
  (`moe-ep-dispatch.md` for the collective half).
- evalscope / ais-bench metas are third-party files: pin the version they ran and grep for `few_shot_num`,
  `prompt_template`, `max_out_len` instead of trusting this page's snapshot; harness defaults change with
  every release of the harness.
