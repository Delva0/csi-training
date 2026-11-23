# Same body, degenerate answer on the DP replica only (`tp1/dp2` vs `tp2`)

Scope: one checkpoint and one request body, and the answer is **degenerate** —
`content` is only `\n` characters, or one token repeated, or the run scores far
below the same body without DP — under `--data-parallel-size N` with
`--tensor-parallel-size 1`, while the same body on `--tensor-parallel-size N`
(DP off) answers normally. Engine healthy, no traceback, no device text. Worked
case: #6671 (image `v0.14.0rc1-a3`, 910C, Qwen3-30B-A3B-Instruct-2507,
`temperature: 0`, evidence = two chat-UI screenshots).

Not this page — take these splits from the paste before ranking anything:

- the command carries `--enable-expert-parallel` and EP-on garbles while EP-off is
  correct → `ep-mode-reduce-double-count.md` (its EP-on/EP-off rerun is rung 1);
- DP > 1 hangs, or a *tensor* is wrong while the layer output looks sane — the
  pad-target / owner-slice arithmetic →
  `../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`;
- **one** in-flight request stalls at graph replay under DP + ACL graph while
  concurrency is fine → `../hccl-and-hardware/dp-aclgraph-single-request-hang.md`;
- the two runs are two *images* or two releases → `harness-vs-engine-attribution.md`;
- a **score** collapse with EP on, where the reporter's own
  `additional_config.expert_tensor_parallel_size` is the suspect → the
  `expert_tensor_parallel_size`-is-not-a-knob bullet in `index.md` (#1791).

## Step 1 — read the two responses at field level, not the prose

A screenshot pair is enough when it carries the response bodies. Read the
`/workspace/attachments/<issue>/*.ocr.txt` **whole** and take four fields per
side: `content`, `finish_reason`, `completion_tokens`, `prompt_tokens`.

- `finish_reason: "length"` with 199 completion tokens retires "empty answer /
  content filter / client display" — the engine really emitted ids.
- `content` made only of `\n` means **one token id repeated** (a `\n` id is the
  Qwen tokenizer's high-id filler, not `!`'s id 0) — the degenerate-forward
  reading of `constant-token-degeneracy.md` §Step 1, so the id-level check below
  is what turns the screenshot into evidence.
- OCR noise is not the run's parameters: #6671's OCR renders
  `"temperature": 0` as `temperature' ： 9` and the body's `max_tokens: 100` as
  `199`. Annotate the corrupt field; never quote it as a knob the reporter passed
  (and never build a "temperature is nonzero" row on it).

## Step 2 — the layout difference is the control that decides the topic

Both sides of #6671 run the same weights, `--dtype bfloat16`,
`--max-model-len 32768`, `--max-num-batched-tokens 10240`,
`--gpu-memory-utilization 0.95`, `--no-enable-prefix-caching`,
`--enable-auto-tool-choice --tool-call-parser hermes`, `"temperature": 0` and the
same Chinese prompt. They differ in **two** knobs, and they are two rungs, not one
fact:

1. `--tensor-parallel-size 1 --data-parallel-size 2` → `--tensor-parallel-size 2`
   (the DP replica);
2. the failing side alone carries `--compilation-config
   '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,
   16,32,48,64,96,128,160,192,256,512,1024,2048,4096,8192,10240]}'` (a 19-bucket
   capture list).

So model choice, bf16, the Hermes parser, prefix caching and the prompt are out in
one sentence, and the ranking is DP-replica × graph-capture. Say that, then
separate them with Step 4's restarts.

## Step 3 — coverage, not a regression (verified at vllm-ascend v0.14.0rc1 `52d4acfa51fb` + vllm v0.13.0)

Three facts decide what the reporter is entitled to hear, and each is one grep:

- **The model's own tutorial has no DP row.** `docs/source/tutorials/Qwen3-30B-A3B.md:51`
  is `vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 4 --enable_expert_parallel`
  (underscore spelling), with `:48` requiring TP ≥ 2 on a 64 GB card / ≥ 4 on
  32 GB. `tp1/dp2` with no EP is therefore off the documented matrix at this pin.
- **The only `tp1/dp2` + graph e2e asserts non-empty text.**
  `tests/e2e/multicard/2-cards/test_single_request_aclgraph.py` (`DATA_PARALLELS = [2]`,
  models Qwen3-0.6B and DeepSeek-V2-Lite-W8A8) is wired at
  `.github/workflows/_e2e_test.yaml:228`, and its only output assertion is
  `assert choices[0].text, "empty response"` (`:85`). A `\n`-only body passes it,
  so "dp2 + ACL graph is in CI" is a liveness statement, not correctness — quote
  it as coverage, never as a bisect.
- **The DP MoE lane at the pin is the all-gather/reduce-scatter strategy.**
  `AllGatherCommImpl` is the default and "compatible with all scenarios"
  (`vllm_ascend/ops/fused_moe/moe_comm_method.py:222`), whose
  `PrepareAndFinalizeWithAllGather` (`prepare_finalize.py:305`) pads to
  `max_tokens_across_dp` and `dp_group.all_gather`s hidden states and router
  logits (`:422-425`), then `get_dp_group().reduce_scatter(hidden_states, 0)`
  (`:483`). The pad target is a **per-step** value: the CPU all-reduce max of
  `[total_num_scheduled_tokens, with_prefill]` — `torch.max` over the DP group's
  **cpu_group** (`worker/model_runner_v1.py:480`), then
  `ascend_forward_context.py:136-145` and the same block in `platform.py:619-646`.
  With `dp_size == 1` this whole lane is skipped, which is why the TP2 control
  cannot show the symptom.

Reading a per-step host value inside a captured decode graph is a *plausible*
mechanism and nothing more: label it hypothesis, and let Step 4's graph-off rerun
decide it. Do not derive the capture-bucket arithmetic first — the paste's
symptom already separates the two axes more cheaply.

## Step 4 — ladder, cheapest first (each is one restart or none)

1. **No restart: id-level grade on the failing serve.** Same body with
   `"return_token_ids": true` and `"max_tokens": 16, "logprobs": true,
   "top_logprobs": 5`. All-equal top-5 entries, `-inf`, or 16 identical ids ⇒ a
   degenerate forward (owner = the DP lane). A normal distribution with a bad
   answer ⇒ the text/detokenizer layer of
   `constant-token-degeneracy.md` §Step 1, which is a different page.
2. **`--enforce-eager` on the failing command, same body** (proof line
   `Compilation disabled, using eager mode by default`). Removes the
   `FULL_DECODE_ONLY` axis and leaves the DP replica: still `\n` ⇒ DP lane;
   answers ⇒ the captured lane, and the capture list is the next variable.
3. **Flip only DP, keep everything else.** `--data-parallel-size 1` on the same
   `--max-model-len/--max-num-batched-tokens/--compilation-config` (needs one
   card fewer), then `--data-parallel-size 2 --enable-expert-parallel` if the
   cards allow. This is the A/B the accuracy index already asks for on #1791, and
   it is the only axis that separates replica from capture.
4. **The documented recipe instead of the layout**: `--tensor-parallel-size 4
   --enable_expert_parallel` for this weight set, or the minimum TP `:48`
   states. Answering correctly there makes the verdict "unvalidated layout", not
   "broken model".
5. Only if the answer is *still* one repeated token **with EP on** →
   `ep-mode-reduce-double-count.md` §Step 0.

## What to send back

Both serve commands are already in the report — quote them, do not request them.
Ask for: the failing request's raw response JSON (or the `return_token_ids` dump)
at `max_tokens: 16`; the same dump from the `--enforce-eager` run; and the
per-rank log window of the first decode step. Nothing else is needed to place
this case, and a report of this shape usually has no log at all.

## Recorded run, and what it cost

The recorded #6671 session classified the track correctly (accuracy/corruption,
screenshot-only evidence) and read the two OCR files on turn 3, but then took the
`\n` body to the DP-MoE padding page and the DP-ACL-graph hang page — neither
owns a *degenerate answer* — greped the release notes, and re-found the #1791
precedent only at turn 18; it shipped one provisional paragraph and was cut at the
16 384-token cap with no final message. The routing miss was the locate call's
`| head -80`: with 13 all-token members the printout continues past line 80, and
the grades that named `accuracy-benchmarks/index.md` were cut. The four restarts
above are the whole deliverable and need no source read beyond Step 3.
