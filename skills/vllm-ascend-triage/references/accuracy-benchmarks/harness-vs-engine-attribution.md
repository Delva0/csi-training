# Harness vs engine: attribution checklist

## Verified facts in the pinned tree (vllm-ascend `da421afa` = v0.19.1rc1, vllm `b1388b1f`)

- The project's own ais-bench recipe is *sampled*, not greedy: `using_ais_bench.md` shows
  `generation_kwargs = dict(temperature=0.6, top_k=10, top_p=0.95, seed=None,
  repetition_penalty=1.03)` plus `pred_postprocessor=dict(type=extract_non_reasoning_content)`,
  and advises `max_out_len = 32768` with `max_out_len + input < max_model_len`.
- `tools/aisbench.py` rewrites the request config: `temperature=0.6` for `accuracy` tasks and
  `temperature=0, ignore_eos=True` for `performance`; it can set `top_p`, `top_k`, `seed`,
  `repetition_penalty`, and `chat_template_kwargs={"thinking": True}` when `thinking` is on.
- The in-repo accuracy regression is lm_eval based: `tests/e2e/models/configs/*.yaml` +
  `tests/e2e/models/test_lm_eval_correctness.py` with `RTOL = 0.05` (5% relative), tasks like
  `gsm8k` / `ceval-valid`, serve knobs incl. `enable_thinking`. Baselines are per-model yaml values, not MMLU-Pro numbers.
- `docs/source/developer_guide/evaluation/using_evalscope.md` documents only gsm8k; there is
  no evalscope MMLU-Pro baseline anywhere in the tree. So "evalscope vs ais-bench on
  MMLU-Pro" is outside the project's regression coverage — that is a legitimate finding to
  report, and it means the reporter's harness configs are the primary evidence.

## Knobs to align before calling it a defect

1. Thinking mode: on/off per side, and *how* it was switched (see `reasoning-output-routing.md`).
2. Decoding budget: `max_tokens`/`max_out_len` per harness vs the server's `max_model_len`;
   a CoT-heavy model truncated mid-thought can lose the answer entirely.
3. Sampling: temperature/top_p/top_k/repetition_penalty/seed. Greedy vs temp 0.6 changes
   results in both directions; recall that Qwen3-family guidance discourages greedy decoding
   for thinking models — cite it as a hypothesis, not as verified repo fact.
4. Prompt/answer format: official dataset prompt vs harness template; how the letter is
   extracted (first vs last `ANSWER:` match; regex on full text vs post-processed text).
5. Aggregation and coverage: micro vs macro, `--limit`, subset list, retries/failed requests
   scored as wrong.
6. Deployment: same image/model rev, TP/DP/EP, quantization, graph mode (`aclgraph` vs eager),
   `max_model_len`, batch/concurrency.

## Decisive experiments (ask the reporter, or state them as the plan)

- Rerun harness B with harness A's parameters only (one change at a time); the parameter that
  closes the gap is the cause.
- Dump raw requests and responses for both runs and compare on identical prompts: count
  `finish_reason == "length"`, empty/`null` content, missing extracted answer, and repeated
  n-grams per subset.
- Compute the floor: if k of N samples lost their answer, the aggregate damage is ~k/N — check
  whether the observed gap is explained by the measured truncation/extraction failure rate.
- Only after harness effects are excluded, look for engine-side causes (custom-op kernels,
  quantization path, graph mode), and prefer a clue that predicts a *uniform* drop.

## Two images, one command line: matching flags is not matching the numeric path

A report of the form "同样的权重、评测脚本和部署脚本; 新镜像 0.7500, 旧镜像 0.7843,
MindIE 0.7885" is still a *no* to knob 6, because two images execute one command
line differently: a mode or flag the older image does not implement is ignored or
rejected there, and a default the newer image switched on is part of its serve flags
even though it never appears on the command line. At vllm-ascend v0.19.1rc1
(`da421afa`) the asymmetric default is the compile-time graph-fusion stack:

| Fact | Evidence at v0.19.1rc1 |
|---|---|
| `cudagraph_mode: FULL` / `FULL_DECODE_ONLY` turns Npugraph_ex **on by default**; `PIECEWISE` and `NONE` turn it off | `docs/source/user_guide/feature_guide/graph_mode.md:18,115`; `vllm_ascend/platform.py:389-392,411` (FULL branch keeps it on, `:413-415`) |
| that stack also flips three fusion passes on: `fuse_norm_quant=True`, `fuse_qknorm_rope=True`, `fuse_muls_add=True` (`fuse_allreduce_rms` stays `False`, `enable_static_kernel` stays `False`) | `vllm_ascend/ascend_config.py:350-353,383-388` |
| the **narrow** ablation — keep the capture mode, drop only the FX/fusion layer | `--additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'` (`graph_mode.md:159`) |

`cudagraph_mode: PIECEWISE` is the *coarse* ablation: it disables Npugraph_ex too
(`platform.py:392,411`) but also changes capture geometry and the splitting-op list,
so it cannot separate fusions from kernels — use the `enable_npugraph_ex:false` line
first and PIECEWISE only as the second rung.

Order, and the first two cost one launch each: (1) the **noise floor** — the newer
image twice on the same body (`repeated-run-nondeterminism.md`, #221); (2) the
`enable_npugraph_ex:false` rerun on the newer image; (3) model-specific numerics.
Rung 3 is where a session drowns: recorded **#9324** (QwQ-32B, TP4, A3, 0.7500 vs
0.7843, `--reasoning-parser deepseek_r1`, `--max-model-len 131072`,
`cudagraph_capture_sizes [1,8,24,48,60]` + `FULL_DECODE_ONLY`) reached rungs 1-2 only
at turn 17 of 18, after turns 9-15 inside `AscendYaRNRotaryEmbedding`; the answer then
shipped cut at ~2.7 K characters inside its fifth rung. Two hypotheses that walk
retires, so a later session need not re-derive them:

- **`--block-size 128` is not a knob between two vllm-ascend images.** It is the
  platform's own default, and it is forced whenever prefix caching or chunked prefill
  is on (`vllm_ascend/utils.py:1189-1216`); hybrid attention+mamba models return from
  that function even earlier. Say so once and spend the turn elsewhere.
- **The dense-Qwen2 YaRN path is a real divergence with a sub-token effect.** QwQ-32B
  is a dense Qwen2 checkpoint whose published config uses YaRN (factor 4,
  original_max 32768 — the `config.json` is not in the snapshot, so verify it with one
  command before citing). The downstream `AscendYaRNRotaryEmbedding`
  (`vllm_ascend/ops/rotary_embedding.py:253`, wired in at `vllm_ascend/utils.py:662`)
  defaults `truncate=False` (`:270`, with its own "current not support actual truncate"
  comment) where upstream defaults `True`
  (`vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py:31`), and the
  factory forwards the key only when the model config carries it
  (`rotary_embedding/__init__.py:236-278`). That moves the floor/ceil rounding of the
  ramp band — label it a divergence, not the cause of a 3.4-point gap.

## When the older image is the good one

The two-image section above assumes the **newer** image is the suspect and one flag can
bring it back onto the old numeric path. #4730 runs the other way — image request
(Qwen2.5-VL-3B, one card), same weights and the same serve command, the **older** tag
structuring the document correctly and the newer one returning "返回的html结构都错乱" — and
the reporter's public follow-up (2025-12-05) adds that rc1, **rc2 and rc3** all fail and
that they now suspect 精度. Two consequences:

- **No upgrade rung exists.** The reporter is already on the newer side, so "re-run on
  the current image" is answered by construction, and a delta that survives three rcs is
  not a one-release regression that the next rc repairs — do not put either in the first
  message. Say instead that a single-commit pin cannot diff the two tags and give the dump
  below.
- **Enumerate the stack rather than guessing which component moved.** A tag moves `vllm`,
  `vllm-ascend`, `torch_npu`, CANN and `transformers` together, so "the code changed
  between the tags" is not yet a claim about vllm-ascend. One command per image, then
  diff the outputs:

```bash
pip show vllm vllm-ascend torch-npu transformers | grep -E 'Name|Version'
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import vllm_ascend.envs as e; print('USE_OPTIMIZED_MODEL', e.USE_OPTIMIZED_MODEL)"
cat /usr/local/Ascend/driver/version.info
docker inspect --format '{{index .RepoDigests 0}}' quay.io/ascend/vllm-ascend:<tag>
```

  The env line is the one a request cannot show: a **default flipped in the newer image**
  is part of its behaviour without appearing on the command line. At v0.11.0rc3
  `USE_OPTIMIZED_MODEL` defaults to `'1'` (`vllm_ascend/envs.py:96-97`) and selects the
  head-dim-padded Qwen2.5-VL tower (`models/__init__.py:21-34`,
  `models/qwen2_5_vl.py:48-49,221-227`), while the rc0 default is not in any snapshot —
  so the same line run inside the *old* image is what says whether that flip **is** the
  delta.

Then the within-image rows, one restart each on the same body: that env default,
`--enforce-eager`, a text-only request (clean ⇒ the vision path owns it), and — for a
quality rather than a score symptom — the request's own token envelope before any numeric
theory (`finish_reason` plus `usage.prompt_tokens` against the tutorial's `max_pixels`;
#4730 ran 2 038 400 px/image × up to 3 images against `--max-model-len 8192`). A
*structure* regression is also invisible to both instruments CI has at that pin — an
aggregate MMMU row and a "non-empty output" assert — so CI green is not a refutation:
`vl-vision-tower-degeneracy.md` §*Fifth pin*.
