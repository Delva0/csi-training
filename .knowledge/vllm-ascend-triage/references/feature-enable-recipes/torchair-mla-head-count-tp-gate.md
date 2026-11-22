# MLA graph mode head-count gate: `queries per kv … only support {32, 64, 128}`

## Scope

A DeepSeek-family MLA serve with `additional_config={"torchair_graph_config":
{"enabled": true}}` dies **while the layers are constructed**, i.e. before any
device op, on this assertion:

```text
  File "vllm_ascend/models/deepseek_v2.py", line 430, in __init__
    self.mla_attn = Attention(
  File "vllm_ascend/attention/mla_v1.py", line 575, in __init__
AssertionError: The allowed number of queries per kv when enabling both MLA and Graph
mode only support {32, 64, 128}, Thus this is not supported for DeepSeek-V2-Lite, as it
only has 16 attention heads. And if you're using DeepSeek-V3 or DeepSeek-R1, please make
sure after the tensor parallel split, num_heads / num_kv_heads in {32, 64, 128}.
```

Runtime twin, when the Python gate is absent: `EZ9999: Inner Error! …
numHeads / numKvHeads = N, MLA only support {32, 64, 128}.[FUNC:CheckMlaAttrs]
[FILE:incre_flash_attention_tiling_check.cc][LINE:1218]`.

Not this page: the first-request `UnboundLocalError: 'decode_hs_or_q_c'`
(`torchair-mla-graph-and-chunked-prefill.md`), a capture-time dynamo failure
(`../startup-failures/torchair-dynamo-compile-unsupported.md`), or the ACL-graph
"does not support deepseek" refusal, which is one grep and appears below.

## Verdict first — the paste carries the rule

Say this in the first paragraph: it is **heads-per-rank arithmetic**, not a
V2-Lite-only defect, not the checkpoint, and not a kernel bug. The MLA layer
hardcodes `num_kv_heads=1` and is handed `num_local_heads = num_heads // tp_size`
(`vllm_ascend/models/deepseek_v2.py:359-362`, `:430-434` at v0.9.1rc1), so
`num_queries_per_kv` **is** the heads on that rank. For a 128-head model
(DeepSeek-V3 / R1): TP ∈ {1, 2, 4} pass (128/64/32) and **TP=8 → 16 → refused**;
V2-Lite's 16 heads fail at every TP. Compute it from the **serve command, not the
title** — recorded #1451's title says V2-Lite while the command is
`--tensor-parallel-size 8 --pipeline-parallel-size 4` on DeepSeek-V3, which is the
same ratio reached a different way.

## Levers, cheapest first

1. **Give up graph mode, keep the TP layout:** add `--enforce-eager` **and delete**
   the `torchair_graph_config` key — the two are mutually exclusive
   (`vllm_ascend/ascend_config.py:129-140`, `RuntimeError: Can't enable graph mode
   and eager mode at the same time`). `ascend_scheduler_config` may stay. Eager
   decode takes `torch_npu._npu_paged_attention_mla` (`mla_v1.py:1007`) instead of
   the graph path's `torch_npu.npu_fused_infer_attention_score` (`:988`). If the
   reporter's own attempt was `export ENFORCE_EAGER=1`, say in the same paragraph
   that no process reads that name (0 `rg` hits in either tree at this pair; the gate
   reads `model_config.enforce_eager`, `platform.py:151`): the missing log line
   `Compilation disabled, using eager mode by default` (`platform.py:163-165`) is the
   whole diagnosis, and the doc gap that invited the guess belongs in the answer
   (`../startup-failures/graph-mode-not-applied.md`, recorded #1456).
2. **Keep graph mode, lower TP:** TP ≤ 4 with PP/DP absorbing the remaining cards.
   Say what is validated: the only graph e2e case at that pin is
   `vllm-ascend/DeepSeek-V3-Pruning`, `tensor_parallel_size=4`, `mp`, **no PP, no
   ray** (`tests/e2e/multicard/test_torchair_graph_mode.py:50`) — exactly on the
   `32` boundary, so TP=8+graph was never covered and any PP/ray change is untested.
3. **TP=8 + graph: not available at that ref.** The kernel's `CheckMlaAttrs` is the
   authority (the assertion is only a pre-flight mirror of it), so bypassing the
   assert moves the failure to capture/replay. Treat as a capability gap to track,
   and cite the tree's own `# TODO: support numHeads / numKvHeads < 16 in MLA kernel`
   (`attention/mla_v1.py:573`) so the follow-up is an upstream request rather than a
   local patch.
4. **ACL graph is not the alternative:** `ascend_config.py:148-157` raises
   `NotImplementedError: ACL Graph does not support deepseek. Please try torchair
   graph mode … Or set enforce_eager=True`, i.e. torchair is the only graph route for
   a DeepSeek model there.

## Anchors (vllm-ascend v0.9.1rc1 `c30ddb8`, vllm v0.9.1 `b6553be`)

| fact | anchor |
|---|---|
| allowed set | `vllm_ascend/attention/attention.py:44` `_ALLOWED_NUM_QUERIES_PER_KV = [32, 64, 128]` |
| the gate (two copies) | `mla_v1.py:573-580` and `attention.py:1037-1045`, both under `if self.torchair_graph_enabled` |
| the project's own TODO, one line above the gate | `attention/mla_v1.py:573` `# TODO: support numHeads / numKvHeads < 16 in MLA kernel` — quote it when the question is "is the assert the bug / can I patch it": the sub-16 case is a known unimplemented kernel path, so the answer is a capability request with this line, not a patch |
| heads per rank | `models/deepseek_v2.py:359-362`; `num_kv_heads=1` at `:430-434` |
| eager vs graph decode op | `mla_v1.py:1007` vs `:988` |
| the project's own statement of the rule | `docs/source/faqs.md:118` (and the kernel text at `:124`) |
| documented gap | `docs/source/user_guide/graph_mode.md:40-51` tells DeepSeek users to enable torchair and states only the chunked-prefill caveat — the head constraint lives **only** in the FAQ, so the recommended path is where people meet it. Two more defects on that page, both worth naming in a `[Doc]` answer: `:10` gives only the Python `enforce_eager=True` form (the CLI `--enforce-eager` appears just in a Qwen example at `:82`, and nothing says no env var exists), and the TorchAirGraph "online example" at `:58` serves `Qwen/Qwen2-7B-Instruct`, contradicting `:14` ("only DeepSeek series models are supported") and `ascend_config.py:143-147` ("Torchair graph mode only works with deepseek model") — a copy-paste defect (#1456) |

## Version scope: the Python gate was deleted, the rule was not

`rg -n "_ALLOWED_NUM_QUERIES_PER_KV" vllm_ascend/` returns three hits at v0.9.1rc1
and **zero** at v0.10.0rc1 `4604882a` (`attention.py` was split into
`attention_v1.py` / `attention_v1_torchair.py`, and `mla_v1.py:618` still computes
the ratio with no assertion), while `docs/source/faqs.md:121` still states the rule
and quotes the same kernel line at `:127`. Graph decode still calls
`npu_fused_infer_attention_score` (`mla_v1.py:1038`) there. So at a ref after the
deletion what remains is the kernel's own tiling check: whether the CANN package in
that image accepts 16 heads is **not decidable from this snapshot** — the check is
the reporter's capture/first-request run, and the removed assert must not be quoted
as the mechanism at that ref. Re-grep the constant *and* the FAQ before repeating
either form; recorded case: #1451 (v0.9.1rc1, DeepSeek-V3 TP8×PP4, ray + torchair).

## What to send back

- Rung 1 (eager): does the engine start and does the first request return tokens?
  Nothing in-tree proves the eager `_npu_paged_attention_mla` path accepts a
  heads-per-rank **below 32** — 16 at TP8 (#1451) or 8 at TP16 on a 128-head model
  (recorded #1458: v0.9.1rc1, `DeepSeek-R1-w8a8`, `--quantization ascend`, the
  `ACL Graph does not support deepseek` paste under a reporter's own
  `export ENFORCE_EAGER=1`) — no Python gate exists on it, so the reporter's run is
  the only evidence, and it is the one open item this page cannot close. Say in the
  same paragraph that the refusal's own advice is already spent at that layout: the
  message recommends torchair, and the division above refuses it too, so
  `--enforce-eager` is the only *startable* route and the TP layout is the second
  question.
- Rung 2 (TP ≤ 4): the capture lines (banner, `Graph capturing finished …`) and the
  full serve command, since PP/ray on top of graph mode is outside CI coverage.
