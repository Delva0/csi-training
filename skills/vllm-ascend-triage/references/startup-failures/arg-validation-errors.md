# Config-validation errors raised before any worker starts

Signature family — the sentence names a **flag combination or parallel mode**,
never an op, tensor, collective or model file:

- `ValueError: Non-MoE models do not support external data parallel mode. For
  external load balancing, launch independent vLLM instances without
  --data-parallel-* arguments.` (worked case: #11863)
- assert texts from the same function: `data_parallel_hybrid_lb and
  data_parallel_external_lb cannot both be True.`, `data_parallel_size_local must
  be 1 or None when data_parallel_rank is set`,
  `world_size=N must be divisible by nnodes=M.`, `node_rank=N must be less than
  nnodes=M.`

Tells that it is this lane and not a defect: there is no `WorkerProc`, no rank
prefix and no `Starting to load model` in the log, and the raise site is
`vllm/engine/arg_utils.py` (`EngineArgs.create_engine_config`) — upstream argument
validation, before weights are touched and before any device or HCCL work. The
deliverable is a launch-command fix plus one self-proof the reporter can run, not
a code walk. Compare with `connector-config-attributes.md` (a Python error raised
after workers exist) and `../hccl-and-hardware/index.md` (device text).

## Procedure

1. Grep the exact string in the pinned **vllm** tree (`rg -n '<fragment>'
   vllm/engine/arg_utils.py`), then `sed -n 'A,Bp'` around it and write the raising
   boolean expression in the answer. Nothing else about the run matters until
   that expression is understood.
2. Expand each term in the expression to the CLI argument that sets it, one grep
   per term. The trap in #11863: `--data-parallel-rank` is not "just a rank",
   it implies external load balancing —
   `data_parallel_external_lb = self.data_parallel_external_lb or
   self.data_parallel_rank is not None` (vllm v0.21.0
   `engine/arg_utils.py:1793-1794`, still `:1943-1945` at v0.24.0).
3. Decide the model's MoE status from code, not from the model name:
   `model_config.is_moe` = `get_num_experts() > 0` and `get_num_experts()` =
   `model_arch_config.num_experts` (`vllm/config/model.py:1824-1825`, `:1253-1254`
   at v0.21.0). A dense model has no `num_experts`, so the gate fires. Give the
   reporter a one-line proof:
   `python -c "from transformers import AutoConfig as A;
   c=A.from_pretrained('<path>',trust_remote_code=True);
   print(c.model_type, getattr(c,'num_experts',0))"` → expect `0`.
4. Open the tutorial/example the reporter followed **in the pinned tree** and say
   whether its flag block is MoE-shaped. A DP block that also carries
   `--served-model-name dsv3` / `--enable-expert-parallel`
   (`examples/external_online_dp/run_dp_template.sh:18-25` at v0.21.0rc1) is the
   DeepSeek-MoE template, not a dense-model recipe.
5. Answer with the cheapest launch change (ladder below) and name the doc the
   reporter cited as the actual defect if it recommends the rejected shape.
6. One history grep for the error string; if it returns nothing (it did for
   #11863) say "no precedent" instead of implying the class is documented.

## Worked case #11863 — dense Qwen3.6-27B under the multi-node PD recipe

| Fact | Evidence (pin) |
|---|---|
| The gate: `data_parallel_size > 1 and data_parallel_external_lb and not model_config.is_moe` → the reported `ValueError` | vllm v0.21.0 `ad7125a4` `engine/arg_utils.py:1793-1805`; same code at vllm v0.24.0 `ee0da84a` `:1943-1952`, so re-grep per version |
| Providing `--data-parallel-rank` alone turns external LB on | `engine/arg_utils.py:1793-1794` (v0.21.0) |
| Qwen3.5-27B / Qwen3.6-27B are **dense** hybrid Mamba-Transformer models; the sparse variants are `qwen3_5_moe` | vllm-ascend v0.23.0rc1 tutorial `docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md:5` |
| No vllm-ascend code bypasses the gate (`vllm_ascend/` only *reads* MoE-ness elsewhere) | `rg -n 'data_parallel_external_lb\|Non-MoE' vllm_ascend/` empty at v0.23.0rc1 |
| The vllm-ascend external-DP docs never state the MoE restriction | `docs/source/user_guide/feature_guide/external_dp.md:31-36` (manual launch with `--data-parallel-rank`), `examples/external_online_dp/run_dp_template.sh` (v0.21.0rc1) |
| Historical precedent for the string | none at v0.21.0rc1 history (0 hits) |

Fix ladder for a dense model that wants PD + DP-style scale-out:

- **A (recommended, and what the error text says):** drop every
  `--data-parallel-*`; run 2×4 independent instances, TP=2 each, distinct
  `--port` and `engine_id`, prefill `kv_role=kv_producer` / decode
  `kv_consumer`, and balance outside with
  `examples/external_online_dp/dp_load_balance_proxy_server.py`. The
  `kv_connector_extra_config` `dp_size`/`tp_size` then describe instances, not
  vLLM ranks.
- **B (closest to the documented single-node recipe):** one instance per node,
  `--tensor-parallel-size 8`, node 0 prefill / node 1 decode.
- **C (only if DP inside one vLLM is required):** internal DP —
  `--data-parallel-size N --data-parallel-size-local K
  --data-parallel-start-rank 0|K`, **no** `--data-parallel-rank`, so
  `data_parallel_external_lb` stays false and the gate is never reached. Label
  this a hypothesis unless the reporter confirms it: internal DP + Mooncake PD on
  a dense model has no recorded run.

## Doc drift: `/latest` docs are not the pinned version

- A report that cites `docs.vllm.ai/.../latest/...#<section>` is quoting a doc
  that may postdate the pinned release. Verify the section exists in the pinned
  tree before treating it as the instructions the reporter ran; if it is absent,
  say so, answer within the pinned version, and ask for the section text instead
  of assuming it.
- #11863 shows both sides of this. At vllm-ascend v0.21.0rc1 the tutorial is
  `docs/source/tutorials/models/Qwen3.5-27B.md` (196 lines, **single-node only**,
  `--data-parallel-size 1` at `:108`) — no multi-node PD section. At v0.23.0rc1
  the same tutorial is `Qwen3.5-27B-Qwen3.6-27B.md` and §5.2
  (`:406`) *recommends DP-first scaling for these small dense models* while its
  launch scripts pass `--data-parallel-size $3 --data-parallel-rank $4 …`
  (`:480-483`, `:544-547`) — exactly the shape the upstream gate rejects for a
  non-MoE model (gate still present at vllm v0.24.0 `:1943-1952`).
- So: the reported error is a usage error from the reporter's side **and** a real
  doc bug in the pinned/later trees. Report both, with the version range and the
  recheck (`rg -n 'Non-MoE models do not support external data parallel'` in the
  installed `vllm`), and do not let the doc citation pull the answer into a
  version-archaeology detour before the launch fix is stated.
