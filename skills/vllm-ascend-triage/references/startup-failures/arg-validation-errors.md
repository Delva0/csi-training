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
- `RuntimeError: shape '[-1, <dp>, <pp>, <tp>]' is invalid for input of size <W>` —
  same class, raised by `torch.reshape` inside upstream
  `initialize_model_parallel`, so the "flag combination" is spelled as tensor
  dimensions (third raise site below).
- `KV transfer 'prefill' config has a conflicting data parallel size. Expected 1,
  but got 2.` — same class (a launch config the process rejects at startup), but
  raised by **vllm-ascend** against the connector's `kv_connector_extra_config`
  blocks, never by upstream `arg_utils`; the `tensor parallel size` twin and the
  `'decode'` role come from the same function (worked case: #8345, below).

Tells that it is this lane and not a defect: there is no `WorkerProc`, no rank
prefix and no `Starting to load model` in the log, and the raise site is
`vllm/engine/arg_utils.py` (`EngineArgs.create_engine_config`) — upstream argument
validation, before weights are touched and before any device or HCCL work. The
deliverable is a launch-command fix plus one self-proof the reporter can run, not
a code walk. Compare with `connector-config-attributes.md` (a Python error raised
after workers exist) and `../hccl-and-hardware/index.md` (device text).

There is a **second raise site in the same class**: vllm-ascend's own
`check_kv_extra_config`, called from `AscendPlatform.check_and_update_config`
when `VllmConfig.__post_init__` runs a second time at the EngineCore handshake.
There the log does *not* look like a flag rejection — connector creation and
per-rank load lines precede the raise and there is no `arg_utils` frame — so
decide by the sentence (it begins `KV transfer '<role>' config`) rather than by
the log's shape.

## Procedure

1. Grep the exact string in **both** pinned trees at once
   (`rg -n '<fragment>' <vllm-root>/vllm <vllm-ascend-root>/vllm_ascend`): half
   this family is upstream `engine/arg_utils.py`, half is vllm-ascend's own
   platform validation, and a grep of only one tree returns an empty result that
   reads like a snapshot gap. Then `sed -n 'A,Bp'` around the hit and write the
   raising boolean expression in the answer. Nothing else about the run matters until
   that expression is understood. One member of this family has **no** `raise` to
   grep — a bare `torch.reshape`/`arange` in upstream `parallel_state.py` — so when
   the frames are `worker → ensure_model_parallel_initialized → initialize_model_parallel`,
   decode the numbers in the message instead (third raise site).
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

## Second raise site: `kv_connector_extra_config` parallel sizes (#8345)

Signature: `ValueError: KV transfer 'prefill' config has a conflicting data
parallel size. Expected 1, but got 2.` on a PD deployment whose
`--kv-transfer-config` carries
`kv_connector_extra_config.prefill/decode.{dp_size,tp_size}`. Reported as "PD
分离 DP2/TP8 起不来", often with DP=2, DP=4 … all failing identically, and no
worker or device text anywhere.

Worked case #8345: Qwen3-**32B** (dense), vllm-ascend v0.18.0rc1 (`99e1ea0`) +
vllm v0.18.0 (`bcf2be9`), `vllm serve … --data-parallel-size 2
--data-parallel-rank 0 --tensor-parallel-size 8 --kv-transfer-config
'{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"36000",
…,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":8},
"decode":{"dp_size":2,"tp_size":8}}}'`. The only evidence was a screenshot; its
OCR text is the paste (`ATTACHMENTS.txt` → `*.ocr.txt`, procedure step 2).

| Fact | Evidence (v0.18.0rc1 / vllm v0.18.0) |
|---|---|
| The check compares the block against `parallel_config.tensor_parallel_size` / `data_parallel_size` and raises the `KV transfer '<name>' config has a conflicting … size` text | `vllm_ascend/utils.py:1125-1149` |
| Only the **local** role is checked: `kv_producer` → `prefill`, `kv_consumer` → `decode` | `utils.py:1146-1149` |
| Call site is platform config validation at engine start, not `arg_utils` | `vllm_ascend/platform.py:249` in `AscendPlatform.check_and_update_config` |
| That runs a *second* `VllmConfig.__post_init__` inside the EngineCore handshake — why model-load and connector lines already appear before the raise | `vllm/v1/engine/core.py:941` |
| A **non-MoE** engine rewrites its DP to 1 there: `if data_parallel and model_config.is_moe: DPEngineCoreProc` **else** `data_parallel_size = 1; data_parallel_size_local = 1; data_parallel_rank = 0` | `vllm/v1/engine/core.py:1061-1073` |
| `is_moe` = `get_num_experts() > 0`; a dense 32B has none, so the `else` branch is taken and no `dp_size > 1` can ever validate | `vllm/config/model.py:1705-1710` |
| Both blocks must keep both keys (`assert "tp_size" in …`, `assert "dp_size" in …`), so a block cannot be emptied to dodge the check | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1162-1180` |
| In this version the two parsed values have **no other reader** (`rg -n '_prefill_dp_size\|_decode_dp_size' vllm_ascend/` = 2 hits, both assignments), so only the number the check sees matters | same file, `:1167-1178` |
| The recipe that teaches `dp_size: 2` is a **MoE** recipe — it states cards = `dp_size × tp_size` | `examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md:73`; DeepSeek-V3.2 tutorial `kv_connector_extra_config` block |

Fix ladder:

- **A (one-line, what the check accepts):** put the *per-engine* DP in the local
  role's block — `"prefill": {"dp_size": 1, "tp_size": 8}` on a dense model —
  and keep `--data-parallel-size 2` if two API-server-spawned engines are wanted;
  each engine now has `data_parallel_size == 1`, which is what the check reads.
  The `decode` block must keep its own numbers (the D node validates those), and
  in this version a `decode.dp_size` on the P node is inert — recheck per version,
  later connectors use the remote sizes for rank arithmetic.
- **B:** keep `dp_size > 1` only on an **MoE** model (DeepSeek, Qwen3-MoE), where
  the engine keeps `DPEngineCoreProc` and `data_parallel_size` stays N.
- **C:** `tp_size` must equal `--tensor-parallel-size` too; the twin error names
  tensor. Do not delete the keys (the connector asserts), and do not report the
  trailing `init engine … took N seconds` line as proof the engine came up — on
  this path the EngineCore raises during construction.
- Recheck per version with `rg -n 'conflicting data parallel size' vllm_ascend/`;
  `--data-parallel-rank` sets external LB (`data_parallel_size_local = 1`) and is
  *not* the compared number, so "drop `--data-parallel-rank`" is not the fix here.

## Third raise site: the model-parallel group reshape (world size vs TP)

Signature: engine start dies on every rank with

```
File "…/vllm/distributed/parallel_state.py", line 1098, in initialize_model_parallel
    all_ranks = torch.arange(world_size).reshape(
RuntimeError: shape '[-1, 1, 1, 4]' is invalid for input of size 2
```

and the frames above it are `vllm_ascend/worker/worker_v1.py` `init_device` →
`_init_worker_distributed_environment` → `ensure_model_parallel_initialized`. In the
recorded case the frames also carry `vllm/executor/uniproc_executor.py` — each
spawned process is a **one-rank** executor, which is the first hint that the process
count, not the card count, set the world size.

**The two numbers in the message are the whole diagnosis.** `arange(world_size)`
holds the *real* world size (`size 2`), and the reshape literal is
`(-1, dp, pp, tp)` — trailing number = the requested `--tensor-parallel-size` (4),
per the source comment `# the layout order is: ExternalDP x DP x PP x TP` (export
`:1006`). Nothing else about the run matters until those two are read.

Why no validation catches it: the launcher that owns `external_launcher` mode creates
the process group itself, so upstream `init_distributed_environment` hits
`if not torch.distributed.is_initialized():` (export `:937`) and **skips** its own
`init_process_group(..., world_size=parallel_config.world_size)` — the argument is
never compared against the real group, and `_WORLD` is built from
`get_world_size()`.

### Worked case #2973 — `examples/offline_external_launcher.py`, TP 4 vs 2 processes

`[Doc]`-tagged, vllm-ascend v0.10.2rc1, single node:
`python examples/offline_external_launcher.py --model="Qwen/Qwen3-30B-A3B"
--tp-size=4 --proc-per-node=2 --enable-expert-parallel`, substrate vllm v0.10.2.

| Fact | Evidence (vllm-ascend v0.10.2rc1 `048bfd5`; vllm export v0.9.1 `b6553be`) |
|---|---|
| The script computes the world size itself, from the node and process counts, and takes `--tp-size` as an independent value | `examples/offline_external_launcher.py:239-241`, `:251` `world_size = node_size * proc_per_node` |
| It spawns exactly `proc_per_node` children per node, in `range(proc_per_node*node_rank, proc_per_node*(node_rank+1))` | same file, `:254-255` → one node, 2 procs, TP 4 |
| Each child exports `WORLD_SIZE` and calls `init_process_group(backend="cpu:gloo,npu:hccl", world_size=world_size, rank=rank)` | same file, `:160-168` |
| The worker then calls `init_distributed_environment(parallel_config.world_size, …)` and `ensure_model_parallel_initialized(tp, pp)` | `vllm_ascend/worker/worker_v1.py:318-325` |
| Upstream skips its own init when the group exists, so the mismatch is never compared | export `parallel_state.py:937` (`init_distributed_environment`), `:1072` `ensure_model_parallel_initialized` |
| The reshape is `torch.arange(world_size).reshape(-1, dp, pp, tp)` | export `:1015` (reporter's v0.10.2 build reports `:1098`) |
| The paste corroborates world size 2: exactly two `non-default args` blocks and two `Initializing a V1 LLM engine` lines | the report's own log |
| The documented usages are `--tp-size=1 --proc-per-node=2` (dense) and `--tp-size=2 --proc-per-node=2 --enable-expert-parallel` (this model) — both satisfy `tp ≤ world_size`, so the invariant is never stated | docstring `:22-56` |
| CI runs the second one and asserts two TP ranks | `tests/e2e/multicard/test_external_launcher.py:87-89`, `:104` |

Fix ladder:

- **A (documented, CI-covered):** `--tp-size=2 --proc-per-node=2
  --enable-expert-parallel`. Match the process count to the intent:
  pure TP needs `--tp-size == --node-size × --proc-per-node`.
- **B (TP 4 on one box, if it has 4 NPUs):** `--tp-size=4 --proc-per-node=4`.
- **C (real multi-node):** `--node-size=2 --node-rank=0|1 --proc-per-node=2
  --master-addr=<ip> --master-port=<free port>` on **both** nodes, same `--tp-size`.
- **Not the fix:** `--enforce-eager`, the model, the card count, HCCL/driver — the
  raise happens before a weight is loaded or a collective is issued.

### Doc defects to report alongside (this shape is usually a `[Doc]` report)

1. The invariant is never stated: `--proc-per-node` is documented as "Number of
   processes per node" (`:96-99`), which reads like a hardware knob, while it is
   half of the world size.
2. The multi-node **Node 1** block omits `--proc-per-node=2` (`:47-55`): Node 1 then
   computes `world_size = 2 × 1 = 2` against Node 0's 4 and spawns only rank 1, so
   the job mismatches or hangs in `init_process_group` instead of erroring.
3. Warn whoever writes the validation: when the world size **is** a multiple of TP
   (e.g. `--tp-size=2 --proc-per-node=4`) the reshape succeeds and the extra ranks
   become independent external-DP groups — the verl use case — so a wrong command can
   appear to work with half the intended parallelism. Only the non-divisible case
   raises.

Recheck per release with `rg -n 'world_size = node_size \* proc_per_node'
examples/offline_external_launcher.py` and the `reshape(` line in the installed
`vllm/distributed/parallel_state.py`; the export here is vllm v0.9.1 while the
reporter ran v0.10.2, so upstream line numbers differ by ~80 lines and structure is
identical.

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
