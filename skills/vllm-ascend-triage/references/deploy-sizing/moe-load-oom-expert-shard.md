# Load-phase OOM on a 1T-class MoE: the expert **count** per card, not TP

Scope: the traceback dies **inside model construction**, not in a collective and not
in the KV pool — `get_model` → `.../models/kimi_k25.py` → `deepseek_v2.py:make_layers`
→ `vllm_ascend/ops/fused_moe/fused_moe.py` → `vllm_ascend/quantization/method_adapters.py`
`create_weights` → `vllm_ascend/quantization/methods/w4a16.py:get_weight` →
`torch.empty(...)` → `RuntimeError: NPU out of memory. Tried to allocate ~0.7 GiB
(NPU 4; 60.96 GiB total capacity; 60.30 GiB already allocated; 132.82 MiB free …)`,
on a checkpoint whose weights are 0.4-0.6 TB (Kimi-K2 / K2.5 W4A16/W4A8, any
384-512-expert DeepSeek-V3-class MoE), with `--enable-expert-parallel` and a
`--data-parallel-size` in the launch dict — typically two nodes of 8×64 GB A2.
The reporter says "A2 多机" (multi-node) and the pasted "parameters" are a launcher
dict (`model_tag`, `api_server_count`, `data_parallel_size`, …), not a serve line.

Not this page: two commands that differ by `--tensor-parallel-size` where the smaller
serves (`tp-scaling-oom-at-startup.md` — a budget question on one node); an OOM after
a healthy serving period (`../runtime-memory/index.md`); a paste carrying a
collective/device error (`../hccl-and-hardware/index.md`).

## Step 1 — the one arithmetic that decides it: EP shrinks the expert count, not the expert width

Under EP, upstream sets the intra-layer `tp_size` to **1** and shards *experts*:

| Fact | Site (vllm v0.15.0 `f1764434`) |
|---|---|
| `use_ep = dp_size_ * pcp_size_ * tp_size_ > 1 and enable_expert_parallel`; then `ep_size = tp_size` and the returned config has `tp_size=1` | `vllm/model_executor/layers/fused_moe/config.py:980-1035` |
| `flatten_tp_size = dp_size * pcp_size * tp_size` — the EP width is the TP *flattened across DP and PCP* | `vllm/model_executor/layers/fused_moe/config.py:901-909` |
| `dp_size_` defaults to `get_dp_group().world_size` (not the CLI number) | `vllm/model_executor/layers/fused_moe/layer.py:377` |
| `intermediate_size_per_partition = intermediate_size // self.tp_size` → **full** `moe_intermediate_size` per expert under EP | `.../fused_moe/layer.py:506` |
| packed int4 layout `num_experts × 2·I × H/8` (w13) and `num_experts × H × I/8` (w2), int32 | `vllm-ascend vllm_ascend/quantization/methods/w4a16.py:115-140` |
| bf16 group scales, `group_size` default 32 | same file `:144-170` |

So per card, per MoE layer:

```
bytes ≈ (E/ep_size) × [ 3·H·I·0.5 (int4) + 3·H·I/(group_size)·2 (scales) ]  … times n_layers
```

For the K2/K2.5 backbone as filed (H=7168, moe_intermediate=2048, 384 routed experts,
61 layers, group 32) that is **67.6 GiB of expert weights per card at EP=8 and 33.8 GiB
at EP=16** — i.e. at EP=8 the checkpoint cannot be materialised on a 60.96 GiB card at
all, which is exactly the shape of the paste (`60.30 GiB already allocated`, then a
672 MiB `w13_weight_packed` refused). Say the two numbers and the ratio (2×); that is
the verdict, and it does not need the allocator.

## Step 2 — `ep_size` is `dp × tp` only when the DP ranks really share a DP group

`FusedMoE` reads `get_dp_group().world_size` (`.../fused_moe/layer.py:377`), so the DP
half of the EP width exists only if the DP ranks really joined one world. What joins
them: `init_distributed_environment` re-expands the world to
`world_size_across_dp = tp × pp × pcp × dp` and offsets the rank by the DP rank
whenever `data_parallel_size > 1` **or** `nnodes > 1`, and for `nnodes > 1` it takes
the store address from `parallel_config.master_addr` / `master_port`
(`vllm/distributed/parallel_state.py:1180-1200`) — i.e. from the flags that say which
node is node 0. So a multi-node attempt that carries `--data-parallel-size 2` and
**not** `--data-parallel-size-local 1 --data-parallel-address <node0 ip>
--data-parallel-rpc-port <port> --data-parallel-start-rank <0|1>` has no cross-node
address to join: `dp_size_` falls back to a single-engine DP group → `ep_size = tp =
8` → per-card expert bytes double and the load OOMs. Two defaults make this silent:
`data_parallel_size_local: int = 1` (`vllm/config/parallel.py:110`) and
`world_size = pp × tp × pcp` in `ParallelConfig.__post_init__`
(`parallel.py:551-558`) — the engine banner shows the pre-expansion number, so
nothing the reporter pasted has to look wrong. The four flags are also the tree's own
shipped multi-node DP+EP form
(`../feature-enable-recipes/moe-expert-parallel-with-dp.md` step 4,
`docs/source/tutorials/Qwen3-235B-A22B.md:177-185`).

## Step 3 — two log lines settle the shard width; the OOM byte does not

The paste's "674.00 MiB" is corroborative only: decoding it needs the checkpoint's
`config.json` (hidden size, `moe_intermediate_size`, routed-expert count, group size),
which the report never carries — the K2 assumption gives 672 MiB for 48 experts/rank, a
2 MiB miss, and every error in those inputs moves the tail. Ask for one grep instead:

```bash
rg -n 'EP Rank .* Local/global number of experts' <worker log>   # e.g. 48/384 → 8-way EP
rg -n 'rank .* in world size .* is assigned as DP rank .* EP rank' <worker log>
```

- first line: `[EP Rank %s/%s] Expert parallelism is enabled. … Local/global number of
  experts: %s/%s` (`vllm/model_executor/layers/fused_moe/layer.py:476-492`; vllm-ascend
  prints the same from `vllm_ascend/ops/fused_moe/fused_moe.py:284-295`);
- second line: the per-rank assignment `logger.info_once` in
  `vllm/distributed/parallel_state.py:1428-1436`, which prints the **world size** the
  rank saw; `EP rank x/N` with `N = TP` is the proof that the DP half never joined.

`N = dp × tp` (16 here) means the shard is right and the residual OOM is a different
term — then go to the knob list below. `N = tp` (8) is the answer to the report.

## Step 4 — read the pinned tutorial's TP floor per **node**, not per deployment

`docs/source/tutorials/models/Kimi-K2-Thinking.md:113` at v0.15.0rc1 (`3d43ed99`):
"For an Atlas 800 A3 (64G*16) node, tensor-parallel-size should be at least 16", with
`:113-121` serving `--tensor-parallel-size 16 --enable-expert-parallel`. That is one
A3 node's 16 cards. It does **not** mean "16 cards in one engine": the card count
enters through `ep_size`, which a 2×8 A2 deployment reaches as dp2 × tp8. Precedent
that the shape exists: #10503 runs Kimi-K2.5-w4a8 `tp8/dp2/EP` on 2×A2
(`../evidence-workspace/recorded-cases-brief-mentions.md`).

## Reporter-confirmed resolution (#7123, 2026-03-25, same host)

The reporter got the same 2-node A2 deployment serving with this line (verbatim, hosts
elided). It is the strongest artefact for this class: the same TP8/EP/DP2 layout, with
the DP group actually wired and the remaining knobs set.

```bash
vllm serve kimi-k2-5 --trust-remote-code --host XXX --port 1025 \
  --tensor-parallel-size 8 --seed 1024 --served-model-name kimi_2_5-1.0.0 \
  --max-num-seqs 16 --max-model-len 72000 --max-num-batched-tokens 12288 \
  --gpu-memory-utilization 0.95 --enable-prefix-caching --enable-expert-parallel \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-address XXX --data-parallel-rpc-port 2357 --data-parallel-start-rank 0 \
  --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false}}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --allowed-local-media-path / --async-scheduling \
  --mm-processor-cache-type shm --mm-encoder-tp-mode data
```

(`--additional-config` keys are matched by name, so a mistyped sub-key is a silent
no-op — worth one look when a symptom says the setting "did nothing",
`../startup-failures/graph-mode-not-applied.md`.) Deltas from the failing dict worth
naming, in this order: the four `--data-parallel-*` flags, `gpu_memory_utilization`
0.86 → 0.95, `ascend_scheduler_config`/torchair graph off, `FULL_DECODE_ONLY` with an
explicit capture-size list. `--max-model-len 4096 → 72000` moved the *other* way, so it
is not the memory lever.

## Answer shape

1. Verdict first: at this card size the deployment fits only with a 16-wide expert
   shard; if your EP width equals TP (8), each card must hold 2× the expert bytes and
   the load cannot succeed — that is the OOM, not a broken card.
2. The two per-card numbers (step 1) for the reporter's `config.json`, and the two log
   lines (step 3) that tell you which one you have.
3. Ladder, cheapest first: wire the cross-node DP group (four flags) and re-check the
   `[EP Rank r/N]` line; then the working line's `gpu_memory_utilization 0.95` +
   scheduler/torchair off + `FULL_DECODE_ONLY`; then more EP ranks, or a smaller quant.
4. Gaps to send back: the `[EP Rank r/N]` and `rank … in world size …` lines, the real
   serve line (the launcher dict hides which flags reached vLLM), `config.json`.

## Stop rule — this class is two rules and one grep

Recorded #7123 (`v0.15.0rc1` / vllm v0.15.0, 2×A2): 30 tool results, reasoning blocks
of 4.2/9.2/9.7/9.9 K characters, **zero message text on four assistant turns**, and it
was cut at the 16 384-token cap while still re-deriving upstream DP/EP group algebra
(`parallel_state.py:1330-1421`) to explain the 674 MiB request — the two log lines in
step 3 would have answered it in the reporter's own log, and the reporter-confirmed
line was already the thing to hand over. Do not decode allocation bytes into expert
counts (step 3), and do not walk upstream group algebra before asking which flags
reached the process.

## Scope and recheck

Verified at vllm-ascend v0.15.0rc1 (`3d43ed99`) + vllm v0.15.0 (`f1764434`), the #7123
pins. Re-grep per version: `flatten_tp_across_dp_and_pcp` and `ep_size = tp_size` in
`FusedMoEParallelConfig.make`; `intermediate_size_per_partition = intermediate_size //
self.tp_size`; the packed-int4 layout and `group_size` default in
`vllm_ascend/quantization/methods/w4a16.py`; `data_parallel_size_local`'s default; and
the Kimi tutorial's TP-floor sentence. The per-card byte table is arithmetic on
`config.json`, not a measured constant; the reporter's `gpu_memory_utilization` values
are one host's margin.
