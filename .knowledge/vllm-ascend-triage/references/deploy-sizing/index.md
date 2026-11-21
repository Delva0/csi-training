# Deployment sizing: "can this run on N cards?" and "why does my stage layout fail?"

Scope: the report is a *plan*, not a crash — "the GPU docs do a 2-device
stage-based deploy, mine does not work on 2 NPUs", "the bundled `platforms.npu`
yaml has three stages and I only have two cards", "does model M fit on 2×64 GB",
"how do I pin stage 1 to one card". Nothing has crashed yet, or the crash text is
an allocation failure that the layout predicts. The same topic owns the *KV* half
of a plan — a measured per-token footprint several times the reporter's own
calculation, or "is there a path to 512K/1M context on this card in this release"
(`dsv4-kv-footprint-terms.md`).

Leave this topic when there is a real startup traceback with a lane
(`../startup-failures/index.md`), a collective/device error
(`../hccl-and-hardware/index.md`), or a long-run memory report
(`../runtime-memory/index.md`); when the question is a latency/throughput claim
(`../accuracy-benchmarks/index.md`); or when the deliverable is a report about a
tutorial page (`../tutorial-verification/index.md`).

Inherit the rules in SKILL.md, plus these:

1. **Answer with arithmetic, then options.** One engine's weights per card
   (`params × bytes-per-param ÷ TP`) against the HBM that engine may use
   (`cards' HBM × gpu_memory_utilization`), and the sum of
   `gpu_memory_utilization` over every process that shares a device. Report both
   numbers; reproduce them with
   `bash <skill_root>/scripts/card-budget.sh` instead of by eye (`--params` is in
   **billions** — `671` for DeepSeek-R1, not `671e9`; the script refuses the raw
   count, because every result of that mistake is 1e9× and the verdict inverts). A sum above
   1.0 on any device is **infeasible by construction** and is not a scheduler,
   connector or device-visibility bug. Each ratio is **per-instance**
   (`vllm/config/cache.py:66-74` at v0.22.1: "only applies to the current vLLM
   instance"), so the sum is a feasibility check — never a licence to inflate one
   instance to cover another's usage, and a start-up that fails at KV sizing while
   the device has free HBM is that other instance's memory being charged twice
   (`../runtime-memory/kv-budget-per-instance-vs-device-global.md`).
2. **The device-visibility env var is the second half of every such report.**
   On Ascend the control variable is `ASCEND_RT_VISIBLE_DEVICES`
   (`vllm_ascend/platform.py:101` at v0.19.1rc1; every multi-node launch script
   in the same docs tree does `export ASCEND_RT_VISIBLE_DEVICES=$1`, e.g.
   `GLM5.md`, `DeepSeek-V3.1.md`, and `CUDA_VISIBLE_DEVICES` occurs nowhere in
   `docs/source/`). A command copied from a GPU page therefore exposes **all**
   cards to **every** process, which looks exactly like a stage-collision bug.
3. **A sibling project's config that the snapshot does not carry is a gap, not a
   hypothesis — but ship what reporters already established.** vllm-omni itself is
   in no `SOURCES.json`, so its parsing is unverifiable. What is known without the
   source, and belongs in the first answer: the NPU stage configs are
   `vllm_omni/platforms/npu/stage_configs/<model>.yaml`, they are overridable with
   `--stage-configs-path` (not `--deploy-config`), each stage carries its own
   `devices`, and copying the file, editing every stage's `devices` and exporting
   `ASCEND_RT_VISIBLE_DEVICES` brought the three-stage 30B-A3B omni up on **two**
   cards (#8068, reporter-confirmed). Do not answer "you need a third card" or
   "the 2-NPU layout is infeasible" before reading the reporter's yaml: the
   per-stage `gpu_memory_utilization` values are the number that decides, and
   that recipe contradicts the guess. Ask only for the still-open pieces (the
   edited yaml, one log line per stage).
   Precedent: #9336 files this exact absence as a `[Doc]` gap ("Missing NPU
   device allocation strategy for multi-NPU deployments") and asks for the NPU
   equivalent of `CUDA_VISIBLE_DEVICES` — so "the docs do not answer this" is a
   publishable finding.
4. **Quantify the options you offer** (more cards / quantised weights / drop a
   stage / pin devices) with the same script, and state the limit: the fit check
   excludes activations, KV growth and fragmentation, so a PASS is "not yet
   excluded" while a FAIL is decisive.

5. **Ask which checkpoint the recipe was priced for before diffing flags.** A tutorial
   command carries the *quantized* weights in its model path (`vllm-ascend/DeepSeek-R1-W8A8`,
   `…-w4a8`) while the reporter serves the unquantized BF16 download — that is 4× the bytes
   per parameter and a different minimum card count, and no flag diff closes it. Price both
   with `scripts/card-budget.sh --params <B> --dtype bf16|w8a8 --tp <TP×PP> …` (**only the
   product** shards the weights; `--tp 8` on a TP8×PP4 layout reports 4× the real per-rank
   bytes) and quote the project's own
   minimum when it exists: reporter-confirmed on #4948 that full-weight DeepSeek-R1-0528-BF16
   needs **at least 4 Atlas 800I A2 servers (32 × 910B4)**, and that the DeepSeek-R1 tutorial
   page suits the W8A8 weights, not BF16. "Your layout is at the documented minimum and the
   recipe you followed is for other weights" is a complete sizing answer.

6. **A 1T-class MoE under `--enable-expert-parallel` is priced in *experts per
   card*, not in TP.** Under EP the intra-layer TP collapses to 1 and the expert
   *count* is split over the flattened `dp × pcp × tp` (`ep_size`), with each expert
   keeping its full `moe_intermediate_size` width — so doubling the EP width halves
   the per-card bytes while a TP-only expert shard doubles them against your
   expectation. `ep_size` is `dp × tp` only when the DP ranks share one DP group:
   `--data-parallel-size N` alone, without `--data-parallel-size-local
   --data-parallel-address --data-parallel-rpc-port --data-parallel-start-rank`,
   leaves each node's engine with `dp_size = 1`. Two log lines give the width —
   `[EP Rank r/N] … Local/global number of experts` and `rank … in world size … is
   assigned as … EP rank …` — and they are one grep, while decoding the OOM's
   byte count needs a `config.json` the report never carries:
   `moe-load-oom-expert-shard.md`.

7. **A pasted multi-node launch line is audited before any layout is designed.**
   Four structural checks, each with its own error text, all cheaper than the
   sizing arithmetic — and the reporter has usually already run one such line
   ("I tried the official 5.2 section and can hardly run it up"):
   - *World size vs the cards in that one command.* `TP × PP × DP` must be
     satisfiable by the devices the command owns. A `--tensor-parallel-size 16` on
     an 8-card node means one engine spanning nodes, which at vllm v0.18.0 is
     `--nnodes N --node-rank k` — accepted **only** with
     `--data-parallel-backend mp` (`assert self.data_parallel_backend == "mp" or
     self.nnodes == 1`, `vllm/engine/arg_utils.py:1588-1589`) — or the Ray backend
     (`../startup-failures/ray-npu-resource-label.md` owns that one's failure).
     The shape the vllm-ascend tutorials ship is the other one: **DP per node × TP
     inside the node** (`../feature-enable-recipes/moe-expert-parallel-with-dp.md`
     §4, and `moe-load-oom-expert-shard.md` for what the missing DP group does to
     the expert shard).
   - *`--headless` is a role, not a mode.* It sets `api_server_count = 0`
     (`vllm/entrypoints/cli/serve.py:59-67` at v0.18.0) and `run_headless` then
     demands `data_parallel_size_local > 0` (`serve.py:167-168`), so a pasted
     headless line with no `--data-parallel-*` group produces a worker-only
     process and **nothing listening on the client's port** — which is why "it
     failed" arrives with no engine error at all. The node whose address the
     client uses must not be headless.
   - *Long-flag spelling.* vLLM rewrites `_`↔`-` inside `--` flags
     (`vllm/utils/argparse_utils.py:237-246`), so `--speculative_config` is not a
     typo; a misspelled word (`--pipeline-paralle-size`) is not repaired and dies
     at parse time as `unrecognized arguments` before any device work. Diff the
     paste's long flags against the tutorial block first.
   - *Which checkpoint, and which per-card HBM* — rules 5-6 — and note the docs may
     ship **no** recipe for the weights asked about: on Qwen3.5-397B-A17B every
     serve block in the pinned tutorial is the `-w8a8-mtp` artifact, while its BF16
     requirement line is 2× A3 (64G×16) or **4× A2 (64G×8)**
     (`docs/source/tutorials/models/Qwen3.5-397B-A17B.md:21-22`, MP block `:151`,
     Ray section `:262`, at v0.18.0 `e18643f8`). `Ascend910B4/-1` is a real device
     id (`cann/hcomm/src/platform/inc/adapter/adapter_hal.h:47-48`) but its HBM
     size is not in that tree — `npu-smi info` (65536 vs 32768 MB per card)
     decides the arithmetic. "No recipe for the weights you asked about, and the
     document's own minimum is double the HBM you name" is a complete sizing
     answer.
   Precedent: #13591 asked for a non-quantized 4×800I A2 recipe, and its next
   public message was a hand-assembled Ray line carrying three of these defects
   (`--tensor-parallel-size 16 --pipeline-paralle-size 2 --headless`, no
   `--nnodes`, no `--data-parallel-*`). Audit that shape in the first answer
   rather than choosing the path for the reporter: "don't use Ray" is a
   preference the tutorial's own Ray section does not support — say which path
   your recipe uses and why, then correct the line they already have.

8. **A per-token KV number is a term list, not one formula.** On a later-generation
   MoE with sparse attention the serve allocates several KV cache kinds (latent/rope,
   indexer K, indexer scale, a sliding-window cache, compressor state), each with its
   own dtype gate and its own padded page size — and two of those gates flip by SoC
   (`float8_e4m3fn` on A5, `bfloat16`/`int8`/`float16` otherwise), so a footprint
   computed from "the fp8 layout" under-counts by construction and a report that
   measures 3-5× it is describing the layout, not a defect. Reconcile the reporter's
   division against those terms before ranking any promotion hypothesis, and answer
   the context question from the pinned tutorial's per-SoC envelope rather than from
   the arithmetic alone: `dsv4-kv-footprint-terms.md`.

Conditional routes:

- The report prices KV per token from the checkpoint layout, or asks for a 512K/1M
  context on a 64 GB card → `dsv4-kv-footprint-terms.md` (the cache-kind term list,
  the A5-vs-else dtype gates, the DCP indexer replication factor, and the DSV4-Flash
  A2 envelope 133120/135000 against 1M being an A3/950DT row).
- The OOM line is inside model construction (`get_model` → `create_weights` →
  `quantization/methods/w4a16.py:get_weight` → `torch.empty`) on a 384-512-expert
  MoE with `--enable-expert-parallel` and a `--data-parallel-size` in the launch
  dict → `moe-load-oom-expert-shard.md` first: experts-per-card arithmetic, the
  `[EP Rank r/N]` greps, the DP-group flags, and the reporter-confirmed 2-node A2
  serve line (#7123). The TP-scaling page above is the one-node, TP-only sibling.
- Two serve commands that differ by one resource knob (`--tensor-parallel-size`,
  `--max-model-len`, `--gpu-memory-utilization`) where the smaller one comes up and
  the larger OOMs at engine start → `tp-scaling-oom-at-startup.md` first: the
  tutorial's validated serve line, **the traceback frame that separates a
  post-sizing phase from the KV pool's own allocation loop**, the OOM line split
  against `gmu × total` (`../../scripts/oom-budget-split.sh`), the per-rank terms
  that do not shrink with TP, and the stop rule against an allocator-mechanism hunt
  (#6572 for the post-sizing shape, #4474 for the pool-loop shape, where
  `already allocated > gmu × total` is expected and the number to check is
  `#tensors × page_size × num_blocks` against the `Available memory:` line).
- The worked #9335/#8068 case: the 30B-A3B Omni model on 2×64 GB 910B, the
  three-stage NPU yaml, the reporter-confirmed `--stage-configs-path` override
  that runs it on two cards, the `ASCEND_RT_VISIBLE_DEVICES` traps, the card
  arithmetic and the answer shape → `card-budget-and-device-visibility.md`.
- The model's own documented envelope and CI config (which TP the project
  actually runs) → `../startup-failures/max-model-len-budget.md` and
  `../tutorial-verification/tutorial-command-audit.md` (checklist item 5).
- The layout is right and the process still dies during profiling/load →
  `../runtime-memory/index.md` (capacity vs growth vs a dead config key).
