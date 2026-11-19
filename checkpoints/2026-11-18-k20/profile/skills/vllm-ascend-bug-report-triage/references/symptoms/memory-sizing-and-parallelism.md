# Per-card memory: "the whole model on every card", "TP is not splitting it"

## Scope

Reports that are a **memory observation, not a crash**: "inference on 4 NPUs loads the entire model on
each card instead of splitting it", "each card uses ~60 GB for a 7B model", "`--tensor-parallel-size 4`
does not reduce memory", "it OOMs immediately while 90 % of the card is already used". They usually
carry no start command, no log and no traceback, so the deliverable is the arithmetic, the two shapes
that produce the observation, and a lever that changes what the reporter sees. Worked report: #1316
(`issue-ledger/index.md`). A number that **grows after** startup until it OOMs is not this page:
`post-startup-oom.md`. Related arithmetic lives in `startup-and-stall.md` §3 (KV-cache OOM at startup)
and `attention-mask/` (a mask-sized allocation); this page is the *expected* memory instead.

## Common rules

- **After a successful start, the resident per-card number is mostly KV cache, not weights.** The
  worker sizes the cache to *fill* the utilization fraction:

  ```python
  # vllm_ascend/worker/worker.py:259-272, determine_num_available_blocks(), v0.7.3
  # (the V1 refs do the same inside determine_available_memory)
  peak_memory = self.init_npu_memory - free_npu_memory      # weights + activations, this rank
  num_npu_blocks = int(
      (total_npu_memory * self.cache_config.gpu_memory_utilization - peak_memory)
      // cache_block_size)
  ```

  with `total_npu_memory` from `NPUPlatform.mem_get_info()` = `torch.npu.mem_get_info()`
  (`platform.py:100-101` at `v0.7.3`). Every rank therefore ends near
  `gpu_memory_utilization × device total` **whatever** the sharding is: 0.9 of a 64 GB 910B is
  ≈ 57.6 GB per card, and a 4-way TP of a 7B bf16 model puts only ≈ 3.8 GB of weights inside that. Same
  statement, from the maintainer of #1316: "memory usage is related to gpu-memory-utilization … kvcache
  size = (total memory size − model size) * gpu-memory-utilization" (`reported`). The sharding itself is
  visible elsewhere in the same file: `worker.py:182` binds `torch.device(f"npu:{self.local_rank}")` per
  rank and `:498-503` calls `init_distributed_environment(parallel_config.world_size, …)` +
  `initialize_model_parallel(parallel_config.tensor_parallel_size, …)`.
- **Weights per rank = parameters × dtype bytes ÷ tp.** A 7B bf16 checkpoint is ≈ 15.2 GB, so TP=4
  puts ≈ 3.8 GB of weights on each rank and TP=1 puts all 15.2 GB on one. Do that division before
  believing either story.
- **`--tensor-parallel-size` defaults to 1** (`vllm/engine/arg_utils.py:121` region at `v0.7.3`:
  `tensor_parallel_size: int = 1`, `gpu_memory_utilization: float = 0.90`). Nothing in
  `vllm_ascend/platform.py:check_and_update_config` rewrites `parallel_config` at `v0.7.3` — it only
  forces the compilation level, picks `worker_cls`, sets `block_size = 128`, the prefix-caching rule
  and `cache_dtype = 'int8'` for FA-quantised checkpoints — so a TP value the reporter passed is
  honoured by vLLM's parallel layers.
- **DP, or one `vllm serve` per card, replicates the weights by design.** Four processes each with
  `ASCEND_RT_VISIBLE_DEVICES=i` each load the whole model: that is data-parallel serving, and the
  "not split" observation is correct but is the expected behaviour, not a defect. `--data-parallel-size`
  does not exist at vLLM `v0.7.3` (it appears by `v0.9.1`), so on `0.7.x` that shape is just N
  independent replicas.

## Separate the two shapes before answering

| observation | what it means | lever |
| --- | --- | --- |
| every one of the N cards near `0.9 × total`, one process, `-tp N` passed | KV cache sized to utilization on top of weights/N — expected | lower `--gpu-memory-utilization` (e.g. 0.5–0.6); KV capacity shrinks, throughput per card drops |
| one card near the full weight size, the others near idle | TP was not requested (`tp=1`) in one process | one process with `-tp N`, or accept one card |
| N processes, each card holding a full model | replicas (DP-style), by design | keep, or move to TP in a single process for one larger KV pool |
| per-card number ≈ `util × total` but requests OOM later | not this page: `startup-and-stall.md` §3, `attention-mask/` for a long prompt, or `post-startup-oom.md` when the message names `NPUWorkspaceAllocator` after minutes of load | — |

Ask for two things when the report allows a follow-up: the `# NPU blocks: N, # CPU blocks: M` and
`Maximum concurrency for … tokens per request: X.XXx` lines printed at cache init
(`vllm/executor/executor_base.py`, `initialize_cache`, `v0.7.3`), and `npu-smi info`'s
**per-process** memory table (it lists each pid's MB per device, which separates weights from KV
immediately). Confirm the effective TP from the log's parallel-config banner rather than from the command
the reporter *intended* — the two differ in exactly the mixed-up reports.

## Workaround and status

- **Workaround** (proposed; the reporter never confirmed on #1316): lower `--gpu-memory-utilization`
  so the KV pool stops filling the card, and keep the model split by running a **single** process with
  `--tensor-parallel-size N`. Cost: smaller KV cache → fewer concurrent tokens; the weights shard is
  unchanged by the utilization flag.
- **Status**: #1316 closed 2025-06-24 by maintainer `Yikun` with the memory identity and a
  matched-CANN/vLLM install link; no follow-up from the reporter. **No code defect was found** at
  `v0.7.3` (see the `check_and_update_config` read above). Report it as a documentation/expectation
  gap — the FAQ explains the identity, the guide's hardware tables do not — not as a TP bug.
- **Falsifiers**: per-process MB ≈ weights/tp + KV (this page) versus one card holding the full weight
  size (TP not requested) versus N full copies in N processes (replicas). A run where lowering
  `--gpu-memory-utilization` does *not* move the number is a different class: look for an allocation
  before `profile_run` (`quantization-load.md`, `attention-mask/`).
- Never conclude "vllm-ascend ignores TP" from a memory screenshot alone: the two shapes above are
  observationally identical without the pid table.
