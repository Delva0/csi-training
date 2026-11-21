# "The whole model is loaded on every card instead of being split"

Report shape (recorded #1316, title only, no command, no log): *"Inference on 4 NPU
cards with `<model>` is currently loading the entire model on each card, rather than
splitting the model across the NPU cards."* Nothing crashed. Sometimes one
`npu-smi info` table is attached showing near-identical HBM on every chip, sometimes
nothing at all.

This is a **sharding-is-on** question with three answers that look identical from
outside, and exactly one log line separates them: the per-rank device-measured
weight footprint. `npu-smi` HBM is **not** that line (see cause 3).

## One number decides it

Every rank prints the device memory its own model load consumed:

```
Loading model weights took X.XXXX GB
```

emitted in the vllm-ascend model runner, once per worker process
(`vllm_ascend/worker/model_runner.py:907` on the V0 runner and
`vllm_ascend/worker/model_runner_v1.py:820` on V1, both at vllm-ascend v0.7.3
`779eebb8`; the value is a `DeviceMemoryProfiler` delta, i.e. device bytes). Compare
it with `params × bytes-per-param ÷ TP`:

| what the line says on each of the N cards | reading |
|---|---|
| ≈ full-model bytes (e.g. ~15 GB for a 7.6 B bf16 checkpoint) | the rank really holds the whole model → cause 1 or 2 |
| ≈ full-model / N (e.g. ~3.8 GB at TP=4) | sharding is working → cause 3, and the reporter is reading a different number |

On the **V0** engine a richer block follows in the same log
(`vllm/worker/worker.py:249-265` at vllm v0.7.3 `ed6e9075`): `Memory profiling takes …`
/ `the current vLLM instance can use total_gpu_memory (X GiB) x gpu_memory_utilization
(0.90) = Y GiB` / `model weights take Z GiB; … the rest of the memory reserved for KV
Cache is W GiB`. On vllm-ascend's own V1 worker there is **no** such block
(`vllm_ascend/worker/worker_v1.py:165 determine_available_memory` computes the peak and
logs nothing), so on V1 ask for `Loading model weights took` instead of promising a
line the build does not print.

## Ranked causes

1. **TP was never requested, so only one card is used.** Nothing defaults to more than
   one card. Offline `LLM(..., tensor_parallel_size=4)`, online
   `vllm serve <model> --max-model-len 4096 -tp 4`
   (`docs/source/tutorials/multi_npu.md:96` and `:53` at vllm-ascend v0.7.3). *Check:*
   the engine's own config line — `Initializing a V0 LLM engine (v…) with config: …,
   tensor_parallel_size=1, pipeline_parallel_size=1, …`
   (`vllm/engine/llm_engine.py:235`; a sample line carrying `tensor_parallel_size=1`
   is in `docs/source/installation.md:255`). `tensor_parallel_size=1` there ends the
   investigation. This is not the same symptom as the title — with one card the other
   three sit at idle, so ask what made the reporter say *each* card.
2. **One process per card.** A shell loop or launcher that starts `N` `vllm serve`
   processes with `ASCEND_RT_VISIBLE_DEVICES=i` gives `N` independent
   `world_size=1` engines, i.e. literally the whole model on every card. At v0.7.3 that
   is the *only* way to get per-card parallelism without sharding: Data Parallel is
   `🟡 Planned` / "Will support in v0.8.x"
   (`docs/source/user_guide/suppoted_features.md:28`), while Tensor Parallel is
   `🟢 Functional` (`:25`). *Check:* `ps -ef | grep -c "vllm serve"` and the process
   table of `npu-smi info` (one process per chip vs one process on `N` chips). Fix =
   one process with `-tp N`; do not tell them to wait for DP.
3. **Sharding is correct and the observation is a false positive.** Two vLLM
   behaviours combine, and both are documented in the pinned tree:
   - **every rank reads the whole checkpoint.** The loader's weight iterator
     (`vllm/model_executor/model_loader/loader.py` `_get_weights_iterator`) walks all
     safetensors files (`vllm/model_executor/model_loader/weight_utils.py:414-422` at
     vllm v0.7.3) on every rank and only the parameter's `weight_loader` keeps that
     rank's slice (`QKVParallelLinear` / `RowParallelLinear` /
     `MergedColumnParallelLinear`, e.g. `vllm/model_executor/models/qwen2.py:136,145`).
     Host RAM therefore looks full-model per process, and on a slow disk every rank
     touches every shard. At that ref the `Loading safetensors checkpoint shards` bar
     is also printed **by rank 0 only** — `enable_tqdm = not
     torch.distributed.is_initialized() or torch.distributed.get_rank() == 0`
     (`weight_utils.py:417-421`) — so "the bar appeared once" is not evidence either
     way.
   - **every card reserves `gpu_memory_utilization × HBM` regardless of TP**
     (`memory_for_current_instance = total_gpu_memory *
     self.cache_config.gpu_memory_utilization`, `vllm/worker/worker.py:233`), and the
     KV pool fills whatever the weights leave. On 8×64 GB cards every chip therefore
     reads ~57 GB used at the default 0.90 with a perfectly sharded model.
   One-flag ablation that needs no source read: rerun with
   `--gpu-memory-utilization 0.5`. If the HBM figure drops on all cards while
   `Loading model weights took` is unchanged, the number the reporter was reading was
   the reservation, not the weights.

## Exclusions worth saying out loud

- **Head divisibility.** `total_num_attention_heads % tensor_parallel_size != 0` raises
  `ValueError: Total number of attention heads (…) must be divisible by tensor
  parallel size (…)` at config time (`vllm/config.py:715-721`). So a 28-head / 4-KV-head
  Qwen2-7B checkpoint can run TP=1/2/4 but not TP=8 — "use 8 cards" is not a free
  option, and KV-head count matters separately when TP exceeds it.
- **Device mapping.** The `local_rank`-to-chip binding is `torch.device(f"npu:{...}")`
  (`vllm_ascend/worker/worker.py:182`) with `ASCEND_RT_VISIBLE_DEVICES` as the control
  variable (`vllm_ascend/platform.py:57`), so "all ranks landed on one chip" is a
  different fault (a single busy chip, other chips idle) and is not this report.
- **`VLLM_USE_V1=1`** is an experimental lane at v0.7.3 (`suppoted_features.md:9`), so
  a V1 run is a legitimate one-line ablation before blaming a sharding defect — but it
  is not a cause of literal replication.

## Answer shape

Name the discriminating line first (`Loading model weights took` / the V0 memory
block), give the arithmetic for the reporter's own checkpoint
(`scripts/card-budget.sh --params <B> --dtype bf16 --tp <TP>`), then the ladder:
confirm `tensor_parallel_size` in the config line → one process (not N) → read the
per-rank weight figure → only then treat it as a defect, and ask for the serve command,
the config line and all N weight lines.

## Scope and limits

Verified at vllm `v0.7.3` @ `ed6e9075` + vllm-ascend `v0.7.3` @ `779eebb8` — the pins
in the recorded report. Line numbers move between releases, and two of the facts above
are era-specific: the rank-0-only progress bar is that ref's `safetensors_weights_iterator`, and "DP not available" stops being true once `data_parallel_size` lands, at
which point "one process per card" becomes a deliberate `--data-parallel-size` choice
rather than a workaround. Re-grep both before quoting them at another pin. Not
verifiable locally: which of the three causes the recorded #1316 run was — the report
carried no command, no log and no card model, so the deliverable was the ranked list
plus the artefact request, not a verdict.
