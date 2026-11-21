# The worker stops right after `Encoder cache will be initialized with a budget of N tokens`

Scope: engine start on a **VL/omni** serve. A worker's last line is

```text
(Worker_TP0 pid …) INFO … [gpu_model_runner.py:NNN] Encoder cache will be
initialized with a budget of <N> tokens, and profiled with <M> <modality> items
of the maximum feature size.
```

nothing is raised, and the EngineCore side then repeats
`[shm_broadcast.py:NNN] No available shared memory broadcast block found in 60 seconds`
every minute. Recorded **#10872**: vllm-ascend `v0.21.0rc1` (`80610e44`), image
`quay.io/ascend/vllm-ascend:v0.21.0rc1-310p-openeuler`, 4×310P, `Qwen3.6-35B-A3B`
(title says `30b-a3b`; a VL MoE), serve with `--tensor-parallel-size 4 --max-model-len 65536
--max-num-batched-tokens 65536 --max-num-seqs 4 --gpu-memory-utilization 0.85
--async-scheduling --no-enable-prefix-caching --enforce-eager --compilation-config '{"mode":0}'`,
and the only log evidence was a screenshot OCR (`budget of 65536 tokens … profiled
with 4 image items … [gpu_model_runner.py:592]`).

Not this topic: a **traceback** out of the same line (→
`../hccl-and-hardware/collective-error-chain.md`); silence that starts at `Loading
model weights took X GB` with no encoder line (→ `silent-stall-after-weight-load.md`);
`shm_broadcast` warnings that accompany a *serving* error rather than engine start
(→ `../hccl-and-hardware/rpc-watchdog-timeouts.md`).

## 1. The two numbers in the line are the reporter's own flags — read them, do not derive them

At vllm **v0.13.0** (the tree this snapshot exports; re-grep per image tag):

| quantity | where it comes from |
|---|---|
| `<N>`, the budget | `min(max_num_encoder_input_tokens, encoder_cache_size)`, and both of those are assigned `max_num_batched_tokens` (`vllm/config/scheduler.py:226-227`; floor of the per-item max kept at `vllm/v1/core/encoder_cache_manager.py:342-348`, `min` taken at `vllm/v1/worker/utils.py:82-83`) |
| `<M>`, the item count | `min(N // max_tokens_per_item, max_num_seqs × max_items_per_prompt)` (`vllm/v1/worker/utils.py:85-128`); `max_items_per_prompt` is `--limit-mm-per-prompt` for that modality |

So `budget 65536 … 4 image items` on #10872 is `--max-num-batched-tokens 65536` with
four concurrent image items on `--max-num-seqs 4`. State that mapping in the first
message: it converts the stall into two flag values the reporter can change.

## 2. What is running when the line stops

`vllm/v1/worker/gpu_model_runner.py:4428-4465`: unless `skip_mm_profiling`, the runner
builds a dummy batch of `M` items at the maximum feature size and calls
`self.model.embed_multimodal(**batched_dummy_mm_inputs)` (`:4462`) — the vision tower
forward, inside `profile_run`. Nothing printed for >60 s after the log line means that
forward (or the dummy preprocessing ahead of it) never returned. The `shm_broadcast`
lines are EngineCore waiting for a worker reply, i.e. the consequence — never rank them
as the cause.

## 3. One flag decides whether the encoder is the owner

- `--limit-mm-per-prompt '{"image":0,"video":0}'` (every non-text modality at 0) makes
  the budget 0, so the whole block is skipped and the log prints `Encoder cache will not
  be initialized` (`vllm/v1/core/encoder_cache_manager.py:321-327`). Startup proceeds →
  the vision-tower dummy forward is the stall; still stalls → the encoder lane is not
  the owner.
- `--skip-mm-profiling` takes the same branch directly (`vllm/v1/worker/gpu_model_runner.py:4428-4432`,
  flag at `vllm/engine/arg_utils.py:969` at this pin). It is a diagnostic: the KV budget
  is then no longer corrected for encoder memory.
- `--limit-mm-per-prompt '{"video":0}'` keeps the cache and changes which modality is
  profiled (`get_modality_with_max_tokens`, `vllm/v1/worker/utils.py:76-80`) — the lever
  the #10872 reporter used in their own next attempt.

## 4. Shrink the profiling batch to the tree's validated recipe

At vllm-ascend `v0.21.0rc1` the project's own 4×310P test of a 35B-A3B MoE runs
`max_model_len=16384, max_num_batched_tokens=2048, max_num_seqs=256`, TP4, fp16, eager
(`tests/e2e/pull_request/four_card/_310p/test_moe_model_310p.py:57-70`
`Qwen/Qwen3.5-35B-A3B`; `:22-37` `Qwen/Qwen3-30B-A3B` with the same knobs) — a batched-token
budget 32× smaller than the recorded report's. Ladder, cheapest first:

1. `--max-num-batched-tokens 2048` (the vision dummy batch shrinks with `<N>`/`<M>`).
2. `--max-model-len 16384`.
3. `--limit-mm-per-prompt` down to one image, or a video-free launch.
4. `--max-num-seqs 1`, then the text-only control.

On 310P the family itself is *documented preview* (`docs/source/tutorials/hardwares/310p.md`,
closing note → #7394), so say that the model×SoC pair has no validated VL recipe at this
pin before ranking flags — `../soc-platform-paths/atlas-300i-310p.md` holds the per-release
document blocks.

## 5. What the paste cannot show

- **Hang or merely slow.** `npu-smi info` while it stalls separates them: flat AICore with
  no HBM growth is a deadlock; rising AICore is an enormous dummy forward. Ask for it —
  one command, and it is the only artefact that decides.
- **Per-rank lines.** A screen capture normally shows TP0 only. One rank's line is not
  evidence that the others stopped at the same place.
- Which of dummy *preprocessing* (host-side decode/resize of synthetic images) and the
  tower *forward* is slow: `VLLM_LOGGING_LEVEL=DEBUG` on one restart bounds it.

## 6. Delivery

This shape is answered in the turn it is recognised: phase (`profile_run` → vision-tower
dummy forward), the two flag values read off the line, the one-flag discriminator, the
recipe-backed ladder, and the artefact requests. Recorded #10872 instead spent 20
assistant turns and ~46 K characters of reasoning in upstream source, emitted one
215-character status note and no answer before the 16 384-token cap. Its later public
trajectory: the reporter dropped `--async-scheduling`, set `--max-num-seqs 1` and
`--limit-mm-per-prompt '{"video":0}'` and moved past the stall into a *different* error
(readable only as a screenshot), so which flag moved it is **not** established.
