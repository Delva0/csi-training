# Memory, throughput and sizing cases

HBM growth and OOM after a healthy period, host RAM and the attention-mask
allocation, container working set, KV-init and allocator unmap faults, queueing and
throughput arithmetic, the capture-size cliff, TP scaling budgets and LoRA decode
overhead. Routes: `../runtime-memory/index.md`, `../performance-throughput/index.md`,
`../scheduler-admission/index.md`, `../deploy-sizing/index.md`,
`../lora-adapters/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file.

**#12534** vllm-ascend v0.23.0rc1 (`f4a08bbd`, `v0.23.0rc1-310p-openeuler` image),
torch_npu 2.10.0.post2, CANN 9.1.0-beta.1, Ascend310P3 ×4, Qwen3.5-35B-A3B w8a8 with
MTP: engine start died on all four ranks in `torch.npu.empty_cache()` with
`unmapHandles … 507899` / `ERR00100` / `EL0003`, and the reporter's own suspicion
(`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`) was the trigger. The recorded run
verified the injection block (`platform.py:761-779`), the `memory_profiling` exit site
and the segment Map/Unmap evidence, then offered `max_split_size_mb:256` / explicit
`:False` first — but keep the driver/CANN pairing as an unverified hypothesis.
Reporter-confirmed afterwards: `expandable_segments:False` deploys; the same host then
ran TP=2 and stalled at `shm_broadcast` on TP=4. Route:
`../runtime-memory/allocator-expandable-segments-unmap-crash.md`, which also carries
the 310P graph-budget follow-up and the "the 60 s broadcast line during startup
compilation is not yet a hang" reading.

**#2873** vllm-ascend v0.10.1rc1 (`7e16b4a`) + vllm v0.10.1.1 (`1da94e6`), one
910B host, Qwen3-Coder-30B-A3B at `-tp 8 --max-model-len 262144`: the machine
died during engine start — the log stops at the eight
`rank N in world size 8 is assigned ...` lines with no `Starting to load model`,
then ssh drops ("machine crash and can not restart"; the reporter later added
"已经跑挂了我两台机器" and `npu-smi` showing every card near-empty, then "work well
with 0.10.0rc1"). Cause: every worker builds the eager `max_model_len²` host
attention mask in `NPUModelRunner.__init__` (`model_runner_v1.py:234`,
`attention_mask.py:18-33`) — 576 GiB peak per rank at 262144, so the box dies
before weight load. The mask refactor is #2442 in the v0.10.1rc1 release note
(v0.9.1rc3's #2326 for the other branch) and 0.10.0rc1 predates it. Route now:
`../runtime-memory/attention-mask-host-allocation.md` +
`scripts/attn-mask-budget.sh`. The recorded run found this mechanism on its own
around the 21st tool call, but emitted **zero** message text on every turn and
was cut at the 16 K output cap with the mechanism in the tail and no answer —
the delivery lesson again, on a report where the first line of the log tail was
already enough to state the host-RAM verdict.

**#3314** vllm-ascend v0.11.0rc0 (`00ba071`) + vllm v0.11.0rc3 (`8ce5d319`), image
`quay.io/ascend/vllm-ascend:v0.11.0rc0`, 8 × 910B2, `free -h` 1.5 TiB, Qwen3-30B-A3B
(2507 refresh, ceiling 262144), `--tensor-parallel-size 8` then `4`, both with
`--enable-expert-parallel`, **no `--max-model-len`**: rank assignment lines, nine
minutes of silence, then `Parent process exited` and `WorkerProc initialization
failed …` with no worker traceback; the reporter's own note is "the system memory is
at 100% when the error occurs" and `--max-model-len 65536` works. Same eager host
mask as #2873 (576 GiB peak/rank at 262144 × 4 ranks > 1.5 TiB; 36 GiB at 65536).
The recorded run answered correctly and in budget: it read the mask page first,
verified at the pin that the builder is sized `max_model_len` **only when CANN is
not 8.3+** (`model_runner_v1.py:328-334`), gave the `9·N²` arithmetic plus
`scripts/attn-mask-budget.sh`, and answered the literal "how can I use
max_model_len > 65536" as a host-RAM budget (TP4 ≈ 148K) rather than an engine cap.
The later reporter follow-up confirms the reading and the ceiling ("reduce
max-model-len can circumvent this problem, qwen3-30b model length is 256k"); the
148K figure itself was never reporter-confirmed, so keep it as a budget estimate.
Route: `../runtime-memory/attention-mask-host-allocation.md`.

**#6279** vllm-ascend v0.13.0rc2 (`a17059a`) + vllm v0.13.0 (`72506c9`), 910B,
Qwen3-32B with multi-LoRA in graph mode: a request that selects the **base**
model kills the workers with `Worker failed with error 'CUDA graph capturing
detected at an inappropriate time. This operation is currently disabled.'`
(relayed by `patch_core.py`, owned by upstream `vllm/compilation/monitor.py`).
Request-time lazy capture for a `has_lora=False` key that startup never captured:
the fork's `_dummy_run` takes `activate_lora` and ignores it
(`model_runner_v1.py:2107` vs `:2124`), so both capture cases write the
`has_lora=True` descriptor while the dispatcher advertises both. The recorded run
reached this mechanism around its 20th tool call, then wrote the answer inside
`reasoning_content` and was cut at the cap with no message text — the delivery
lesson, not the diagnosis, is what failed. Route now:
`../lora-adapters/graph-capture-descriptor-mismatch.md`.

**#1812** vllm-ascend v0.9.2rc1 (`b5b7e0e`) + vllm v0.9.2 (`a5dd03c`),
Atlas 300I Duo / 310P, Qwen3-8B: 9 tok/s on the base model, 2 tok/s with
`--enable-lora` and an adapter, no error text. The LoRA path at that ref is the
PyTorch punica backend (`punica_npu.py:6` imports `vllm.lora.ops.torch_ops`;
no AscendC `lora_ops.py` and no `csrc/kernels/bgmv_*.cpp` in the tree) and 300I
is eager-only, so a few thousand tiny launches per decode step cannot be
amortised. The recorded run had the backend within its first few calls (one read of
`punica_npu.py`), then kept walking the 310P gates, `set_active_loras` and
`add_lora_logits` until the cap; its verdict was written and truncated
mid-table. Route now:
`../lora-adapters/decode-overhead-vs-base-model.md`.

**#6833** v0.14.0rc1 + vllm v0.14.1 Qwen3-Next-80B-A3B on 4×910B4, "多并发 请求排队
那么久 + 缓存命中为 0", a seven-sample `loggers.py` metrics block: the recorded run
computed the right ratios by its fourth turn (queue prefill-bound, ~5.5 K tok/s
prefill over ~50 s, `gen tok/s ÷ Running` = 0.5-1 during prefill vs 26 after, KV
peak 47.9 % so no capacity story), then spent fifteen more turns inside the
hybrid-KV-cache, mamba-config and prefix-caching sources and shipped no answer; the
gate it needed is one file (`../performance-throughput/index.md`). The replay of the
same paste, once that file existed, wrote its verdict, arithmetic and gate chain by
turn 19 — but it opened only the cache page, re-derived the queue arithmetic from
`scheduler.py` at identical pins, and was cut mid-sentence in its action ladder.

**#16563** reported 0.23.0RC1 (snapshot pins vllm-ascend v0.26.0rc1 / vllm v0.26.0,
`reported_ref_status: no_named_release`) — DeepSeek-V4-Flash on one 800I A2, TP8 +
`--enable-expert-parallel`, MTP n=1, `--async-scheduling`,
`--compilation-config FULL_DECODE_ONLY`, prefix caching on, served at 20-40 req/s:
"容器内 `container_memory_working_set_bytes` 使用量一直在增加，7 天增加了 80G",
no crash, no log, `npu-smi` flat. Two runs, both cut at the output cap. The run
that shipped no answer emitted nothing but a 69-character opener on its 20 turns
(~48 K characters of reasoning) and was cut inside a reasoning block long after it
had the class right (cgroup host memory, not HBM, turn 2) and the arithmetic in
hand (turn 7). The other wrote `/workspace/scratch/findings.md` and did put a
verdict, five causes, four precedents and the ladder in messages, then was cut
inside its closing version-scope section. Facts across the two runs: 80 GiB/7 d =
8.1 MiB/min ≈ 3.5-7 KB per request and < 16 B per decode step, so per-step host
buffers do not explain the slope; the A2 recipe diff found by the run that read
the tutorial is `--enable-prefix-caching` vs the recipe's
`--no-enable-prefix-caching` (`DeepSeek-V4-Flash.md:170,321`),
`additional_config.enable_flashcomm1` vs the reporter's
`VLLM_ASCEND_ENABLE_FLASHCOMM` (no consumer at the pin; `enable_dsa_cp` without
SP raises `ValueError`, `utils.py:1386-1405`), and
`PYTORCH_NPU_ALLOC_CONF` vs their `PYTHON_NPU_ALLOC_CONF`
(`platform.py:767-785`) — device-side either way. The MTP path's five per-step
`torch.from_numpy(...).pin_memory()` allocations (`model_runner_v1.py:1411-1416`,
under its own TODO, against preallocated buffers at `:407,:452,:458,:564`) are a
real asymmetry, priced out as this metric's owner by the arithmetic but still the
MTP-off ablation row (`../runtime-memory/container-working-set-growth.md`).

**#16365** reported vLLM 0.23.0 + a vllm-ascend `glm-5.3-flash` image (snapshot pins
vllm-ascend v0.26.0rc1 / vllm v0.26.0), one Atlas 800 A2, TP8 + `--enable-expert-parallel`,
GLM-5.3-Flash (no `Glm5*` class in the pinned registry — only `Glm4*`): identical
~8K-token prompts served on build `26bc8ca3` and hanging on build `7f569517` the same
day, then a 7-token probe unanswered for 12+ min while `/v1/models` answered and a
sibling model on the same gateway stayed healthy. A same-reporter follow-up to
#15984, which this corpus does not carry (history stops at 15419), so its "kpool
compressor-state" numbers are unverifiable here. The run had the shape pinned by its
middle turns — `scheduler_reserve_full_isl` default True (`vllm/config/scheduler.py:140`)
passed as `full_sequence_must_fit` on waiting admission (`scheduler.py:944`) and the
waiting loop's `break` on the `None` return (`:949-956`), plus the downstream admission
cap and its #40863 comment (`vllm_ascend/core/single_type_kv_cache_manager.py:273-325`) —
and then spent its whole remaining budget reasoning, so it shipped **no** answer and was
cut at the output cap on its 20th turn. Route:
`../scheduler-admission/waiting-queue-wedge.md`.

**#4766** vllm-ascend `v0.11.0rc3` (`b6d63bb`) / vllm `v0.11.0` (`b8b302c`), 2×910B4-64G,
Qwen3-VL-32B under an ais-bench textvqa loop: "显存一直异常增长，最后 OOM" with no serve
log, no `npu-smi` sample and no crash tail (runtime-memory growth lane). Delivered at
turn 12 from the two-sample split plus a flag audit the pages now carry
(`../runtime-memory/hbm-growth-and-oom.md` 4b/4c): at this ref nothing injects
`PYTORCH_NPU_ALLOC_CONF`, `--block-size 16` is rewritten to 128 by
`platform.py:269-285`, `--disable-mm-preprocessor-cache` is already deprecated and sizes
a host shm buffer rather than HBM, and the effective `max_num_batched_tokens` is 2048
because `_set_default_args` fills it before `SchedulerConfig`, so the multimodal 5120
floor never fires. **Delivery cost**: four assistant turns with zero message text before
the first text, ~16.0 K of the 16 K completion budget.

**#6572** vllm-ascend `v0.14.0rc1` (`52d4acf`) + vllm `v0.14.1` (`d7de043d`), 910B3-64G,
Qwen3-Coder-Next at `--max-model-len 262144 --gpu-memory-utilization 0.80
--max-num-batched-tokens 4096 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`:
TP4 serves, TP8 OOMs at engine start; the body was the two serve commands and the
environment block, no log. Track: TP-scaling budget
(`../deploy-sizing/tp-scaling-oom-at-startup.md`). What the tree settled and the
run found: the pinned tutorial runs this family at **TP4 with `--max-model-len
32768`** and the same four flags (`docs/source/tutorials/Qwen3-Next.md:64`, CI
`tests/e2e/multicard/4-cards/test_qwen3_next.py`), so the reporter's working
command was already 8× outside the documented context and TP8 was never validated;
`get_num_kv_heads = max(1, kv_heads//TP)` with `num_key_value_heads=2` makes the
attention KV per token identical at TP4 and TP8, so the KV requirement is
TP-invariant while weights halve. Delivery: **no answer at all** — 20 assistant
turns, all 8.2 K output tokens, the last 28.4 K characters inside a single
reasoning block that was still oscillating over the allocator mechanism when the
cap hit, with the tutorial envelope, the reporter's gmu ceilings and the OOM line
already in hand. Its first `skill_view` also paid 107 KB for one `SKILL.md` body
(the whole catalogue is 25 K tokens per call) — the reason the support-entry
catalogue is now a topic table. Later reporter follow-ups (same issue, not replies
to these investigators): full graph is not the differentiator (they run it on many
models), and the model was not retested on a newer release.

**#4357** vllm-ascend v0.11.0rc2 (`a2e4c3fe`) / vllm v0.11.0, Qwen3-Next-80B-A3B on
8 visible cards with TP=4, `--max-model-len 32768 --gpu-memory-utilization 0.7
--enforce-eager`, hand-built image adding `Ascend-BiSheng-toolkit` +
`triton_ascend-3.2.0.dev20250914`: "Qwen3-Next-80B too slow 太慢",
`Avg generation throughput: 5.3 tokens/s, Running: 1 reqs`. Track: single-stream
decode throughput on a hybrid GDN MoE
(`../performance-throughput/qwen3-next-decode-throughput.md`). The arithmetic and
the v0.11.0rc2 map were right (the project's own command carries `--enforce-eager`,
`FULL_AND_PIECEWISE` is degraded to `PIECEWISE`, prefill GDN is the pure-torch chunk
loop, `tests/` has no `qwen3_next`), but delivery was cut mid-answer on its 14th
turn (~38 K of reasoning) and the reporter answered "并没有" two days later — so the
graph flip it led with is a hypothesis with a named confirmation line, not a fix,
and the later release's `FULL_DECODE_ONLY` tutorial command is the real lever.
Also missed in-flight: the paste's `served_model_name` is
`Qwen3-235B-A22B-Instruct-2507`, not the reported model.

**#14440** vllm-ascend v0.23.0rc1 (`f4a08bd`) + vllm v0.23.0 (`0fc695f`), 8×910B2,
DeepSeek-V4-Flash-w8a8-mtp (tp=8, EP, MTP `num_speculative_tokens: 1`, FlashComm1,
`FULL_DECODE_ONLY`, `--max-num-seqs 264`): decode throughput falls in one step as
*engine* concurrency crosses 256 (3048 tok/s at 256, 1182 at 258, 1100 at 264),
with `Waiting` 0, KV 81-84 %, AICore 32 % → 8 % and no error at all. Track:
throughput. Mechanism (source-verified at the pins, not yet reporter-confirmed by
A/B): the decode step is `running_seqs × (1 + num_spec)` tokens against
`max_cudagraph_capture_size`, which Ascend defaults to
`min(max_num_seqs × decode_query_len, 512)` (`platform.py:241`), so 264 requests
= 528 tokens > 512 and `cudagraph_dispatcher` returns `NONE` with no log
(`v1/cudagraph_dispatcher.py:276-285`) — every step eager. The reporter's later
public follow-up supplied the localisation this page needed and one shape rule:
the cliff is per engine (dp=2/tp=4 healthy at ≤224 with the aggregate ceiling
doubling), it survives `--max-model-len 1024`, and a bare-HCCL all-gather
microbenchmark is flat at ~0.3 ms from 256 to 640 tokens, which retires the
transport lane. Their seven-row "collapses" ablation matrix is **not** a
refutation: none of its rows carries its own below-cliff baseline, and
`--enforce-eager` deletes the graph entirely. Delivery: the recorded run reached
the ceiling argument by its 11th assistant turn and then shipped **no answer** —
18 turns, 0 characters of message text, ~45 K characters of reasoning, cut
inside a reasoning block; the arithmetic script
(`scripts/seq-cliff-capture-budget.sh`) now produces the same verdict in one call,
which is what `../performance-throughput/sequence-count-cliff-capture-size.md`
tells the next session to do first.

**#3479** vllm-ascend v0.11.0rc0 (`00ba071`) + vllm v0.9.1 (`b6553be1`),
`v0.11.0rc0-310p-openeuler`, 310P3, Qwen3-4B at `--max-model-len 4096
--gpu-memory-utilization 0.8 --enforce-eager`: KV init dies in
`_convert_torch_format` with `Memory async copy failed … length=748683264`,
`sdma copy error`, SMMU Terminate, `507013`. Track: allocation at KV init. The
recorded run loaded `../runtime-memory/kv-init-and-format-cast-alloc.md` plus the
310P page and answered in ~10 turns with the capacity arithmetic (per-token
147,456 B reproduces the log's 26.95 GB pool; the failing copy is one layer's
K+V; `7.56 + 2 × 26.95 GB` vs a 46.43 GB card) and a `--gpu-memory-utilization`
ladder — the page worked as written, and its peak-arithmetic block now carries
that table so the next session does not re-derive it with an ad-hoc snippet.

**#16600** vllm-ascend v0.23.0 tree (`5cb98ca`), 910B2 single card, Qwen3.5-9B
(`/workspace/models/Qwen3.5-9B`), `--max-model-len 16000 --max-num-batched-tokens
32000 --max-num-seqs 20 --gpu-memory-utilization 0.95 --async-scheduling
--skip-mm-profiling --mm-processor-cache-gb 0 --mm_processor_cache_type shm`,
`--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":
[1..20]}`, `--additional-config {"enable_cpu_binding":true,
"multistream_overlap_shared_expert":true,"enable_weight_nz_layout":true}`:
"[Misc] 910B2 + Vllm ascend 0.23.0 + Qwen3.5 9B 速度很慢", asking for a quantized
build for 910B since only 310P ones are published. Track: cross-SoC capability, not
a defect and not measurable from the paste (no metric block). Evidence reached:
`Qwen3.5-Dense` exists only in the `Atlas 300I DUO` tab of
`support_matrix/supported_models.md:161` and is absent from `A2/A3` (`:142-152`);
`tutorials/models/Qwen3.5-Dense.md:9` names Atlas 300I DUO/200I Pro;
`tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py:45,55-60` is the non-310P
launch shape (bf16, capture sizes [2,4,6,8], `method: mtp`) against the 310P
`float16` + `mamba_ssm_cache_dtype` + `qwen3_5_mtp`
(`_310p/test_spec_decode_mtp_310p.py:27,29,31`), with the test-path→SoC routing in
`.github/workflows/scripts/test_config.yaml:22-24`; `enable_weight_nz_layout` is
read by no code at this ref (`ascend_config.py:249-253` uses `weight_nz_mode`;
rename table `configuration/additional_config.md:25`) and
`multistream_overlap_shared_expert` is MoE-only (`:78`). The run put a provisional
verdict in a message at turn 9 and then spent ~15 turns inside `platform.py`
capture-mode gates and hybrid-KV gates before the output cap landed inside the
ranked list — no final answer. Page
`../feature-enable-recipes/per-soc-recipe-and-quantized-weights.md`. Open: the
reporter's token/s, TTFT and `Running`.

**#1138** vllm-ascend v0.8.5rc1 (`ec27af3`) + vllm v0.8.5.post1, DeepSeek-R1-Distill-Qwen-32B
(dense Qwen2) TP=4 on 4 of 8 910B, `--gpu-memory-utilization 0.98 --max-num-seqs 1024`,
body = three screenshots (OCR) of the metrics block and two `npu-smi` pastes: "低 NPU
利用率 (<15 % AI Core)、吞吐有限", asking for extra flags, what caps concurrency at 6, and
profiling advice. The paste decides it: `Running: 6 reqs, Swapped: 0, Pending: 0, GPU KV
cache usage: 0.2 %, CPU KV cache usage: 0.0 %` is the **V0** logger (`vllm/engine/metrics.py:489`),
so 57 tok/s ÷ 6 = 9.5 tok/s per request with nothing queued — no server-side cap — and the pin
hard-sets `enforce_eager = True` (`platform.py:124`) while graph mode is a V0-only
`additional_config` opt-in that `supports_dynamo()`/`VLLM_USE_V1` can void (`:151-160`).
The recorded run spent 12 assistant turns and the whole 16 K output budget on the V0/V1
selection and the V0 runner and shipped **no answer text** (3 status notes). Page
`../performance-throughput/pre-graph-mode-eager-era.md`. Open: the load driver's concurrency
and whether `enable_graph_mode` was ever applied on the reporter's host.

**#7351** vllm-ascend v0.13.0rc1 (`1b5d5ab`) + vllm v0.13.0 (`72506c9`), 2 cards
(`ASCEND_RT_VISIBLE_DEVICES=0,1`, card model not stated), QwQ-32B-W8A8
`--quantization ascend`, TP=2, `--max-model-len 131072`, `FULL_DECODE_ONLY` with
capture sizes `[1,8,24,48,60]`, `--async-scheduling`: "单请求推理时
`QuantBatchMatmulV3_ND` 被调用了 98 万次", evidence = one MindStudio op-statistics
screenshot (OCR is the paste); the public follow-up turned it into a cross-stack
claim, "msprof analyze compare: NPU 上 matmul 调用次数是 GPU 上的 50 倍", with a
second screenshot that is **not** in the case view. Track: profiler-table evidence
(`../performance-throughput/profiler-opcount-evidence.md`). The recorded run
recovered the window exactly from the table's own ratios (`988224/192` =
`329408/64` = `653669/127` = `334555/65` = 5147 steps, ≈256 matmuls/step, the same
shape as upstream's 4 fused matmuls × 64 layers) and answered "cumulative
per-request counts, not a bug", then read the 50× as a window ratio and asked for
the compare columns; both attempts were otherwise sound. Its first turn was cut by
the 16 K output cap **mid-derivation with no message text** (11 steps, arithmetic
still in `reasoning_content`); the retry answered in 3 steps. The reporter's later
public comment confirms the two load-bearing points and adds the one the run
missed: on **v0.13.0-release** the collection *errored* because of a bug in the
shipped `libruntime_v100.so` — a CANN runtime artefact (`cann/runtime`
`cmake/package.cmake:271` installs the `runtime_v100` target), not vllm-ascend —
and after replacing it the data still showed **NPU idle 95 %**, while the same
requests ran at "性能几乎无差异" on 0.13.0rc1 vs 0.13.0. So: equal end-to-end
performance plus an impossible idle reading means the *table* is the fault, which
is why collection health is rule 1 on the page, ahead of the arithmetic. Side
finding kept on the page: `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` was removed by
v0.13.0rc1 (`release_notes.md:35`, PR #5272) and
`VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` has no `vllm_ascend/` reader, so two exports in
that launch command were silent no-ops.
