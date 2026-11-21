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

**#14274** vllm-ascend v0.22.1rc1 (`5f6faa0`) + vllm v0.22.1 (`0decac0d`),
`v0.22.1rc1-310p` image, Ascend310P3 EP, Qwen3-0.6B fp16 TP1, `max_model_len`
4096, `gpu_memory_utilization: 0.5`: `Total non KV cache memory: 1.65GiB` then
`Available KV cache memory: 0.09 GiB (halved for workspace)` and the `ValueError:
To serve at least one request with the model's max seq len (4096) …`, with
`npu-smi` at 20,687/44,278 MB and no process visible in the container. The body
already named the mechanism and the precedents (`#7308`, `#7427`, `#7372`), so the
marginal work was verification — and the recorded run did that well: it placed the
cause on `_310p/worker_310p.py:96-130`, showed the generic worker (`#7427`'s shape)
and upstream keep no device-global term, kept the startup check and the RC branch
out of the patch, found that `self.non_torch_memory` is read by the inherited
budget log, and reported the missing `--kv-cache-memory` fast path. Two misses,
both now covered by the page: it never opened `#7308`'s body (the twin carries the
failing/working ratio pair — 0.3/0.3 fails, 0.3/0.6 starts both, v0.15/0.17 print
`-1.33GB`), and it offered "raise `gpu_memory_utilization` above 0.5" as the
workaround without the per-instance framing. The reporter's later public comments
settled both: `gpu_memory_utilization` is a per-instance limit
(`vllm/config/cache.py:66-74`), the 310P override never inherited `#7427`, and they
opened `#14275` with this patch. Route:
`../runtime-memory/kv-budget-per-instance-vs-device-global.md`.

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

**#9026** vllm-ascend v0.18.0 (`e18643f8a4d5`) + CANN 8.5.0, Atlas A3, GLM-5 W8A8
(`hidden_dim=6144`), PD + MTP, `VLLM_ASCEND_ENABLE_MLAPO=1`: `kernel_details.csv`
prints `Input Shapes: N/A` / `Block Dim: 0` for `mla_preprocess_0_mix_aic` (2028
calls, ~103 µs) between rows that carry shapes, and the report self-diagnoses
`OpCommand::SetCustomHandler` as the cause, with an `aclprofRangePushEx` patch
attached. The recorded run refuted the middle link — vllm-ascend's own
`EXEC_NPU_CMD` also calls `SetCustomHandler`
(`csrc/aclnn_torch_adapter/op_api_common.h:584`) — and found `mla_preprocess` in no
`CUSTOM_OPS` build list (so no `aclnnMlaPreprocess` exists to route to), the
runtime's 5 × 8 tensor-info record (`aprof_pub.h:34-35,322-326`), and `acl_prof.h`
stopping at `aclprofMarkEx` while the asc-devkit export carries no `.asc` files at
all (so the proposed API is unverifiable, not disproved). Delivery: **no answer** —
`status=output_truncated` on `output_tokens` while still drafting the ranked list;
the verdict sentence belonged in the first message. Page:
`../performance-throughput/profiler-shape-columns-custom-op-launch.md`.

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

**#7123** vllm-ascend v0.15.0rc1 (`3d43ed99`) + vllm v0.15.0 (`f1764434`), A2
multi-node (2×8 cards, 60.96 GiB reported per card), Kimi-K2.5 W4A16 served from a
launcher dict (`tensor_parallel_size: 8`, `enable_expert_parallel: true`,
`data_parallel_size: 2`, `gpu_memory_utilization: 0.86`, `max_model_len: 4096`, and
no `data_parallel_size_local` / `address` / `rpc_port` / `start_rank` keys): the
engine dies **during model construction** — `get_model` → `kimi_k25.py:349` →
`deepseek_v2.py:1069 make_layers` → `fused_moe.py:320` →
`method_adapters.py:202 create_weights` → `w4a16.py:132 get_weight` →
`torch.empty` → `NPU out of memory. Tried to allocate 674.00 MiB (60.96 GiB total;
60.30 GiB already allocated)`. Track: load-phase capacity under EP →
`../deploy-sizing/moe-load-oom-expert-shard.md`. The recorded run spent 30 tool
results and four message-text-free assistant turns on upstream DP/EP group algebra
(`parallel_state.py:1330-1421`) and on decoding the 674 MiB into an expert count,
and was cut at the 16 384-token cap with **no deliverable** — the two lines that
settle the shard width (`[EP Rank r/N] … Local/global number of experts`,
`rank … in world size … assigned as … DP rank … EP rank …`) were never requested.
Reporter-confirmed afterwards (2026-03-25, same host): the deployment serves with
tp8 / `--enable-expert-parallel` / `--data-parallel-size 2
--data-parallel-size-local 1 --data-parallel-address XXX --data-parallel-rpc-port
2357 --data-parallel-start-rank 0`, `--gpu-memory-utilization 0.95`,
`--max-model-len 72000`, `ascend_scheduler_config`/`torchair_graph_config` off and
`FULL_DECODE_ONLY` with `cudagraph_capture_sizes
[1,2,4,8,16,32,64,96,128]` — i.e. the same 16 cards, with the cross-node DP group
that makes the expert shard 16 wide instead of 8. Also note the pinned tutorial's
"For an Atlas 800 A3 (64G*16) node, tensor-parallel-size should be at least 16"
(`docs/source/tutorials/models/Kimi-K2-Thinking.md:113`) is a per-node A3 statement;
answering "you need TP16" would have been wrong for this A2 host.

**#9443** vllm-ascend `v0.19.1rc1` (`da421af`) + vllm `v0.19.1` (`b1388b1`),
Qwen3.5-122B w8a8-mtp (title says `122B-A3B`, the weights URL says
`Eco-Tech/Qwen3.5-122B-A10B-w8a8-mtp`), 1P1D on A2: "Profiling shows that Full
Attention doesn't have quantization kernel, so this model can't get the W8A8
performance benefits — the performance of bf16 model is as same as w8a8". The
recorded run found the right two-part structure (per-layer int8 linears decided by
`quant_model_description.json`, versus the checkpoint-driven, decode-only
`fa_quant_type` int8 attention/KV) and source-proved the loud-failure property —
`is_layer_skipped_ascend` only skips on a `"FLOAT"` value, and the fused shard
lookups are bare dict reads — so an engine that started with a bf16 class has a json
that says so, and no serve flag can change it. What it under-delivered: the verdict
framed the gap as that artifact's (or its converter's) choice, and the reporter's
public follow-up was "Please check this issue. This issue should be a common
problem that affects performance gains". For an *architectural* layer class
(every-Nth full-attention in a hybrid GDN model) the hole is in every artifact of
the family: say so, hand over the sibling-size `jq` plus the same-request bf16/w8a8
A/B, and route the report to the conversion pipeline. Also note the report's own
name mismatch and that only `Qwen3.5-397B-A17B` and `27B` are in the matrix/nightly
w8a8 configs at that ref. Route:
`../performance-throughput/quantized-artifact-layer-coverage.md`.

**#13591** vllm-ascend `v0.18.0` image (`e18643f8`), 4× Atlas 800I A2 with
`910b4-1` cards ×8 each, non-quantized `Qwen/Qwen3.5-397B-A17B`, no log pasted:
asks for an official example ("DP=? TP=?" in the MP scene, "PP=? TP=?" for Ray) and
the recommended vllm-ascend version, adding "I've already tried to change arguments
on the official docs of the w8a8 version but can hardly run it up". Two facts decide
the sizing without any source read: the pinned tutorial's every serve block is the
`Eco-Tech/…-w8a8-mtp` artifact while its BF16 requirement line is 2× A3 (64G×16) or
**4× A2 (64G×8)** (`docs/source/tutorials/models/Qwen3.5-397B-A17B.md:21-22`, MP
section `:151`, Ray section `:262`), and `910B4/-1`'s per-card HBM is not in the
snapshot — `npu-smi info` (65536 vs 32768 MB) decides whether a 4×8-card layout is
the documented minimum or half of it. The recorded answer delivered the MP
DP-per-node × TP-inside-node block with `--data-parallel-size{,-local,-address,-rpc-port,-start-rank}`
and ruled Ray out; the reporter's next public message was instead a hand-assembled
Ray line carrying three structural defects (`--tensor-parallel-size 16`,
`--pipeline-paralle-size`, `--headless` with no `--data-parallel-*`, no `--nnodes`).
Lesson: audit the launch line they already have (`../deploy-sizing/index.md` rule 7)
rather than choosing the path for them — the tutorial has a Ray section, so "don't
use Ray" is a preference, not a prohibition.
**#5859** reported 0.11.0rc2 image (snapshot pins vllm-ascend v0.13.0rc1 `1b5d5ab` /
vllm v0.13.0 `72506c9`), single 8-card A2 64G, GLM-4.7-W8A8, 32K-in/1K-out: "KV cache
usage 接近 100 % 时 TPOT 激增四五倍，排队 waiting 机制有效但保不住服务性能" — 6/7/8
offered concurrency at 107/114/**440** ms TPOT with KV 79.4/92.4/98.9 %, and 12
offered running at 8 with `Waiting` 4. The run back-solved the pool from two of the
reporter's own rows (33 792 × 6 / 0.794 ≈ 255 K and × 7 / 0.924 ≈ 256 K tokens, so a
≈7.5-request ceiling), pinned preemption-as-recompute (`scheduler.py:775`
`num_computed_tokens = 0` at vllm v0.13.0), the `Preemptions`-only-when-non-zero
logger line, the downstream SLO fork repeating the same reset, and the fact that the
default capture list already contains 8 — so an 8-sequence step with
`num_speculative_tokens = 0` is not the graph ceiling. Route now:
`../performance-throughput/kv-capacity-step-and-preemption.md` (the KV column at the
step is the discriminator against `sequence-count-cliff-capture-size.md`). No reporter
confirmation: the later message only answered the checkpoint-provenance and
MTP-version questions (GLM-4.7 MTP needs main; 0.11-0.13 run without it), which is
why the MTP-era graph candidate is retired for that image but the preemption check
stays owed.
