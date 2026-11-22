# Memory, throughput and sizing cases

HBM growth and OOM after a healthy period, host RAM and the attention-mask
allocation, container working set, KV-init and allocator unmap faults, queueing and
throughput arithmetic, the capture-size cliff, TP scaling budgets and LoRA decode
overhead. Routes: `../runtime-memory/index.md`, `../performance-throughput/index.md`,
`../scheduler-admission/index.md`, `../deploy-sizing/index.md`,
`../lora-adapters/index.md`.

Inherits the scope and rules of `recorded-cases/index.md`, whose page for this
family carries one line
per case in this file.

**#6862** vllm 0.14.1 + vllm-ascend v0.14.0rc1 (`d7de043` / `52d4acfa`), 910B3 x8
idle, Qwen3-32B-W8A8 dense, CANN 8.5.0, torch_npu 2.9.0: "根据Qwen3-Dense文档，开发
FLASHCOMM1后采集prof，显示的仍然是allreduce算子" — a capability report with no launch line,
no profile table and no traceback. The recorded run read the per-forward gate from the
tree and the pinned doc (`num_tokens > 1000` for a dense target) and shipped a complete
ranked answer in eight assistant turns, which is the shape this track should keep: the
verdict is "expected, not a bug", the deciding launch is one profile of a >1000-token
schedule, and the second-tier explanation is a branch that claims the layer before
`enable_sp()`. No reporter confirmation came back, so the gate and the branch order are
source-verified (`../feature-enable-recipes/sequence-parallel-flashcomm.md` §*other entry
shape*), not reporter-confirmed.

**#2099** vllm-ascend v0.9.1rc1 (`c30ddb83`) + vllm v0.9.1 (`b6553be1`), torch_npu 2.5.1,
CANN 8.1.RC1, one 910B3, a ~1B Qwen-VL-class model ("mineru") served over `/v1/chat/completions`
with a base64 image and "Document Parsing:". The body has a client snippet and one screenshot whose
OCR is the only log evidence: every 5 s a `[metrics.py:417]` block with `Swapped` / `Pending` /
`CPU KV cache usage`, prefill 518.9 tok/s in one window then decode steady at 27.9-28.5 tok/s,
`Running: 1 / Pending: 0 / GPU KV cache usage 0.2-0.3 %`, AICore 0-10 % on `npu-smi`. That field
set is the **V0** logger (`vllm/engine/metrics.py:418` at v0.9.1; V1 prints `Waiting` and no
`Swapped`, `vllm/v1/metrics/loggers.py:120-124`), and the reason is upstream, not Ascend: the
`_is_v1_supported_oracle` non-CUDA branch (`arg_utils.py:1443-1450`) calls `_warn_or_fallback`,
which at INFO logs `npu is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.` and returns
`True` whenever `VLLM_USE_V1` is unset (`:1645-1657`). Verified at the pin that V1 is both
available and the recommended engine — Ascend `supports_v1()` → `True` (`platform.py:254-258`),
worker class by env (`:186-187`), the `PIECEWISE … ACL Graph` / `Compilation disabled` banners
(`:161-181`), `release_notes.md:52` ("V0 Engine is frozen … set `VLLM_USE_V1=1`"), and the
singlecard CI running `test_multimodal` on `Qwen/Qwen2.5-VL-3B-Instruct`
(`.github/workflows/vllm_ascend_test.yaml`, `tests/e2e/singlecard/test_offline_inference.py:39,87`)
under `VLLM_USE_V1: 1`. V0 also disables chunked prefill for multimodal
(`arg_utils.py:1455-1464`). The run reached all of this but wrote no user-visible text for ~8
turns and emitted the ranked answer last, cut inside cause 2. Route:
`../performance-throughput/pre-graph-mode-eager-era.md` §Step 3b.

**#8940** vllm-ascend v0.15.0rc1 (`3d43ed99`) + vllm v0.15.0 (`f176443`), Qwen3.5-0.8B on
910B4, no log: "为什么 Qwen3.5 的 linear attention 在 Ascend 上不能像 CUDA 一样融合成一个更大的
算子" — design intent (platform limit vs engineering tradeoff) plus "any plan to fuse". The
recorded answer corrected the reporter's own links (`main` paths `vllm_ascend/ops/gdn.py`,
`patch/worker/patch_qwen3_5.py` and the `releases/v0.19.1` CUDA links do not exist at 0.15.0rc1),
then source-verified the per-lane stage map (`chunk_gated_delta_rule` prefill at `:248`, the single
`fused_sigmoid_gating_delta_rule_update` decode kernel at `:279`, the `is_cuda_graph and
divide_grid < 65536` gate on `fused_qkvzba_split_reshape_cat` at `:58-60`) and refuted "Ascend
splits more" against upstream v0.15.0. Delivered at turn 21. What it could not know (and the
maintainers' later public reply supplied): the roadmap is a mega GDN fusion op (PR #8872)
replacing the chunk op, and the reporter's next question targets the decode lane — that op is
prefill-only, decode falls back to Triton. That per-lane roadmap split belongs in the first
answer. Route: `../performance-throughput/gdn-fusion-lanes-and-roadmap.md`.

**#12534** vllm-ascend v0.23.0rc1 (`f4a08bbd`, `v0.23.0rc1-310p-openeuler` image),
torch_npu 2.10.0.post2, CANN 9.1.0-beta.1, Ascend310P3 ×4, Qwen3.5-35B-A3B w8a8 with
MTP: engine start died on all four ranks in `torch.npu.empty_cache()` with
`unmapHandles … 507899` / `ERR00100` / `EL0003`, and the reporter's own suspicion
(`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`) was the trigger. The recorded run
verified the injection block (`platform.py:761-779`), the `memory_profiling` exit site
and the segment Map/Unmap evidence, then offered `max_split_size_mb:256` / explicit
`:False` first — but keep the driver/CANN pairing as an unverified hypothesis.
Reporter-confirmed afterwards: `expandable_segments:False` deploys — and the run the
reporter calls `两张卡可以跑` is that same TP=4 launch on `davinci4-7`, four dies =
**two** 300I Duo cards; scaling to `--tensor-parallel-size 8` (four cards, eight dies)
stalls at `shm_broadcast`. That follow-up launch carries no `--compilation-config` and
sets `enable_npugraph_ex:false` / `enable_static_kernel:false`, which is **not** an
eager switch: the pin's O2 default is `FULL_AND_PIECEWISE` and takes the piecewise
branch, so it still captures. Route:
`../runtime-memory/allocator-expandable-segments-unmap-crash.md` (post-fix section:
die-count reconciliation, the default-mode fact, the export table) plus
`../startup-failures/mm-encoder-profile-stall.md` §*Scope variant* for the repeating
watchdog line (#12985, same `2 卡能部署 / 4 卡不行` wording).

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

**#1629** vllm-ascend v0.9.1rc1 (`c30ddb83`) + vllm v0.9.1 (`b6553be1`), 2×910B3
TP2, DeepSeek-R1-Distill-Qwen-32B (dense Qwen2, 64 layers) under
`--enable-lora --max-loras 3`, CANN 8.1.RC1: one report with two faces — the
adapter serve runs at 2-3 tok/s while the base-only serve reaches **10-12 tok/s**
(the reporter's own later measurement), and `VLLM_USE_V1=1` never starts because
the memory-profiling forward dies on a dynamo guard inside `bgmv_shrink`'s
`torch.einsum` (`punica_npu.py:6` binds upstream `torch_ops` unconditionally; no
AscendC LoRA kernel at this ref). The recorded run read the attached
`errorlog.txt` (the crash) and screenshot OCRs (the metrics, `lora_request:
LoRARequest(...)`) by its third turn, source-verified the binding and the
upstream Triton contrast, and shipped verdict + anchors + ladder + scope — the
delivery worked; the gap it left open is the one this page's step 3 names (which
engine the *slow* run used, and whether base *requests* on the LoRA server pay
the per-step cost). Routes: `../lora-adapters/pytorch-punica-compile-guard.md`
(the crash) and `../lora-adapters/decode-overhead-vs-base-model.md` (the ratio).

**#2451** vllm-ascend v0.9.1rc2 (`b9f715d6ce30`) + vllm v0.9.1 (`b6553be1`),
910B, Qwen2.5-32B-Instruct TP4 with `VLLM_USE_V1=1 … --enforce-eager`:
"vllm 0.9.1 搭配 mindie-turbo 2.1.rc1 没有吞吐提升，vllm0.7.3 + 2.0.rc2 有提升",
plus the reporter's own observation that the `MindIE Turbo is installed.` log line
they saw on 0.7.3 is missing on 0.9.1. That line is the only probe
(`utils.py:75-85` `try_register_lib` at this pin, silent `except Exception: pass`),
so the plugin never loaded in the 0.9.1 worker and the two 0.9.1 numbers are one
baseline; the recorded run read the helper, the release policy table (MindIE
Turbo column empty from `v0.8.4rc2` up, `2.0rc1` only beside `v0.7.3`) and
`faqs.md:72` in four turns and shipped the verdict, the swallowed-`ImportError`
precedent #1144, and the independent finding that the baseline was measured with
`--enforce-eager` (not the 0.9.x graph-mode path). Route:
`../image-dependency-faults/mindie-turbo-graph-break.md` (variant B).

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

**#17953** vllm-ascend v0.18.0rc1 (`99e1ea0fe685`) + vllm v0.18.0 (`bcf2be961200`), one
910B3 aarch64, Qwen3-0.6B bf16 TP1/DP1 eager: `VLLM_ASCEND_BALANCE_SCHEDULING=1` with
`--max-num-seqs 1` starts and admits nothing (health 200, zero computed tokens), with the
env-off and capacity-2 controls serving; filed as a pin-scoped report whose fix was
already merged, asking for a capacity-one regression test. The run delivered at its 7th
turn and then spent two more turns on correspondence: the guard is real at the pin
(`patch_balance_schedule.py:269` over a zeros queue `:49-52`), and the two facts that
made the answer better than the report were the *writer* gate — `balance_gather` runs
only from `BalanceDPEngineCoreProc.run_busy_loop` (`:631`), which `run_engine_core`
instantiates only for `data_parallel_size > 1 or dp_rank > 0` (`:673-681`), so on DP=1
the queue never leaves zeros and the guard is inert above capacity 1 — and upstream's
`max_num_running_reqs = max_num_seqs` (`vllm/v1/core/sched/scheduler.py:105`), which
widens the wedge to any DP size. Route:
`../scheduler-admission/balance-scheduler-admission-stall.md`.

**#18691** vllm-ascend v0.23.0 (`5cb98caaadeff`) + vllm v0.23.0 (`0fc695fc6d1d`), image
`quay.io/ascend/vllm-ascend:v0.23.0` (local ID `sha256:660ce23a…`), 2×910B3, TP=2/EP=2,
eager, synchronous scheduling, a generated two-layer dummy-weight Qwen3-MoE,
`eplb_config.dynamic_eplb=True`: a controlled failure injection — both planner
subprocesses SIGTERM'd (`/proc` state `Z`), after which the next completion times out at
45.0075 s while `/health` returns 200 both immediately and 10 s later, and the API exits
0 on SIGTERM. A 45 s client deadline proves nothing about the engine, so the report
already carried its own source quotes and a shortened-deadline control; what it could not
see was that `/health` is not a liveness probe at this pin (`AsyncLLM.check_health` reads
`errored` = `engine_dead or not is_running(output_handler)`, and the executor's 10 s
`check_health` RPC has no caller) and that the 211-line
`patch_multiproc_executor.py` is a thin subclass, so the upstream monitor/timeout lines
the report cites are the code that runs. The run reached both facts by its tenth turn and
shipped **no** message at all (21 turns / 34 tool calls, 48.3 K characters of reasoning,
cut at 16 384). Route: `../scheduler-admission/background-helper-death-wedge.md`.

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
models); BF16 on 64 GB caps at gmu ~0.60 for TP8 against TP4 at 0.80, and a W8A8
checkpoint on 32 GB cards needs TP4 gmu 0.85 (more will not start, less does not fill
256K) — both measured points now live in the owner page; and the model was not
retested on a newer release. **Replay with that page**: two messages (the envelope +
budget verdict, then a delta on the reporter's OOM line and ceilings), 10 assistant
turns / 3.6 K output tokens, sub-shape A reached without an allocator walk; its one
defect was quoting the reporter's own ceilings as 「已记录的实测」 before they arrived,
now guarded in the owner page and in `case-layout.md`.

**#6611** vllm-ascend `v0.13.0` (`6281c12`) + vllm `v0.13.0` (`72506c9`), as reported
(`collect_env`: transformers 4.57.6, torch 2.8.0, torch_npu 2.8.0.post2,
triton-ascend 3.2.0, CANN 8.5.0, 8×910B3-64G), Qwen3-Next-80B: startup OOM,
`NPU out of memory. Tried to allocate 2.57 GiB (NPU 0; 60.96 GiB total capacity;
57.49 GiB already allocated; 57.49 GiB current active; 1.96 GiB free; 57.62 GiB
reserved in total by PyTorch)`, frames `worker.py:455` → `model_runner_v1.py:2426` →
`:2450` → `:2499` `torch.zeros(kv_cache_tensor.size)` — the **KV-pool** frame
(sub-shape B of `../deploy-sizing/tp-scaling-oom-at-startup.md`), where `:2499` is the
`linear_attn` branch's full-size tensor, i.e. inside the divergent per-layer
allocation loop. The body carried **no serve command at all**. What the pin settled and
the run found: all four frame line numbers match the tag exactly (so the environment
question is closed and `torch_npu` is `.post2`, not the #6299 leak); the validated line
at this ref is TP4 / `--max-model-len 32768` / gmu 0.8 / `--max-num-batched-tokens
4096` / `FULL_DECODE_ONLY` (`docs/source/tutorials/Qwen3-Next.md:64`), with the v0.11
card-count note gone and CI still 4-cards TP4; `# TODO: REFACTOR ME to sharing hybrid
cache` survives at `:2492` against upstream's one-tensor-per-`kv_cache_tensor`
allocator (`vllm/v1/worker/gpu_model_runner.py:5056-5075`); and `max(1, kv_heads//TP)`
makes the attention KV TP-invariant. Delivery: answered in 5 assistant turns / ~7.7 K
completion tokens — envelope row, frame, identity, `--num-gpu-blocks-override` ladder,
residual left open, missing serve command as the ask — which is the shape the page's
stop rule asks for. Blemish: the answer's opening line was Chinese for an
English-language report (`SKILL.md` rule 1 asks for the report's language).

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
public follow-ups supplied the localisation this page needed and one shape rule:
the cliff is per engine (dp=2/tp=4 healthy at ≤224 with the aggregate ceiling
doubling), it survives `--max-model-len 1024`, and a bare-HCCL all-gather
microbenchmark is flat at ~0.3 ms from 256 to 640 tokens, which retires the
transport lane. The **capture grid** is the missing half of the arithmetic:
upstream floors the injected default to 8/16-step sizes, so the boundary is the
per-engine `--max-num-seqs` while `mns × dq ≤ 512` and `512 // dq` above it — the
tp8 index (`264 × 2 = 528 → 512 → 256`) and the dp2/tp4 index (`224 × 2 = 448 →
224`) both reproduce from their serve lines, and their MTP-off row is predicted
too (`264 × 1` floors back to 256, with `Truncating max_cudagraph_capture_size to
256` in that log). Their second follow-up adds the startup command
(`--max-num-batched-tokens 8192`, `enable_dsa_cp`, `multistream_overlap_shared_expert`,
`enable_cpu_binding`, `block-size 128`), a benchmark table in which TTFT **falls**
across the step (3.2 s → 2.0 s) while per-request wall time goes 65 s → 185 s →
202 s for the same 1024 output tokens and MTP acceptance stays 1.6-1.8 on both
sides — a decode-step cost, with prefill, admission and the drafter all retired —
and three plog/process logs at default level (startup warnings only). Their
seven-row "collapses" ablation matrix is **not** a refutation: none of its rows
carries its own below-cliff baseline, `--enforce-eager` deletes the graph
entirely, and `enable_npugraph_ex: false` is a compile-backend switch, not the
graph mode. Delivery: the recorded run reached
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
`--max-model-len 72000`, `ascend_scheduler_config`/`torchair_graph_config` off,
`FULL_DECODE_ONLY` with `cudagraph_capture_sizes
[1,2,4,8,16,32,64,96,128]`, plus the VL-side `--mm-processor-cache-type shm
--mm-encoder-tp-mode data` — i.e. the same 16 cards, with the cross-node DP group
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

**#2833** vllm-ascend v0.10.1rc1 (`7e16b4a7cdb1`) + vllm v0.10.1 (`aab5498`), 2 nodes
× 8×910B2, GLM-4.5 (MoE, `glm4_moe`), `--data-parallel-size 2
--data-parallel-size-local 1 --tensor-parallel-size 8`, `--max-num-batched-tokens
2048`, `--no-enable-prefix-caching`, no `--quantization`, and `--enable-expert-parallel`
on **node 1 only**: "性能较差" against an H800 run of the same eval, three screenshots
and no log. The body also carries the half the reporter did not ask about — "用
tp+dp+ep 会胡言乱语，所以改成 tp+dp" — i.e. the reported configuration is an accuracy
workaround, and it is what moved every MoE layer off MC2. Source at the pin:
`ep_size = get_ep_group().world_size if enable_expert_parallel else 1`
(`ascend_forward_context.py:87-88`) → `1` = `NaiveMulticast` prefill / `AllGather`
decode, `1<ep_size<16` or any prefill = `All2All`, `≥16` decode = `MC2`
(`_get_fused_moe_state:26-40`), and **nothing logs the choice** at this ref. Recipe
anchor: `release_notes.md:140` points GLM-4.5 at `tutorials/multi_node_kimi.md`
(DP4/2 local, TP8, EP, `--quantization ascend`, mbt 4096, `HCCL_BUFFSIZE=1024`), whose
`torchair_graph_config: {enabled: true}` **raises** for `glm4_moe`
(`ascend_config.py:20-28,189-196`), leaving the experimental aclgraph path
(`:211-215`); no MTP proposer for the family either (`worker/mtp_proposer_v1.py:20-28`
imports only `CustomDeepSeekMTP` while vllm registers `glm4_moe_mtp`,
`registry.py:267`). The EvalScope screenshots reconcile to per-request 4094/100
(Ascend, step 138.7 ms) vs 2041/1026 (H800, step 68.5 ms) and prefill 0.80 vs 1.01 ms
per input token — decode is the axis, so the 2× is the MoE comm path, not prefill
tuning. Delivery: 22 assistant turns, the full 16 384-token budget, 656 characters of
message text and **no ranked answer**; the two fragments shipped were
"The report has two halves …" and "Verified the key source facts". Routes:
`../performance-throughput/moe-comm-path-when-ep-off.md` and
`../performance-throughput/cross-run-ratio-normalization.md`; the garbled EP output
stayed a reporter-confirmed symptom (its later comment is a screenshot of it) with the
mechanism unestablished.

**#14291** reported vllm-ascend 0.24.0rc1 vs 0.23.0rc1 (snapshot pins only
v0.23.0rc1 `f4a08bdd` + vllm v0.23.0 `0fc695f`), A3, GLM-5.2-w4a8c8 PD split
(P DP2 TP16 + `num_speculative_tokens 1`, D DP16 TP2 + `num_speculative_tokens 3`,
`FULL_DECODE_ONLY` capture `[4,8]`, `--max-num-batched-tokens` 8192/128,
`--enforce-eager` on P, DSA c8 flags in `--additional-config`): a `1+1=?` request
returns in 12 s on 0.24.0rc1 against 4 s on 0.23.0rc1, and w8a8c8 is also 12 s. Track:
measurement. The run's ranking (completion_tokens/thinking length first, then the
`num_speculative_tokens:3` acceptance multiplier, then a fixed per-request cost in the
PD chain, then a 0.24 default flip) needed no source and shipped as a full answer in
9 turns; what it could not do is name a 0.24 change — that build is absent from the
snapshot, so the deliverable is the diff command plus the two numbers. Its own
control-finding is the reusable part: w8a8c8 was never run on 0.23.0rc1, so "both
quant widths are 12 s" retires the quant hypothesis and nothing about the version.
Routes: `../performance-throughput/cross-run-ratio-normalization.md` and
`../spec-decode-mtp/acceptance-collapse.md` (the degraded-acceptance row).

**#6299** vllm-ascend v0.13.0rc1 (`1b5d5ab`) + vllm v0.13.0 (`72506c9`), CANN 8.3.RC2,
`torch==2.8.0+cpu` / **`torch_npu==2.8.0`**, 8×910B3 serving GLM-4.7-W8A8 TP8 + EP
with `--gpu-memory-utilization 0.9` and `FULL_DECODE_ONLY` capture sizes: "memory leak
… Memory used keeps increasing", with a `top` paste of eight `VLLM::Worker_TP` rows at
172.8-175.6 GB RES and `npu-smi` flat at `59718/65536 MB` on every chip. The recorded
run classified the paste correctly as host-memory (not device OOM), then spent the
session on per-rank RSS arithmetic, an `RssAnon`/`RssFile` split recipe and a
dead-`additional_config`-key finding, ranked as an unverified hypothesis — and never
read the pta line. The pinned tree already carried the answer:
`docs/source/user_guide/release_notes.md:197` (under `## v0.11.0rc1`) says "The memory
may be leaked and the service may be stuck after long time serving. This is a bug from torch-npu, we'll upgrade and fix it
soon." What settled it publicly: `--no-enable-prefix-cache` changed
nothing, then the maintainer's "upgrade torch-npu to 2.8.0.post1 or 2.8.0.post2. There
is a known issue that 2.8.0 will lead mem leak", then "upgraded pta to 2.8.0.post2 and
rebuilded vllm-ascend. Bug fixed." The ref's own `requirements.txt:31` pins the leaking
`torch-npu==2.8.0`, so the version table matched the release exactly and still carried
the defect. Route: `../runtime-memory/host-ram-and-per-rank-rss.md` §Step 0 (the pta
line before the arithmetic); the dead-key half is still valid and now checked against
`tests/` too (`../runtime-memory/index.md` rule 5).

**#9372** vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be96`), a 32 GB vNPU
serving Qwen3-14B at `--max-model-len 4096 --enforce-eager`: `[Misc]` request to make
the memory message actionable, because `Loading model weights took 27.5376 GB` is
followed by `Available KV cache memory: -0.87 GiB`. The answer shipped complete and
is the reference for this shape: the number is bookkeeping-correct (gmu × 32 GiB
requested vs weights + peak activation + non-torch ≈ 29.7), the *message* comes from
vllm-ascend's own `determine_available_memory` (`worker/worker.py:355-359`, mirrored in
`_310p/worker_310p.py:91-101`, both needing the fix), and a ≤ 0 budget already reaches
upstream's `_check_enough_kv_cache_memory` `ValueError` with suggestions — so the
missing traceback, not the missing text, is what the reporter should check next. Side
finding kept in `../runtime-memory/index.md` rule 5: `--kv-cache-memory` is accepted
and ignored on the Ascend worker at this ref.

**#5124** vllm-ascend v0.11.0rc3 (`b6d63bbd5294`) / vllm v0.11.0 (`b8b302cde434`):
"AclGraph mode underperforms eager mode in Qwen speculative decoding scenario". The
reporter had done the ablation — the padded decode `query` is truncated at
`vllm_ascend/attention/attention_v1.py:674-675`
(`num_tokens = attn_metadata.query_start_loc[-1]`), and deleting it made graph mode
beat eager. The recorded answer found the gate one block above (metadata padding at
`:228-248`, conditioned on `attn_state == DecodeOnly`) and the two cost mechanisms
(a per-layer device→host sync from the device-tensor `query_start_loc`, `:151`/
`:250`, and host-built `actual_seq_lengths_q`/`seq_lens_list` at `:271-272`), then
recommended widening the gate over the deletion. One correction left for this page:
`_build_attn_state` (`worker/model_runner_v1.py:1609-1633`) assigns `SpecDecoding`
**only** for `method == 'deepseek_mtp'`, so the Qwen MTP batch at that ref reaches
the branch as `ChunkedPrefill`/`DecodeOnly` and the "widen to `SpecDecoding`" patch
as phrased would miss it. Page (new):
`../performance-throughput/graph-mode-slower-than-eager.md`.

**#770** reported vllm-ascend `v0.8.4rc2` ("[Usage]: 推理速度太慢"), snapshot pins
vllm-ascend v0.8.5rc1 (`ec27af3`) + vllm v0.8.5.post1. The body was the template stub plus
one screenshot; the attachment OCR is a benchmark table whose `并发数`/`成功率` columns were
lost (`RPS 14.791 / 24.522`, `均延迟 15.752 / 24.555`, `平均TPS 0.402`), so nothing in the
paste fixes model, card count or ISL/OSL. The recorded run answered the era question well
(`platform.py:124` forces `enforce_eager`, `:133` logs `Compilation disabled`, torchair is
the V0-only opt-in for DeepSeek, `pytest.ini:60` blocks `VLLM_USE_V1=1` on NPU) and shipped
it in Chinese — but it also published a Little's-law `in-flight ≈ 233 → 602` from two OCR
cells, and it never reached the scaling lane. The reporter's later comments on the thread
give the setting: 8×910B3 Qwen3-30B-A3B ≈ 6 tok/s, one-card Qwen3-4B 17 tok/s "比 mindie
慢了一半多", "多卡跑更慢 应该是多卡通信机制上没有优化". At the pin that hypothesis is
source-verified for every non-DeepSeek MoE (per-layer TP all-reduce via
`ops/common_fused_moe.py:27-67` → `ops/fused_moe.py:668-670`, MC2 DeepSeek-only and
`VLLM_ENABLE_MC2`-gated, no `enable_expert_parallel` anywhere, eager decode), and the tree
holds no 8-card MoE recipe or throughput number — multi-card tutorials are QwQ-32B TP4
(`multi_npu.md:45`) and DeepSeek-V2-Lite-w8a8 TP4 (`multi_npu_quantization.md:78`) while
`supported_models.md:8-9` prints Qwen3/Qwen3-Moe ✅ and CI covers only `Qwen/Qwen3-0.6B-Base`
single card (`tests/singlecard/test_offline_inference.py:35`). Page (new):
`../performance-throughput/multicard-scaling-and-vendor-baseline.md`.

**#5605** vllm-ascend v0.13.0rc1 (`1b5d5abf865f`) + vllm 0.13.0, Atlas 800 A2
(910B2 × 8) 双机 PD, `DeepSeek-R1-W8A8` (ModelScope), body = two EvalScope screenshots
(`vllm-ascend 0.13 mtp=1 双机性能`, `… mtp=2 双机`): "compared with mtp1, mtp=2 or mtp=3
has some negative performance". No serve command, no log, no `SpecDecoding metrics`. What
the evidence can carry: the decode axis (K changes only the D node), the pin's matrix
(every PD/multi-node DeepSeek MTP recipe is K=1 —
`docs/source/tutorials/pd_disaggregation_mooncake_multi_node.md:255,315,375,435,504,564,624,684`,
`tests/e2e/nightly/multi_node/config/models/DeepSeek-R1-W8A8-A2.yaml:33,55` — while the
colocated page uses K=3), and #3097's pricing (+8.6 % output throughput on the D node with
MTP on, 9.78 TPS/NPU under single-op dispatch against 42.67). **Both** recorded runs
shipped **no message text**. The first: 33 messages, three reasoning blocks of 10.4 K /
10.6 K / 7.6 K characters reconstructing the OCR'd percentile rows and re-deriving `tokens
per step` from `ITL_p50 / TPOT_p50` that the paste did not contain, cut at the cap — while
the summary fields (wall time, output tok/s, request count, concurrency) were in the first
two files it opened. The second (a replay of this same report): reached this entry at
assistant turn 6 and still spent 8.2 K + 10.4 K characters re-verifying the owning page's
citations at the reported pin before writing anything, then lost the remaining budget to
an 8.4 K-character answer skeleton cut mid-sentence — i.e. the ledger here is right that
the ladder is the message, and the failure is deferring it for verification. Ask for the
EvalScope summary text and one `SpecDecoding metrics:` line per K instead of decoding a
screenshot —
`../performance-throughput/spec-decode-gain-accounting.md` §Worked case.

**#8285** vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be9`), 8×910B3-64G,
`[Usage]: 怎么查看当前HBM上kvcache的使用情况` plus "curl metrics 的接口中有相关的统计信息吗" — the
paste is the env dump and `npu-smi` only (whole-card 61.6/65.5 GB, one 58.2 GB process
row), no engine log. An **observability lookup**, not a defect: the answer is two startup
lines (`Available KV cache memory`, `GPU KV cache size`/`Maximum concurrency`),
`vllm:kv_cache_usage_perc` (a block-pool occupancy ratio, not HBM), the
`vllm:cache_config_info` labels, and `--kv-cache-metrics` for per-block residency — while
`npu-smi` cannot split KV out of the card (the componentized split is the open feature
request #2662). The recorded run assembled all of it and kept the right honesty (it
labelled the `vllm:gpu_cache_usage_perc` rename as recalled, and that name indeed occurs
in no file of either export), but used 16 tool turns inside `vllm/v1/core` and
`vllm_ascend/worker/`, emitted no message text until the final turn, and told the reporter
that `--disable-log-stats` leaves `/metrics` usable — the route survives, the **gauge
freezes** (`vllm/v1/core/sched/scheduler.py:1910-1911` returns no stats, and the gauge is
set only from them, `vllm/v1/metrics/loggers.py:1045`). Page (new):
`../observability-and-logs/kv-cache-capacity-and-usage-metrics.md`.

**#1427** vllm-ascend `v0.7.3rc1` (`f025df05f614`) + vllm `v0.7.3` (`ed6e9075d31e`),
torch/npu 2.5.1, CANN 8.1.T18, Qwen2.5-VL-3B on two 910B4 (`ASCEND_VISIBLE_DEVICES=3,6`):
`--dtype bfloat16 --max-num-seqs 16 --max_model_len 32768 --gpu_memory_utilization 0.8
--enable-prefix-caching --multi-step-stream-outputs True -tp 2`, body = env dump plus the
client's latency table at concurrency 1 (200 req / 135.42 s, mean 677 ms, max 12 780 ms)
and 4 (200 / 74.16 s, mean 1 483 ms) and "even worse than transformers based inferring"
with **no transformers numbers**. Sendable at once: 1.48 and 2.70 req/s, Little's law
reproducing in-flight 1.0 and 4.0, 4× concurrency buying 1.82× throughput at ×2.19
per-request latency — the engine batches, something serialises per request. Two flags in
that command are inert at the pin (`--enable-prefix-caching` force-disabled for a
multimodal model under V0, `arg_utils.py:1129-1134`; `--multi-step-stream-outputs`
already default, `:162`) and chunked prefill is off for MM in V0 (`:1170-1173`). The one
pinned up/downstream gap is the vision tower: only `Qwen2VLForConditionalGeneration` is
registered (`vllm_ascend/models/__init__.py:5-9`) and its override calls
`torch_npu._npu_flash_attention_unpad` (`qwen2_vl.py:87-101`), while Qwen2.5-VL keeps
`get_vit_attn_backend` → TORCH_SDPA with one SDPA per image in a Python loop
(`qwen2_5_vl.py:227,304-320`) and both paths pay a TP all-gather per vision block
(`:238-239`). No graph key, no `enforce_eager` override and no V1 worker exist at that
ref, so the v0.8/v0.9 lever lists must not be carried back. Recorded run: loaded the
0.8.x-scoped pages, read the ViT backend, the vision attention and the registry, and
shipped **no message text after its second turn** — 20 assistant turns / ~875 K
cumulative prompt tokens, cut at the output cap with the ranked answer unsent. Page
`../performance-throughput/pre-0.8-vl-vs-transformers.md` (new).

**#8600** vllm-ascend **v0.18.0** reported (pin v0.18.0rc1 `99e1ea0`, vllm v0.18.0
`bcf2be961200`), A2(A+K) 910B host, Qwen3.5-35B-A3B at TP2/DP1 with
`--additional_config '{"enable_cpu_binding": true, "multistream_overlap_shared_expert":
true, "enable_shared_expert_dp": true}'`: "A2相同亲和性的NPU分别执行单卡推理时绑核范围重合
导致性能下降". The body carries `npu-smi info -t topo` (its `CPU Affinity` column pairs
cards: 144-167 for NPU0/NPU2, 0-23 for NPU1/NPU3, 96-119 for NPU4/NPU6, 48-71 for
NPU5/NPU7) and two single-card plan blocks — `NPU0` and `NPU2` both
`main=[146 … 189] acl=[190] release=[191]`, i.e. the same 48-core pool 144-191, and
both `[migrate] … -> NUMA [6]`. The recorded run derived the whole mechanism from
source (`topo_affinity` sizing the pool from the *visible* card's band plus
`extend_numa`'s next NUMA node, `average_distribute` only inside one process, no
cross-process state, A2 not on `global_slice`) and shipped a complete ranked answer in
its final turn — but **no message text until that turn**, and it never read the pin's
own `release_notes.md`, whose two `Known Issues` rows (`:547` at `## v0.11.0`, `:650`
at `## v0.11.0rc2`) say the CPU-bind feature does not work for the multi-instance case.
That pair outranks any source derivation (the project admits the shape), so a
feature-perf report owes one `rg -n 'Known Issues' -A20 release_notes.md`-style pass
before the code walk. Page `../performance-throughput/cpu-binding-multi-instance-overlap.md`
(new); check `scripts/cpu-bind-plan-audit.py`. No follow-up arrived in the snapshot, so
the mechanism is source-verified + the paste's own arithmetic, not reporter-confirmed.

