# Startup cases

Engine-start failures: a silent stall, a worker exit, the metadata/capture phase,
device-open ACL/GE bring-up, and the two device-side cases whose answer starts at
startup (a sleep-mode instance conflict, an MC2 collective at capture). Routes:
`../startup-failures/index.md`, `../hccl-and-hardware/index.md`,
`../runtime-memory/index.md`.

Inherits the scope and rules of `recorded-cases/index.md`, whose page for this
family carries one line
per case in this file. Cases whose owner is an image, package, SoC/version,
capability or document question live in `recorded-cases-install-and-environment.md`.

**#8224** image v0.18.0rc1 + vllm v0.18.0 (`bcf2be96`), Atlas 800I A2,
Qwen3-VL-8B-Instruct in graph mode: engine dies in `profile_run` → `_dummy_run`
with `torch._dynamo.exc.Unsupported: Attempted to call function marked as
skipped` (`posix.putenv`) raised from `ops/rotary_embedding.py:500`
`os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"` behind `query_shape[0] > 65535`,
with `--max-num-batched-tokens 81920`. The model is fine and the reporter's
Qwen2.5-VL control never enters `forward_triton` (no `mrope_interleaved`). This
run found the owning line, the `fullgraph=True` wrapper and the coverage gap, and
shipped a complete ranked answer; the dispatch condition, the launch-side rung
and the check that the code is unchanged at v0.23.0rc1 are on
`../startup-failures/env-write-in-compiled-forward.md`.

**#10481** image v0.18.0rc1 (`99e1ea0f`), Atlas A2 with one visible card,
`ASCEND_RT_VISIBLE_DEVICES=7`, Qwen3.5-0.8B: every rank's `WorkerProc` dies at the first
statement of `Worker.__init__` (`worker.py:109 check_ascend_device_type()` →
`utils.py:723 get_soc_version()` → torch_npu `_lazy_init` → `_npu_init`) with
`ERR99999 PTA call acl api failed` + `set default device id failed, ret: 107001`
(`[FUNC:SetDefaultDeviceId]`, `reason=device id error`) + `invalid device ID` and the
`107002` `ctx is NULL` cascade. `107001 = ACL_ERROR_RT_INVALID_DEVICEID`; the guard never
compared a SoC, so no serve flag and no model question is reachable from that log. The run
also showed the reporter's entire flag set is verbatim
`tests/e2e/multicard/2-cards/test_qwen3_performance.py:61-82`, which retires the flag
hypothesis in one grep — and it over-reached once, calling the `--device /dev/davinci0 \  #注释`
line a docs-template defect although both pinned trees list `davinci0` … `davinci7` on clean
continuations (`installation.md:434-441`). The shell behaviour itself is real (the `#` ends
the `docker run`, so only `davinci0` is mapped and the `-v` lines fail as separate
commands), which is why `ls /dev/davinci*` inside the container is the first check —
`../startup-failures/device-id-invalid-at-worker-init.md`.

**#2473** vllm-ascend v0.10.0rc1 (`4604882a3e2c`) + vllm v0.10.0
(`6d8d0a24c02b`), Qwen3-30B-A3B-W8A8 (msmodelslim) on 910B4 ×8, TP8 +
`--quantization ascend` + `--enable-expert-parallel`, `--max-model-len 4096
--max-num-batched-tokens 8192 --max-num-seqs 32`, chunked prefill + prefix
caching: engine start dies at graph capture with
`AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027`, `ERR00100
PTA call acl api failed`, `EE9999 … Not allow to synchronize captured-stream,
stream_id=9`, deepest frame `compilation/piecewise_backend.py:194` →
`entry.runnable`. Track: capture-phase host sync inside the W8A8-dynamic MoE
all2all dispatch (`w8a8_dynamic.py:1001` → `:384` `torch.bincount` in the
`init_routing_quant` fallback, `:446-448` still syncing on the newer branch);
`ERR00100` is the generic wrapper, not a fabric fault. The report carried **no
serve command** — `--enable-expert-parallel` was inferred from the code path,
which the reporter's later comment confirmed along with the cheapest
discriminator nobody had asked for: the unquantized Qwen3-30B-A3B-Instruct-2507
with `--quantization None` and identical flags serves clean, so the quantization
axis owns the extra sync (and the bf16 EP path's own `bincount` at
`ops/fused_moe.py:369` is the follow-up question that keeps it honest). The run
delivered its ranked answer on the 12th turn. Page:
`../startup-failures/captured-stream-host-sync.md`.

**#2064** vllm-ascend reported as `v0.9.2rc1(e7d32ed3f18d)` with vllm v0.9.2, image
path `/__w/vllm-benchmarks/.../vllm-empty/vllm/` (a source build), OpenGVLab/InternVL3-9B,
`--trust_remote_code`: eager dies at worker import with `ImportError: cannot import name
'bind_kv_cache' from 'vllm.v1.worker.utils'`; graph mode dies earlier, in
`check_and_update_config` → `update_aclgraph_sizes` → `get_max_hidden_layers` →
`hf_config.to_dict()`, inside the checkpoint's own `configuration_internvl_chat.py:87`
(`AttributeError: 'InternVLChatConfig' object has no attribute 'llm_config'. Did you mean:
'sub_configs'?`). Track: version **pair** plus a composite-config read. The reporter's
`utils.py:292/321` and `model_runner_v1.py:71` match the **v0.10.0rc1** tree (`utils.py:291-292/321`, `model_runner_v1.py:70-71`) to within one line
(`4604882a3e2c`), whose vllm counterpart is 0.10.0 — where `bind_kv_cache` moved to
`vllm/v1/worker/utils.py:120` (vllm 0.9.2 has it at `vllm/v1/utils.py:278`, and v0.9.2rc1's
`model_runner_v1.py:66` imports it from there). So the printed tag is only the nearest tag.
The recorded run answered "your build is not the release tag" and called the graph half a
residual hypothesis; the sibling snapshot settles it — `get_max_hidden_layers` exists at
v0.10.0rc1 and the flat read is recorded raising for InternVL2_5-38B at v0.9.0rc2 (#1276,
`utils.py:134`), so the class is real at a pinned release. Pages:
`../startup-failures/environment-correspondence.md`, `../startup-failures/vl-layer-count-attribute.md`.

**#7352** v0.14.0rc1 (still broken at v0.17.0rc1) DeepSeek-R1-Distill-Qwen-1.5B
1P1D Mooncake PD, 910B4: prefill came up while the decode pod died in
`register_kv_caches` → `KVCacheRecvingThread.__init__` with
`AttributeError: 'Qwen2Config' object has no attribute 'head_dim'` (a flat config
whose `head_dim` is derived, not a nested-VL config — `hf_text_config` was already
in use), and the same reporter's next round hit
`ImportError: ascend_transport.so` that turned out to be the launcher's effective
`LD_LIBRARY_PATH` rather than the image; both lanes are now in
`../startup-failures/connector-config-attributes.md` and
`../image-dependency-faults/launch-env-and-library-path.md`. The first message named
the owning line correctly; the **follow-up segment** then re-derived the Mooncake
plugin's `DT_NEEDED` chain from the `kvcache-ai/Mooncake` sources over 18 tool calls
and was cut with no message text, although both pages above already held that
segment's answer — a new-turn rule now sits in `SKILL.md` rule 1;

**#1496** vllm-ascend v0.8.5rc1 (`ec27af3`) + vllm v0.8.5.post1 (`3015d563`),
Qwen2.5-Math-1.5B, `tensor_parallel_size=4` with
`distributed_executor_backend="external_launcher"`, torch 2.5.1 / torch_npu 2.5.1 /
CANN 8.1.rc1: "TP=1 works, TP>1 deadlocks during driver initialization", the paste
ending after the worker's device line with no traceback, and both an `LLM()` example
and an `AsyncLLM` script reproducing it. This ref carries no `external_launcher`
string at all, so the running mechanism is upstream's; `init_process_group` is passed
`world_size=parallel_config.world_size` explicitly, which is why fewer processes than
TP hang instead of raising the `reshape` error. The recorded run shipped **no message
text** in 12 turns (~49 K characters of reasoning spread over four mechanisms), so the
lane now has its own ordering and its own page:
`../startup-failures/multiprocess-launch-hang.md`.

**#18130** vllm-ascend `v0.27.1rc1` (`3b31886`) / vllm `v0.27.1` (`6e448d0`),
GLM-5.3-w8a8c8 on 4×Atlas 800I A2 (910B2) multi-node co-located, command copied
from the v0.27.1rc1 GLM5.3 tutorial: `fuse_muls_add / Unexpected keyword argument
[type=unexpected_keyword_argument, input_value=True, input_type=bool]` at startup.
Answered in 17 turns with the whole mechanism: the key is nested-only
(`AscendCompilationConfig`, default `True`), the tutorial's serve command still
prints it flat although its own header says the scripts are based on v0.23.0, and
`AscendConfig` is `extra="forbid"` since the pydantic migration (#15418). The
reporter's later comment quotes exactly that tutorial page with that JSON, so the
document as the source is confirmed (whether deleting the key then started the
service is not reported). Lane worth copying: classify from the pydantic error *type* plus the
missing `WorkerProc`/`Starting to load model` lines, then one grep for the key's
owning class (`../startup-failures/additional-config-schema.md`).

**#1895** vllm-ascend `v0.9.2rc1` (`b5b7e0e`) + vllm `v0.9.2` (`a5dd03c1`), 4×910B3,
Qwen2.5-32B, TP=4: v1 graph mode dies at startup with
`NotImplementedError: _C::rotary_embedding: attempted to run this operator with Meta
tensors`, after weights load and before capture. **Answered completely** in 29 turns
with a `path:line` table and correct confidence labels: the Ascend rope monkeypatch
(`ops/rotary_embedding.py:286`) takes the fp16-gated `torch.ops._C.rotary_embedding`
branch (`:29-31, 60-70`) whose only exemption is torchair (`:42-48`), and
vllm-ascend's own binding registers it `PrivateUse1`-only
(`csrc/torch_binding.cpp:290,299,302`, no `kMeta` anywhere in `csrc/`); the
`Failed to import from vllm._C` warning is normal on NPU and the crash itself proves
`vllm_ascend_C` loaded. Ladder given: `--enforce-eager`
(`platform.py:160-162`), then `--dtype bfloat16` as a labelled hypothesis (discriminator:
the new error names `npu_rotary_embedding`), then a maintainer-side Meta impl. Cost
note: the verdict text arrived only at the 7th of 11 substantive turns, after ~12 K
characters of reasoning; the pasted traceback's missing middle frames were correctly
listed as the open gap. Later-ref correspondence added from #8590's tree: v0.14.0rc1
moved the op to `_C_ascend` and registers Python Meta impls
(`vllm_ascend/meta_registration.py:108-111`, imported at `utils.py:255-258`) plus
`csrc/torch_binding_meta.cpp` — `../startup-failures/meta-kernel-missing.md`.

**#4309** vllm-ascend v0.11.0rc1 (`c5fe179cef7f`) + vllm v0.11.0
(`b8b302cde434`), 8×910 64 GB, `Qwen/Qwen2.5-7B-Instruct-1M` served with
`--max-model-len 200000 --dtype float16`: `Starting to load model` then
`AssertionError: Torch not compiled with CUDA enabled` at upstream
`rotary_embedding/dual_chunk_rope.py:37`, reached from `get_rope`'s
`dual_chunk_attention_config` branch; the reporter had already found that deleting
the key from `config.json` works. Track: checkpoint-config → upstream CUDA-only
path (`../startup-failures/checkpoint-config-cuda-only-path.md`, page written from
this case). The session shipped a complete answer in 12 assistant turns with the
verdict on turn 7, and got the useful extra facts: the siblings in the same package
use `current_platform.device_type`, `vllm_ascend/ops/rotary_embedding.py` patches
only four rope classes, and nothing under `vllm_ascend/attention/` mentions the
feature's fields. Two wasted calls worth avoiding: `rg -rn "dual_chunk"
/workspace/history` (the `-r` replacement trap of SKILL.md rule 7 — it printed
`self.n_attention_config` and nearly became a lead) and a `history-titles.sh`
alternation that ORed a common model name into the regex (53 hits, none relevant).

**#8345** vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be9`), Qwen3-32B
(**dense**) under PD with `MooncakeConnectorV1`, `--data-parallel-size 2
--data-parallel-rank 0 --tensor-parallel-size 8` and
`kv_connector_extra_config.prefill/decode.dp_size = 2`: `ValueError: KV transfer
'prefill' config has a conflicting data parallel size. Expected 1, but got 2.`,
identical for DP=2 and DP=4, evidence only a screenshot (OCR = the paste). Track:
startup config validation, downstream raise site
(`../startup-failures/arg-validation-errors.md`, new second-raise-site section).
The mechanism the tree supports: upstream rewrites a non-MoE engine's DP to 1 in
`vllm/v1/engine/core.py:1061-1073`, the check compares against
`data_parallel_size`, and the extra-config numbers have no other reader — so
`dp_size: 1` is the one-line fix and the keys cannot be deleted (connector
asserts). Delivery: answered on turn **15** (G1/G2 missed), after ~41 K characters
of reasoning against 3.4 K of message text, and cut inside its final section; four
turns went into `ParallelConfig`/`arg_utils` before the twelve-line
`run_engine_core` branch resolved it — the cheap route is
`rg -n 'is_moe\|data_parallel_size = 1' vllm/v1/engine/core.py`.

**#2015** vllm-ascend v0.9.2rc1 (`b5b7e0e`) + vllm v0.9.2 (`a5dd03c`), 910B
(Ascend910B2C) 16 cards, `DeepSeek-R1-bf16-hfd-w8a8`, offline `LLM(...)` with
`enforce_eager=True` and **no** `quantization=`: `RuntimeError: Engine core
initialization failed. See root cause above.` with no worker traceback. Track:
startup failure, quantization gate in the *absent* direction
(`../startup-failures/quantization-method-gate.md`, section B). Mechanism the
pinned tree documents: the msmodelslim W8A8 layout deliberately has **no**
`quantization_config` in `config.json`
(`docs/source/user_guide/feature_guide/quantization.md:37`), the feature is enabled
"by specifying `--quantization ascend`" (`:5`, `:94`), and the page's own FAQ maps
`KeyError: 'xxx.layers.0.self_attn.q_proj.weight'` to "first, make sure you specify
`ascend` quantization method" (`:99-101`) — so upstream's auto-detect
(`vllm/config.py:881-895`, gated on `quantization_config`/`compression_config`)
has nothing to read and the int8 tensors meet a dense model inside `load_model`
(precedent #813, same `KeyError` shape). Second cause, with arithmetic: 671 GB of
int8 weights / 16 cards = 41.9 GB against a 51.2 GB budget (`0.8 × 64`), leaving
9.3 GB for activations *and* an unsharded MLA KV of several GB per rank at the
default 163840-token context. Excluded: #1458's ACL-graph refusal, because
`enforce_eager=True` satisfies `ascend_config.py:158-167`. Delivery: **no answer
shipped** — 17 turns / 25 tool calls, ~45 K characters of reasoning against 4.0 K
of message text, cut on a tool call, after the documented quantization gate had
already been read at turn 11. Cost detail worth keeping: `scripts/card-budget.sh`
was called with `--params 671e9` (a raw count, not billions) and printed a
confident inverted FAIL which the run then had to disown and redo by hand — the
script now refuses that input.

**#3098** vllm-ascend v0.10.2rc1 (`048bfd5`), ray over 2 nodes x 8 910B4,
Qwen3-Next-80B-A3B-Instruct: "该ray集群部署Qwen3-30B-A3B-Instruct正常运行，而部署
Qwen3-Next-80B-A3B-Instruct报错", with no error text at all — the body pastes only the
ray cluster resource view. Track: the model-specific support surface of the first
release that claims the model. Facts at that ref: the release note
(`user_guide/release_notes.md:9`, PR #2917) says EP and MTP "doesn't work with this
release"; the family tutorial makes triton-ascend + the BiSheng toolkit a manual
per-node, aarch64-only install (`tutorials/multi_npu_qwen3_next.md:41-68`, `:75-78`)
and validates exactly one shape (`:89`: `--tensor-parallel-size 4 --max-model-len
4096 --gpu-memory-utilization 0.7 --enforce-eager`); the model asserts no prefix
caching (`vllm_ascend/models/qwen3_next.py:1186`). The recorded run answered in 7
turns, ranking "node 2 lacks triton-ascend / has a different vllm-vllm_ascend" first
— and the reporter's later clarification (the working sibling is
`Qwen/Qwen3-30B-A3B-Instruct-2507`, not the original 30B) took away its third
candidate: the 2507 refresh carries the same 262,144 context as Qwen3-Next, so the
eager `max_model_len²` mask arithmetic could not separate the pair
(`../runtime-memory/attention-mask-host-allocation.md`, section on works/fails
pairs). Route now: `../startup-failures/environment-correspondence.md` (per-worker
environment on ray) plus the Qwen3-Next pages. Delivery: the answer shipped, but the
cap landed inside its own ladder and dropped the closing artefact list.

**#12015** vllm-ascend v0.22.1rc1 (`5f6faa0`, `v0.22.1rc1-310p-openeuler` image) +
vllm v0.22.1 (`0decac0d`), Atlas 300I DUO (310P), TP=1: the title says
Qwen3-14B-w8a8sc, the command serves `/model/Qwen3-VL-2B-Instruct/` in fp16 with
`--enforce-eager`, a `FULL_DECODE_ONLY` `--compilation-config`, prefix caching left
on, and the 310P doc's `--additional-config … fuse_norm_quant: false`. After
`rank 0 in world size 1 is assigned …` the EngineCore child died:
`!!!!!!! Segfault encountered !!!!!!!` with a dump whose top frame is a **CPU**
`torch.zeros` fp16 fill (`at::native::fill_kernel` → `VectorizedLoop2d` →
`fill_non_native_type<c10::Half>`), then eight
`[ERROR] TBE Subprocess[task_distribute] … main process disappeared!` lines and the
parent's `Failed core proc(s): {'EngineCore': -11}`. Track: signal death at engine
start on a combination the release's own 310P table does not validate. Delivery:
**no answer at all** — 20 turns and the whole 16 K output budget went into
investigation (fusion-config plumbing between `additional_config` and upstream
`pass_config`, the 310P mask builder, the doc's validated rows) and the cap landed
on a status note; it also broke G1/G2 by emitting no verdict text until turn 19.
What the next run should have: the startup section of
`../hccl-and-hardware/worker-segfault-no-python-frames.md` (child = EngineCore
spawn, `-11` = SIGSEGV, TBE block = messenger, trivial CPU fill = surfacing site)
and the v0.22.1rc1 block of `../soc-platform-paths/atlas-300i-310p.md` (the
`_310p/` subpackage, the doc's validated table, the inert fusion flag, the
`max_model_len`-sized mask). Root cause stays unresolved — no plog in the paste and
no maintainer answer in the corpus; the ranked candidates are the image/CANN/
torch_npu pairing (#10136 family), the unvalidated model/feature combination, and
host-memory pressure during init. **A later re-run of the same paste at the same pin
shipped it in 7 turns / 13 tool calls** off exactly those two pages: verdict first
(SIGSEGV in the EngineCore spawn; the CPU `torch.zeros` fill is the surfacing site,
the eight TBE lines are messengers; the title's 14B-w8a8sc is not the served
checkpoint), then ranked candidates with one check each and the doc's 14B row as
rung 1. Two facts it still had to derive by hand, now written into the v0.22.1rc1
block of `../soc-platform-paths/atlas-300i-310p.md`: that pin's doc carries no
eager-only/float16-only note and its rows pass `FULL_DECODE_ONLY` without
`--enforce-eager`. Its one wrong first guess was
`../soc-platform-paths/310p-vl-rope-cos-nullptr-startup.md` (VL + 310P looked right,
but that lane needs `Check cos != nullptr`), now excluded on that page.

**#6972** vllm-ascend v0.14.0rc1 + vllm v0.14.1, 8×910, Qwen3-Omni via
`vllm serve --omni`: `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",
"cudagraph_capture_sizes":[1,2,4,8,16,24,48]}'` and a stage yaml
`engine_args.compilation_config: {level: 1, …}` both "未生效". Track: config gate
audit, not a crash. The run shipped **no answer at all** — 54 messages, 1.27 M prompt
tokens, the whole 16 K output budget inside reasoning and tool calls, G1/G3 both
broken — although it had the decisive finding at hand: `level: 1` maps to
`CompilationMode.STOCK_TORCH_COMPILE` (`vllm/config/compilation.py:748-758`,
`:355-369`), which vllm-ascend refuses by setting `cudagraph_mode = NONE`
(`platform.py:241-245`, warning only). It also found that vllm-omni is absent from the
snapshot (`--omni` has no hit in vllm v0.14.1, so the CLI→stage merge is unreadable)
and that the Qwen2.5-Omni tutorial's `--compilation-config '{"full_cuda_graph": 1}'`
names a key this ref's `CompilationConfig` (`extra="forbid"`) does not define. Page
`../startup-failures/graph-mode-not-applied.md`.

**#698** image `vllm-ascend:v0.8.4rc1` (pin `086423d`), vllm v0.8.4 (`dc1b4a6f`),
910B4, DeepSeek-R1-Distill-Qwen-1.5B, `--max-model-len 10000 --dtype float16
--tensor-parallel-size 1 --enable-reasoning --reasoning-parse deepseek_r1`: the log
ends at `[model_runner.py:904] Loading model weights took 3.3461GB`, nothing else,
HTTP port refused. Track: startup silent stall. The paste's line number reproduces
in the pin's **V0** runner (`worker/model_runner.py:904`; V1 is
`model_runner_v1.py:781`, and vllm's own `model_runner.py` carries no such string),
and the next code is `worker.py:240-256` `determine_num_available_blocks` →
`profile_run()` ending in `torch_npu.npu.synchronize()` (`model_runner.py:1078`) —
nothing captures graphs at that ref (`worker.py:325-326`). Both flags the reporter
used are valid at vllm v0.8.4 (`arg_utils.py:216-217,979-994`), so the command is
not the fault. The recorded session shipped **no answer** (14 turns, 11 bare tool
calls, ~47.6 K characters of reasoning against 322 characters of text); the public
thread moved only when the maintainer asked for the startup command and the
reporter supplied the `vllm serve` line — the ask, not a mechanism, was the
productive step. Page `../startup-failures/silent-stall-after-weight-load.md`.

**#11095** vllm-ascend v0.21.0rc1 (`80610e44`) + vllm v0.21.0 (`ad7125a4`), image
`quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler`, 2×910B3 (64 GB), TP=2,
Qwen3.5-27B, `--max-model-len 133000`,
`--speculative-config '{"method": "qwen3_5_mtp", …}'`,
`--additional-config '{"enable_cpu_binding":true}'`, container
`-v /mnt/xxx/model:/model`: the serve hangs after
`Loading safetensors checkpoint shards:   0% Completed | 0/11`, no error, both
`VLLMWorker_TP` processes alive at 26 618 MB each, port never opens. Track:
startup, silent stall **inside** the shard loop (`model_runner_v1.py:3635`
`get_model`, before `Loading model weights took` at `:3666`). The session shipped
**no answer** (11 turns, 606 K prompt tokens, the full 16 K output budget in
reasoning): it derived "HBM ≈ checkpoint/TP ⇒ the weights are already loaded, the
`0/11` frame is stale", and every later turn was post-load code (`qwen3_5`
registration, `gdn.py` `_warmup_prefill_kernels`, NZ cast, cpu binding) — an
inference the pin refutes, since `vllm/model_executor/model_loader/base_loader.py:53-64`
builds the model under `with target_device:` before `load_weights`, so the
parameters are on the device before the first shard is read. It also used
`weight_utils.py:938` `Filesystem type for checkpoints: EXT4` to dismiss the
storage lane; that line is `_get_fs_type`'s `/proc/mounts` lookup (`:784-812`),
used only to decide prefetching. The reporter's own later comment
(issuecomment-4831844713, 2026-06-29) resolved it by copying the checkpoint into a
**new host directory** and re-running the container against it — `--name` changed,
*"其他容器启动命令不变"*, *"真的是奇奇怪怪！！！"* — i.e. a reporter-confirmed outcome
on the checkpoint **source** with the mechanism unsettled. Page
`../startup-failures/weight-load-stall-in-shard-loop.md`, one call
`../../scripts/shard-source-probe.sh`.

**#2973** `[Doc]` vllm-ascend v0.10.2rc1 (`048bfd5`; upstream export is vllm
v0.9.1 `b6553be`, `dated_reference_only`), one node, running the repo's own
`examples/offline_external_launcher.py --tp-size=4 --proc-per-node=2
--enable-expert-parallel` on Qwen3-30B-A3B: every rank dies in
`initialize_model_parallel` with `RuntimeError: shape '[-1, 1, 1, 4]' is invalid for
input of size 2`. Not a defect and not lane C — the script sets the world size from
`--node-size × --proc-per-node` (`:251`) while `--tp-size` is an independent value,
and upstream skips its own `init_process_group` when the group already exists, so
nothing compares the two. Delivered: the two numbers decoded (`[-1, dp, pp, tp]` vs
real world size), the documented `--tp-size=2 --proc-per-node=2` ladder, and three
doc defects found in the same file (the invariant is never stated; the multi-node
Node 1 block omits `--proc-per-node=2`, so it computes world 2 against Node 0's 4;
a world size that *is* a multiple of TP silently becomes external-DP groups).
16 tool calls / 9 turns / 9.3 K completion tokens, and the walk was longer than the
message needed: one grep of `world_size|tp_size|node_size|proc_per_node` in the
example gave the whole mechanism at call 2, and the `docs/source/` grep for the
example's name returned nothing (it is not documented outside the docstring). Page
`../startup-failures/arg-validation-errors.md` (third raise site). Open: how many
NPUs the reporter's node has, and whether the 2-process gloo group formed cleanly.

**#955** vllm-ascend image `v0.8.4rc2` (env dump says `vllm-ascend 0.8.5rc1`,
vllm 0.8.5.post1, CANN 8.1.RC1, torch-npu 2.5.1), Qwen2.5-VL-7B-Instruct on 910B2C
×10, `tensor_parallel_size=4`: "[Bug] … Failed with HeaderTooLarge Error". All four
ranks die in the same second as `Loading safetensors checkpoint shards: 0% … 0/5`
with `safetensors_rust.SafetensorError: Error while deserializing header:
HeaderTooLarge`, reached from `vllm_ascend/worker/worker.py load_model` →
`model_runner.py get_model` → upstream `qwen2_5_vl.load_weights` →
`model_loader/loader.py` → `weight_utils.safetensors_weights_iterator`
`safe_open`, plus `_pickle.PicklingError: Can't pickle <class
'safetensors_rust.SafetensorError'>` masking it on ranks 1-3. Track: the checkpoint
*file* is not a safetensors container (Git-LFS pointer / partial copy / wrong
file) — not TP, not model support (`supported_models.md:11` lists Qwen2.5-VL), not
the image/CANN pair, not the title-vs-env version disagreement. The corpus has no
`HeaderTooLarge` hit at all (title grep and issue search), so there was no product
defect to find. The recorded run computed the LFS arithmetic locally (first 8
bytes `b"version "` → u64 2336927755350992246 ≫ the 100 MB cap), could not verify
the cap itself (no network for `uv --with safetensors`), and delivered the verdict,
`ls -l`/`head -c 80`, a per-shard header script and the `--max-model-len` aside in
one answer at turn 8. Page `../startup-failures/checkpoint-file-not-safetensors.md`,
one call `../../scripts/checkpoint-header-scan.py <model dir>`. Open: how the
weights were obtained and the shard sizes.

**#986** image `quay.io/ascend/vllm-ascend:v0.8.5rc1-openeuler` (tree v0.8.5rc1
`ec27af34`; the snapshot's vllm pin is v0.6.6.post1, **not** the 0.8.x in the image),
`DeepSeek-R1-bf16-hfd-w8a8` from modelers.cn, 2 nodes, TP8 × PP2 + ray,
`--enforce-eager`, **no** `--quantization` flag, the checkpoint's
`quant_model_description.json` hand-merged into `config.json` per #856: all 16 ranks
die while loading (shards `2/157`) at upstream
`vllm/model_executor/models/deepseek_v2.py:788 param = params_dict[name]` with
`KeyError: 'model.layers.23.mlp.experts.w2_weight_offset'` on the PP0 ranks and
`layers.40…` on PP1. Track: the index-free name is upstream's own remap of a
per-expert key (`make_expert_params_mapping` + `name.replace(…, "experts.w2_")`), and
at this ref only `W8A8_DYNAMIC` allocates MoE offsets (`w8a8_dynamic.py:417,426`),
the static `W8A8` MoE path raising `NotImplementedError` before load
(`quantizer.py:266-270` + `:210-212`) — so the experts resolved to the unquantised
method, i.e. the description in effect did not say `W8A8_DYNAMIC`. The recorded run
read the fused name as proof of a stacked msmodelslim export, then walked
quantizer/create_weights, and ended **output-truncated at 16 384 tokens with zero
answer text** (913 K cumulative prompt tokens); it never asked for `ls` of the model
dir or the experts' description entry. Page:
`../startup-failures/weight-scale-keyerror-quant-variant.md` (`.weight_offset`
section).

**#3325** vllm-ascend v0.11.0rc0 (`00ba071`) + vllm v0.11.0rc3 (`8ce5d31`), image
`quay.io/ascend/vllm-ascend:v0.11.0rc0`, Qwen3-Next-80B-A3B-Instruct at TP4:
`ERROR [registry.py:547] Error in inspecting model architecture
'Qwen3NextForCausalLM'` followed only by `Traceback (most recent call last):`. Answered
in 11 assistant turns / 19 tool calls with the verdict first and in the report's
language: not a version gate (the arch is registered at
`vllm_ascend/models/__init__.py:59-60` and a tutorial exists), inspection runs in a
subprocess so the cut tail is the finding, then triton-ascend pairing ranked first with
`#2889`'s `tl.exp` inner frame, `#2929`'s `ModuleNotFoundError` second, and the
tutorial's `--enforce-eager` / shorter context as the follow-up. What it added beyond
the paste: an absent triton cannot produce `AttributeError: module 'triton.language'`
(vllm substitutes placeholders), so a triton is installed and mispaired. Cost: one
`cat SOURCES.json` (~21 KB, rule 7) and a full read of the tutorial including its
x86_64 "Coming soon" tab. Route:
`../startup-failures/model-type-not-recognized.md` (inspection-crash section).

**#4206** image `quay.io/ascend/vllm-ascend:v0.11.0rc0` (snapshot pin: vllm-ascend
`v0.11.0rc1` `c5fe179` + vllm `v0.11.0` `b8b302c`), Qwen3-Next-80B-A3B, body in
Chinese: start dies in `create_model_config` with
`pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig` /
`Value error, Model architectures ['Qwen3NextForCausalLM'] failed to be inspected.
Please check the logs for more details.` — the pydantic wrapper around #3325's crash,
with no `[registry.py]` line pasted at all. Track: the model-inspection look-alike,
**not** the version gate (`../startup-failures/model-type-not-recognized.md`). What the
pin settles: `failed to be inspected` is raised only for an architecture that is
registered (`_raise_for_unsupported`); the replacing class is downstream
(`vllm_ascend/models/__init__.py:46-48` → `CustomQwen3NextForCausalLM`), so the
subprocess import goes through `vllm_ascend/models/qwen3_next.py`, which at rc1 imports
**only** upstream `vllm.*` names and no `vllm_ascend` internal module; and Qwen3-Next
was supported from v0.10.2 and "stable" at `v0.11.0rc0` (`release_notes.md:22`), so the
architecture is not the question. Delivery: **no answer at all** — 37 messages, every
turn after the opener a bare tool call, the whole output budget spent on two dead
lanes: a missing `triton` ranked first off the tutorial's mandatory `triton_ascend` +
BiSheng install step, then refuted against `find_spec`/`HAS_TRITON` placeholders four
turns later; and reconstructing the `v0.11.0rc0` image's `qwen3_next.py` from the
`v0.11.0rc1` export, which a one-commit snapshot cannot contain. Owed at turn 5: the
classification, one `python -c "import vllm_ascend.models.qwen3_next"`, and the ask for
the `[registry.py]` lines.

**#4948** vllm-ascend v0.11.0rc3 image (`quay.nju.edu.cn/…`, tree `b6d63bb`) + vllm v0.11.0,
"部署满血版 DeepSeek R1 0528 BF16 失败" on Atlas 800I A2 / 910B4, `--tensor-parallel-size 8
--pipeline-parallel-size 4 --enable-expert-parallel --distributed-executor-backend ray
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":
{"enabled":true}}'`, `--max-model-len/--max-num-batched-tokens 65536`: every rank printed
`Capturing torchair graph …` and then `Error executing method 'compile_or_warm_up_model'` with
`torch._dynamo.exc.Unsupported: call_function DelayGraphBreakVariable()` from
`dispose_tensor` → `x.set_` (`utils.py:504-505`, called at
`torchair/models/torchair_deepseek_v2.py:985-986`) inside the `fullgraph=True` compile
(`torchair_model_runner.py:423,446`); precedent #1615 carries the same signature. Co-equal
cause on this checkpoint: the paste's `Available memory: 5080964608, total memory:
65452113920` (`worker_v1.py:257`) leaves 7.8 % headroom at a 65 K-token profile, and
reporter-confirmed later on the same issue: full-weight BF16 R1-0528 needs ≥4 A2 servers
(32 × 910B4) and the DeepSeek-R1 tutorial suits the W8A8 weights. Known issue in the pinned
tree: `release_notes.md:383` still lists "Pipeline parallel does not work with ray and graph
mode" (#1751/#1754), and both recorded reports ran PP=4. The recorded run reached the verdict
by turn 10 but shipped **no answer** (one 270-character paragraph, the rest reasoning) before
the 16 K cap. Page `../startup-failures/torchair-dynamo-compile-unsupported.md`. Open: the
image's `torch`/`torch_npu` versions against the pinned 2.7.1 pair.

**#5484** vllm-ascend v0.11.0 image (`quay.io/ascend/vllm-ascend:v0.11.0-my`, a
`pip install vllm-ascend==0.11.0` build over the CANN 8.2.rc2 base, tree `2f1aed98`),
DeepSeek-V3.2 `w8a8-QuaRot` on 2 × A2 (910B) DP2/TP8/EP with
`torchair_graph_config` `{enabled, graph_batch_sizes:[16]}`: the torchair compile dies in
GE **shape inference** — `RuntimeError: E89999: Inner Error!` + `op[Transpose_5],
attr[perm], has wrong value[1], it should be less than input shape size[0]` +
`Call InferShapeAndType for node:Transpose_5(Transpose) failed` + `[Compile][Graph]Compile
graph failed … [FUNC:ge_api.cc][LINE:1365]` — from `_compile_torchair_graph` →
`_dummy_run(is_torchair_compile=True)` (`torchair_model_runner.py:286,245,223` at v0.11.0),
i.e. host-side before any kernel. Verified locally: `ascend_config.py:285` refuses
`torchair_graph_config` + `--enforce-eager`, so the graph-off ablation is deleting the
key; the pin has **no** rotation handling (`rg -il 'quarot|qua_rot|hadamard' vllm_ascend/`
→ 0) while `docs/source/tutorials/multi-node_dsv3.2.md:298-340` validates
`vllm-ascend/DeepSeek-V3.2-Exp-W8A8` on exactly this DP2/TP8 shape; the
`graph_batch_sizes` torchair fix (#2760) predates the pin, and no `tests/e2e/**` case
compiles DS3.2 in torchair mode. `cann/ge` (the error's own files) and the
`torch_npu/dynamo/torchair` package are both outside the case view's exports, so the
producing `.permute` cannot be sourced locally. Recorded run: verdict text at turn 35,
then ~10 calls hunting transpose sites, and a final 10.9 K-character reasoning block —
1.2 K characters of message text, cut at the cap with the ranked answer unsent.
Reporter's public follow-up refutes the upgrade rung: `0.11.0`, then an
`0.11.0-dev` source build, same error. Page
`../startup-failures/torchair-ge-infershape-rejection.md`. Open: which node produced the
rank-1 operand (decided by the compile's GE graph dump) and whether the rotated
checkpoint's `quant_model_description.json` is the owner.

**#2315** vllm-ascend image `v0.9.2rc1-310p` (tree `b5b7e0e`) + vllm v0.9.2 (`a5dd03c`),
2×`310P3` (host `NPU-SMI 24.1.0.1`), Qwen3-32B TP2: engine start dies inside torch_npu
device open — `RuntimeError: SetPrecisionMode: … AclSetCompileopt(aclCompileOpt::
ACL_PRECISION_MODE, precision_mode), error code is 500001`,
`[Error]: The internal ACL of the system is incorrect.`,
`EH9999 … [PLATFORM]PlatformInfoInit runtime platform info failed. SocVersion Ascend@910B`,
`GEInitialize failed`, ending `ERR99999` — with no `vllm_ascend` frame, and the paste is
OCR'd from screenshots. Same two lines in #302/#564/#692 on 910B2 hosts. Track: lane A
startup, environment-owned (`../startup-failures/acl-init-500001-and-soc-version.md`).
The recorded run confirmed the "not vllm-ascend code" verdict in a turn-17 status note but
shipped **no answer**: 18 turns / 26 tool calls / 232 characters of message text, the
budget spent in `torch_npu` `LazyInitAclops.cpp`, the `SocVersion` enum and `is_310p()`.
The three facts that answer it are the printed SocVersion against `npu-smi`, the image's
CANN base (`Dockerfile.310p:18`, `requirements.txt:27`) against the host driver, and the
framework-free `torch.npu.set_device(0)` probe, plus — on a run whose `docker run`
already matches the docs — the same command + `--privileged=true`. Semantics established
here, not by a reported fix: `SetPrecisionMode` only *supplies* the precision string for
the SoC it resolved, so the numeric code is not a flag problem. The 910B2 sibling **#302**
(below) is the one case in this family with a reporter-confirmed ending, and it is a
container flag: keep the driver-pairing theory under the privilege re-run.

**#6578** vllm-ascend v0.14.0rc1 (`52d4acfa`) + vllm v0.14.1 (`d7de043d`), Qwen3-32B on
910B, title `no attribute qkv_rmsnorm_rope`, body's first line an
`AttributeError: '_OpNamespace' '_C_cache_ops' object has no attribute 'reshape_and_cache'`:
the `torch.compile` pass `qknorm_rope_fusion_pass.py:214` traces its replacement and the
triton op is imported only `if HAS_TRITON` (`ops/__init__.py:25-26`) while the pass is
appended by default (`fuse_qknorm_rope` True) and gated only on dtype + `head_size == 128`.
`_C_cache_ops` has zero hits in the vllm-ascend tree, so it is a second launch's line, not
this traceback's. Track: lane A/B graph-compilation
(`../startup-failures/missing-triton-op-in-compile-pass.md`). The run reached the mechanism
by turn 5 and shipped a ranked answer on turn 11, but the cap landed inside its closing
scope paragraph — the fix (`--additional-config '{"ascend_compilation_config":
{"fuse_qknorm_rope": false}}'`, the project's own line at
`docs/source/user_guide/configuration/additional_config.md:79`) did ship.

**#16581** vllm-ascend v0.23.0rc1 (`f4a08bd`, not byte-verified) + vllm v0.23.0
(`0fc695f`), one 910 (64 GB), ModelSlim W8A8 Qwen3-32B (32.03 GiB) on
`--max-model-len 16384 --gpu-memory-utilization 0.9`: `ACL graph capture fails with
error code 507015` at `torch.npu.graph.__enter__` → `torch.npu.synchronize()`
(`torch_npu/npu/graphs.py:519`), wrapped as `NPUModelRunner failed, error is …`, with
`fftsplus aivector error … error code = 0x800000` / `The DDR address of the MTE
instruction is out of range` / `retCode=0x26, [aicore exception]`; the reporter's own
workaround changed util 0.9→0.85, capture sizes 51→`[1,2,4,8]` and `--max-model-len`
16 K→40 K at once. Both frame numbers match the pin (`acl_graph.py:192`,
`model_runner_v1.py:5177`). The recorded run found the two decisive facts — the advice
gate only knows `207008` (`acl_graph.py:28-51`) and nothing reserves the graph pool
(upstream `profile_cudagraph_memory()` is `is_cuda()`-guarded, downstream's
`determine_available_memory` at `worker.py:527-597` drops it) — and correctly refused to
name one cause because the workaround moved three variables, but it spent a turn on a
442-hit history listing for `graph capture` and its answer was cut **mid-sentence inside
its scope block** (3.9 K characters delivered against a 16.4 K completion-token cap), so
the "what to send back" half never shipped. Route now
`../startup-failures/acl-graph-capture.md` (v0.21–v0.23 section) and the new
signature row for `NPUModelRunner failed … AclrtSynchronizeDeviceWithTimeout`.

**#11498** vllm-ascend release/v0.23.0 (`79928c2`) + vllm v0.23.0, DeepSeek-V4-Flash
w4a8 recipe on 4 cards, TP4 + `--enable-expert-parallel`,
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` + `enable_flashcomm1: true`, DP1, PP1: engine
start dies in `profile_run` with `RuntimeError: Promotion for Float8 Types is not
supported, attempted to promote Float8_e4m3fn and Bool` at
`ops/fused_moe/token_dispatcher.py:393 topk_weights = topk_weights * mask`. Track:
W4A8-MXFP activations (quantised only on the `_prepare_with_ep_group` path, i.e.
flashcomm1/SP) cast the routing weights to fp8 at `w4a8_mxfp4.py:202`, and the
AllGather dispatcher multiplies them by a bool EP mask. The run derived all three
conditions correctly and named the right owning line, but exhausted its response
room inside its own reasoning and shipped no answer — every mechanism it needed was
three `sed` reads plus one ablation. Page now holds them:
`../startup-failures/fp8-topk-weights-mask-promotion.md` (new).

**#12675** image v0.23.0rc1, DeepSeek-V4-Flash, TP8 + EP on one 8-card node,
`--dtype bfloat16`, `--block-size 128`, `--additional-config` with
`enable_dsa_cp`/`multistream_overlap_shared_expert`: every rank dies during weight
load at `vllm_ascend/models/deepseek_v4.py:1535 param = params_dict[name]` with
`KeyError: 'model.layers.<n>.self_attn.wkv.weight_scale'` /
`…self_attn.indexer.wq_b.weight_scale` — a different layer per rank, only two
module families across eight ranks. Track: the reporter is serving a plain
`deepseek-ai/DeepSeek-V4-Flash` directory (`-v /data/models/DeepSeek-V4-Flash:/root/.cache`,
`vllm serve /root/.cache`) with **no** `--quantization ascend`, while the tutorial
lists only the quantised `DeepSeek-V4-Flash-w8a8-mtp` (and the PR test the w4a8
sibling) and every documented serve line carries the flag. The recorded run
inferred "the model must be quantised, so exactly these two modules are skipped in
code" from the short key set and spent its budget in `modelslim_config.py`
(`is_layer_skipped_ascend` → `create_scheme_for_layer` → `quant_prefix_mapper`)
without ever asking for the serve command; the answer arrived in the reporter's own
later paste. Page: `../startup-failures/weight-scale-keyerror-quant-variant.md`
(new), whose first step is the scale-key census
(`scripts/checkpoint-key-prefix.py --scales <dir>`).

**#1579** vllm-ascend v0.9.1rc1 (`c30ddb8`), vllm v0.9.1
(`b6553be`), torch_npu `v7.1.RC1.alpha002` (`d33df55`), image
`quay.io/ascend/vllm-ascend:v0.9.1rc1-310p`, one 310P node × 8, Qwen3-32B dense,
`--tensor-parallel-size 8 --enforce-eager --dtype float16`, no `--max-model-len`
(40960 default). The paste opens with a passing `hccl_test all_gather_test`; engine
start then dies in the memory-profiling window: ranks 1-7 raise
`EOFError: Ran out of input` at the same second from
`worker/worker.py:288 determine_num_available_blocks` → `model_runner.py:1205
profile_run` → `:1472 compute_logits` → `logits_processor.py:99 _gather_logits` →
torch_npu `distributed_c10d.py:126` `dist.broadcast_object_list` →
`_tensor_to_object`, and the driver follows 119 s later with `HcclAllgather` `EI0006`
socket timeout / `DestLink` (the collective it was left waiting in). Track: the
gather **emulation**, not the fabric — HCCL has no `gather`, so `_gather` broadcasts
the per-rank sizes first and the non-destination ranks unpickled an empty payload;
the V0 notice above it is upstream `_warn_or_fallback` on non-CUDA hardware and
`VLLM_USE_V1=1` is a live lever (`supports_v1()` returns True there). The recorded run
read the torch_npu frame by turn 5, then spent nine turns inside torch's own
`distributed_c10d.py` — which this snapshot does not contain — and ended at the output
cap with zero answer text (14 turns, 16 384 completion tokens of reasoning). Public
follow-up: a third party reported the same setup starting with `VLLM_USE_V1=1`, the
reporter said they would try and later reported it **unresolved**, having moved
Qwen3-32B to MindIE on a 310P T18 image. Page (new):
`../startup-failures/profile-run-logits-gather-eof.md`.

**#302** image `quay.io/ascend/vllm-ascend:v0.7.1rc1` (`Dockerfile:17` → CANN
`8.0.0-910b`), 910B2 ×8, driver 24.1.rc3, `vllm serve Qwen/Qwen2.5-7B-Instruct
--max_model_len 26240` in a `docker run` that repeats `docs/source/quick_start.md:15-30`
mount for mount (`davinci0`, `davinci_manager`, `devmm_svm`, `hisi_hdc`, `dcmi`,
`npu-smi`, the driver's `lib64/` + `version.info`, `/etc/ascend_install.info`,
`/root/.cache`, `-p 8000:8000`): the engine dies at `init_device` → `torch.npu.set_device`
with `AclSetCompileopt(ACL_PRECISION_MODE) 500001` + `PlatformInfoInit … SocVersion =
Ascend910B2` + `GEInitialize failed` (`ERR99999`), no `vllm_ascend` business frame and no
`Starting to load model`, and the printed SoC matches the card. The recorded run answered
with five ranked causes and the right exclusions (`--max_model_len`, `PYTORCH_NPU_ALLOC_CONF`,
`VLLM_USE_MODELSCOPE`, the `No platform detected … UnspecifiedPlatform` line, the expected
`vllm._C`/Triton absences) but ranked the driver/firmware × image-CANN pairing first and
put the container lane third — as `--privileged` **plus** an extra whole-driver mount, two
variables at once. The reporter's own closing comment the next day was
「已解决，加个--privileged=true即可」: that one flag on the unchanged command, no driver or
CANN change. The ordering is the lesson, not the exclusions — the documented mount list is
a minimum, and the privilege/device set is the first rung
(`../startup-failures/acl-init-500001-and-soc-version.md`).

**#2960** vllm-ascend v0.9.0rc2 (`8dd686d`) + vllm v0.9.0 (`5873877`), `Qwen2.5-32B-Instruct-GPTQ-Int8`
served as a capability question ("请问是否支持…?"): the log's engine args already show
`quantization=ascend` with no `--quantization` on the command line, and startup dies at
`vllm_ascend/quantization/quant_config.py:123 is_layer_skipped_ascend` with
`KeyError: 'model.layers.0.self_attn.q_proj.weight'`. The recorded run derived the chain
correctly — the platform list is `["ascend"]` alone (`platform.py:67`),
`override_quantization_method` renames any declared method once the NPU is up
(`quant_config.py:82-86`), and upstream's `get_quant_config` passes the GPTQ
`quantization_config` dict straight into `AscendQuantConfig` as `quant_description`
(`vllm/model_executor/model_loader/weight_utils.py:154-165`), so
`quant_model_description.json` is never read — and answered "not supported; only
msmodelslim-exported W8A8/W8A8_DYNAMIC", with the model itself supported. Accounting:
15 assistant turns, the verdict only on the last one (3.7 K characters against ~30 K of
reasoning), several calls spent re-deriving that chain from upstream source and reading
`weight_scale` precedents that are a different signature. Section C of
`../startup-failures/quantization-method-gate.md` now carries the two-call version.

**#902** vllm-ascend v0.8.5rc1 (`ec27af3`) + vllm v0.8.5.post1 (`3015d56`), image
`quay.io/ascend/vllm-ascend:v0.8.5rc1`, QwQ-32B-w8a8 TP4, `--max-model-len 32768`:
`vllm serve … --quantization ascend` exits with `vllm serve: error: argument
--quantization/-q: invalid choice: 'ascend' (choose from 'aqlm', 'awq', …, 'torchao',
None)` and nothing else — no `APIServer pid=` line, no worker, no device or weight text.
Parse-time lane, not the platform gate of §A: `--quantization` is added with
`choices=[*QUANTIZATION_METHODS, None]` (`vllm/engine/arg_utils.py:600-604`), a copy of
upstream's static `QuantizationMethods` `Literal`, and the plugin hook that is supposed to
extend it runs *afterwards* (`:1640-1646`). At this pin `NPUPlatform.pre_register_and_update`
takes `parser` and never uses it (`vllm_ascend/platform.py:66-75`), while
`supported_quantization = ["ascend"]` (`:61`) and the offline path both accept the method:
`ModelConfig._verify_quantization` gates on `supported_quantization = QUANTIZATION_METHODS`
(`vllm/config.py:752-753`), which the import-time `@register_quantization_config("ascend")`
(`vllm_ascend/quantization/quant_config.py:41`) has already extended because
`EngineArgs.create_engine_config` calls `pre_register_and_update()` (`arg_utils.py:1096`)
before `create_model_config()` (`:1099`). Failing and fixed states verified across pins:
v0.17.0rc1 (`e20f0b1a`) rewrites the choices through
`parser._option_string_actions.get("--quantization")` (`platform.py:134-147`, pinned by
`tests/ut/test_platform.py:54-100`). The run delivered a complete ranked answer in 9
assistant turns (evidence table, the action ladder — drop the flag when the checkpoint
declares `quant_method`, import the config module before `make_arg_parser`, the offline
`LLM(quantization=...)` spelling, upgrade — and the unverifiable "first fixed release"
left open), and its own closing note flagged the
gap this page then had. Route: `../startup-failures/quantization-method-gate.md` §D.

**#4696** vllm-ascend v0.11.0rc0 (`00ba0710`) + vllm v0.11.0rc3
(`8ce5d319`), Ascend 910B: `[Doc]` report that the Qwen2.5-Omni tutorial's
"Graph Mode / full graph" section cannot be enabled, with the serve log carrying
`CUDAGraphMode.FULL cudagraph_mode is not support on NPU. falling back to NONE`.
The pasted line is the plugin's own `else` branch, the legacy `full_cuda_graph`
is what lands there, and the tree's unit test asserts the fallback — so the doc's
command is the defect and `FULL_DECODE_ONLY` is the replacement. The run delivered
a complete ranked answer including the
release note that already recorded the failure on A2 (910B1, #2182); the one
correction the library now carries is the quote-style point (the Python-literal
dict parsed, so it is cosmetic) — `../startup-failures/graph-mode-not-applied.md`
(`FULL` section).

**#1364** vllm-ascend v0.8.5rc1 + vllm v0.8.5.post1, ray: `AsyncLLM.from_engine_args` hangs inside an RLHF server and the paste ends on the `utils.py:2522 … not implemented in <…NPUWorker object>` constructor warning. The named frame is only the waiter (`_wait_for_engine_startup` raises only when a core proc exits); the stall is worker init, and the recorded run shipped 0 characters of message text. `../startup-failures/ray-programmatic-init-hang.md` (new).

**#4863** image `quay.io/ascend/vllm-ascend:v0.11.0rc3-openeuler`, `Qwen3-Reranker-8B`
with `--task score`, vllm-ascend v0.11.0rc3 (`b6d63bbd5294`):
`ImportError: cannot import name 'STR_DTYPE_TO_TORCH_DTYPE' from 'vllm.utils'` raised
while `vllm serve` builds its parser (`make_arg_parser` → `pre_register_and_update` →
`adapt_patch` → `patch/platform/patch_mamba_config.py:6`). The recorded run found the
two facts that decide the shape: the same symbol is imported unguarded at two more
sites (`worker/worker_v1.py:38`, `worker/model_runner_v1.py:72`), so an edit to line 6
only relocates the error to worker start; and the traceback's `vllm/utils/__init__.py`
(package) versus the pin's monolith `vllm/utils.py:174` names the upstream generation.
No reporter follow-up in this snapshot. Page: `../startup-failures/environment-correspondence.md`
(the `from vllm.utils import …` section, which also carries the newer pin's migrated
form `from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE`).

**#4847** image `quay.io/ascend/vllm-ascend:v0.11.0rc2` (`a2e4c3f`) + vllm v0.11.0
(`b8b302cd`), two nodes × 8 cards, `Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot` with
`--data-parallel-size 4 --data-parallel-size-local 2`, `--tensor-parallel-size 4`,
`--enable-expert-parallel` and `--additional-config '{"torchair_graph_config":{"enabled":true}}'`:
`EngineCore` fails in the memory-profiling dummy run with `IndexError: index 0 is out of
bounds for dimension 0 with size 0` at `torchair_attention.py` `if kv_cache is not None
and kv_cache[0].numel() > 0:`. The recorded run spent its whole 16 K output budget inside
`torchair/*` and `model_runner_v1` and shipped **no message text**; the reporter's later
public comments carry both fixes — the container's `docker run` omitted
`-v /usr/local/dcmi` (weight loading failed until it was added) and dropping the
`torchair_graph_config` key then let the service start. Mechanism and ladder:
`../startup-failures/torchair-graph-profile-run-kv-cache-indexerror.md`; the container
diff: `../startup-failures/no-npu-in-container.md` §*A responding `npu-smi info` is not a
mount check*.

**#8367** image v0.18.0rc1 (`99e1ea0f`) + vllm v0.18.0 (`bcf2be96`),
`Qwen3-VL-32B-Instruct-w8a8-new` served with `--quantization ascend
--tensor-parallel-size 4 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", …}'
--mm_processor_cache_type shm --async-scheduling`: engine start dies at model construction
with `KeyError: 'lm_head.weight'` from
`vllm_ascend/quantization/modelslim_config.py` `get_linear_quant_type`. The recorded run
read the packed-modules table, `apply_vllm_mapper`, the upstream `qwen3_vl` mapper and the
corpus for `lm_head` precedents (5 hits, none matching) and was truncated with **no
message text**. The decisive pair is `qwen3_vl.py:1327-1332` (a literal
`prefix="lm_head"`) against `:1374-1381` (`"lm_head."` →
`"language_model.lm_head."`): the key the engine asks for cannot exist in a
mapper-rewritten description, so the fix is the model file, not the export —
`../startup-failures/empty-prefix-quant-keyerror.md` §*The renamed-prefix variant*.

**#14467** image `quay.io/ascend/vllm-ascend:qwen3.8-a5`, Ascend 950PR ×1, vllm and
vllm-ascend 0.23.0, `Qwen/Qwen3.8-27B-FP8` served with `qwen3_5_mtp` spec decode and
`FULL_DECODE_ONLY`: `EngineCore failed to start` in the second after `Starting to load
model …` and `Using the vLLM Ascend fp8 Quantization now!`, deepest frame
`vllm_ascend/quantization/methods/fp8.py:45 self.n_groups = hf_config.o_groups`.
The reporter's own body already named the chain (detection → `AscendFp8Config` → a
DeepSeek-V4-only `ds_linear` scheme), so the recorded run's contribution was
verification at `5cb98ca` plus the action ladder it delivered: the registration scope
(`fp8_config.py:68` takes over the generic name, `:137` reuses the class for
`deepseek_v4_fp8`), the A5-only detection gate (`utils.py:132-133`, `soc_version ==
260`), the unread `ignore` list (`:77`), and the acceptance rule that a
`getattr(..., "o_groups", 0)` stopgap must be scored by one real request against a BF16
serve because `process_weights_after_loading` re-blocks every linear for the DS layout.
Its one unasked artefact is the checkpoint's `config.json` (`quantization_config.quant_method`,
`weight_block_size`, `ignored_layers`), which decides whether that stopgap is
numerically safe; the later public trail on the issue is a code fix (a linked PR), i.e.
the "no serve flag fixes this" verdict held —
`../startup-failures/native-fp8-config-scope.md`.

**#7088** vllm-ascend v0.15.0rc1 (`3d43ed997e71`) + vllm v0.15.0 (`f176443446f6`),
4×910B3, `vllm-ascend/Qwen3-235B-A22B-W4A8` served with `--tensor-parallel-size 4
--gpu-memory-utilization 0.95 --enable-chunked-prefill --max-num-batched-tokens 8192
--max-num-seqs 2 --async-scheduling --compilation-config '{"cudagraph_mode":
"FULL_DECODE_ONLY"}'`: every rank dies at `Loading safetensors checkpoint shards: 3%`
with `RuntimeError: The expanded size of the tensor (4) must match the existing size
(16) at non-singleton dimension 1.  Target sizes: [4096, 4].  Tensor sizes:
[4096, 16]`, deepest frames `qwen3_moe.py:628 load_weights` → `fused_moe/layer.py:1310
weight_loader` → `:979 _load_per_channel_weight_scale` `expert_data.copy_(loaded_weight)`.
The reporter's own reading was half right (the `w2_scale_bias` allocation and
`update_bias`); the missing half is the **tag**: `method_adapters.py:210` stamps every
dynamic-quant param `CHANNEL` and re-tags `scale_bias` only when `group_size > 0`
(`:211-226`), so a description carrying `version: "1.0.0"` **and** `group_size: 0`
leaves the `[E, hidden, 16 // tp_size]` param (`w4a8.py:317-322`, `:202` divisor = 1
under `--enable-expert-parallel`) on upstream's unsharded per-channel w2 copy. The
recorded run shipped the ranked answer (the reporter's own `#3214` RFC as the format
precedent) and flagged the gap this page fills: **the library had no route for a
load-phase `expanded size` error**, only the DP-padding and EAGLE-hidden ones. Its one
unasked artefact is the checkpoint's `quant_model_description.json`, i.e. the
`version`/`group_size` pair it inferred from the error —
`../startup-failures/quant-aux-param-tp-shape-mismatch.md` (new).
