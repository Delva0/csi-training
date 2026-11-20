# Recorded cases: version, symptom, track, and how the run ended

## Scope

The per-issue index behind rule 1 of `SKILL.md`. One entry per issue this library
has triaged: the version, the symptom as filed, the track it landed in, and — where
it is instructive — what the recorded run got right or wrong. Read it to place a
new report beside a precedent, or to check what a delivery failure looked like
before deciding a session is going well; do not read it end to end. The
classification table in `SKILL.md` routes by signature, topical pages by track, and
`scripts/history-titles.sh '<regex>'` by earlier-issue title.

## Rules for using it

- A precedent supports a hypothesis; it does not replace reading the source at the
  pinned commit (`SKILL.md` rule 8).
- Pick precedents by whether the title names a *mechanism*, not by symptom match.
- An entry that ends "shipped no answer" or "cut mid-table" is a delivery lesson
  about that session, not evidence about the fault.

## Cases

**#8944** v0.19.1rc1
`aclnnAddRmsNormBias` device fault; #10068 evalscope vs ais-bench MMLU-Pro gap;

**#3543** v0.11.0rc0 silent worker death; #7477 v0.17.0rc1 HCCL allreduce ending in
an HCCS link fault; #11661 v0.22.1rc1 missing `attention_type` on a patched
decoder layer; #8409 v0.13.0 `model_type` not recognised (version gate); #16015
v0.26.0rc1 GLM-5.3 w8a8c8, PD disaggregation + sparse sfa c8 + DCP=16, 400K+
prompt → 乱码; #5572 v0.12.0rc1 Qwen3-VL-30B-A3B PD disaggregation, Mooncake
connector `AttributeError` on a nested config; #3389 v0.11.0rc0 Qwen3-VL-235B
TP=16 whose posted cause ("insufficient available streams") is canned text around
a capture-phase HCCL timeout; #10503 v0.20.2rc1 Kimi-K2.5-w4a8 on 2×A2 with
tp8/dp2/EP, 乱码 on a 3.6K-token prompt; #17357 v0.26.0rc1-a3 DeepSeek-V4-Pro
w4a8 `SUSPECT REMOTE ERROR`/507057 surfacing at a compile-phase pattern-registration
sample op (lane C, reporting rank unrelated to the faulting chip); #5585 msmodelslim
W4A4 Qwen3-32B with one blown-up decoder layer (converter-owned, conversion phase);

**#5609** v0.12.0rc1 `corrupted size vs. prev_size` abort on
`from mooncake.store import MooncakeDistributedStore` in the official image (image-owned
native module, before any vllm-ascend code); #11863 v0.21.0rc1 dense Qwen3.6-27B
launched from the 27B tutorial's multi-node PD recipe → upstream non-MoE external-DP
gate (usage error plus a real doc bug); #16022 v0.23.0rc1 4×910B4 Qwen3.6-27B
long-run HBM growth then crash, with `torchair_graph_config`/`ascend_scheduler_config`
keys that never engaged; #6784 v0.14.0rc1-a3-openeuler LongCat-Flash
`AttributeError: 'ZeroExpertFusedMoE' object has no attribute 'select_experts'`
(an upstream MoE refactor outside vllm-ascend's name-keyed OOT map, with the
failing file absent from the pinned upstream tree); #11519 v0.18.0 Step-3.5 on
A3 `AclNN_Parameter_Error(EZ1001) Get regInfo failed … socVersion
[ascend910_93] does not support opType [AddRmsNormBias]` (custom-op package
built/installed for another SoC — same op as #8944, opposite failure class);
#12141 vllm-ascend v0.21.0rc1 Qwen3.5-9B on an 8-die **Ascend950PR** host,
`EngineCore failed to start` at `worker.py:113 check_ascend_device_type()` with
`Current device type: AscendDeviceType.A5 does not match … AscendDeviceType.A2`
(an A2-build wheel on A5 silicon — packaging, not a code defect) — verdict stated
in turn 3, then 25 more turns pinning the
build stamp, the a2/a3/310p-only wheel matrix, the A2/A3 support matrix and four
precedents (#7157, #9354, #10206, #8238) before a turn-28 answer;

**#14506** v0.18.0rc1 single node, 8×910B3 with every RoCE link DOWN, 1P1D in two
non-hostNetwork Pods → `HIXL Connect failed | 103900` because two Pod IPs became
two HCCL servers (transport topology, not a connector bug); #12339 v0.22.1rc1
Qwen3.5-397B-W8A8-MXFP8 on Ascend950, `MooncakeConnectorV1` PD without MTP,
alternating good/bad answers on a hybrid GDN model (the run spent its whole
session on source walks and delivered no answer); #10985 v0.18.0
`--enable-log-requests` + `--enable-log-outputs` printing neither prompt nor
generation (an upstream frontend gate checklist, not an Ascend defect — the run
spent the whole session and shipped no answer); #4812 v0.11.0.rc2-310P,
Qwen2.5-VL-3B fp16 eager, which the reporter's later dump showed as
`Memory_Allocation_Failure(EL0004)` + `507013` / SDMA at a KV-cache
`npu_format_cast` during `initialize_kv_cache_tensors`; #9335 v0.19.1rc1
Qwen3-Omni-30B-A3B on 2×910B, where vllm-omni's 2-NPU stage layout is
memory-infeasible and `CUDA_VISIBLE_DEVICES` is inert on NPU (a sizing answer,
not a fault); #11739 v0.18.0 community-test report on the
Qwen3-Omni-30B-A3B-Thinking tutorial, where the run spent the session auditing
the page and shipped no report; #9183 v0.18.0rc1 A3 1P1D
DeepSeek-V3.2-w8a8 whose quoted `107030` capture-mode plog block turned out to be
the **previous day's** crash, while the reporter's later plog put an `EZ9999`
fftsplus-aivector / MTE-DDR exception inside the reported window on three chips at
once; #6800 v0.13.0 MiniMax-M2.5-AWQ rejected by the platform quantization gate
(`awq` is not in `[ascend, compressed-tensors]`) — a re-quantize answer, not a
defect; #1865 v0.9.2rc1 16-card DP4×TP4+EP DeepSeek-R1-W8A8 titled
多进程卡间通讯超时, whose pasted `TimeoutError` was the framework's own 300 s
`execute_model` RPC watchdog around a stalled rank (answer delivered); #253 v0.7.1rc1 DeepSeek-R1-BF16 TP=16/PP=2 under
`--distributed_executor_backend ray`, `TypeError: 'NoneType' object is not
callable` because the MoE monkeypatch is imported only by
`check_and_update_config`, so the Ray workers kept upstream `forward_oot`
(diagnosed correctly, but the run over-reasoned and shipped a one-line reply); #13439 v0.23.0rc1
2×8×910B2 DeepSeek-V4-Flash-w8a8-mtp 1P1D with `MultiConnector` =
`MooncakeHybridConnector` + `AscendStoreConnector`, HDK 25.2.3: `503900` /
`HcclCommPrepare 0x13` on every transfer plus 乱码, resolved by the reporter when
`global_segment_size` went 40 GB → 2 GB (the pool's store segment, not the
transport; both p2p connectors failed with the pool and PD without it was fine) —
the two recorded runs before that answer ranked `HCCL_INTRA_ROCE_ENABLE` and the
RDMA/connect timeouts instead; #9583 v0.18.0 + vllm-ascend v0.18.0rc1 Qwen3.5 on
8×910B2, `--profiler-config '{...}'` then `POST /start_profile` → `404`, a
route-registration question answered by naming `serve/profile/api_router.py`
`attach_router` and one `openapi.json` request, but the run spent 20 turns / 30
tool calls on "is `args.profiler_config` still a dict" and shipped one line (no
answer); #12253 vllm-ascend v0.18.0-a3
Qwen3.5-397B-A17B-w4a8-mtp, output "被截断" to tens of tokens after `"pass_config":
{"enable_sp": true}`, answered from the SP reader table plus the doc support
matrix (at that pin SP = VL+Dense only, graph, no quantization; FC1 is
complementary, not a rename) in 14 turns with a 5.3K answer — the lead it chose,
a "half-enabled SP" mechanism, is labelled there as unverified; #12278
**v0.22.1rc1** (`5f6faa0`) the same family one release later: body says `stop` /
`completion_tokens: 97` / `prompt_tokens: 15` with CoT in `content` and
`reasoning: null` (early EOS **plus** missing `--reasoning-parser qwen3`), and the
reporter's follow-up repeats #12253's outcome verbatim (delete
`pass_config.enable_sp`; "该参数 v0.18.0 及以上已不支持"; `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`).
The v0.22.1rc1 recheck moved three things: `sp_threshold` → upstream
`pass_config.sp_min_token_num` (MoE still 1), `SequenceParallelismMoePass` now
registered too, and the tutorial's own Qwen3.5-397B command never sets
`enable_sp`; the run delivered the lane in turn 1 and the corrected mechanism
(not dropped, activated outside the matrix; the key also picks the MoE all2all
backend) two turns after the follow-up; #8984 v0.19.1rc1
DeepSeek-V3.2-w4a8/w8a8-mtp-QuaRot on A2, "如何正确开启 PCP/DCP" with no log pasted (a
capability question — the run never stated the support verdict, and the
reporter's later public comment asked whether anyone had actually run PCP on
A2); #12487 v0.22.1rc1 A3 3P1D DeepSeek-V4-Flash long-run crash whose title
named `port 65536 have already been bound`, a three-day-old re-printed EI0019
record, while the in-window error was an AICPU `reduce_scatter`
parameter-invalid on a chip the reporting rank did not own.

**#9355** v0.18.0 Qwen3.6-27B bf16 on 2×910B3, whose first request after start returned
`content: ""` with `completion_tokens: 1` and `finish_reason: "stop"` while later
requests answered (the first run shipped only a half-answer; its lead mechanism was a "server-lifetime first decode" story that the
reporter's later comments leave unestablished: the follow-up changed the prompt and
`temperature` and posted to a different port, and the second comment only confirmed the
image tag). Two later runs against the page this case produced both answered in full (18
and 31 messages) but neither read the draft-side config gates or offered `min_tokens`;
the sibling `Qwen3.5-27B.md` recipe was opened only by the second, and used there as a
copied workaround rather than as the deviation table
(`../spec-decode-mtp/empty-first-response.md`); #10495 v0.20.2rc1 Qwen3.6 MoE + `qwen3_5_mtp` on 2×A2 with ~64K prompts,
acceptance collapsing to exactly 0.000 with drafts still produced and throughput
stalling, where the run ranked KV capacity/queue thrash first and the reporter then
reported MTP as the owner (MTP off → normal); #10943 v0.20.2rc1 custom
`nvidia/LocateAnything-3B` served through `TransformersMultiModalForCausalLM`,
`terminate called after throwing an instance of 'std::logic_error'` /
`basic_string::_S_construct null not valid` at EngineCore init — a native
null-string abort; the run spent the session inside the exported
CANN/torch_npu/triton C++ and shipped a one-line reply, and the reporter's later
comment established that the unmodified
checkpoint crashes identically (AutoModel loads it fine), so none of the eight
model-side changes was the cause; #3902 v0.11.0rc0-a3 Qwen3-VL-235B-A22B TP=16 on
one 910C, "the machine crashes" with no log pasted, answered through the
ACL-graph capture lane (host memory was its rank 3) — the reporter's later `top`
then showed 16 python3 ranks at 88-100 GiB RES each, 1472 of 2014 GiB host RAM
with swap 0, so host memory is the number that lane owes first.

**#7352** v0.14.0rc1 (still broken at v0.17.0rc1) DeepSeek-R1-Distill-Qwen-1.5B
1P1D Mooncake PD, 910B4: prefill came up while the decode pod died in
`register_kv_caches` → `KVCacheRecvingThread.__init__` with
`AttributeError: 'Qwen2Config' object has no attribute 'head_dim'` (a flat config
whose `head_dim` is derived, not a nested-VL config — `hf_text_config` was already
in use), and the same reporter's next round hit
`ImportError: ascend_transport.so` that turned out to be the launcher's effective
`LD_LIBRARY_PATH` rather than the image; both lanes are now in
`references/startup-failures/connector-config-attributes.md` and
`references/image-dependency-faults/launch-env-and-library-path.md`;

**#10961** v0.22.1rc1 Qwen3.X "PD 分离+MTP 存在精度问题" filed as an empty template
(no log, no serve commands), where the run's top-ranked source-verified candidate
— the `method == "mtp"` gate missing `qwen3_5_mtp` — is refuted by upstream's
`MTPModelTypes` → `"mtp"` rewrite (`vllm/config/speculative.py:520-524`); the
artifact list plus ablation rows were the right deliverable, the ranking was not
(`references/pd-disaggregation/hybrid-state-transfer.md` §3).

**#7580** v0.17.0rc1 A3 大EP GLM-5 + `deepseek_mtp`, `num_speculative_tokens: 3` →
`JSON Parse error: Unterminated string` while `1` was normal: the run reached
`sample/rejection_sampler.py:114 using_block_verify = max_spec_len >= 3` on its
tenth turn, then spent its remaining nine turns inside the acceptance kernel, its
pytorch twin and the test files, and was cut inside its evidence table with no
ablation plan; the reporter's later reproduction showed why the
first artefact owed was the assembled arguments — they stop at
`{"filePath":"/…/K8sYamlAgent.java`, a lost tail, which `return_token_ids: true`
splits between engine and delta assembly. #8122 v0.14.0rc1 Qwen3-Next-80B-A3B
`stream: true` + `--async-scheduling`, "流式 function call 大概率截断、小概率格式
错误、重复输出": answered late but in full, ranked on the reporter-confirmed #5683
precedent (async scheduling on → only CoT, no `tool_calls`) plus the CI gap that
`async_scheduling` never appears in `tests/e2e/multicard/4-cards/test_qwen3_next.py`;
the same audit on its 97-chunk paste returns valid JSON, so the posted sample was
not the failing request and the answer had to say so.

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

**#4057** v0.11.0rc0 Qwen3-Next-80B-A3B, "问到数字相关问题时精度较差" on a 19-token
prompt with no serve command in the report: the run found the downstream fp32
`torch_chunk_gated_delta_rule` prefill replacement and the tutorial's experimental
Triton-Ascend note, then spent its output budget re-deriving the FLA chunk algebra
and shipped nothing — its 14th assistant turn was cut mid-plan, with ~49 K
characters of reasoning and zero message text. The reporter's later public comment
supplied the deploy command (TP=4, `--max-model-len 8192`, `--enforce-eager`, no
MTP/EP/quantisation), which is the narrowing the answer owed
(`../accuracy-benchmarks/qwen3-next-gdn-numeric-path.md`).

**#13853** v0.20.2 + vllm-ascend v0.20.2rc1 Qwen3.5-0.8B on one 910B, 乱码 at
`--block-size 2048` with a "正常" command at `--block-size 512 --enforce-eager` (two
knobs changed in one comparison): the run had the two-clamp story (a hybrid model's
cache block is never clamped to 128 and is only *raised* by the mamba page-size
constraint) by its 10th turn, but kept walking turns 8-22 through `patch_gdn_attn.py`
chunk metadata, `gdn_chunk_meta` and the block-table plumbing, and was cut inside a
reasoning block — 22 assistant turns, ~11 K of a 16 K-token output budget, every
message a tool call, zero message text. The confound and the matrix were the
deliverable (`../accuracy-benchmarks/hybrid-block-size-corruption.md`).

**#1144** vllm-ascend v0.7.3.post1 (`c69ceac`) / vllm v0.7.3, torch+torch_npu 2.5.1,
aarch64 EulerOS: `import mindie_turbo` →
`ImportError: …/mindie_turbo/_ops.cpython-310-aarch64-linux-gnu.so: undefined
symbol: _ZNK6AsdOps3Ops18GetOperationByNameERKSs`, while the env dump selects
`ATB_HOME_PATH=…/nnal/atb/latest/atb/cxx_abi_1` — a `std::string` ABI mismatch
between a prebuilt wheel and the selected NNAL/ATB root. Answered on its 9th
assistant turn with the demangled symbol, the three `nm -D` checks and the
`cxx_abi_0` export, and the
**reporter confirmed on 2025-06-10: "Thanks, the env setting works for me."** The
run also pinned the impact split (worker `try_register_lib` warning vs
`NotImplementedError: There is no available ascend quantizer` on the quantised
path); the weak spot was guessing which ABI torch_npu's own `libop_plugin_atb.so`
needs instead of asking for the one `nm -D` count, which its answer then correctly
listed as the deciding check (`../image-dependency-faults/abi-undefined-symbol.md`).

**#1696** vllm-ascend `v0.7.3` (`779eebb8`) + vllm v0.7.3, image
`quay.io/ascend/vllm-ascend:v0.7.3`, host CANN 8.1.RC1, an **Ascend 910C1** card
(`npu-smi` name), `LLM("Qwen/Qwen2.5-0.5B-Instruct")`: model construction dies in
upstream `rotary_embedding.py:109 _compute_inv_freq` → `torch.arange` →
`RuntimeError: call aclnnArange failed … EZ9999 … Parse dynamic kernel config fail /
AclOpKernelInit failed opType / ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed`, no
`vllm_ascend` frame; the reporter's question was literally "does this image support
910C?". Answerable from docs in four calls (`Dockerfile:16` base
`cann:8.1.rc1-910b`, `docs/source/conf.py:78`, `faqs.md:14-17` "ONLY Atlas A2",
`installation.md:87-89` the `-910b` kernels run) plus one standalone op probe, and
the run also disposed of the `Failed to register custom ops` /
`No module named vllm_ascend.vllm_ascend_C` red herring (`setup.py:296-298`,
`platform.py:29-38`). Answered in full on its 13th assistant turn, with the verdict already in the
seventh turn's message text; the
sibling of #11519's `EZ1001` on the same axis — built-in op here, custom op there
(`../soc-platform-paths/kernel-package-vs-card.md`).

**#13719** vllm-ascend v0.23.0rc1 (`f4a08bdd`) + vllm v0.23.0 (`0fc695fc`) as the
snapshot's reference, but the report's build was a daily "0.23.0": GLM-5.2 W8A8C8 on
910C, 4-node PD disaggregation (`MooncakeConnectorV1`) with `deepseek_mtp`
(n=1 P / n=5 D), 64 concurrency MMLU-Pro, `max_tokens` 32000, "2 小时后 5 条响应存在
句子的重复", both serve commands pasted and no logs at all. Delivered in full at its
10th assistant turn: the class is decode-time repetition (short
prompt, long completion — not the prompt-length corruption pages), and the
deliverable was the tutorial diff (`GLM5.2.md` 4-node 1M PD example) plus the
`--max-num-seqs 1` identity row. The run's useful correction: at this ref
`using_block_verify` also requires `rejection_sampler_config.enable_block_verify`
(default `False`, `ascend_config.py:766`), so the n≥3 block-verify story does not
transfer from #7580 (`../accuracy-benchmarks/long-output-repetition.md`).

**#13002** v0.23.0 on A2, Kimi-K2.6 W4A8, 1P1D `MooncakeConnectorV1` + eagle3,
≥16 concurrent / ≥23K-token prompts: `</think>` leaking into `content`, repetition
collapse, 乱码 in 6-10 % of requests, both serve commands pasted. **Shipped no
answer**: 21 assistant turns, 1.06 M prompt tokens and the whole 16 K output
budget, ~8 turns reading `mooncake_connector.py` in 2600-line slices; the model's
own tutorial (`docs/source/tutorials/models/Kimi-K2.6.md`, validated P/D commands
and §9.1 table) was opened only at turn 17, though its diff — `--no-enable-prefix-caching`
on both nodes vs the reporter's `--enable-prefix-caching`, `--max-model-len 32768`
vs 131072, P `HCCL_BUFFSIZE 800` vs 256, D `--max-num-batched-tokens 32/4` vs 256,
P eagle3 `num_speculative_tokens 1` vs 3 — plus the ablation matrix was the answer
(`../accuracy-benchmarks/garbled-long-context-output.md`).

**#7197** vllm-ascend v0.16.0rc1 + vllm v0.16.0, 910B3, Qwen3-Next on a customer
dataset: `ERROR [serving.py:315] Error in preprocessing prompt inputs` with
`RuntimeError: Already borrowed` raised at `hermes_tool_parser.py:63` (the tool
parser's `__init__` encoding its special tokens) under `_preprocess_chat` — a
fast-tokenizer borrow conflict between the event-loop thread and
`AsyncMicrobatchTokenizer`'s single-worker executor, with no `vllm_ascend` frame
and no downstream tokenizer code to blame. The recorded run answered on its 14th
assistant turn with the mechanism, vLLM's single existing retry for the same
exception (`vllm/multimodal/processing/context.py:440-459`, tokenizers#537), the
serial-concurrency/retry ladder and the transformers-4.57 line-number gap; the
prior-month history has no `Already borrowed` precedent at all
(`../serving-frontend-faults/tokenizer-already-borrowed.md`).

**#5295** vllm-ascend v0.12.0rc1 (`42ceaf0`) / vllm v0.12.0 (`4fd9d6a`), GLM-4.5-W8A8
on one 16-card 910B node: `--data-parallel-size 2 --tensor-parallel-size 8
--enable-expert-parallel --distributed-executor-backend mp
--compilation-config '{"cudagraph_capture_sizes": [4], "cudagraph_mode":
"FULL_DECODE_ONLY"}'` fails engine init with `RuntimeError: Engine core
initialization failed … Failed core proc(s): {}`, a 16-rank interleaved paste whose
deepest frames are `fused_moe/layer.py … moe_forward_shared` → `auto_functionalize.py:856`
→ `prepare_finalize.py:191`, behind `No available shared memory broadcast block found
in 60 second(s)`; the reporter asks how to run EP **and** DP together. Feature-enable
/ capability track on a lane-A/B paste. **Shipped no answer**: 16 assistant turns,
every message a tool call, ~46 K characters of reasoning, cut inside a reasoning
block. It had the selector predicate (`world_size_across_dp/pp >= 16` on A2), the
"MC2 inherits `All2All.finalize`" correction and the capacity arithmetic in hand by
turn 10; the route and the refuted uneven-split theory are in
`../feature-enable-recipes/moe-expert-parallel-with-dp.md`.

**#7689** vllm-ascend v0.13.0rc1 (`1b5d5abf`) / vllm v0.13.0 (`72506c98`), dots.ocr
on 2×910B4 (32 G) vs 910B3 (64 G) with an identical serve command: the B4 host answers
garbled or off-task text where B3 is correct. VL corruption track; the run delivered a
complete ranked answer (first ablation `USE_OPTIMIZED_MODEL=0`, then TP=1,
`--async-scheduling` off) and paid for one useful version correction: at this pin
`vllm_ascend/models/` no longer exists, the vision tower is patched by class name
(`utils.py:697` → `ops/mm_encoder_attention.py:31`) and the head-dim pad moved from
weights to **activations** (`:101-108`), so #976's weight-padding mechanism cannot be
quoted at v0.13.0rc1 — see `../accuracy-benchmarks/vl-vision-tower-degeneracy.md`.

**#15815** vllm-ascend `v0.26.0rc1` (`f2f74a16`) + vllm `0.26.0` (`568afb3a`),
Qwen3-Coder-30B-A3B BF16 TP4 with EP on and off: startup dies in `profile_run`
with `AttributeError: 'UnquantizedFusedMoEMethod' object has no attribute
'is_monolithic'. Did you mean: 'apply_monolithic'?`, raised from upstream
`moe_runner.py:564`, because `patch/platform/patch_fused_moe.py:87-88` rebinds
only the layer and package modules and `qwen3_moe.py:45` had already cached the
pre-patch factory; the inner `'NoneType' object has no attribute 'is_monolithic'`
is the real message (`oracle/unquantized.py:209` returns `(OOT, None)`, and the
base property dereferences `experts_cls`). 15 upstream modules share the hazard,
so the fix is a `sys.modules` sweep, not a qwen3 special case
(`../upstream-downstream-drift/stale-module-binding.md`). The run delivered a
complete answer, but only at its 15th turn with 31 K of its 50 K output characters
spent on reasoning — a delivery case, not an evidence case.

**#11759** vllm-ascend `v0.18.0` image, community-test row for
yxr2_qwen3-32B (Qwen3-32B-W4A4 on 1×Atlas 800I A2, HDK 25.5.0.B078), i.e. the same
task shape as #11739 a batch later and the same outcome: **no report file and no
message text**, cut at the 16,384-token cap on its 16th turn while writing the
report heredoc. Every section of the answer existed before that: the absent
"Run basic benchmark" section by turn 6, the curl's ignored
`max_completion_tokens` (`/v1/completions` has only `max_tokens`, default 16, at
vllm v0.18.0) by turn 8, the sizing pass by turn 11 (16.4 GB of W4A4 weights
against a 57.6 GB per-card budget, so one card is right and bf16 would not fit),
and the `pip install transformers==4.51.0` line against the image's own
`>=4.56`/`>=4.57.4` pins by turn 14. See
`../tutorial-verification/community-test-report.md` for the page's verified
defect set and the write-granularity rule this run cost.

**#8724** vllm-ascend `v0.13.0rc3` (`bbe9bc0d`) / vllm `v0.13.0` (`72506c98`), one
A3 16-card node, DeepSeek-V4-Flash-w8a8-mtp with
`--speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}'`,
DP4×TP4 + EP, `--async-scheduling`, `vllm bench serve --temperature 0.6`
(0 or unset runs clean): three tracebacks at the same second — worker `KeyError:
'chatcmpl-bench-…'` at `patch/worker/patch_model_runner.py:154`, EngineCore
`'NoneType' object has no attribute 'sampled_token_ids'`, APIServer `Counters can
only be incremented by non-negative amounts`. One provenance (a *discarded* request
gets a `-1`-filled sampled row → count 0 → `num_accepted = -1`, and
`prev_req_id_to_index` omits exactly those requests), and the failing statement is
upstream v0.13.0's, carried verbatim by the patch, whose only added hunk
(`is_kv_consumer`) a non-PD command never runs. **Shipped no answer**: 29 turns,
every message a bare tool call, ~48.5 K characters of reasoning against 73
characters of message text, cut inside a reasoning block on the very turn it began
the patch-vs-upstream diff — the first step that should have been taken
(`../upstream-downstream-drift/patch-copy-vs-upstream.md`,
`../spec-decode-mtp/async-spec-bookkeeping-desync.md`). A later replay that
started from the classification row and that page **answered it in 8 assistant
turns (8.4 K characters of message text)**: four confirmation reads (the
`KeyError` line, the `-1` fill in `eagle_proposer.py:825`, the index-map filter,
the `vllm_version_is('0.13.0')` install gate), then the ranked answer with the
upstream identity, the temperature anchor and the honest gap (which step discards
a row). The page, not more source, is what closed it.

**#17991** vllm-ascend `v0.23.0rc1` (`f4a08bdd`) / vllm `0.23.0` (`0fc695fc`),
8×910B3, FA3 NPU `v0.4.2`, Qwen3-0.6B with `attention_config={"backend":
"FLASH_ATTENTION"}` + `ngram` spec decode and `max_num_batched_tokens=256`:
`RuntimeError: ACL stream synchronize failed, error code:507015` (`EE9999 … aicore
exception`) at `vllm_ascend/attention/fa3_v1.py:122`. Real downstream defect one
line away from correct: the prefill half slices the request-indexed `block_tables`
by `num_decode_tokens` and passes an absolute `query_start_loc` as `cu_seqlens`, so
with spec decode the offset triples and the kernel reads outside the page table;
line 122 is the decode half's blocking `.npu()`, not the faulting kernel
(`../mixed-batch-splits/fa3-mixed-batch-split.md`, one call:
`scripts/batch-split-audit.sh`). **Shipped no answer**: 17 turns, ~51 K characters
of reasoning against 255 characters of message text — the complete source-verified
diagnosis was already written in the reasoning at the turn whose message text was
"Confirmed the fault mechanism. Let me pin two last facts", and the truncation kept
the plan and dropped the answer.

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

**#10405** vllm-ascend `v0.20.2rc1` (`367b8e6`) / vllm `v0.20.2` (`bc150f5`), 4×910B2C,
Qwen3.5-35B-A3B with `qwen3_5_mtp`, `--async-scheduling` and a 1M window set through
`--hf-overrides` yarn RoPE: "回答中英混杂". The screenshot OCR was decisive —
`"content": null`, all 500 completion tokens inside `reasoning`, `finish_reason: length`
— so the answer is routing plus budget first, model/template thinking language second
(`#6848`), a hand-written rope override third (unproven row), and the variant is absent
from the support matrix (no validated recipe to diff against). Two corrections the run
established: a missing `--mamba-cache-mode align` is auto-supplied by upstream
(`vllm/model_executor/models/config.py:349-377`), and `enable_thinking: false` carries a
version gate (`#7094`). **Delivery cost**: 24 assistant turns, most of them bare tool
calls, and 16.3 K completion tokens — the answer landed at the cap.

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

**#18315** vllm-ascend `v0.26.0rc1` (`f2f74a1`) pinned, title says v0.27.1RC1, env
block v0.26.0rc1; GLM-5.3-w8a8c8 on 4×A2 混部, `HCCL_OP_EXPANSION_MODE=AIV` +
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` → `aclnnMoeDistributeDispatchV4`
`107020` after an `event wait timeout … task_timeout`, starts when both are removed.
Real finding (built in reasoning, never delivered): the AIV × MC2 dispatch/combine
defect family #14928/#13688/#15195, plus the CaMem pool's
`expandable_segments:True` assert. **Shipped no answer**: 18 turns, 16 K output
tokens (~42 K characters of reasoning against 259 characters of message text), 17 bare
tool-call turns, no `findings.md`, cut inside a reasoning block whose last step was
rebuilding the paste into a scratch file to feed
`plog-timeline.sh` — a gate that needs a long-form timestamped record
(`[PID: n] <YYYY-MM-DD-HH:MM:SS>`) the paste did not have. The ranked answer
existed from about turn 8 (`../hccl-and-hardware/aiv-mc2-env-ablation.md`).

**#10546** vllm-ascend `v0.20.2rc1` (`367b8e6`) / vllm `v0.20.2` (`bc150f5`),
DeepSeek-V4-flash 1P1D, decode instance DP2×TP4 + EP, 10-way concurrency: the
decode worker dies with `rtMemcpyAsyncEx … reason=the model stream execute
failed`, an `aivec … timeout or trap error … subErrType:4` dump and an HCCL
`AllGather` task line (`rankSize 2`, `numBlocks 2`, `size 65536`), inside a
traceback that ends at `model_runner_v1.py:1393 counts_cpu[: counts.shape[0]]
.copy_(counts, non_blocking=True)` in `_copy_valid_sampled_token_count`
(`srclen=24` → six `int32` rows on that rank). The recorded run got the
load-bearing reading right by turn 3 — the copy is a sync point, the collective
is the fault — and then verified the DSV4 config surface
(`VLLM_ASCEND_APPLY_DSV4_PATCH` default `0` gating `patch_speculative_config`,
the `deepseek_v4`→`deepseek_mtp`/`DeepSeekV4MTPModel` remap, the nightly A3 yaml
as the only in-tree recipe, `disable_padded_drafter_batch` default `False`)
before being cut with **no answer**: 14 assistant turns, 23 tool results, 16.4 K
completion tokens, ~41.6 K characters of reasoning against ~2.6 K of message
text (every turn's text was a one-line status note). **Delivery cost**: from
turn 4 on, each turn opened with "I need to answer now" and made one more call;
turn 12 spent itself writing `/workspace/scratch/findings.md` instead of message
text; turn 14 carried 3.6 K characters of reasoning and zero text. It was also heading for a wrong
mechanism: it read `rankSize 2` as "the shared-expert DP AllGather", while
`fc3_quant_x`/`mc2` are `DP×PCP×TP` groups (`parallel_state.py:85-110`) and
cannot be 2 ranks on DP2×TP4. Both lessons are in
`../spec-decode-mtp/drafter-count-copy-device-fault.md` and in the classification
row that now routes there.

**#16111** vllm-ascend `v0.26.0rc1` (`f2f74a1`) + vllm `v0.26.0`; 800I A3 single node,
DeepSeek-V4-Flash-0731-w8a8, DP4×TP4, AscendStore + MemCache three-tier cache. Two
cases in one report: `device_sdma` → first curl fine, second crashes with
`HCCL watchdog thread terminated with exception` / `ERR02005 DIST internal error`
(reporter suspects "vllm-ascend 懒加载"); `host_shm` → dies at memcache init with
"device id 转换异常". A later public comment by the original reporter (2026-09-14)
adds a wiki log link for the device-id case — unreadable here, so it changes which
sub-case is live but supplies no quotable line. Real findings (source-read at the
pin, never delivered): lazy init is the documented A3+DSV4+`device_sdma`
behaviour (#9975 item 2), the resolved `mmc-local.conf` carries two keys that
exist nowhere (`ock.mmc.local_service.standalone.dram.size`, `hugepage_mem_gb`)
while `dram.size` is per-die and should be `0GB` on A3/HCCS (`kv_pool.md:511`,
`:761-790`), and the HCCL line is downstream of a swallowed rank-local save
failure (#14106). **Shipped no answer**: 36 turns, every message a bare tool call,
cut inside a reasoning block that was composing the answer ("Verified facts
(pinned v0.26.0rc1 …)"). Lane: classify from the protocol value in the resolved
config file, then the config/version diff (`../kv-pool-store/memcache-a3-dsv4-init.md`).

**#8788** vllm-ascend `v0.13.0rc3` (`bbe9bc0`) + vllm `v0.13.0`; A3 16 cards,
"Cannot install from source in an offline pod: custom-ops step pulls dependencies
from the public network". Real finding (source-read, never delivered): the chain
`setup.py` `build_aclnn` → `csrc/build_aclnn.sh` → `csrc/build.sh --pkg` →
`csrc/CMakeLists.txt` fetches a `catlass` git submodule (always, since
`csrc/third_party` is gitignored) plus json/abseil-cpp/ascend_protobuf/makeself
from `gitcode.com`, each behind an existence gate under `CANN_3RD_LIB_PATH`;
`CANN_PKG_SERVER` reaches only two orphan cmake files, and `setup.py` never
forwards `CANN_3RD_LIB_PATH` to CMake. **Shipped no answer**: 30 turns, every
message a bare tool call, cut inside a block that had already written
"**Answer**: Yes, real…". Lane: name the fetch legs and the pre-seed directory
(`../image-dependency-faults/offline-source-build.md`).

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

**#8590** vllm-ascend `v0.14.0rc1` (`52d4acfa`) + vllm `v0.14.1` (`d7de043d`),
single 910B-class card, TP=1, Qwen2.5-1.5B: repeated 猞猁 from the first generated
token whenever `<|im_start|>`/`<|im_end|>` are in the prompt; reporter later confirmed
"0.14.0rc1 through 0.18.0rc1, same". **Shipped no answer — cut twice** (11 steps at a
16 K cap, then 3 more at 8 K after the reporter's ping), with ~21 K characters of
reasoning in one block and every message a one-line status note. The paste already
carried both discriminators it never used in message text: the printed `Input ids` are
identical on both input routes (so token ids cannot be the mechanism), and
`Using max model len 131072` is the checkpoint's own value
(`vllm/config/model.py:1545`, `_get_and_verify_max_len` at `:1898`), which stock
Qwen2.5-1.5B does not declare. Real findings, source-read and never delivered:
no id-keyed branch in `vllm_ascend/` at that ref and no `vllm_ascend/models/` at all;
`AscendVocabParallelEmbedding._forward_origin` skips the mask at `tp_size == 1`
(`ops/vocab_parallel_embedding.py:174-184`), so an out-of-range id reaches the
embedding call instead of being zeroed; and Qwen2.5-1.5B *generation* has no e2e coverage (only the pooling
checkpoint at `tests/e2e/singlecard/pooling/test_classification.py:10`). Lane:
`../accuracy-benchmarks/short-prompt-corruption.md` step 0.
**#17797** vllm-ascend `v0.23.0` (`5cb98ca`) + vllm `v0.23.0` (`0fc695f`), image
`quay.io/ascend/vllm-ascend:v0.23.0-310p`, Ascend 310P; `Qwen/Qwen3-Embedding-0.6B`
with `--runner pooling --dtype float16`, single request at a time, prompts
of ~4-10 K characters (~3-8 K tokens): wrong embedding vectors for long inputs, cosine vs three
reference services collapsing from `>0.999` to `0.2-0.6`, deterministic per serve
session and different across restarts, gone with `--no-enable-chunked-prefill`.
Real findings (source-read at the pin, never delivered): the chunked-prefill
accumulator is inherited upstream (`PoolingStates.hidden_states_cache`,
`tokwise/methods.py:35-84`, `gpu_model_runner.py:3330-3343`) so "Ascend never
populates `pooling_states`" is refuted *at this pin*; the Ascend-specific pooling
sites are `attention_v1.py:1535-1538` (`and not causal` -> the prefix-blind
`_forward_encoder_attention`), the fixed-size pooling mask branches
(`attention_mask.py:75-77`, `_310p/attention/attention_mask.py:139-145`) and the
310P dispatch `_310p/attention/attention_v1.py:329-342`; and the mask theory is a
dead end for the chunked step because `forward_chunked_prefill_310` builds its own
`get_splitfuse_mask`. **Shipped no answer**: 18 assistant turns, all 16 K output
tokens spent inside reasoning, one short status note on each of the first 17,
cut mid-sentence while it was still ranking the causes (its own notes had said
"I must deliver now" three turns earlier). Lane: pooling + chunked prefill (`../pooling-embedding/index.md`).

**#5525** vllm-ascend `v0.11.0` (`2f1aed9`) + vllm `v0.11.0` (`b8b302c`),
`quay.io/ascend/vllm-ascend:v0.11.0`, Qwen3-Embedding-8B served with `--task embed
--max-num-seqs 256 --max-num-batched-tokens 8192`; `vllm-bench`
(`openai-embeddings`, sharegpt) answers ~1800 requests, then the engine dies and
everything after gets `400 Bad Request`. Evidence is four bench screenshots only;
the `KeyError` key is unreadable, so the mechanism is source-derived, not
reporter-confirmed. Real finding (source-read, never delivered): at v0.11.0
vllm-ascend ships **its own copy** of upstream's `InputBatch`
(`vllm_ascend/worker/npu_input_batch.py:94`) with a pooling fast path upstream
lacks (`add_request` -> `num_reqs` at `:316-324`, `remove_request` skipping
`removed_append` at `:462-465`, trim-only `condense` at `:573-577`, forced by
`assert request.sampling_params` at `:303`), which leaves a `None` hole plus a
duplicated slot under concurrency and raises at
`vllm/v1/attention/backends/utils.py:798`; by v0.23.0 the copy is gone
(`class NPUInputBatch(InputBatch)`, `__init__` only). **Shipped no answer**: 20
assistant turns, 11 of them with zero message text (bare tool calls), the 16 K
output budget entirely in reasoning, cut while it was still verifying
`add_request`'s index assignment after having named the root cause two turns
earlier. Lane: pooling batch bookkeeping
(`../pooling-embedding/batch-index-desync.md`).
