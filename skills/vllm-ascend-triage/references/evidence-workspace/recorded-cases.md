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
matrix (SP = VL+Dense, graph, no quantization; FC1 is complementary, not a
rename) in 14 turns with a 5.3K answer — the lead it chose, a "half-enabled SP"
mechanism, is labelled there as unverified; #8984 v0.19.1rc1
DeepSeek-V3.2-w4a8/w8a8-mtp-QuaRot on A2, "如何正确开启 PCP/DCP" with no log pasted (a
capability question — the run never stated the support verdict, and the
reporter's later public comment asked whether anyone had actually run PCP on
A2); #12487 v0.22.1rc1 A3 3P1D DeepSeek-V4-Flash long-run crash whose title
named `port 65536 have already been bound`, a three-day-old re-printed EI0019
record, while the in-window error was an AICPU `reduce_scatter`
parameter-invalid on a chip the reporting rank did not own.

**#9355** v0.18.0 Qwen3.6-27B bf16 on 2×910B3, whose first request after start returned
`content: ""` with `completion_tokens: 1` and `finish_reason: "stop"` while later
requests answered (the run shipped only a half-answer; its lead mechanism was a "server-lifetime first decode" story that the
reporter's later comment leaves unestablished, since only the prompt and `temperature`
changed); #10495 v0.20.2rc1 Qwen3.6 MoE + `qwen3_5_mtp` on 2×A2 with ~64K prompts,
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
