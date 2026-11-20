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

**#1784** vllm-ascend v0.9.2rc1 (`b5b7e0e`) + vllm v0.9.2 (`a5dd03c`), "installing
vllm-ascend 0.9.1dev and `pip show vllm-ascend` shows 0.9.0rc2": packaging metadata,
no lane — `setup.py:305-311` derives the version with `setuptools_scm` from the build
tree's tags (`LookupError → "0.0.0"`), `vllm_ascend/_version.py` is gitignored, no
`vllm_ascend/` code reads its own version (`utils.py:283-295` `vllm_version_is()`
reads `vllm.__version__` / `VLLM_VERSION`), and `-dev` is a branch name, not a pip
version. Answered on the last turn over 13 tool calls, ranking stale dist-info →
wrong checkout → tag/branch expectation; the recorded gap is the snapshot's own
one-commit export with no tags, so the tag-ancestry half is the reporter's
`git describe` to run, not a local check. Route now
`../image-dependency-faults/installed-version-vs-checkout.md`.

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
backend) two turns after the follow-up. A later recording of the same body (pinned
v0.22.1rc1, 35 messages / ~15 K completion tokens) got the lane and the
`reasoning-parser` half right but **ranked a `stop_reason` forensics finding first** —
"the w4a8 export declares a secondary eos 248044, so the engine stops mid-CoT", all
`vllm/` lines verified at `0decac0` — and demoted the reporter's own
single-variable result (delete `enable_sp` ⇒ output returns) to hypothesis 3, so it
never made rung 1 the opening action; #8984 v0.19.1rc1
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

**#1262** vllm-ascend v0.9.0rc2 (`8dd686d`) + vllm v0.9.0 (`5873877`). Body = the
PR #1251 link plus one review-comment link: "Remove `VLLM_ASCEND_ENABLE_NPU_MROPE`
before 0.9.1 final release". No log, no environment dump — a code-change request.
The identifier and any `mrope` op are **absent** from the pinned tree (only the
`mrope_positions` plumbing in `worker/model_runner{,_v1}.py`, and `csrc/` has
`pos_encoding_kernels.cpp` only), so the snapshot predates the PR and the
deliverable is the removal sweep on the branch that carries it. Found on the way:
at vllm v0.9.0 `MRotaryEmbedding.forward`
(`vllm/model_executor/layers/rotary_embedding.py:1008`) bypasses `CustomOp`
dispatch, so the tree's `RotaryEmbedding.forward_oot = rope_forward_oot`
(`vllm_ascend/ops/rotary_embedding.py:273`) never covers mRoPE — a downstream shim
can be unreachable without any error. Shipped a good answer, but its first
message text came on its **15th assistant turn**: 14 bare tool-call turns, no
truncation pressure. Track: code-change / flag retirement
(`../code-change-requests/`).

**#5509** vllm-ascend v0.13.0rc1 (`1b5d5ab`) + vllm v0.13.0 (`72506c9`),
DeepSeek-V3.1-Terminus-w4a8, 2 nodes × (DP4, `--data-parallel-size-local 2`),
TP4, `--enable-expert-parallel`, `--async-scheduling`, MTP 3, FULL_DECODE_ONLY,
`VLLM_ASCEND_ENABLE_MLAPO=1`, `VLLM_ASCEND_BALANCE_SCHEDULING=1`,
`HCCL_INTRA_PCIE_ENABLE=1`/`HCCL_INTRA_ROCE_ENABLE=0`. Weights and
`torch.compile` succeed, then ~10 min of `No available shared memory broadcast
block found in 60 seconds` and `507014` aicore timeout at
`torch.npu.synchronize()` in `profile_run` → `determine_available_memory`, fault
kernel `MoeDistributeDispatchV2_…_high_performance_…`, title blaming
`aclnnInplaceCopy`. The session read the selector, corrected its own first-turn
blame of the `HCCL_INTRA_*` pair (the dispatcher comment and the tutorial both set
it), and named the real reason the kernel appears during profiling: `profile_run`
runs an extra `_dummy_run(mc2_tokens_capacity, with_prefill=True)` at
`model_runner_v1.py:2202-2208`. Answer shipped in Chinese with the DP4×TP4 = 16
boundary arithmetic, the tutorial diff, a refuted-rows list and a four-rung ladder.
Track: lane C, MC2 dispatch during profiling
(`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md`); the hang's own cause
stayed open (no `npu-smi`, no headless-node log in the paste).

**#9986** vllm-ascend v0.18.0rc1 (`99e1ea0fe6`) / vllm v0.18.0 (`bcf2be9`), image
`quay.io/ascend/vllm-ascend:v0.18.0rc1`, one 910 card, dense Qwen3-0.6B
(`--quantization ascend --tensor-parallel-size 1 --async-scheduling`), the
FLASHCOMM1 `export` commented out in the pasted script; the report's only evidence
was a screenshot (误报信息) and its complaint was the startup line `Unknown vLLM
environment variable detected: VLLM_ASCEND_ENABLE_FLASHCOMM1`. Track: observability
/ env registry (`../observability-and-logs/env-var-validation-warning.md`). The run
got the mechanism right — upstream `validate_environ`, warn-only, vllm-ascend's
registry never merged, merge belongs in `pre_register_and_update` — and still
**lost its closing scope section to the cap** after 13 assistant turns and ~40 K
characters of reasoning (the 8th turn was a bare tool call, so G2 carried no ranked
answer). Two factual errors came from the calls it skipped: it told the reporter
"your screenshot cannot be read in this snapshot" although
`/workspace/attachments/9986/screenshot-*.ocr.txt` was present and carries the
emitting line `[envs.py:1717] (APIServer pid=231)`, and it stated that
`VLLM_DISABLE_SHARED_EXPERTS_STREAM` (`platform.py:31`) is the only `VLLM_*` name
vllm-ascend writes, missing `VLLM_ASCEND_ENABLE_NZ` /
`VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` (`batch_invariant.py:79-81`) that its own
grep had already printed. One call went to a paginated `github` issue search
(17 K characters, `display_complete: false`) that produced no precedent the corpus
lacked. Delivery: verdict and action arrived; version scope and open questions lost.

**#9984** vllm-ascend v0.19.1rc1 (`da421afad7`) / vllm v0.19.1 (`b1388b1fbf5a`),
GLM-5.1-w4a8 started with the tutorial's FLASHCOMM1 + MTP flags:
`AttributeError: 'AscendRowParallelLinear' object has no attribute
'aclnn_input_scale'` at `sfa_v1.py:707` ← `:544`. Track: DSA-CP o_proj weight
switching (`../dsa-sparse-attention/o-proj-quant-attribute-crash.md`). Delivered a
complete ranked answer in 10 assistant turns: `enable_sp()` ← FLASHCOMM1 →
`enable_dsa_cp()` (needs `index_topk`) → the non-layer-shard branch clones three
tensors only a static-W8A8 method creates, so the fix is FLASHCOMM1 off
(reporter-confirmed) or the W8A8 checkpoint; the pinned GLM5 tutorial ships
FLASHCOMM1 only with `glm5-w8a8`, and no `tests/e2e/**` FLASHCOMM1 config is W4A8.
The decisive detour was one `history-titles.sh 'aclnn_input_scale|o_proj_tp|dsa_cp'`
call (4 hits, one exact precedent: #7937, v0.18.0-a3 nightly, same frame on a
decode instance).

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

**#3979** vllm-ascend v0.11.0rc0 (`00ba07102212`) + vllm v0.11.0rc3
(`8ce5d3198d00`), QwQ-32B-w8a8 on Atlas 800I A2: any request with
`extra_body={"guided_choice": [...]}` kills EngineCore with
`RuntimeError: Invalid request status: WAITING_FOR_FSM` at
`vllm_ascend/core/scheduler.py:274`, client sees 500, then the `nanobind: leaked …
GrammarMatcher` teardown noise. The reporter attached the full serve log, so the
decisive line was in hand by the 5th assistant turn; the answer is the escape from
the fork plus the three-site patch. The recorded run guessed the fork was opt-in
("you must have enabled it via `--additional-config`"), which the pinned tree
contradicts: `platform.py:139-150` **forces** `ascend_scheduler_config.enabled =
True` for a non-MLA, non-multimodal model with the default `auto` structured-outputs
backend, so a plain default serve lands in the fork and `enabled: false` is
overwritten; the block is gone by `v0.11.0rc1`. That one grep, and the escape flags
it implies, are the page's main added value over the recorded answer. Track: downstream
scheduler-fork drift (`../upstream-downstream-drift/scheduler-fork-missing-status.md`,
page written from this case). Delivery: the verdict text came on turn 15 of 16 —
the turns in between pinned the docs/matrix claim, the e2e coverage gap and the
`enabled` default, all of which the answer needs, but the verdict and cheapest
action could have gone out at G2 (turn 8) and been corrected in place. The case was
also a version mismatch: the reporter ran `0.11.0` final while the snapshot pins
`rc0` (line numbers matched), so the "is it default-on / fixed" half was correctly
left open.

**#5750** image `vllm-ascend-0.13.0-rc1`, 4×910B, Qwen3-Next-80B-A3B-Instruct:
"Failed to compile custom operators with Triton Ascend" on the first request
(`MLIRCompilationError` / `[ConvertLinalgRToBinary] … operand is unlinked` /
`bishengir-compile … exit status 1` inside
`vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py`), in a container that had
`pip install`-ed `triton-ascend-3.2.0rc4` from a mirror over the image's own dev
package. Track: image/dependency pair
(`../image-dependency-faults/triton-ascend-pair.md`). The reporter's later
follow-up carries the maintainer answer — do not install triton-ascend in these
images, and v0.14.0rc1 needs none — which is exactly the action the session was
circling. Delivery: **no ranked answer at all**; 10 assistant turns, ~40 K
characters of reasoning against 3.8 K of message text, cut on the turn where it
had `findings.md` and the verdict in hand. Its first bash call paid 104 KB for
`cat SOURCES.json` + `ls /workspace/history` (SKILL.md rule 7), and the pinned
tree is v0.11.0, which has no `vllm_ascend/ops/triton/` directory — the version gap
it flagged correctly.

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

**#6275** vllm-ascend v0.14.0rc1 (`52d4acf`) + vllm v0.14.1 (`d7de043`), 2×A2
910B2 64G, `DeepSeek-V3.2-Exp-w8a8`, DP2 × TP8 + EP + MTP `num_speculative_tokens:
2`: `fftsplus aivector … The DDR address of the MTE instruction is out of range`
at `_dummy_sampler_run` (`model_runner_v1.py:2158`, `profile_run`), followed by
the UCE-repair block (`507899` / `rtGetMemUceInfo` / `DSARandomNormal` /
`aclnnInplaceNormal`) and a `WorkerProc hit an exception`. Track: lane C device
fault at a profile-run sync point
(`../hccl-and-hardware/error-report-timestamps.md`). The launch command is
verbatim the tree's A2 tutorial block, so the config diff is exhausted; the paste
does carry two environments (`npu-smi 24.1.rc3` / openEuler 5.10 versus
`25.3.rc1` / Ubuntu 5.15). Delivery: **no message text at all** — 10 assistant
turns, ~48 K characters of reasoning, 2.7 K characters of status lines, cut with
an empty final message; much of it spent deciding whether `aclnnInplaceNormal` was
a random-number defect while the paste's own async caveat and the
`checkUceErrAndRepair` frame already said it was not.

**#12261** vllm v0.20.2 (`bc150f5`) + vllm-ascend v0.20.2rc1 (`367b8e62`), GLM-5.1
(`GLM_5_1_MEP-1.0.0`) w4a8/w8a8 on 910C 560T, `stream: true` only: the client's
assembled `tool_calls[0].function.arguments` is `{"hvd": Pura900"}` — balanced
braces, one quote short, missing the *opening* quote of the value. Reporter pasted
the raw SSE, both `/v1/completions` captures and the request body, which is what
made it answerable. Track: tool-call output integrity, the schema-type-gate branch
(`../tool-call-streaming/arg-type-gate-unquoted-value.md`, new). Mechanism, source-
verified at that pin: `hvd` is declared `anyOf: [{"type":"string"},{"type":"null"}]`,
so `_is_string_type` (`vllm/tool_parsers/glm4_moe_tool_parser.py:127-148`, only a
top-level `type: "string"` counts) returns `False`, the partial value is streamed
raw (`:408`), and when the pair closes the quoted form arrives one character
*earlier* than the streamed length, so `_compute_args_diff` (`:424`) drops the
quote for ever; `stream: false` is clean because `extract_tool_calls` (`:223`)
always `json.dumps`es the whole dict. Ownership: upstream vLLM — both Ascend GLM
patches are excluded by their own conditions (`patch_glm47_tool_call_parser.py:39`
needs a region with no `<arg_key>`; `patch_glm_tool_call_streaming.py` needs
`finish_reason == "tool_calls"` *and* non-empty `arguments` in the same chunk).
Delivery: **no answer shipped** — 15 assistant turns / 23 tool calls, ~35 K
characters of reasoning against 3.5 K of message text, cut on a tool call. It had
the mechanism at turn 11 and then spent four turns trying to `import vllm`
(`ModuleNotFoundError: torch`) and `import regex` and porting the two functions into
a simulation that never ran; the fragment arithmetic needed for the proof is one
subtraction (the streamed length 14 against the completed string's 18, so the diff
is the last three characters `0"}`) and can be done on paper.

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
