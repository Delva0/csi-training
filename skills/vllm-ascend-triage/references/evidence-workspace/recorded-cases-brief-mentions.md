# Brief mentions

Cases recorded as compact mentions: one paragraph listing several issues where the
classification, the one-line fact and the delivery outcome are all that is known.
Use them as precedent ids and route to the topical page named in each clause; the
full entries live in the other five case pages.

Inherits the scope and rules of `recorded-cases/index.md`, whose page for this
family carries one line
per case in this file.

**#3859** v0.11.0 line (pin `v0.11.0rc0` `00ba071`), 910B3, no log: "910B3 是否支持权重
nd 到 nz 的转换，提高 matmul 的性能？0.11.0 的代码好像只支持 310P" — a capability question
whose premise is a guard read; at that ref quantized Linear (`w8a8.py:139`,
`w8a8_dynamic.py:104`) and all MoE weights (`not is_310p()` / unconditional) are already
NZ on 910, and only the unquantized **dense** cast is 310P-gated
(`model_runner_v1.py:2633-2643` with `ACL_FORMAT` NZ only on 310P); no `weight_nz_mode`
exists at that ref. Answered completely in 14 assistant turns, but three of them chased a
gate function (`_should_trans_nz`) that is **absent** from that tree because
`nz-format-gate-check.sh` reported "SHAPE GUARD: absent in _should_trans_nz" for a
function that does not exist there — the script now detects the absent gate
(`../feature-enable-recipes/weight-nz-layout-by-soc.md`). The reporter's follow-up was an
image with no text, so nothing was confirmed.

**#8944** v0.19.1rc1
`aclnnAddRmsNormBias` device fault; #10068 evalscope vs ais-bench MMLU-Pro gap;

**#3543** v0.11.0rc0 silent worker death; #7477 v0.17.0rc1 HCCL allreduce ending in
an HCCS link fault; #11661 v0.22.1rc1 missing `attention_type` on a patched
decoder layer (re-filed #11926 and #11991; the guard fix starts the service and breaks recognition); #8409 v0.13.0 `model_type` not recognised (version gate); #16015
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
parameter-invalid on a chip the reporting rank did not own; #133 vllm-ascend 0.7.1rc1 + vllm 0.7.1, open-r1/`trl` GRPO rollout failing
on step 0 with `RuntimeError: Prefix cache and chunked prefill are currently not
supported` — a downstream capability gap whose trigger was the trainer's own
`LLM(enable_prefix_caching=True)`; answered in 12 calls from the raise site plus the V0
defaults, the shape to copy — `../upstream-downstream-drift/unsupported-path-raise.md`.

**#1609** vllm-ascend v0.9.0rc2 (`8dd686d`) + vllm v0.9.0, Atlas 300I Duo, no log:
"does vllm-ascend v0.9.0rc2 support 300i + qwen2.5-vl-7b?". The pin is **one release
before the lane exists** — `git grep -n -i -E '300i|310p'` returns one hit
(`docs/source/faqs.md:18`, "NOT supported yet … might be supported on 2025.Q2"), there is no
`single_node_300i.md`, no `Dockerfile.310p` and no `is_310p` — so the 300I model list the
reporter quotes (Qwen3-0.6B 1 card / Qwen2.5-7B-Instruct 2 cards / Pangu-Pro-MoE-72B 8
cards) is a later ref's page: `Pangu` occurs in no file at this pin. Answered at its 9th
assistant turn from that grep, the unconditional `qwen2_5_vl` registration, and the move to
a `-310p` tag; #1383 (v0.9.1rc1, `TransData` op-store refusal on a 0.5B text model) is why
the ladder must not promise the upgrade fixes it — `../soc-platform-paths/atlas-300i-310p.md`.

**#8920** vllm-ascend v0.18.0 (`e18643f8`) + vllm v0.18.0, Qwen3.5 MoE under PD "per the
docs", curl answer mixes Chinese and English, **no** env dump / serve command / log. One
documented PD recipe exists for this family and it is in the model page
(`Qwen3.5-397B-A17B.md`), while the generic PD page names two different connectors in
different sections, the generic one being the connector whose `register_kv_caches` derives a
single page geometry from the first cache tensor. Answered with ranked causes plus an
ablation ladder whose rung 1 is the single-instance control run (the report carried no
working-vs-failing control) — `../pd-disaggregation/hybrid-state-transfer.md`. The run
emitted text on its first and last turns only (13 silent turns, ~30 K characters of
reasoning against 6.3 K delivered); the ranked facts were in hand by the 11th turn, so
three more confirming reads were all the margin it had.

**#6549** vllm-ascend v0.14.0rc1 (`52d4acfa`) + vllm v0.14.1 (`d7de043`), report filed
on 0.13.0: "0.13.0 Release vllm-ascend MLA interface is different form 0.13.0 vllm",
whole body = that sentence + two `<!-- Failed to upload -->` images, no log, no version
dump. Not a crash report — the question form of the drift topic, and the answer is
which class actually runs: `AscendMultiHeadLatentAttention` (OOT map
`vllm_ascend/utils.py:715`) routes through `torch.ops.vllm.mla_forward` to
`AscendMLAImpl.forward(layer_name, hidden_states, kv_cache, attn_metadata,
need_gather_q_kv, output)` (`mla_v1.py:1448-1456`), while upstream's
`MLAAttention.forward` → `impl.forward(self, q, kv_c_normed, k_pe, …)` and every
`vllm/v1/attention/backends/mla/*` impl are never reached on NPU (`platform.py:467`).
The recorded run answered in 13 turns and delivered the table — the good shape for
this family — and #868 ("Why not `AscendMLAImpl` just inherit `MLACommonImpl`?") is the
earlier reporter asking the same design question; #3138 is the crash form of the same
subject (`get_attn_backend() got multiple values for argument 'use_mla'`). The
reporter's later public comment is two more images, so nothing was confirmed —
`../upstream-downstream-drift/impl-and-oot-signature-drift.md` Rule 6 (new).

**#5158** vllm-ascend v0.12.0rc1 (`42ceaf08a1ef`) + vllm v0.12.0 (`4fd9d6a85c00`), image
`quay.io/ascend/vllm-ascend:v0.12.0rc1`, `[Doc]` report: the DeepSeek-V3.2 tutorial says
"install extra operator" and gives no how, and the reporter asks whether #3278's manual
`deepseek_v3_ops` build is what they must do. The sentence is inherited boilerplate
(`Qwen2.5-7B.md:27`, same text) while the sibling DeepSeek pages say "run … directly" and
the same release's own `release_notes.md:8` says "you don't need to install any other
packages now" (`:15` custom ops mandatory, `:14` the triton-ascend wheel is the only
optional extra) — tier stale, the real gap being the wheel with no URL. The recorded run
delivered that verdict; the reporter's later public comment (2025-12-19) is "I switch to
latest main branch and hit another error", which is why the answer must name the
image↔vllm pair and ask for the runtime error —
`../tutorial-verification/missing-guidance-and-contributor-docs.md` §#5158.
