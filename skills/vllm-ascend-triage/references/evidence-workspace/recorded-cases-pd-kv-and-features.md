# PD, KV-pool and feature cases

PD disaggregation that connects and then corrupts or cannot move KV, KV-pool store
segments and the MemCache init lane, EP+DP and DP>1 MoE recipes, logging and
profiling flags, flag-retirement requests and downstream patch/registry drift.
Routes: `../pd-disaggregation/index.md`, `../kv-pool-store/index.md`,
`../feature-enable-recipes/index.md`, `../observability-and-logs/index.md`,
`../code-change-requests/index.md`, `../upstream-downstream-drift/index.md`.

Inherits the scope and rules of `recorded-cases/index.md`, whose page for this
family carries one line
per case in this file.

**#15402** vllm-ascend v0.23.0rc1 (`f4a08bd`) + vllm `0.23.0+empty`, torch_npu
2.10.0.post2, CANN 9.0.1; MiniMax-M2.5-w8a8 at 40-60 K context; PD with `MultiConnector` +
`MooncakeLayerwiseConnector` (`use_layerwise: true`, `use_ascend_direct`, P dp1/tp8, D
dp1/tp8): 「偶现TTFT变短，并且输出乱码」, with an attached HTML report holding the
reporter's own wait-idiom A/B (`event.synchronize()` → intermittent short TTFT + garbage;
stream-side wait → stable), i.e. the attachment *was* the experiment. Delivered on the ninth
assistant turn, on source alone: `mooncake_layerwise_connector.py:483-490` is the only branch that
waits host-side while the two `resharding_stream` siblings insert `reshape_cache_event.wait()`
as a device dependency (`:1736-1737`, `:492`); `pd_head_ratio` is `ascend_config.py:211`
(default 1, MLA checkpoints skip the block, so the host branch is the ordinary path); the
`(TTFT, output)` pairing is the closing check and the mamba/linear coverage question sits
below it. Two corrections to the recorded ranking are on the page: KV quantization is a
**confounded** discriminator (it also changes dtype and buffer registration), and the
pin's `:484-489` ADXL comment describes a *hang*, not an early return. Reporter later
confirmed publicly that both sides ran the current `0.23.0+empty` / `0.23.0rc1` pair, so
the version lane is closed → `../pd-disaggregation/layerwise-send-sync-race.md` (new).

**#7182** vllm-ascend v0.13.0 (`6281c12`) + vllm v0.13.0 (`72506c9`), A2 and A3: "[Bug]:
DSv3.1, the TTFT of Prefill-Decode Disaggregation is 2 times longer than TTFT of
Multi-node Deployment … the typical model of qwen no this issue", no log, no metric block,
no serve commands, empty `collect_env`. Measurement lane, delivered as a verdict: the TTFT
chain plus **#3099** (the proxy returns the first token only after D produces one, so
PD > colocated is structural), the pinned tree's own anchor
(`DeepSeek-V3.1.md:692-706`: 6.16 s @ 3.5k/1.5k, P DP2+TP8 / D DP32+TP1), the recipe flag
asymmetry the comparison hides (PD prefiller `--enforce-eager` `:319` + `--max-num-seqs 8`
`:318` + ns 1 `:324` vs colocated `--async-scheduling` `:121` + `FULL_DECODE_ONLY` +
`--max-num-seqs 16` + ns 3 `:128`), the concurrency×`max-num-seqs` trap
(`long_sequence_context_parallel_single_node.md:114`) and
`VLLM_ASCEND_BALANCE_SCHEDULING` (`:135`). (run: seven reasoning-only assistant turns
before the first message text, then one 5.7 K-character answer cut inside `## Scope and
gaps` — the same content was a two-turn answer) → `../pd-disaggregation/pd-vs-colocated-ttft.md`
(new).

**#6950** vllm-ascend v0.15.0rc1 (`3d43ed99`) + vllm v0.15.0 (`f176443`), body only
"[Bug]: v0.15.0rc1, Flash Comm V1 is not supported for VL models … Assertion failed … while
v0.13.0 supports this" with an empty `collect_env` block. The string is the project's own guard —
`assert not is_vl_model(vllm_config), "Flash Comm V1 is not supported for VL models. Please
disable it by setting VLLM_ASCEND_ENABLE_FLASHCOMM1=0. For optimal performance with VL models, we
recommend enabling Sequence Parallelism via --compilation-config '{"pass_config": {"enable_sp":
true}}'."` at `platform.py:399-403`, under `if enable_flash_comm_v1():` (`utils.py:727-733`) — so
the answer is a capability verdict plus the replacement route, not a regression hunt. Verified at
the pin: `is_vl_model()` now counts `vision_config` as well as `thinker_config`
(`utils.py:816-827`); `docs/source/user_guide/feature_guide/sequence_parallelism.md:41-57` marks
SP × VL+Dense = `graph`, SP × VL+MoE = `x`, FC1 × VL = `x` (all `x` with quantization);
`enable_sp_by_pass` requires graph mode (`utils.py:736-738`, `sp_threshold` 1000); the in-tree
`tests/e2e/multicard/2-cards/test_sp_pass.py:9-50` serves `Qwen/Qwen3-VL-2B-Instruct` at TP2 with
SP off/on and compares outputs (not wired in `.github/workflows` at this ref). The "v0.13.0
worked" premise: `release_notes.md:498` places the guard under `## v0.11.0rc2` (#4222), i.e.
*before* the reporter's working version, so the plausible change is the predicate (thinker-only →
`vision_config`) or the error class (ValueError → assert) — not diffable from a one-commit
snapshot, so it was delivered as that open comparison with
`sed -n '/def is_vl_model/,+12p' vllm_ascend/utils.py` for the reporter's tree. The Qwen3-VL
tutorial's glossary still explains the FC1 export without noting the VL refusal
(`tutorials/models/Qwen3-VL-235B-A22B-Instruct.md:197`; its launch block does not carry it).
Answer shipped in 26 turns / 10 tool results — route
`../feature-enable-recipes/sequence-parallel-flashcomm.md` §*The VL / Omni guard*.

**#3959** vllm-ascend `v0.11.0rc0` (`00ba071`) + vllm `v0.11.0rc3` (`8ce5d31`),
"`v0.11.0rc0 Atlas 800T A3双机PD分离部署，当前是否支持走参数面网络通信？`" — no log, no
serve command, only "两台机器通过参数面 ROCE 网络通信，超平面不通，没有灵衢交换机". Plane
capability track, not a failure lane. Source-verified: the tree's only PD tutorial is
A2 three-node (`multi_node_pd_disaggregation.md:7`), the feature table says PD is
🟢 Functional with no SoC qualifier (`supported_features.md:25`), the A3 branch of
`examples/disaggregated_prefill_v1/gen_ranktable.py:78-101` fills `device_ip` from
`hccn_tool -vnic -g` where A2 uses `-ip -g`, and `llmdatadist_c_mgr_connector.py:681-762`
declares `server_count "2"` + `device_ip` to HCCL and adds `super_pod_list` only on A3.
Unverifiable at this ref: `cann/hccl`/`hcomm`/`hixl`/`driver` are in `SOURCES.json`'s
unavailable list and the Mooncake export has no `ascend` string at all, so which plane
HCCL selects is a plog question — the recorded answer said so, correctly. Delivery:
verdict + the `hccn_tool` ping block landed in the opening paragraphs, then four
expanding ranked causes consumed the 16 K cap and it was cut inside cause 4 with no
scope/send-back close; 26 assistant turns, 25 of them bare tool calls (2.3 K chars of
prose, all in the last message), ~8 of the 39 calls
spent in `kvcache-ai/Mooncake` and `Ascend/mind-cluster` after a one-grep answer
existed. Route now `../pd-disaggregation/plane-selection-a3-parameter-plane.md`.

**#10961** v0.22.1rc1 Qwen3.X "PD 分离+MTP 存在精度问题" filed as an empty template
(no log, no serve commands), where the run's top-ranked source-verified candidate
— the `method == "mtp"` gate missing `qwen3_5_mtp` — is refuted by upstream's
`MTPModelTypes` → `"mtp"` rewrite (`vllm/config/speculative.py:520-524`); the
artifact list plus ablation rows were the right deliverable, the ranking was not
(`../pd-disaggregation/hybrid-state-transfer.md` §3).

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

**#3107** vllm-ascend v0.9.1rc2 (`b9f715d6`) + vllm v0.9.1 (`b6553be1`), Pangu Pro
MoE on **DP=8 × TP=1 with `--enable-expert-parallel`**, `--enforce-eager`,
`--no-enable-prefix-caching --no-enable-chunked-prefill`, `--max-model-len 4096`: the
dummy run works, a real request prints `max_tokens_across_dp 68`, gathers
`[544, 5120]` (`68 × 8`), then `layer1 output` is `[68, 5120]` of ≈`-5.9e37` and the
log stops at layer 2's `input_layernorm`; the reporter's own hypothesis was the
migration from `attn_metadata.max_tokens_across_dp` to
`get_forward_context().max_tokens_across_dp`. Track: DP MoE bookkeeping, with an
out-of-tree model file — the paste's `vllm_ascend/models/pangu_moe.py` does not
exist at the pin (only a torchair comment mentions PanguProMoE,
`vllm_ascend/ascend_config.py:78`; the class is registered by v0.11.0rc0), so the
model arithmetic is the reporter's and only the framework calls are citable. The
session shipped **no answer** (12 turns, ~49.6 K characters of reasoning, 661
characters of text) though it had the writer (`ascend_forward_context.py:91-98`),
the three consumers and the `544 = 68 × 8` arithmetic by turn 6. What the next run
should have: the gathered width matching means the pad target *agreed*, so the
suspects are the owner slice (`reduce_scatter_tensor`, `ops/fused_moe.py:1446-1452`)
and the `get_fused_moe_state` branch (EP at `ep_size 8 < 16` is `All2All`, not the
DP `AllGather`) — page `../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`.
Cause unconfirmed; the reporter never replied.

**#12432** vllm-ascend `v0.23.0rc1` (`f4a08bdd`) + vllm `v0.23.0` (`0fc695fc`), GLM-5.2
1P1D per `docs/source/tutorials/models/GLM5.2.md` §PD Disaggregation 1M (P and D each
`DP4 TP8 DCP8` over 2 nodes, `--api-server-count 1` on node 0 / `--headless` on node 1),
client `evalscope` at 30 concurrency, `max_tokens: 135000`: "proxy通信超时，D实例未接收到
请求", with only four P-side `mooncake_connector.py:233 Force freed expired request … Reason:
Request exceeded timeout threshold (480 seconds)` records pasted. The 480 is the recipe's own
`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` export, and the send thread is producer-only, so the lines
prove P finished and no decoder came. The reporter's proxy listed both prefiller hosts on `9081`
and **eight** decoder entries (`9900 9901 9902 9903` per node), while `--api-server-count 1` +
`--headless` leave exactly one listening address per side (`vllm/entrypoints/cli/serve.py:61-70`,
`:105-113`) — reporter-confirmed fix, same thread: "api-count被限制为1了，所以只对外暴露一个api口，
proxy里只写master ip 和start port就行", which is the doc's own `:1460-1472` proxy line. Route:
`../pd-disaggregation/proxy-endpoint-vs-api-server-count.md`. Delivery: no answer ever shipped —
23 assistant turns / the full 16 K output budget, mostly inside `mooncake_connector.py` plus the
recompute path, a "draft verdict" cut mid-sentence in a message, and the decisive comparison
(doc proxy line vs DP layout) never made.

**#11923** vllm-ascend 0.18.0rc1 + the reporter's own #7874/#10098 patches (snapshot
pins v0.18.0rc1 `99e1ea0`), 910B/A2, GLM-5.1 (DSA/MLA) 2P1D on 4+4 servers with
`AscendStoreConnector` + `memcache_hybrid`/`memfabric_hybrid` 1.1.1: "带 memcache 的
版本初始 TTFT 比不带时慢约 6-7 倍，命中缓存的后续请求快很多", the numbers in an
attachment that was not readable (no attachments in that workspace) and no log. The
run pinned at the ref that the miss path returns before any store work
(`pool_scheduler.py:56-111`), that `load_async` defaults `False` so the *hit* path is
the blocking one, that the save runs off-step but syncs an NPU event before
`m_store.put`, the A2-only `all_gather` + `register_buffer` branch, the blocking
per-request lookup RPC, and the doc trap that the sample `mmc-local.conf` sets
`device_sdma` (`kv_pool.md:530`) while the table recommends `device_rdma` for A2
(`:566`, default `host_rdma`). Delivery: a complete ranked answer did ship, then the
run reopened with "Final answer — one correction …" and was cut mid-sentence, so the
A2 correction never reached the reporter. Route now:
`../kv-pool-store/pool-attached-cold-request-cost.md` (same-prompt-twice is check 1).

**#10724** vllm-ascend `v0.21.0rc1` (`80610e44`) + vllm v0.21.0, DeepSeek-V4-Flash-w8a8
MTP in PD-mix on 2×2 Atlas 800 A2 (910B4, 32 GB) with Mooncake `AscendStoreConnector`
`kv_both`, `--enable-prefix-caching`, `--no-disable-hybrid-kv-cache-manager`,
`--async-scheduling`, MTP 3, `--block-size 16`; healthy ~3 h, then 1 of ~10 instances dies
with `AssertionError: prev_free_block of fake_free_list_tail should always exist` at
upstream `vllm/v1/core/kv_cache_utils.py:337 append_n`
(`allocate_slots → remove_skipped_blocks → block_pool.free_blocks`), the APIServer side
repeating `EngineDeadError`. Both node scripts were already folded into the opening body,
and the plog the reporter later supplied is the teardown of the one-sided service
(`one_sided_service_adapt.cc … memHandle is invalid`,
`[HcclOneSidedService][DeInit] There are memories still bound to this comm`,
`rtFreeHost … 507899`) — a consequence, not the cause; the thread closed non-reproducible
after the reporter changed the deployment. Delivery: 19 assistant turns, 50.6 K characters
of reasoning deriving the free-list invariant from the paste, **100 characters** of
message text, cut inside a block that began "Let me step back and decide: budget." Pages:
`../upstream-downstream-drift/kv-block-free-list-assert.md` (new — it carries the stop rule
and the four per-trigger ablations).

**#15394** — image `v0.23.0-a3`, Atlas 800I A3 双机 PD over RoCE, Qwen3.5-122B-A10B-w8a8-mtp, P and D both
TP16/DP1 (`--enable-expert-parallel`, `--enforce-eager` on P, `FULL_DECODE_ONLY` + `recompute_scheduler_enable`
on D, `--no-disable-hybrid-kv-cache-manager` both), `MooncakeLayerwiseConnector` with
`use_layerwise: false`: AISbench 20 k-in/1 k-out, 4 concurrency fine, at 5 concurrency in round 2 one P worker
(`Worker_TP11_EP11`) dies inside `kv_connector.start_load_kv` — `mooncake_layerwise_connector.py:769` →
`:1627` `single_group_transfer_mappings = self._get_kv_split_metada` — and the paste stops there, with no
exception class and D serving normally. Symmetric TP16 and 20 k prompts under `--max-num-batched-tokens 4096`
on P (128 on D) mean chunked prefill, and Qwen3.5 with the hybrid KV manager makes `need_truncate` true, so
the producer truncates by one token while D trims its remote block ids. Delivery: the recorded run pinned the
version, matched `:1627` to the *non-Mamba* `_get_kv_split_metadata` call (the MambaSpec sibling is `:1623-1625`),
enumerated the throw sites and found the `build_connector_meta` DEBUG line that prints exactly
`local_block_ids`/`remote_block_ids`/`chunk_finish` — then spent the rest of the budget deriving
`chunk_finish`/`to_trans_idx` arithmetic and shipped one **509-character** status note, no ranked answer. Page:
`../pd-disaggregation/layerwise-split-metadata-crash.md` (new — a cut traceback is answered by the throw-site
inventory plus the DEBUG ask, not by a mechanism).

**#13363** — vllm-ascend v0.23.0rc1 (`f4a08bd`) + vllm v0.23.0 (`0fc695f`), the
disaggregated example proxy with `MooncakeConnectorV1` kv_producer/kv_consumer: a
**non-streaming** request carrying `prompt_logprobs` kills the decoder with
`OverflowError: out of range integral type conversion attempted`. The report itself named the
upstream mechanism and asked for the vllm-ascend-side mitigation (harvest on the prefiller).
The recorded run confirmed it line by line (`gpu_model_runner.py:5441-5459`, `outputs.py:94-113`
`torch.empty(…, dtype=torch.int32)`, `engine/logprobs.py:147` → `convert_ids_list_to_tokens`)
and answered *real, upstream's, and the mitigation is not in the snapshot* — `rg
'prompt_logprobs|echo'` on `load_balance_proxy_server_example.py` returns nothing — then gave
the three patch sites, the `echo`→`prompt_logprobs` alias on **both** endpoints
(`chat_completion/protocol.py:577-579`, `completion/protocol.py:275-277`), `prompt_text` being
chat-only at that pin, and the one-curl check that decides harvest-vs-no-mitigation. Two
corrections to keep: the detokenize helper is `vllm/tokenizers/detokenizer_utils.py:83,98`
(the run cited the nonexistent `vllm/v1/tokenizers/…:100`), and a strip must cover `echo`.
Delivery: it completed with the verdict text present from turn 3 and the ranked answer at the
end, but spent ~520 K cumulative prompt tokens re-reading trees whose anchors the page now
carries. Page: `../pd-disaggregation/prompt-side-outputs-on-decode-node.md` (new).

**#4866** — vllm-ascend v0.11.0rc2 (`a2e4c3f`), vllm v0.11.0, CANN 8.3.RC2,
8×910B2 per node, image `v0.11.0rc2`: 2P2D `MooncakeConnectorV1` (DeepSeek-R1-W8A8,
TP8, `kv_connector_extra_config` `prefill/decode` `dp=2 tp=8`, `kv_port 20001`,
`engine_id 0/1`), both sides serving their own `/v1/completions`, and the first proxied
request failing on D with `ascend_direct_transport.cpp:820 Failed to connect to target:
<P ip>:<port>, status: 103900` + `:581 Failed to connect to segment` + `mooncake_connector.py:410
Mooncake transfer failed for request … ret: -1` on all 8 TP ranks within 2 ms. The recorded
answer classified it correctly (PD transport, no pool, first pull, nothing timed out) and
priced the DP=2 layout out on the reporter's own 1P1D follow-up, but its **cheapest action**
was `ss`/`nc` on the transport ports plus a `hostNetwork: true` ablation — a cross-node P/D
cannot collapse to one HCCL server, and the config hint another commenter supplied
(`use_ascend_direct` in `kv_connector_extra_config`) is a no-op at the pin, where
`mooncake_connector.py:905-911` already defaults it `True` and only uses it to pick the
engine-name form. The reporter's later `grep -rn ERROR /root/ascend/log` from the decode node
supplies the deciding artifact: `topoinfo_ranktableConcise.cc:446 'device_ip' is not set
correctly, must be set when multi Server or HCCL_INTRA_ROCE_ENABLE enabled` →
`HcclCommInitClusterInfo failed, rankNum[0], server[], device[-1]` →
`comm Name exist.` on retry, i.e. the transport's own HCCL comm never came up and no
segment port was ever bound — and "same error" after the config move is what the source
predicts. Cause left open; the deliverable is the D-side plog plus the `hccn_tool` device-plane
block. Delivery: the ranked answer shipped at the last turn (51 messages, ~100 K prompt tokens)
and was **output-truncated** inside action rung 5. Page:
`../pd-disaggregation/mooncake-adxl-connect-103900.md` (new).

**#9003** vllm-ascend v0.19.1rc1 (`da421af`) + vllm v0.19.1 (`b1388b1`),
Qwen3.6-27B-w8a8 TP2 with `--speculative-config '{"method": "qwen3_5_mtp",
"num_speculative_tokens": 5, "enforce_eager": true}'` **plus**
`--enable-prefix-caching`, `--async-scheduling`, `FULL_DECODE_ONLY`, a hand-picked
`cudagraph_capture_sizes [6…60]` and `--max-model-len 262144`: the bug body is
`<!-- Failed to upload "image.png" -->` and `/workspace/attachments` is empty, so
there is **no error text at all**. The run verified the right facts — the pair is
*documented* (`docs/source/tutorials/models/Qwen3.5-27B.md:137`, with the
block_size-2048 caveat) while every validated row runs it off (the nightly
`Qwen3.5-27B-w8a8-A3.yaml`, `tests/e2e/multicard/4-cards/test_qwen3_5.py`, and
`enable_prefix_caching=False` hard-coded in
`tests/e2e/singlecard/spec_decode/test_v1_spec_decode.py:558,680`); align mode is
auto-selected upstream (`vllm/model_executor/models/config.py:436-472`), so
`--mamba-cache-mode` is redundant rather than missing; `enable_weight_nz_layout` is
read by no code at that pin — and shipped them in one final message, after **20
tool turns that emitted 19 characters of message text**, ending at 13 393 of 16 384
completion tokens. Durable correction: with no error text the **first** message
already carries the deliverable ("the pair is documented, never tested;
`--mamba-cache-mode` is not the gap; drop `--enable-prefix-caching` and send the
traceback") — `../spec-decode-mtp/hybrid-mtp-prefix-cache-unvalidated.md`.

**#10290** vllm-ascend reported 0.18.0 (snapshot v0.20.2rc1 `367b8e62`) + vllm
v0.20.2 (`bc150f50`), GLM-5.1 大EP: `/v1/chat/completions` answers through
`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` while
`/v1/messages` and the management/metrics endpoints cannot be called ("需要对所有
vllm 的接口都做兼容"). Track: an **example-script endpoint-coverage gap**, inherited
from upstream's `toy_proxy_server.py`, not an engine defect. Verified at the pin:
five routes only (`:892-920`), the header self-describes two (`:14`), and the
Anthropic request model carries `kv_transfer_params`
(`vllm/entrypoints/anthropic/protocol.py:116`, forwarded at `serving.py:338/450`),
so `/v1/messages` is a tier-1 route — but with two silent traps the run found by
reading the retry path: the proxy keys its recompute retry on
`choices[0].stop_reason == "recomputed"` (`:793`, emitted by
`vllm_ascend/core/recompute_scheduler.py:837`) which the Anthropic
`stop_reason_map` (`serving.py:93-97`, `:617`) turns into `None`, and the retry
rewrite (`:731-737`, `:795-800`) assumes string `content` where Anthropic allows a
block list. The run also located the in-repo passthrough/fan-out template
(`examples/disaggregated_encoder/disagg_epd_proxy.py:561/570/676/686`) and shipped
a complete ranked answer with the nginx workaround. Page:
`../pd-disaggregation/proxy-example-endpoint-coverage.md`.
**#8664** vllm-ascend v0.18.0rc1 (`99e1ea0f`) + vllm v0.18.0 (`bcf2be96`), QWQ-32B W8A8 on
2×910B (`--tensor-parallel-size 2`, `--block-size 128`, `--max-model-len 131072`,
`--gpu-memory-utilization 0.9`, `FULL_DECODE_ONLY`, `AscendStoreConnector` `kv_role: kv_both`
over Mooncake, 36 requests of 20-60 K tokens with repeated prefixes, 64 K outputs): after 24
good requests the APIServer raises `ValueError: Counters can only be incremented by
non-negative amounts` at `metrics/loggers.py:1120`; the body already guesses
`num_cached_tokens + num_external_computed_tokens > prompt_len`, and the reporter's own
control is "does not occur when KV Pool is disabled". The recorded run got the arithmetic
right — the identity at `stats.py:280-286`, the `num_cached_tokens` latch
(`request.py:150`, `scheduler.py:805-807`) that `_preempt_request` (`:941`) never re-arms,
the connector's `load_async=False`/`discard_partial_chunks=True` defaults pruning the async
branches, and a numeric simulation of the preempt-resume case — then was cut at the
16 384-token completion cap after ~1.38 M prompt tokens with **no** final message. Nothing
it derived had to be re-derived: the identity, the latch, the two checks
(`vllm:num_preemptions`, `--disable-log-stats`) and the ownership split are now the page.
Page: `../kv-pool-store/prompt-token-counter-negative.md` (new).

**#7508** vllm-ascend v0.17.0rc1 (`e20f0b1a`), image
`quay.io/ascend/vllm-ascend:v0.17.0rc1-a3`, A3, Kimi-K2.5 W4A8, `--quantization ascend`
TP8 × DP2 + `--enable-expert-parallel`, `VLLM_ASCEND_ENABLE_MLAPO=1`,
`VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=4`, `additional_config` with
`layer_sharding: ["o_proj"]` and `recompute_scheduler_enable: true`, P node
(`kv_producer`) with `MooncakeLayerwiseConnector`. A worker dies while loading weights:
`ops/layer_shard_linear.py:73 post_process_after_loading` →
`AssertionError: layer_idx must be consecutive`. Both switches register the **same** `o_proj`
linear into one shard series (`"[0,0,1,1,…]"` after the sort): the Flashcomm2 row-parallel
op's `update_attrs` (`ops/linear_op.py:381-386` → `Flashcomm2OShardManager.register_layer`)
plus the attention impl's `layer_sharding` kwargs (`attention/mla_v1.py:719-727`, DSA
`sfa_v1.py:441-450`), both under `is_hidden_layer`. The pin's own
`docs/source/user_guide/feature_guide/layer_sharding.md:53-59` ships exactly this pair, and
`utils.py:1018-1028` accepts TP8 × 4, so neither the value nor the transport is at fault —
`../feature-enable-recipes/flashcomm2-o-shard-layer-sharding.md` carries the two ablations
and the `PARALLEL_SIZE` / `recompute_scheduler_enable` answers the reporter also asked for.
The run reached the double-registration reading only in its last two turns, kept
5-8 K characters of reasoning per turn, and ended `output_truncated` at 16 384 output
tokens: 15 assistant turns, 26 tool results, 48 K characters of reasoning, **zero**
characters of message text.

**#9579** vllm-ascend v0.19.1rc1 (`da421af`) + vllm v0.19.1 (`b1388b1`), Qwen3.5-27B:
"[0.19.1rc1]Qwen3.5-27B use the prefix-cache function fail" — "if the server config
includes prefix-cache, it will abnormal shut down", with **no traceback and no serve
command**, and the linked tutorial is the **v0.18.0** build's merged
`Qwen3.5-27B-Qwen3.6-27B` page, which exists in neither exported tree (both carry
`docs/source/tutorials/models/Qwen3.5-27B.md`). Third report of the documented-but-untested
`qwen3_5_mtp` + `--enable-prefix-caching` pair (#9003, #10326) and the existing page gave
it the right verdict — but only after 21 tool turns with no message text: the
upstream-frame history search returned **`0 hits`** (that corpus ends at #8849, so the
cited ids cannot appear), which the run answered with two `github /search/issues` pages
(both `total_count` 1, the issue it was reading) plus greps that re-derived the tutorial
`:118`/`:137` and nightly `--no-enable-prefix-caching` anchors the page already carried;
the ranked verdict shipped on the 22nd of 24 assistant turns. Durable correction: a 0-hit
corpus is expected and is not a licence to escalate — the precedent ids are the page's own
record, and a lexical search needs a lexical signature. Page:
`../spec-decode-mtp/hybrid-mtp-prefix-cache-unvalidated.md`.
