# PD, KV-pool and feature cases

PD disaggregation that connects and then corrupts or cannot move KV, KV-pool store
segments and the MemCache init lane, EP+DP and DP>1 MoE recipes, logging and
profiling flags, flag-retirement requests and downstream patch/registry drift.
Routes: `../pd-disaggregation/index.md`, `../kv-pool-store/index.md`,
`../feature-enable-recipes/index.md`, `../observability-and-logs/index.md`,
`../code-change-requests/index.md`, `../upstream-downstream-drift/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file.

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
(`references/pd-disaggregation/hybrid-state-transfer.md` §3).

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
