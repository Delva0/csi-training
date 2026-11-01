# Model tutorials: the documented serve line per tag

The triage workflow compares the reporter's command with the reported tag's own model
tutorial. These are the locators and the flags that matter, so a 30k-character tutorial
need not be re-read.

## Where they live

`docs/source/tutorials/models/<Model>.md` at the reported tag
(`git show <tag>:<path>`). Each deployment command sits in an `A2`/`A3`/`A5` `tab-set`
block, so grep for `vllm serve` and read ~40 lines around the hit instead of the file.
Model weight names, node counts and NPU memory requirements are above the command.

Before `v0.17.0rc1` the tutorials are flat: `docs/source/tutorials/<Model>.md` (the W4A8 recipe is
`docs/source/tutorials/Qwen3-8B-W4A8.md` at `v0.13.0rc1`/`v0.14.0rc1`, and `models/Qwen3-8B-W4A8.md`
from `v0.17.0rc1`). A grep naming only `models/...` returns an empty result there and reads as
"the tag has no such doc" — enumerate with `git ls-tree -r --name-only <tag> | grep -i <model>`
first.

`tests/e2e/` beside them carries maintained reference configs per model and hardware:
`tests/e2e/models/configs/<Model>.yaml` (`Qwen3-VL-8B-Instruct.yaml` at `v0.18.0`),
`tests/e2e/nightly/single_node/models/configs/`, `tests/e2e/nightly/multi_node/config/` and
`tests/e2e/nightly/multi_node/internal_dp/config/` (present at `v0.20.2rc1`). Enumerate with
`git ls-tree -r --name-only <tag> | grep -i '<model>'`: when the reporter's topology matches no
tutorial tab, the e2e config for that exact NPU count often does, and it shows the env vars and
serve flags the maintainers actually run in CI.

## DeepSeek-V4-Flash at v0.23.0rc1 (`docs/source/tutorials/models/DeepSeek-V4-Flash.md`)

- Weights: the **quantized** `vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp`. The deployment text
  says "The quantized model ... can be deployed on 1 Atlas 800 A3 (128G × 8) or 1 Atlas 800
  A2 (64G × 8)". The base `deepseek-ai/DeepSeek-V4-Flash` repo is not the documented
  artifact.
- A2 recipe: TP=8, `--enable-expert-parallel`, `--quantization ascend`,
  `--max-model-len 133120`, `--max-num-batched-tokens 8192`, `--no-enable-prefix-caching`,
  `--tokenizer-mode/--tool-call-parser/--reasoning-parser deepseek_v4`,
  `--speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}'`,
  `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`, `--async-scheduling`,
  `--additional-config` with `ascend_compilation_config`/`enable_cpu_binding`/
  `enable_dsa_cp`/`multistream_overlap_shared_expert`; env `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`,
  `HCCL_BUFFSIZE=1024`, `HCCL_OP_EXPANSION_MODE="AIV"`, jemalloc preload.
- A3 recipe: DP=4 / TP=4, `--max-model-len 1048576`, `--max-num-batched-tokens 10240`,
  `--api-server-count 1`, `--max-num-seqs 64`, `enable_cpu_binding` and no `enable_dsa_cp`.
- **Both tabs put `"enforce_eager": true` in `--speculative-config`.** A report that sets
  `method: mtp`/`deepseek_mtp` without it is off the documented path and enters the DSA
  graph-capture bug; why that flag is load-bearing, and the release that fixes it, are in
  [deepseek-v4-flash.md](deepseek-v4-flash.md).
- A `deepseek_v4` serve line claiming 0.18.x/0.19.x is not the tagged tree: DeepSeek-V4 is absent
  from those tags and present from `v0.20.2rc1` (main line) and, as a backport, at
  `v0.13.0rc3` with the tutorial at the flat path `docs/source/tutorials/DeepSeek-V4.md`. See
  [deepseek-v4-flash.md](deepseek-v4-flash.md) before calling the version wrong.
- Serving the base checkpoint with `--dtype bfloat16` and no `--quantization ascend` is off
  this path; see
  [quantization-startup-keyerrors.md](quantization-startup-keyerrors.md) for the
  `weight_scale` KeyError that pairing produces.

## DeepSeek-V4.1-Flash at `v0.30.0rc2` and main (`docs/source/tutorials/models/DeepSeek-V4.1-Flash.md`, 1929 lines)

- The model is a separate package, not a V4 config (`models/deepseek_v41/`, `dsa_v41`/`dsa_v41_cp`
  attention, Engram/DSpark, `deepseek_v41` parsers, V1 runner only). Which tags carry it, the
  parser surface and the per-tag check are in
  [deepseek-v4-flash.md](deepseek-v4-flash.md); the doc facts below are the recipe side.
- The tutorial exists earlier than the package: added in `7a5b24735` (2026-09-11) and present in
  `v0.29.0.rc1` (2026-09-28) and `v0.28.0.rc1` (2026-09-30), while `models/deepseek_v41/` is absent
  from the latter — a tag carrying the doc need not carry the model. Its support sentence also
  moved — `v0.30.0rc2`'s copy requires the `main` branch plus an upstream vLLM revision dated
  2026-09-11 or later (`:23-26`), main's copy says "supported in vLLM Ascend v0.30.0rc1 and later"
  with `v0.30.0rc2` as baseline (`:29`). Quote the revision actually read.
- Sizing fact that decides "why OOM at 1M": 552B MoE, CSA2, FP4 main KV cache, SWA bounded replay,
  Engram, DSpark, and a stated global KV footprint of **890 bytes/token** (`:18-22`). 1M tokens of
  KV is ≈0.9 GB, so an A2 OOM at 1M is not explained by KV size; the raise site stays unknown until
  a log is supplied. Do not present "1M KV does not fit in 64 GB" as the mechanism.
- Tab matrix at main (`1bd23a0a9`, 2026-10-30): §5.1 single-node colocated is **A3-only**,
  `--max-model-len 150000`, `--gpu-memory-utilization 0.9` (`:192-219`); §5.2 A3 1P1D: 1048576,
  `--max-num-batched-tokens` 8192/512, `--max-num-seqs` 16/72, gmu 0.9/0.95 (`:407-508`); §5.2 A2
  four-node 1P1D (Prefill DP4/TP4 on two nodes, Decode DP8/TP2 on two, Mooncake): **1048576**,
  mnbt 4096/400, mns 16/32, gmu 0.9/0.95 (`:786`/`:896`); §5.5 A2 two-node colocated (DP2/TP8/EP):
  1048576, mnbt 8192, mns 32, gmu 0.9, `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'`
  needing `--ipc=host` or a ≥320 GB `/dev/shm`, `HCCL_BUFFSIZE` 1024/1800,
  `VLLM_ENGINE_READY_TIMEOUT_S=360000`, `VLLM_USE_V2_MODEL_RUNNER=0` (`:1645-1708`).
- **The A2-at-1M contradiction is real and predates the report.** `7a5b24735` (2026-09-11) and
  `f804696c0` (2026-09-28) already give the four-A2 recipe 1048576 at gmu 0.90, and `f804696c0`
  introduces "The existing A2 configuration is retained unchanged in this update"; main's §11
  (`:1856`) says "retained unchanged and is not revalidated by this update. Its revised
  configuration will be documented separately". The repo's own A2 ceilings for the V4 family sit far
  lower — `DeepSeek-V4-Flash.md` on main uses 133120 (single node), 135000 (both PD roles) and
  800000 (A2 with DSpark), while this doc's A3 single node uses 150000 — so 1048576 on A2 is the
  outlier among the same repo's A2 recipes. The release notes
  carry the wording another model got: "GLM-5.2 1M-context deployments were validated only on Atlas
  800 A3; the A2 series was not validated for 1M context" (`docs/source/user_guide/release_notes.md`,
  v0.23.0 Known Issues). Conversely, `v0.30.0rc2`'s `support_matrix/supported_models.md:145` lists
  `1048576` for this model on A2/A3 — so no in-repo doc states the A2 1M limit the reporter
  attributes to a v0.18.0 FAQ; say the statement is not reproducible in the clone.
- Verdict shape for this report class: quote the A2 tab value, the doc's own non-revalidation
  sentence, the 133120/135000/150000 siblings and the release-notes precedent; give the conservative
  context/gmu as the workaround; leave the irrecoverable-restart mechanism a candidate (the 100-hour
  `VLLM_ENGINE_READY_TIMEOUT_S` is one) and name the log that would settle it.

## Qwen3.5-397B-A17B at v0.18.0 (`docs/source/tutorials/models/Qwen3.5-397B-A17B.md`, 606 lines)

Report shape: "give me an official example / is it DP=? TP=?" for a very large MoE, usually on
exactly the node count the tutorial's BF16 bullet asks for.

- Footprint bullets (`:21-22`): BF16 `Qwen/Qwen3.5-397B-A17B` needs **2 Atlas 800 A3 (64G x 16) or
  4 Atlas 800 A2 (64G x 8)** nodes; the quantized `Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp` needs 1 A3
  or 2 A2. A "4 x 800I A2, 8 cards each" host is the BF16 minimum, not an over-allocation.
- The doc says the model is "first supported in `vllm-ascend:v0.17.0rc1`" (`:9`), while the file
  itself first exists at `v0.18.0rc1` (`git ls-tree -r --name-only <tag>` per tag) — so a reporter on
  the `0.18.0` image is on the first release whose own docs carry the recipe.
- **Every deployment section ships the w8a8 artifact** with `--quantization ascend` (single-node
  `:111`+; MP node0 `:177`+, node1 `:226`+; PD `:309`+). The flag notes say "To disable
  quantization, remove this option", which is the only documented way to reach a BF16 serve line;
  there is no separate BF16 example to quote, so label the derived line as derived.
- MP multi-node (recommended, `:151-260`) is DP-across-nodes / TP-within-node for 2 A2 nodes: node0
  `--data-parallel-size 2 --api-server-count 2 --data-parallel-size-local 1 --data-parallel-address
  <node0_ip> --data-parallel-rpc-port 13389 --tensor-parallel-size 8 --enable-expert-parallel
  --max-num-seqs 16 --max-model-len 32768 --max-num-batched-tokens 4096 --no-enable-prefix-caching
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager":
  true}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config
  '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'`; node1 adds `--headless`
  and `--data-parallel-start-rank 1` and drops `--api-server-count`. Scale with DP = node count (four
  nodes: `--data-parallel-size 4`, start-ranks 0-3); TP stays the cards per node (8).
- Ray: the section is a single link (`:262-264`) to `docs/source/tutorials/features/ray.md`, a
  Qwen3-235B walkthrough whose stated rule (`:149`) is "`tensor-parallel-size` = NPUs per node,
  `pipeline-parallel-size` = number of nodes" (TP=8/PP=2 for two 8-card nodes, `:154-157`), with
  `--distributed-executor-backend ray`. Four A2 nodes therefore map to TP=8/PP=4; no 397B Ray recipe
  is validated. Read a reporter's Ray line for `--pipeline-paralle-size` (sic) — argparse rejects
  the typo before any model code runs, and TP=16/PP=2 neither matches the formula nor fits a node.
- Known issues (`:604-606`): single-node multi-DP deployment with `fused_mc2` can emit garbled or
  empty output after recompute; the doc says to disable that fusion when DP > 1 (PD decode nodes may
  instead enable the recompute scheduler). Cross-read
  [sampling-and-output-quality.md](sampling-and-output-quality.md) before calling such output a
  sampler defect.

## Kimi-K2.5 at v0.20.2rc1 (`docs/source/tutorials/models/Kimi-K2.5.md`)

- Weights `Kimi-K2.5-w4a8` (`Eco-Tech/Kimi-K2.5-W4A8`) plus Eagle3 draft
  `lightseekorg/kimi-k2.5-eagle3`.
- Both recipes set `VLLM_ASCEND_ENABLE_MLAPO=1`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`,
  `VLLM_ASCEND_BALANCE_SCHEDULING=1`, `--quantization ascend`, `--no-enable-prefix-caching`,
  `--mm-encoder-tp-mode data`, `--allowed-local-media-path /`, `--trust-remote-code`, and
  `--speculative-config '{"method":"eagle3","model":"lightseekorg/kimi-k2.5-eagle3","num_speculative_tokens":3}'`.
- Single node A3 (64G × 16): DP=4, TP=4, `--max-model-len 32768`,
  `--max-num-batched-tokens 16384`, `--max-num-seqs 64`, cudagraph capture sizes up to 256.
- Two nodes A2 (64G × 8 each): DP=4 total with `--data-parallel-size-local 2`, TP=4,
  `--data-parallel-start-rank 2` on node 1, `--enable-expert-parallel`,
  `--max-num-seqs 16`, `--max-model-len 32768`, `--max-num-batched-tokens 16384`, plus
  per-node `HCCL_IF_IP`/`GLOO_SOCKET_IFNAME`/`TP_SOCKET_IFNAME`/`HCCL_SOCKET_IFNAME` and
  `HCCL_INTRA_PCIE_ENABLE=1`/`HCCL_INTRA_ROCE_ENABLE=0`.
- A two-node report that omits FLASHCOMM1/MLAPO/BALANCE_SCHEDULING, keeps
  `--enable-prefix-caching`, runs TP=8 with DP=2/local=1, raises `--max-model-len` to
  262144 while lowering `--max-num-batched-tokens` to 4096 and drops the Eagle3 config is
  several deltas away from the documented recipe; name the deltas before theorising about
  numerics. In #10503 (garbled output, 910B two-node, v0.20.2rc1) the only technical
  suggestion in the thread was a reporter comment: add
  `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, "may become very slow" — which matches the
  tutorial. The maintainer-requested repro is the same weights with FLASHCOMM1 on and off,
  and the issue stayed open awaiting feedback. That makes FLASHCOMM1 a documented-config
  delta and a candidate, not a confirmed cause.
- The topology the tutorial does not show is the nightly config
  `tests/e2e/nightly/multi_node/internal_dp/config/Kimi-K2_5-W4A8-A2-dual-nodes.yaml`: TP=8,
  DP=2 with `--data-parallel-size-local 1` and `--data-parallel-start-rank 0/1`, EP,
  `--max-num-seqs 64`, `--max-model-len 51200`, `--max-num-batched-tokens 8192`, the Eagle3
  config, `--mm-encoder-tp-mode data --mm-processor-cache-gb 0`, and env
  `VLLM_ASCEND_ENABLE_MLAPO=1`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, `HCCL_BUFFSIZE=512`,
  `HCCL_OP_EXPANSION_MODE=AIV`, `HCCL_INTRA_PCIE_ENABLE=1`, `HCCL_INTRA_ROCE_ENABLE=0`,
  `TASK_QUEUE_ENABLE=1`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` — without
  `VLLM_ASCEND_BALANCE_SCHEDULING`, which the tutorial sets. Its shape (TP=8/DP=2/local=1) is the
  one the #10503 reporter ran, while the tutorial's two-node tab is DP=4/local=2/TP=4, so compare
  against the config whose shape matches first, then against the tutorial.
- **The release-note Kimi-K2.5 warning is not this tag's.** `grep -i kimi` on
  `v0.20.2rc1:docs/source/user_guide/release_notes.md` hits `:521`, "Kimi-K2.5 Model Support ...
  **Please note** that vLLM 0.15.0 has a known issue with Kimi-K2.5. To fix this, please apply the
  changes ... pull requests [#33320] and [#34501]" — but `:512` is the header `## v0.15.0rc1`, whose
  `Dockerfile:53` is `ARG VLLM_TAG=v0.15.0`. `v0.20.2rc1`'s own file has **no** `## v0.20.2rc1`
  section (its first header is `## v0.18.0`); that section first appears in `v0.21.0rc1` (`:3-66`),
  and its Known Issues list only GLM5/GLM5.1 W4A8 and the DeepSeek-V4 KV-pool entry. So a v0.20.2rc1
  report has no tag-level Kimi-K2.5 known issue to quote: the upstream-PR pair belongs to the 0.15
  line (its own tag is `ARG VLLM_TAG=v0.15.0`), while the reported tag pairs `ARG VLLM_TAG=v0.20.2`.
  Check the enclosing header and the paired upstream version before attributing either.
- The only Kimi-K2.5-specific plugin patch at `v0.20.2rc1` is
  `vllm_ascend/patch/worker/patch_kimi_k25.py`, and it patches the **vision** encoder (MoonViT
  `Learnable2DInterpPosEmbDivided_fixed` / `get_rope_shape`). A text-only garbled-output report has
  no Kimi-specific plugin patch on its path, so the remaining candidates are the documented
  configuration deltas above plus the DP/EP and sampling surfaces.

## Kimi-K2.6 at v0.23.0 (`docs/source/tutorials/models/Kimi-K2.6.md`, 874 lines)

First tag carrying this tutorial: `v0.22.1rc1` (absent at `v0.20.2rc1`/`v0.21.0rc1`) — a
Kimi-K2.6 report on an earlier tag has no tutorial to compare against. Layout: §5.1 single node
(`:136`), §5.2 multi-node PD separation (`:259`, which delegates deployment to
`tutorials/features/pd_disaggregation_mooncake_multi_node.md`), §9.1 recommended configurations
(`:825`), §10 FAQ (`:868`).

- Weights `Kimi-K2.6-w4a8` plus Eagle3 draft `lightseekorg/kimi-k2.6-eagle3` (`:22`).
- PD tabs at `:336`, `:415`, `:494`, `:573`. All set `--no-enable-prefix-caching`, and the
  Eagle3 line is `--speculative-config '{"method":"eagle3","model":"lightseekorg/kimi-k2.6-eagle3","num_speculative_tokens":1}'`
  in the first two tabs and `...:3` in the later two; the decode tabs add
  `--additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": false}'`
  (`:517`, `:596`) and `--max-num-seqs 8`. `HCCL_BUFFSIZE` is 800 in the single-node block (`:161`)
  and in the first three PD tabs (`:336`, `:415`, `:494`), 1100 in the last one (`:573`).
- The tutorial documents **no** `--reasoning-parser`/`--tool-call-parser`, so a `kimi_k2` pair in
  a report is the reporter's own flag choice; the reasoning-marker triage for that is in
  [serving-toolcall-reasoning-streaming.md](serving-toolcall-reasoning-streaming.md).
- The FAQ's tool-call answer is a version gate: `tool_calls` needs transformers ≤ 4.57.6 unless
  vllm-ascend is v0.21 or later.
- Recorded report #13002 (v0.23.0, P TP=8/DP=2 + D TP=8/DP=4, `MooncakeConnectorV1`, long
  context ≥23k with ≥16 concurrency, 6-10 % degradation in three shapes). Against this tutorial
  the deltas are: P node `HCCL_BUFFSIZE=256`, `--enable-prefix-caching` on both nodes,
  `num_speculative_tokens: 3` on the prefill node, and a `--reasoning-parser kimi_k2` pair. Name
  them as deltas, not causes. The reporter's own theory ("256 is too small → HCCL buffer
  overflow") has no documentary support: the tag's `pd_disaggregation_mooncake_multi_node.md`
  itself sets `HCCL_BUFFSIZE=256` for several of its tabs (600 for others), so 256 is a value
  the tag ships, while the model tutorial uses 800/1100.
- Thread boundary: closed `not_planned`; maintainers answered with an automated reply asking for
  (1) the matched upstream vLLM version/commit and (2) a run with Eagle3 disabled, then
  `/wait-feedback` and a stale close. Those two experiments are the next verification step, and
  no maintainer root cause exists.

## Kimi-K2.7-Code: no doc and no config before `v0.23.0`; the shipped shape is W4A8 + PD with different TP per role

`git grep -l 'Kimi-K2\.7' <ref>` returns nothing at `v0.21.0rc1`/`v0.22.1rc1` and 8 files at
`v0.23.0`; the `tutorials/multi_node_kimi.html` page a reporter cites has no in-tree file at any tag
checked (`git ls-tree -r --name-only <tag> | grep -i multi_node_kimi` is empty at
`v0.21.0rc1`/`v0.22.1rc1`/`v0.23.0`/`main`). So on 0.21/0.22 there is no tutorial, no e2e config and
no support-matrix entry to compare a Kimi-K2.7 deployment against — say that instead of inventing a
recipe.

At `v0.23.0` the artifacts are weekly multi-node configs
`tests/e2e/weekly/multi_node/external_dp/config/Kimi-K2.7-W4A8-*.yaml` (model
`Eco-Tech/Kimi-K2.7-Code-w4a8`), plus the note "Fixed Mooncake KV-transfer grouping for Kimi-K2.7
Code with Kimi-K2.5-DFlash when P/D nodes use unequal TP sizes and target/draft models have
different global KV-head counts" (#11887, `release_notes.md:192`). The 64k PD config is the
documented shape: two nodes, prefiller DP=2/TP=8 and decoder DP=4/TP=4 (roles differ in TP),
`VLLM_ASCEND_ENABLE_MLAPO=1` + `VLLM_ASCEND_ENABLE_FUSED_MC2=1` on both, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`
with `HCCL_BUFFSIZE=512` on the prefill node only, `HCCL_BUFFSIZE=800` on decode. A single TP=16
rank set spanning two nodes with `--enable-expert-parallel` is not a shape the repo ships, so treat
that report as an unsupported topology and point at the PD-divergent-DP config. The single-node A3
variant `Kimi-K2.7-w4a8-A3.yaml` only appears with the later `tests/e2e/cases/models/configs/Kimi/`
reorganisation (added by `69454b03b`, first tag `v0.30.0rc1`).

Recorded instance: #11019 (`v0.21.0rc1`, A2 two nodes × 8×910B-64GB, Kimi-K2.7-Code 1T). Closed
`not_planned`; the maintainer reply is "Kimi-K2.7-Code is a raw floating-point weight and is
incompatible with the `--quantization ascend` parameter. It can run normally on A2 dual-node after
W4A8 quantization." The reporter's own "solved" comment bundles three changes at once — adding
`--additional-config '{"enable_flashcomm1": true, "enable_mlapo": true}'`, switching to
`Kimi-K2.7-Code-w4a8` with `--quantization ascend`, and moving `v0.21.0rc1` → `v0.22.1rc1` — on a
non-PD TP=16 Ray run, so it attributes nothing; and with a BF16 checkpoint on a non-decode instance
the MLAPO half of that bundle cannot have taken effect at all
([env-vars-and-additional-config.md](env-vars-and-additional-config.md)).

## MiniMax-M2.5 at v0.20.2rc1 (`docs/source/tutorials/models/MiniMax-M2.5.md`)

- Weights `MiniMax-M2.5-w8a8-QuaRot`. TP/DP counts differ per tab: one tab (`:148-160`) is
  TP=4/DP=4 with `--max-model-len 40690`, and the other two serve blocks start at `:228` and
  `:282`.
- All three serve blocks set `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` (`:145`, `:219`, `:273`)
  **and** `--enable-expert-parallel` (`:156`, `:241`, `:296`), plus `--quantization ascend`,
  `--additional-config '{"enable_cpu_binding":true}'`, and `--compilation-config
  '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`; the first block also sets
  `VLLM_ASCEND_ENABLE_FUSED_MC2=1` and `VLLM_ASCEND_BALANCE_SCHEDULING=1`.
- The FAQ (`:447-459`) answers three recurring report shapes: garbled output in EP mode →
  "keep `--enable-expert-parallel` and `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`"; empty `reasoning`
  with `minimax_m2_append_think` → expected, use `--reasoning-parser minimax_m2` for
  `/v1/responses` semantics; HCCL `address already bound` at startup → kill stale
  `vllm serve` processes.
- A report that runs TP=8 with no EP and no FLASHCOMM1 is missing both documented flags. At
  this tag that state also arms a double TP all-reduce in the MiniMax MoE patch — see
  [minimax-m2-family.md](minimax-m2-family.md) before writing it up.

## GLM-5.2 at v0.23.0rc1 (`docs/source/tutorials/models/GLM5.2.md`, ~1550 lines)

Section index — jump to the line instead of reading the file: `:15` Environment Preparation,
`:17` Model Weight, `:25` Installation (`:32` A3 tab `IMAGE=quay.io/ascend/vllm-ascend:glm5.2-a3`,
`:79` A2 tab `IMAGE=quay.io/ascend/vllm-ascend:glm5.2` — model-specific image literals, not
`vX.Y.Z` substitutions), `:117` Deployment, `:119` Single-node, `:169` Multi-node (`:176` A3 tab,
`:284` A2 tab), `:406` Co-located 4 nodes 200k context, `:530` Prefill-Decode disaggregation
(`:982` its 8x A2 block), `:1212` 1M context (single-node / dual-node co-located / PD).

- Weight and topology, `:19-20`: BF16 needs 2x A3 (128G x 8) or 4x A2 (64G x 8);
  `GLM-5.2-w8a8` needs 1x A3 (128G x 8) or 2x A2 (64G x 8). The multi-node tabs restate it as
  "2 Atlas 800 A3 (64G x 16)" (`:179`) and "2 Atlas 800 A2 (64G x 32)" (`:287`) — the same
  two-node footprint in a different notation, not a bigger requirement.
- The A3 tab and the single-node recipe are a **different weight format**: `GLM-5.2-W4A8C8-0713-MTP`
  with `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, `VLLM_ASCEND_ENABLE_FUSED_MC2=1` (`:132-133`, `:204-205`),
  `--tool-call-parser glm47 --reasoning-parser glm45 --max-model-len 135000
  --max-num-batched-tokens 8192`, and `--additional-config` `enable_dsa_cp` / `enable_sparse_sfa_c8` /
  `enable_sparse_li_c8` / `enable_balance_scheduling` (`:154`). Do not carry those flags to a w8a8
  report or vice versa.
- The A2 multi-node node-0 block (`:317-345`) is the one w8a8 reports copy: env
  `HCCL_OP_EXPANSION_MODE=AIV`, `VLLM_RPC_TIMEOUT=360000`,
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000`, `HCCL_EXEC_TIMEOUT=200`, `HCCL_CONNECT_TIMEOUT=120`,
  `TASK_QUEUE_ENABLE=1`, `CPU_AFFINITY_CONF=1`, `VLLM_ENGINE_READY_TIMEOUT_S=1200`, FLASHCOMM1 /
  `USE_MULTI_GROUPS_KV_CACHE` / `USE_MULTI_BLOCK_POOL` commented out, `VLLM_VERSION=0.21.0`
  (a doc bug at this tag: [env-vars-and-additional-config.md](env-vars-and-additional-config.md));
  serve with `--max-model-len 40000 --max-num-batched-tokens 4096 --data-parallel-size 2
  --data-parallel-size-local 1 --data-parallel-rpc-port 13389 --tensor-parallel-size 8
  --enable-expert-parallel --quantization ascend --safetensors-load-strategy prefetch
  --block-size 128 --async-scheduling --additional-config '{"multistream_overlap_shared_expert":
  true}' --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' --speculative-config
  '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'`.
- A recorded report pasted exactly that block (2x 800 A2, w8a8, DP2/TP8/EP, MTP k=5), so "diff the
  command against the tag's tutorial" has no delta there; when that happens on GLM-5.2, stop
  diffing flags and answer from the failure tail's class
  ([log-signatures.md](log-signatures.md), the `WorkerProc initialization failed` section).

## Doc-verification reports (众测 task tickets)

Auto-generated task tickets ask for a completeness/usability/correctness verdict on one named
tutorial plus deployment evidence, and their comment thread is a queue of tester reports. Judge the
tutorial at the reported tag, not the published site. At `v0.18.0`,
`docs/source/tutorials/models/Qwen-VL-Dense.md` still opens with "This tutorial uses the
vLLM-Ascend `v0.11.0rc3-a3` version for demonstration" (`:9`), its single-NPU tab serves
Qwen3-VL-8B-Instruct with `--dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384`
(`:324-327`), and the `mmmu_val` table it prints is labelled
`vllm-ascend:0.11.0rc3` (`:504-508`) — stale version claims inside a tag's own doc are the
concrete correctness finding this class wants. Pair it with the model's e2e config above
(`tests/e2e/models/configs/Qwen3-VL-8B-Instruct.yaml`) and the release notes' known issues.

The second shape is a `[Doc]` report with the `documentation` label, often zero comments, no serve
line and a proprietary image, asserting that the current guide recommends something the hardware
cannot do. Judge it as documentation correctness at the report's date *and* at `main`: quote the tab
that carries the value, the doc's own limitation/notes text, the sibling recipe that disagrees, and
the release-notes known issue that states the validation boundary. Do not adopt the reporter's
proposed mechanism (fragmentation, HCCL corruption) as established; name what the doc does and does
not say, give the value the same doc validates for that hardware, and ask for the failing log. The
DeepSeek-V4.1-Flash A2-at-1M instance is the worked example above.

### The task book's doc URL can name a file the tag does not have

The `任务范围` link is normally `blob/releases/vX.Y.Z/<path>` — the release **branch** — while the
ticket also pins an image tag. A task book naming `docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md`
for `v0.18.0` reads as `fatal: path '...' exists on disk, but not in 'v0.18.0'`: that tag carries the
197-line `Qwen3.5-27B.md`, and the merged Qwen3.5/Qwen3.6 file exists only on
`origin/releases/v0.18.0` (`2d61c216e` #9309, then `#11003`, `#12158`). Enumerate before concluding
a doc is missing — `git ls-tree -r --name-only <tag> -- docs/source/tutorials/models | grep -i
'<model>'` — and read the branch copy with `git show origin/releases/v0.18.0:<path>`; a path miss at
the tag is not evidence the guide omits the recipe. The recipe is the same in both copies
(`--net=host` at `:52`, `vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp --host 0.0.0.0 --port 8000` at
`:105-107`, `--data-parallel-size 1 --tensor-parallel-size 2` at `:108-109`,
`curl http://localhost:8000/v1/completions` at `:148`; the renamed branch copy moves them to `:62`,
`:117-119`, `:120-121`, `:196`). What *is* frozen at the tag is the prose image tag — the `v0.18.0` file still
advertises `quay.io/ascend/vllm-ascend:v0.17.0rc1` (`:35`) while the ticket's blue zone is `v0.18.0`.
Also reconcile the card count with the recipe: a "4 cards on Atlas 800I A2" task running the doc's
TP2/DP1 uses two cards and leaves two idle, so ask for the launcher before reading any throughput
number.

### A task problem that is a missing client tool, not a defect

"容器内无法使用postman工具发送推理请求" is a container-contract question. The tutorial starts the
container `-it ... bash` with `--net=host` and no `-p` (`:50-52`), and the server binds
`--host 0.0.0.0 --port 8000`, so a GUI client such as Postman runs on the **host** against
`http://<host>:8000`; the in-container probe the same doc prints is `curl
http://localhost:8000/v1/completions` (`:148`), and a Python `requests`/OpenAI-SDK call is
equivalent. Pages in that release that do publish a port use `-p 8000:8000`
(`docs/source/developer_guide/contribution/testing.md:91`,
`developer_guide/evaluation/using_ais_bench.md:30`, `using_evalscope.md:28`, `using_lm_eval.md:30`),
so a task book copied from one of those needs the flag added; no page under `docs/` mentions
Postman. Candidate only, because the base image is not inspectable here: the tag's `Dockerfile:36`
installs `wget` and not `curl`, so the doc's own `curl` probe may be missing in-container — offer
`wget --post-data` or the SDK rather than calling the image broken. None of this is a plugin defect.

### TTFHW install/usability tickets and the getting-started page

A TTFHW ticket (`opensourceways/ttfhw` automation) carries a fixed header table — community, repo,
branch, verifying machine + image tag, test date, a link to a JSON test report — and then a suggestion
rather than a failure. Judge it against the tag's own getting-started page and that page's history; when
a maintainer answers at all, it is usually that the doc already says it and that the ask is a policy
choice.

- **Locate the page per release before grepping.** `docs/source/quick_start.md` and
  `docs/source/installation.md` exist through `v0.26.0rc1`; from `v0.27.1rc1` the same pages are
  `docs/source/getting_started/quick_start.md` and
  `docs/source/getting_started/installation.md` plus `installation/*.inc.md` (24 files), introduced by
  `0b8cd0aeb` "Rework Quick Start and Installation guides (#15309)", first tag `v0.27.1rc1`. A
  `git grep <ref> -- <old path>` against a moved page prints nothing and exits 0, which reads as "the
  doc never mentions it" — list the tree first
  (`git ls-tree -r --name-only <ref> -- docs | grep -i 'quick\|install'`).
- **Worked instance (`ModelScope` not emphasised, #14502, image `v0.18.0`).** At `v0.18.0`
  `docs/source/quick_start.md:91-97` says "You can use ModelScope mirror to speed up download" with
  `export VLLM_USE_MODELSCOPE=True` and no `pip install modelscope`, because the generic, `a3` and
  `310p` images install it (`Dockerfile:95-96`, `Dockerfile.a3:94-95`, `Dockerfile.310p:71-72`) and
  `docs/source/installation.md:288` still lists `pip install modelscope` for manual installs. That
  omission is deliberate history: `a7a1e5914` (#9484) deleted `pip install modelscope>=1.35.1` from the
  quickstart as redundant with the Dockerfile, and `1383e98c4` (#9493) put the env-var instruction back
  ("the redundant `pip install modelscope` command is omitted as it is pre-installed in the provided
  Docker images"). On `main`, `docs/source/getting_started/quick_start.md:50-59` is now an admonition
  ("If Hugging Face access is restricted") with `export VLLM_USE_MODELSCOPE=True` **and**
  `pip install "modelscope>=1.18.1,<1.38"`. So the report is a policy/wording request that later
  rework partly implemented; answer with the sentence at the reported tag, the commit that produced it,
  and the page on `main`, and do not invent a download defect. The recorded maintainer reply is exactly
  that reading — "the quick start guide clearly states ... and informs users to use ModelScope if they
  cannot connect to Hugging Face. I don't think it's necessary to emphasize the region" — and the
  ticket then closed through the stale bot.
- **Verdict shape for this class.** "Doc says X at tag T (file:line), has said so since commit C,
  because reason R; main now says Y; the reporter's ask is a policy change, not a doc error" — plus the
  issue's own close state, since these tickets are often closed by the stale bot (with or without one
  maintainer comment).

## Image tags and duplicated container blocks in the docs

"Which image do I pull" and "why does the guide start two containers" are documentation-drift
questions. Read the substitution value and container blocks at the relevant ref.

- **Where a doc's tag comes from.** Release-wide pages interpolate a substitution — Sphinx era
  `docs/source/conf.py` `myst_substitutions` (`v0.18.0rc1:72` `vllm_ascend_version: v0.17.0rc1`),
  MkDocs era `mkdocs.yml` `extra:` (`main:167` `vllm_ascend_version: v0.23.0`) — written as
  `|vllm_ascend_version|` inside a `{code-block} bash :substitutions:` or as
  `{{ vllm_ascend_version }}`. The bump list is the repo's own
  `.agents/skills/vllm-ascend-release/references/version-files.yaml`: README, README.zh, mkdocs.yml,
  `docs/source/faqs.md`, release notes, versioning policy, contributors, workflows, Dockerfiles.
  **Model tutorials are not on it**, and its validation step (`grep -r 'X.X.X' docs/ README*`) treats
  a surviving literal version as the defect. So a literal `quay.io/ascend/vllm-ascend:vX.Y.Z` in a
  tutorial is unmaintained drift, while the substituted line beside it moves with every release.
- **A tag's substitution value is not what the page showed.** `v0.18.0rc1`'s own `conf.py:72` still
  says `v0.17.0rc1`; the value is bumped at cut-down on the release branch
  (`origin/releases/v0.18.0:72` says `v0.18.0`). Compare against the ref the URL matches — `latest`
  -> `main`, `/en/vX.Y.Z/` -> that release branch — before calling the reporter wrong.
- **Recorded instance (a `/en/latest/` report).** At `v0.18.0rc1`,
  `docs/source/tutorials/models/Qwen3.5-27B.md:35` names `quay.io/ascend/vllm-ascend:v0.17.0rc1`
  (and `-a3`) in prose while `:45` interpolates `|vllm_ascend_version|` in the command below; the
  release-branch copy, after the rename to `Qwen3.5-27B-Qwen3.6-27B.md`, lists `v0.17.0rc1` /
  `v0.18.0rc1` at `:45` and interpolates at `:55`. One page, two different tags. The fix that landed
  removed the prose tag and added "strongly recommended to use the latest release candidate (rc)
  version or the latest official version"; the PR opened for the issue was still open, unmerged and
  hardcoded `v0.19.1rc1` again — an open PR is not a fix, and a re-hardcoded tag drifts again.
- **Second instance, in a runnable command.** At `v0.18.0`,
  `docs/source/tutorials/models/Qwen3-Coder-30B-A3B.md:30-32` sits inside a `{code-block} bash
  :substitutions:` block yet writes a literal `export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc1`, so
  a reader following the 0.18.0 tutorial pulls a 0.11 image. Substitutions rewrite only the
  `|...|`/`{{ }}` tokens, never the literal text around them — that is why a frozen tag survives in
  the rendered page.
- **One-command check for any tag-drift report:**
  `git grep -ln 'quay.io/ascend/vllm-ascend:v0\.' <ref> -- docs/source/tutorials` — nine files at
  `v0.18.0`, and `main` still carries literals in `DeepSeek-V4-Flash.md`,
  `DeepSeek-V4.1-Flash.md` and `Qwen3.8-27B.md`. So main having *a* literal neither clears the
  reported page nor makes the reporter's tag current: read the literal at the reported ref.
- **Variant matrix.** At `v0.18.0rc1` it is `docs/source/installation.md:197-202`, and on
  `main` the page moved to `docs/source/getting_started/installation/prebuilt_image.inc.md:19-26`,
  so a stale locator goes quiet rather than disproving the claim. Rows: bare tag = A2 series/Ubuntu,
  `-openeuler` = A2/openEuler, `-a3`, `-a3-openeuler`, `-310p` (Atlas 300I DUO and, on main, Atlas
  200I Pro), `-310p-openeuler`, plus `-a5`/`-a5-openeuler` for 950DT on main. There is **no** `-a2`
  suffix. An A2 machine paired with an `-a3` tag, or a 310P device with the bare tag, is a
  wrong-variant report rather than a build bug — the 310P case has its own startup error in
  [310p-atlas-300i-duo.md](310p-atlas-300i-duo.md).
- **The `-a3` family starts only at `v0.10.0rc1`.** `875a920d4` (#1794, "Add support for Altlas A3
  series") adds `Dockerfile.a3`/`Dockerfile.a3.openEuler` and the `image_a3_*` workflows, and it is
  absent through `v0.9.2rc1` (`git ls-tree -r --name-only v0.9.2rc1 | grep -c '^Dockerfile.a3'` = 0);
  every earlier tag ships only the `910b`-based `Dockerfile`/`Dockerfile.openEuler` (at `v0.7.3`,
  `FROM quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10`, `VLLM_TAG=v0.7.3`). A 910C/A3 server on
  such a tag runs an A2 (910B) CANN base and dies at the first `aclnn` op —
  `RuntimeError: call aclnnArange failed ... EZ9999 ... Parse dynamic kernel config fail ... ArangeAiCore
  ADD_TO_LAUNCHER_LIST_AICORE failed` (#1696) — and the plugin cannot compensate: at `v0.7.3`,
  `git grep -i -E '910c|910_93|Ascend910B' v0.7.3 -- vllm_ascend docs` is empty, so hardware support
  comes entirely from the base image. The maintainer's route for such a report is to rebuild that
  tag's Dockerfile with an `-a3` CANN base ("replace the base image with
  `quay.io/ascend/cann:8.1.rc1-a3-ubuntu22.04-py3.11`"); the first released `Dockerfile.a3` uses
  `8.2.rc1-a3-ubuntu22.04-py3.11` with `VLLM_TAG=v0.10.0`. Git tags carry no hardware suffix —
  `git tag --list | grep -i a3` is empty — so `-a3` exists only as a registry image tag published by
  that workflow (`vllm-ascend:vX.Y.ZrcN-a3`).
- **Two `docker run` blocks in one tutorial are alternatives, not steps.** At `v0.18.0`,
  `docs/source/tutorials/models/Qwen3-Dense.md` is the only tutorial carrying both:
  `#### Docker run` (`:53`, under `### Installation`) is the shared boilerplate —
  `--name vllm-ascend-env`, `--net=host`, eight davinci devices, no `-p`, A2/A3 image exports
  commented out — and `### Run docker container` (`:112`, under `## Deployment`) is the model recipe:
  `--privileged=true`, four devices, `-p 8113:8113`, serving on `--port 8113`. `--net=host` already
  publishes the port, so the second block's `-p` is not an extra step, and no sentence says the two
  are alternatives — that silence is the finding. The maintainer answer was "use the first; the
  difference is only some docker start parameters". The repo's own
  `docs/source/_templates/Model-Deployment-Tutorial-Template.md` §4.1 shows the sanctioned shape — one
  `export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}` run block per hardware tab and
  no second container under Deployment — so both blocks in one tutorial is a template violation.
  `main` rebuilt §4.1 that way, while `origin/releases/v0.18.0` still carries both, so the versioned
  page keeps showing them: a doc fix merged to `main` after a tag is absent from that release branch
  unless cherry-picked (the Qwen3-Dense rewrite is contained only in `releases/v0.22.1rc`).
- **A "test passed" 众测 ticket still owes a verdict.** With no failure log, answer from the tag's own
  doc staleness (the frozen image prose above is exactly that class) plus the model's e2e config, and
  say the task thread reports no failure — do not invent a runtime mechanism to fill the slot.

## Use

Quote the tutorial path, tag and the exact differing flags. A byte-identical command points
at a documented-path bug or a stale doc; a delta is the first hypothesis. Do not carry one
model family's code mechanism to another: a missing FLASHCOMM1 is a documented delta in both
Kimi and MiniMax reports, but the double-reduce path is a MiniMax-M2 patch specific to
`vllm_ascend/patch/worker/patch_minimax_m2.py`. Neither the
tutorial nor the thread proves a runtime cause — label each statement accordingly.
