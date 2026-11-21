# Script catalogue: the exact invocation of each one-call check

## Scope

The flag synopsis and the expected reading for every re-runnable script under
`scripts/`, i.e. the detail behind the compact name index in `SKILL.md`'s support
entry points and behind the "one call: `scripts/…`" pointers on the topical
pages. Open it when a page names a script and you need its arguments. Each
script's own header repeats its full flag list and is authoritative; the synopsis
here records the flags as written when the page was added, so a script changed
after that keeps the header, not this text.

## Common rules

- Inherit every rule in `SKILL.md`.
- A named script **is** the check: run it rather than re-deriving its arithmetic
  by eye or with an ad-hoc snippet (`cheap-checks-by-track.md`).
- `<skill_root>` is the path `skill_view` reports; every other link on this page
  is skill-root relative and every script is run from there. Run `scripts/*.sh`
  with `bash <skill_root>/scripts/<name>.sh` and `scripts/*.py` with
  `uv run --no-project python <skill_root>/scripts/<name>.py` — never `bash` on a
  `.py`, and never rely on an executable bit (it is not durable).
- A script reproduces a calculation or extracts a set of sites. It does not
  decide the cause: quote its output as the check and keep the claim-strength
  label (`SKILL.md` rule 6).
- The issue id in brackets is where the arithmetic came from. Re-read the pinned
  source before trusting a derivation at another release.
- Several scripts answer per-track questions in one call; the track entry that
  names one is in `cheap-checks-by-track.md` or in its topical page.

## Entries

- `scripts/image-recipe-inventory.sh <vllm-ascend-root> [--soc <variant>] [--image-tag vX.Y.ZrcN]`
  (`--soc` = `910b` (default) / `a3` / `310p` / `openeuler` / `a3-openeuler` / `310p-openeuler`) — the official image's dependency
  recipe in one call (#2812, #10525 hand-built CANN+conda envs). Prints the chosen
  variant Dockerfile's `FROM` (where CANN **and** NNAL/ATB come from), `ARG VLLM_TAG`,
  both `--extra-index` URLs, `pip uninstall -y triton`, both `set_env.sh` sources and the
  `devlib` `LD_LIBRARY_PATH` append; the `requirements.txt` torch / torch-npu pins and
  the `--pre`/`--extra-index-url` hint lines; the docs' component table, manual
  CANN/kernels/NNAL `.run` block and the container's host `-v` driver mounts; the first
  compatibility-matrix rows (the pair, not one package); and a paste-ready block the
  reporter runs inside their own container plus the diff against the same-tag image.
  Exit 0 always, 2 on a missing checkout or unknown `--soc`. Step 1 of
  `../image-dependency-faults/image-recipe-inventory-and-diff.md`.
- `scripts/logprob-bucket-fit.py --row 'P_LOW,P_HIGH,N,MEAN_ABS' [--row …]
  [--vocab 152064] [--spread 2.0]` — the cross-platform logprob table in one call
  (#18755, 910B2C vs an H100×2 golden). Prints each bucket's `mean|Δ log p| / (1-p)`
  ratio and an implied logit-space delta bracket, then the **flat-ratio verdict**:
  ratios within `--spread` mean one perturbation scale reproduces every row (quote
  `mean|Δ log p| ≈ k × (1-p)` as the platform floor and do not rank a per-bucket
  cause; the `p > 0.99` row is expected to come out *lower*), while a row above the
  flat line is not roundoff and owes a `top_logprobs: 5` dump at that token. Row =
  `P_LOW,P_HIGH,N,MEAN_ABS` (two fields accepted). Exit 0; 2 on bad input. Usage and
  the worked reading: `../accuracy-benchmarks/cross-platform-logprob-parity.md`.
- `scripts/benchmark-table-normalize.py --run 'NAME OUT_TOK_S WALL_S REQS CONC'
  [--run …] [--total-tok-s NAME=V] [--ttft NAME=SECONDS] [--compare A B]` — the
  two-run ratio in one call (#2833 Ascend-vs-H800 EvalScope screenshots, #14291
  v0.23→v0.24 PD "12s vs 4s"). The summary block's only OCR-stable fields are the wall
  time and the two throughput fields; this derives per-request output/input tokens, the
  decode step (`concurrency / output tok/s`), the per-sequence rate, the prefill cost
  (`TTFT / input tokens`), a wall-clock closure check (`wall / (waves × tokens × step)`,
  `>>1` = queueing/warmup/fixed per-request cost) and, with `--compare`, the cost ratio
  per axis plus which axis moved more and on which run. Prints the derived averages to
  quote instead of the table's own transposable columns. Exit 0; 2 on non-numeric or
  zero-throughput input. Rows with no `--total-tok-s` keep only the decode figures.
  Step 1 of `../performance-throughput/cross-run-ratio-normalization.md`.
- `scripts/opcount-reconcile.py --count NAME=N [--count …] [--table FILE|-]
  [--max-factor M] [--layers L] [--steps N] [--elapsed-s S] [--avg-us NAME=US]` —
  profiler-table arithmetic for an op-count report (#7351): prints every row's
  count, the step count that *all* rows' quotients agree on with each row's per-step
  factor, `--steps`/`--layers` per-unit figures, calls/s over `--elapsed-s`, and the
  `count × avg-us` total against the window (over 100 % = rows from different
  windows, under 5 % = those rows cannot be the window's cost). Exit 0 =
  reconciles, 1 = no common step count (mixed windows / broken collection), 2 =
  unusable input. Table lines are `<name> <count>` / `<name>,<count>`; a garbled
  OCR row is listed as skipped rather than guessed.
- `scripts/op-shape-lane.sh <vllm-ascend-root> [op-name …]` — the launch-lane check
  behind a profiler row with `N/A` shapes (#9026): with no op name it prints the
  `CUSTOM_OPS`/`CUSTOM_OPS_ARRAY` lists of `csrc/build_aclnn.sh`, the `EXEC_NPU_CMD`
  and `SetCustomHandler` counts in `csrc/`, and every customHandler site; per op name
  it prints the build-list hits, the csrc files that mention it with their binding
  style, and one verdict — `aclnn-built` (an `aclnn<Op>` entry point exists, shapes
  expected), `raw-kernel-lane` (the op's own adapter launches from a
  `SetCustomHandler` lambda and never calls `EXEC_NPU_CMD`, so `N/A` is the expected
  outcome) or `unknown at this pin`. It extracts sites; it does not prove what the
  installed CANN reports.
- `scripts/moe-gating-topk-probe.py [--experts 256] [--top-k 8] [--rows 64]
  [--dtype float16|bfloat16|float32] [--seeds 5] [--spread 1.0 0.05] [--device npu]
  [--atol 1e-3] [--rtol 1e-3]` — the one-command check behind "fluent but wrong answers on
  a MoE serve" (#10892): runs `torch_npu.npu_moe_gating_top_k_softmax` against
  `softmax(x.float(), -1).topk(k)` over several seeds and two logit spreads (a narrow
  spread concentrates the ties), printing the worst weight error, rows whose selected
  expert **set** differs, and rows where only the order differs (benign ties). Exit 0 =
  within tolerance (gating lane weakened — try larger `--experts` or another `--dtype`),
  1 = mismatch (the lane is live: report the output, no config ladder), 2 = cannot run
  here (no torch / no `torch_npu` / no NPU — a gap, not a pass). Run it through
  `uv run --no-project python …`; the reference is the identity the in-tree e2e test
  asserts, at fp32 instead of the test's discarded `torch.allclose`.
- `scripts/quant-scale-dtype-audit.py <vllm-ascend-root>` — the sibling sweep for a
  quant-scale dtype gate (`EZ1001 … Tensor scale not implemented for DT_FLOAT16`,
  `161002`): prints every weight-side scale handed to `npu_quant_matmul` /
  `npu_grouped_matmul*` / `fused_experts` with `file:line`, whether each parameter has
  a `_fp32` / `_fp32_list` companion in the package, which companions are `CREATED
  BUT NEVER READ` (the v0.17.0rc1 linear site), and which call sites take the op's
  output dtype from a scale tensor (`moe_mlp.py:144`, `device_op.py:142`) — the
  second operand of the same gate (#7443). Run through
  `uv run --no-project python …`; per-token/activation scales are excluded by design,
  and a site it lists still has to be read in `apply()` before it can be cited.
- `scripts/pp-layer-partition.sh --layers N --pp P [--key-layer K]` — the PP layer
  split and stage-start fingerprint (#5724). Mirrors `get_pp_indices`: the
  `layers % pp` remainder goes to `[-2], [-3], …`, never the last partition, so
  61 layers / PP=4 is `[15,15,16,15]` (starts `0,15,30,46`), not the naive
  `0,15,30,45`. With `--key-layer` it says whether a `KeyError: '…layers.<K>…'`
  index is a stage start — the only index a foreign per-worker kv-cache config can
  name — or contradicts that theory.
- `scripts/shard-source-probe.sh <model-dir> [pid] [--tp N] [--bytes MiB]` —
  checkpoint-source probe for a start-up frozen inside weight loading
  (`Loading safetensors checkpoint shards: 0/N`, no error, no `Loading model
  weights took`): the mount behind the path, shard count vs
  `model.safetensors.index.json`, per-rank size, oldest/newest mtime, a timed
  1 GiB raw read (or `dd did not finish within 20 s -> the READ PATH IS BLOCKED`),
  and with a pid a `/proc/<pid>/io` `read_bytes` delta over 5 s plus
  `ps -o stat,wchan`. Its per-rank line is arithmetic, **not** a phase bound —
  vLLM allocates the params on the device before reading
  (`base_loader.py:53-64`), so equal numbers prove nothing (#11095).
- `scripts/recipe-tab-diff.py --page <tutorial-page.md> [--section N.N] [--command FILE]
  [--tab NAME]` — which hardware tab a serve command belongs to (#11300). The model
  pages repeat their deployment section per SoC (`::::{tab-item} A3 series` / `A2
  series`, `:sync:`-paired) and the blocks are not interchangeable (dp/tp,
  `--max-model-len`, `--api-server-count` in one tab only, `enable_dsa_cp` in one tab
  only, and a `-a3` image tag beside an unsuffixed A2 one). Prints per tab the serve
  line's `path:line`, the flags whose value differs from another tab, the flags only
  that tab has, plus its `export`/`IMAGE` lines; with `--command` it tokenises the
  reporter's script, prints the dp/tp/`max-model-len`/`--api-server-count` it actually
  asks for and each tab's distance from it, then names the closest tab. The page,
  not the script, is the evidence — cite the `path:line` it prints. Exit 0 even when
  no tab matches (that is the finding); 2 for a missing page/command file. Step 0 of
  `../image-dependency-faults/native-string-abort.md` uses it on a startup abort.
- `scripts/history-titles.sh [-i] [-b] '<regex>' [root] [limit]` — precedent index,
  one `<file> :: <title>` line per matching history issue (rule 8: titles first).
  **Title-only is the default**: `-b` adds whole-body matching, and a title-only
  pattern that matches more than 30% of the corpus is reported as a pattern problem
  (split the alternation) — a body search for `suffix`/`mtp`/`ngram`/`spec` returns
  almost every issue through the pasted `collect_env.py`, which is what made #9278
  read 207 real hits as noise. The exception is a **code identifier or exception
  string** (`TRITON_ALL_BLOCKS_PARALLEL`, `posix.putenv`): those live in tracebacks
  and launch blocks, so `-b` is cheap — recorded #8224 got 1 title hit and 8 body
  hits from the same pattern. Arguments after the flags are **positional**; `-i` is
  case-insensitive, and any other leading `-x` exits 2 with the usage instead of being
  taken for the regex (a `history-titles.sh -i 'reranker'` call once burned a turn on
  `no such history root: reranker`).
- `scripts/history-body.sh [-n LINES] [-g REGEX] [--env] [--all] [--root DIR] <id>
  [<id> …]` — reads the 2-3 precedent bodies `history-titles.sh` selected, without
  their `collect_env.py` paste: prints `=== #<id> :: <title>` and the body with the
  fenced dump replaced by `[... N lines of collect_env dump omitted ...]`, dropping
  `<details>`/`<summary>` furniture so `-n` (default 80) counts report text. `-g REGEX`
  prints only matching lines with their line numbers (pull the traceback, the serve
  command, or the `Encoder cache will be initialized …` line); `--env` prints the dump
  instead (the artefact when the torch_npu/CANN pin, card count or host RAM is the
  question); `--all` prints everything. Several ids per call. `--root` picks one case
  directory when several share an id, and the header flags that case (`x2`). Use it
  instead of `sed -n '1,60p'`: the bug heading sits below the dump (#3963 line 148,
  #4663 line 168), so a prefix read costs a call and returns no symptom.
- `scripts/310p-rope-branch.py <config.json | model dir>` — which 310P rope
  implementation a checkpoint lands in, from its own `config.json`: multimodal or
  not, `uses_mrope` (`mrope_section` in `rope_parameters`), `head_size` /
  `rotary_dim`, the predicted `_rope_forward_oot` branch (a/b/c at
  `_310p/ops/rotary_embedding.py:136-190`), whether that branch's cos/sin globals
  are populated for a multimodal config, and the one probe that decides it (#11936,
  `../soc-platform-paths/310p-vl-ocr-garbage-output.md`). Exits 2 rather than
  guessing when `head_size` cannot be derived.
- `scripts/init-none-attr-probe.py FILE [FILE ...]` — which `self.<x> = None`
  declared in a class `__init__` is **never assigned again** in that file, with the
  lines that read it. Run it on a failing impl *and* its sibling impl in the same tree
  (torchair vs eager, graph vs non-graph, MLA vs non-MLA): the sibling's extra
  assignment is the fix. Answers the ATB `tensor dimNum 0 … inTensor [N]` family, where
  the op names the victim and the caller passed `None` — at v0.10.1rc1 it prints
  `torchair_attention.py:305: self.key_cache = None …, read at line(s) 416`
  (`../custom-ops-runtime-faults/atb-input-dim0-none.md`, recorded #2779). Exits 1
  when it finds one, 0 when the file is clean.
- `scripts/qwen2-body-drift-probe.py [--config DIR|config.json] [--lines N]` — is a
  downstream *copied* forward written against the installed library body? Prints the
  interpreter's `transformers` version/file, the checkpoint's `layer_types` /
  `sliding_window`, whether a `Qwen2DecoderLayer` exposes `attention_type`, whether
  `Qwen2Model.has_sliding_layers` exists, then the installed `Qwen2Model.forward` and
  `Qwen2DecoderLayer.__init__` source plus a two-line verdict (`INDEXES BY
  config.layer_types`, `mask mapping dict handled: yes|no`). Run it in the reporter's
  interpreter for an `AttributeError` raised inside `vllm_ascend/ops/qwen2_decoder.py`
  (#11661 / #11926 / #11991,
  `../upstream-downstream-drift/patched-forward-api-drift.md`): `attention_type: False`
  beside `INDEXES BY config.layer_types` means the copy is another generation of the
  library, so the repair is the delegation, not a `getattr` guard — the guard starts the
  service and misrecognises images. Exits 2 when `transformers` cannot be imported,
  which is itself the finding (a host run, not the image's interpreter).
- `scripts/sources-summary.sh [SOURCES.json ...]` — the version contract, one line
  per repo with coverage and `runtime_verified`, plus the unavailable repos.
- `scripts/fork-ctor-sweep.sh <repo-root> --base '<regex on the base class name>'
  [--kwarg NAME] [--glob '*.py']` — every subclass of one base class with its
  `__init__` parameter list in one call (#18688,
  `vllm_ascend/core/scheduler_dynamic_batch.py` missing `hash_block_size` while the
  engine passes it unconditionally). Per class: `OK` (declares the kwarg; the
  declaration's line is printed as the fix pattern to copy), `IMMUNE` (`**kwargs`),
  `MISSING` (neither — the bug set), `NO-CTOR`; the tail name of each base is
  matched, subclasses of a subclass are followed (an `Async…` twin counts), and the
  last line summarises the MISSING set plus the in-tree fix pattern. Run it in the
  reporter's vllm-ascend tree with `--base '^Scheduler$'`; the upstream caller side
  is one `rg -n '<kwarg>' <vllm-root>/vllm | head`. Exit 0 always, 2 on bad usage.
  Step 3 of `../upstream-downstream-drift/fork-ctor-arg-sweep.md`.
- `scripts/oom-budget-split.sh --line '<OOM line>' --gmu F [--params P] [--tp N]
  [--dtype D] [--hbm GB]` — splits a device-side OOM into inside vs outside the
  `gmu x total` budget, then prices weights/TP and the remainder (#6572). It does
  not classify the phase: a `torch.zeros` inside `initialize_kv_cache_tensors`
  makes "outside the budget" the expected shape (#4474), so read the frame first
  — `../deploy-sizing/tp-scaling-oom-at-startup.md` step 2.
- `scripts/acl-header-guard-check.sh --torch-npu-root DIR [--torch-npu-root …]
  --cann-root DIR [--sources REPOS_ROOT]` — compares the `acl/acl_base.h` copies a
  host-side C++ JIT TU can include (#6429): per copy it prints the path, the
  include guard, whether it typedefs `aclOpExecutor` and whether
  `acl_op_compiler.h` sits beside it, then the collision verdict (exit 1 =
  collision, 0 = no collision / only one copy, 2 = no roots). Run it on the
  reporter's `torch_npu` include root and CANN root; `--sources` runs the same
  check against the pinned exports. Owner: `../image-dependency-faults/host-cxx-compile-header-conflict.md`.
- `scripts/card-budget.sh --params P [--dtype] [--tp] [--gmu] [--hbm] [--cards]
  [--stage-gmu] [--stage-devices]` — weights-per-card vs budget and the gmu sum
  over processes sharing a device (#9335; the `--stage-*` half is the 3-stage omni
  layout of #8068). `P` is **billions** of parameters (`671`, not `671e9`); the
  script exits 2 on a raw count rather than printing arithmetic that inverts the
  verdict, and it has no default per-stage gmu — those values come from the
  reporter's stage yaml, because a 2-card deploy can pass once `devices` are
  re-pinned (`../deploy-sizing/card-budget-and-device-visibility.md`).
- `scripts/kv-page-size-arithmetic.py --layout {legacy,v023} --mode {li_c8,sfa_c8}
  [--soc A3|A5] [--dcp D] --k-head-dim L --v-head-dim R --index-head-dim I
  [--block-size 128] [--num-kv-heads 1] [--pages N]` — the DSA/sparse-C8 KV page
  identity behind `assert sum_page_size_bytes % page_size_bytes == 0` (#12102):
  recomputes `page_size_bytes` and the virtual terms from the checkpoint's dims,
  simulates the `int(size // ratio)` splits and prints `VERDICT: holds / does not
  hold`. The two layouts are the two arithmetic contracts in the snapshot (legacy
  ≤ v0.22.1rc1 keeps the virtual terms in bf16 elements, so exactness needs an
  even `index_head_dim`; v0.23.0 is byte weighted and carries
  `sfa_dcp_replicated_indexer_size`). `does not hold` = arithmetic, so no flag
  ablation can help.
- `scripts/aclgraph-size-budget.sh --layers N [--tp --dp --pcp --dcp --ep --moe
  --aiv --max-num-seqs --sizes]` — the capture-size budget: maximum captureable
  sizes and the sampled subset (94 layers, TP=16 gives 6 sizes, #3389).
- `scripts/seq-cliff-capture-budget.sh --max-num-seqs N [--num-spec K]
  [--max-num-batched-tokens T] [--running 256,258,264]
  [--max-cudagraph-capture-size V] [--tp --dp] [--enforce-eager]` — the
  concurrency cliff: the Ascend default capture ceiling
  (`min(max_num_seqs × (1+num_spec), 512)`), the capture-size list near the
  boundary, the running-sequence count where dispatch flips to `NONE`, the
  padded size per step, and the `max_cudagraph_capture_size` A/B value with its
  truncation trap (#14440).
- `scripts/fia-tnd-seqlen-check.sh --query-t N --seqlen-last M [--paste <log|->]
  [--max-num-seqs S] [--num-spec K] [--capture-sizes 1,2,4,8,16,32] [--dp D]
  [--requests R] [--mode full|piecewise|eager]` — the FIA TND refusal in one
  call: the capture-size list that launch resolves to (upstream default
  `[1,2,4] + 8..min(max,255) step 8 + 256..max step 16`, max =
  `min(max_num_seqs × (1+K) × 2, 512)` clipped by `max_num_batched_tokens`),
  whether `queryT` is one of those buckets, the request count and accepted-draft
  count implied by the pair under the v0.19 helper's branch rule
  (`r = N/(1+K)`, `M − r` drafts, with the `M == r` ambiguity stated instead of
  resolved), every Lane A request/token decomposition, the drafter-vs-target and
  DP-max readings, and the ordered ablations (#6598 v0.14, #9088 v0.19.1rc1).
- `scripts/working-set-arithmetic.sh --growth 80GiB --days 7 --rps 20-40
  [--tokens N] | --samples <paste|->` — the container-memory lane: bytes/request
  and bytes/decode-step from the slope, or the rss-vs-cache split (#16563).
- `scripts/host-ram-check.sh [--total-gib N] <paste|-> [...]` — host RAM vs the
  per-rank RSS sum, with the VIRT and first-sample `%CPU` caveats (#3902).
- `scripts/attn-mask-budget.sh --max-model-len N [--ranks K] [--dtype] [--host-gib]
  [--log file]` — the eager `max_model_len²` host attention mask: peak (9·N²) and
  retained (2·N²) bytes per rank, the total across ranks, the share of host RAM,
  and the largest `--max-model-len` that fits (#2873, #3002).
- `scripts/kvpool-segment-check.sh <mooncake.json|serve-log|-> [...]` — the
  `global_segment_size` / `local_buffer_size` gate and the HDK band (#13439).
- `scripts/store-put-code-check.sh <paste|-> [...]` — the store **put** lane:
  maps `hccl_result = N` / `error_codes=[…]` to `hcclResult_t` and Mooncake's
  `ErrorCode` (`4` = generic INTERNAL, `-1` = generic store error, `-200` = out of
  space), reads the master's `PutStart`/`PutEnd`/`PutRevoke` as control-plane vs
  data-plane, and flags the canned `Check memory and store capacity` text (#11454).
- `scripts/ranktable-check.sh <rank_table.json|log> [...] | -` — `server_count` /
  `server_id` / duplicate `device_ip`, i.e. multi-server vs local HCCS (#14506).
- `scripts/hccl-link-pair-audit.sh <log|-> [--visible 0-3]` — the distinct failed
  `createLink` edges of a `Transport_Init_Error(EI0009)` / `DestLink` block, the
  `drvRetCode` / runtime-result codes, and same-card vs cross-card classification from
  the paste's `npu-smi` Bus-Id rows; it refuses to read equal address text as a duplicate
  `device_ip` (#4148).
- `scripts/plog-timeline.sh [--since YYYY-MM-DD] <log|-> [...]` — per-error-family
  timeline and the first in-window error of a plog (#9183); an older record it
  lists is a previous run **or** this process's own `init engine (...)` window, so
  read it against the startup lines before discarding it (#10700). It first
  measures the whole-hour offset between each embedded device record and its
  log-line prefix and normalises it (collector prefix local, records UTC; 8 h on
  the recorded hosts, #16652), prints that shift as a `NOTE`, and lists the dump's
  record clusters so a multi-hour/multi-incident dump is anchored with `--since`
  instead of on its newest line.
- `scripts/executor-timeout-check.sh <log|->` — vLLM's RPC/ringbuffer watchdog vs
  a real collective timeout; tolerates screenshot-OCR garble (#1865). It also prints
  the graph-mode evidence that decides the *next* branch — the mode banner, the
  `Graph capturing finished` rank count, the crash dump's `cudagraph_mode` tuple with
  its name (`[2,0]` = `FULL_DECODE_ONLY`), and whether the stalled step was a
  new/prefill request — and names `aclgraph-full-decode-runtime-hang.md` when the
  mode engaged (#4178). When the same paste names a dump/profiler feature
  (`dump_config_path`, `dump_config`, `PrecisionDebugger`, `msprobe`,
  `--profiler-config`, `collect_detailed_traces`), it reports the inverted lane
  instead: the step is slow on purpose and raising
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` before the launch is the fix
  (`../hccl-and-hardware/rpc-watchdog-timeouts.md`, #15015).
- `scripts/mm-placeholder-count.py --tile HxW [--tile … | --grid HxW …] [--patch 14]
  [--mode 16x|4x] [--merge 2] [--stages N] [--placeholders M] [--features N]` —
  the two multimodal token counts of one request side by side (#14014): the
  prompt side's area floor `(W*H)//(patch*patch*divisor)` and the vision side's
  per-stage per-dimension floor `(h//2)*(w//2)` over the patch grid, per tile plus
  the total, the `exact=NO` mark where the merge's `reshape` would raise instead of
  rounding (so a silent gap is never a rounding story), the `M > N` / `M < N`
  direction verdict, and the `mm_processor_kwargs` / `use_vit_merger` lever list.
  Tiles are the *resized* pixel sizes the image processor produced.
- `scripts/dp-single-request-bucket.sh --capture-sizes 8,1 [--dp --tp
  --max-num-seqs --report 1,2,8 --num-spec]` — the DP>1 + ACL-graph
  single-request lane: the `bs_to_padded_graph_size` map for the launch's own
  capture list, which bucket a one-request step replays versus a full step, the
  idle ranks' `execute_dummy_batch()→_dummy_run(1)` (eager, still joins the DP
  all_reduce), and the four-row ablation (#3128).
- `scripts/mc2-capacity-asymmetry.sh --capture-a LIST|default --capture-b LIST|default
  --tp N [--dp --pp --experts --max-num-seqs --num-spec
  --max-num-batched-tokens]` — the cross-node MC2 symmetry check (#8294): each
  node's own capture-size list is turned into `mc2_tokens_capacity =
  ceil(max_capture/tp)*tp` and `global_bs = ceil(max_capture/tp)*ep_world_size`,
  printed side by side with the ratio, plus the A2 selector gates
  (`ep_world_size >= 16`, `experts/ep_world <= 24`) and whether `profile_run`
  still takes the extra `_dummy_run(capacity)`. `default` means the node omits
  `--compilation-config`. It prints `ASYMMETRIC`/`SYMMETRIC` and the next lane;
  that the mismatch *hangs* is a hypothesis (close it with the identical-config
  rerun), and `VLLM_ASCEND_ENABLE_FUSED_MC2` is an A3-only switch
  (`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` Step 1b). It also runs
  the **within-engine** half — the largest per-rank batch (`max_num_seqs ×
  (1 + num_spec)`, and `--max-num-batched-tokens`) against the same capacity — so a
  serve whose two commands are identical is not closed too early: some DP ranks can
  cross capacity in one step and take the fallback MoE method while their peers stay
  on MC2 (#9887); that section also says when a crossing is *not* a bug.
- `scripts/mc2-dispatch-bs-check.sh --tp N [--dp --pp --max-num-batched-tokens
  --max-num-seqs --num-spec --experts --enforce-eager --pin 0.10.1rc1|0.22.1rc1]` — the
  **token-dim** A2 MC2 dispatch tiling refusal (`batchsize is invalid`, `…A2TilingFuncImpl`
  `CheckShapeAndSetTiling`, #2858): from a serve line it prints the EP world size (and
  therefore whether this op can be reached at all — `ep_size < 16` forces
  `allgathercommimpl`), `mc2_tokens_capacity` for the chosen pin, and the per-rank **bs**
  the op would be handed for one extend chunk and for a decode step
  (`bs = ceil(tokens/tp)`), marked *uniform* or *per-rank*: `--enforce-eager` skips the DP
  all-reduce, and at v0.10.1rc1 the op is called in uniform mode on A2 (`global_bs=0`, no
  `x_active_mask`), so unequal counts are the first lane. It also prints the per-card
  expert count against the `<= 24` gate of Step 2b. It cannot print the op's own
  acceptance limit — the CANN op source is not in the snapshot
  (`../feature-enable-recipes/moe-expert-parallel-with-dp.md` §Step 2c).
- `scripts/hccl-stuck-timeline.sh <log-file> [more files …] | -` — the **silent**
  whole-service hang (#8975): per file it prints the group's `[Load] comm config`
  (loaded `bufferSize` in bytes **and** MiB, `execTimeOut`), the
  `HCCL_BUFFSIZE set by environment|default` line the process really read, the
  `[BuildRemoteResByTag] newtag[…]` op family, the `status[STUCK]` span, and the
  `count of currently executed operators` series per rank marked `FROZEN`,
  `CLIMBING`, or `ONE SAMPLE` (a single line cannot show a freeze — ask for the
  series); then one cross-file verdict naming the files that froze on the same
  count in the same second (milliseconds are ignored: the ranks' suffixes differ).
  A CLIMBING rank is not stuck whatever its STUCK lines say — on the recorded case
  the prefill node printed STUCK and still climbed 110822 → 4191386 while the two
  decode nodes froze together on 2706. It reads the
  paste, not a cause: the mechanism and the MC2 selector arithmetic are
  `../hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md`.
- `scripts/device-type-check.sh <paste|serve-log|-> | --soc-version N` — which side
  of the device-type assert is the card and which is the wheel (#12141).
- `scripts/cuda-only-path-audit.sh <vllm-root> [subpath] [vllm-ascend-root]` — the
  unguarded `torch.cuda` lines and whether the downstream tree replaces the class
  (#4309).
- `scripts/batch-split-audit.sh <vllm-ascend-root>` — every site that slices a
  request-indexed tensor by a token count, attention backends **and** the PCP
  schedule metadata (`num_scheduled_tokens`, one entry per request) (#17991,
  #11231); on the v0.21.0rc1 pin it prints `fa3_v1.py:132`, `pcp_utils.py:660`
  and `:759`, and the two `pcp_utils` hits are gone at v0.23.0.
- `scripts/lora-graph-descriptor-audit.sh <vllm-ascend-root> [vllm-root]` — the
  LoRA lane in one call: the punica binding (`torch_ops` vs the AscendC
  `lora_ops`) with its SoC / `max_lora_rank` predicate, every `has_lora`
  producer split into live-batch vs config-constant, whether the capture loop's
  per-case `activate_lora` is ever consumed, and upstream's guard / `lora_cases`
  / `cudagraph_specialize_lora` default (#6279, #1812; the same audit succeeds at
  v0.14.0rc1 + vllm v0.14.0, #7150). Its reading block also spells out what does
  **not** work — `cudagraph_specialize_lora: false` turns the request-time crash
  into a capture-time `Aclgraph runtime mode mismatch at dummy_run` — alongside the
  rungs that keep graphs.
- `scripts/pip-pairing-check.sh <vllm-ascend-root> [vllm-root]` — the install-time
  pin arithmetic behind "could not find a version that satisfies" / "unmatched torch
  version": reads both trees' `torch` / `torch-npu` / `torchvision` pins, flags the
  metadata-vs-tree split for `--pre` / `--extra-index-url`, and names each
  contradiction (a torch-npu line off the torch line, a torchvision cap that excludes
  the vllm pin) (#1796, #4737). It also prints the **release's own contract** in the
  same call — the `docs/source/community/versioning_policy.md` matrix row(s) and the
  `conf.py` pip-pair substitutions — plus a reminder that `installation.md` prints
  `|pip_vllm_version|` tokens, so a version grep there comes back empty by design;
  that block is what answers a pin-bump report (`dependency-pin-bump-request.md`,
  #3266), where there is no pip error to read.
- `scripts/pep440-local-check.py '<spec>' '<spec>' [--candidates v1,v2]` — PEP 440 local-label
  matching for a `Cannot install X and Y … conflicting dependencies` paste: prints which concrete
  versions satisfy every spec, so `==2.9.0` vs `==2.9.0+cpu` is read as compatible (a missing
  index) instead of unsatisfiable. Falls back to `pip._vendor.packaging` when `packaging` is
  absent and the network is closed (#7089, #4621).
- `scripts/abi-symbol-check.sh <paste|->` — the `std::string`-ABI split in a
  bundled wheel (#1144).
- `scripts/image-arch-matrix.sh <vllm-ascend-root> [--soc 310p|a2|a3|a5]` — which
  CPU arch and SoC each published tag was built for, from the tree's CI: the
  `metadata-action` tag suffixes (→ SoC variant, with the bare/generic line
  separated), the `platforms:` / `arch:` matrix and its `push`-only condition, the
  runner legs, every `Dockerfile*` `FROM` base (an arm64 leg needs an arm64 base),
  the wheel-release matrix and the per-SoC `Dockerfile.buildwheel.*` list (#2178).
  It reads intent, not published manifests: the registry question is still
  `docker buildx imagetools inspect`.
- `scripts/checkpoint-key-prefix.py <model-dir|*.safetensors|*.index.json|->` — the
  checkpoint's root tensor-name prefixes from the safetensors header (no device, no
  weight load), plus `lm_head` presence, `config.json` `architectures` and any
  `config_sentence_transformers.json`; exit 1 = root-level names, which cannot load into a
  `…ForCausalLM` class (#5549). `--scales` switches it to the quantisation census:
  every `.scale` / `.weight_scale` / `.weight_scale_inv` / `.quant_bias` key grouped
  by module family, plus `quant_model_description.json` and any
  `quantization_config`; it separates "quantised export, wrong/absent method" from
  "full-precision release whose few FP8-stored projections carry a scale" behind a
  load-phase `KeyError: '<prefix>.weight_scale'`
  (`../startup-failures/weight-scale-keyerror-quant-variant.md`). `--dtypes` switches
  it to the **weight-dtype** census instead: the header `dtype` histogram over every
  shard, the module families carrying `F8_E4M3`/`F8_E5M2` *weight* tensors (scale-like
  suffixes counted apart), and `quant_model_description.json` / `config.json` side by
  side; exit 1 = FP8 weight tensors, i.e. the upstream DeepSeek-V3/R1 release that the
  910 does not execute — the answer `config.json` cannot give, since the conversion
  guide has converters delete those fields first
  (`../startup-failures/quantization-method-gate.md` §E).
- `scripts/checkpoint-header-scan.py <model-dir|*.safetensors>` — is each shard a
  safetensors *container*: per file the size, whether the first 8 bytes are a Git-LFS
  pointer / zeros / HTML, the decoded header length, the tensor count and a `BAD`
  verdict, plus `model.safetensors.index.json` vs disk; exit 1 names the file behind a
  `HeaderTooLarge` / `HeaderTooSmall` / `invalid JSON` report, so the lane closes
  without opening the loader (#955).
- `scripts/mtp-head-check.py <model-dir|config.json> [-q]` — does the checkpoint
  actually carry the MTP head the config promises (`mtp_num_hidden_layers` /
  `num_nextn_predict_layers`)? Reads the sharded index or each shard header and
  reports the three layouts that count — `mtp.*` keys (Qwen3.5/Qwen3-Next, the prefix
  `Qwen3_5MTP.load_weights` remaps), layer ids `>= num_hidden_layers` (DeepSeek), the
  `enorm`/`hnorm`/`pre_fc_norm*` norms — then `PRESENT`/`ABSENT` with the branch that
  follows (unquantized: `ValueError: Following weights were not initialized from
  checkpoint`; quantized: the tracking check is off, so `Accepted: 0` instead). Exit 2
  when it cannot decide (`mtp-head-missing-from-checkpoint.md`, #15201).
- `scripts/sse-toolcall-audit.sh <raw-sse>` — the streamed `tool_calls` argument
  fragments, their assembly, the count of chunks carrying neither `content` nor
  `tool_calls`, and a per-call verdict (`balanced` / `tail-lost` / `head-lost` /
  `quote-gap` / `malformed`) (#17797, #12261, #7682). `head-lost` counts paired
  braces, so a 2-character `"}` with no `{` is reported as a lost **head**
  (suffix-only assembly) and no longer as a `quote-gap` to be sent to
  `tool-schema-type-audit.py`.
- `scripts/seq-cls-arch-check.sh <vllm-root> <Arch>[,<Arch> ...] [--list '<frag>']`
  — resolve a `--hf_overrides architectures` string the way
  `ModelRegistry._normalize_arch` does: registered directly, else the first
  registered suffix rewrite, with the model family (text vs multimodal) that
  decides a `TypeError: Invalid type of HuggingFace config` report (#14817).
- `scripts/nz-format-gate-check.sh <paste|-> [vllm-ascend-root]` — which `EZ1001`
  family the paste is (**aclnn registry** vs **dtype × layout** vs **shape**,
  `Not support mat2 n = 1 or k = 1 when format is FRACTAL_NZ`), the phase of the
  first hit (a load-phase `process_weights_after_loading` /
  `_validate_shared_expert_consistency` hit makes `--enforce-eager` irrelevant), the
  NZ gate's fp32 policy at the checked-out ref (`is_enable_nz` vs
  `_should_trans_nz`), whether that gate tests the weight's **shape** at all, and how
  `weight_nz_mode` resolves there (`0` off / `1` quantised-only default / `2`
  bf16+fp16) — i.e. whether an fp32 pooling head (#4679) or a bf16 MoE
  shared-expert gate (#10014) is being cast to `FRACTAL_NZ`, and which mode is the
  escape. Two version boundaries it now reports instead of guessing: when **no** gate
  function exists at the root (v0.11-line trees decide the layout inline) it prints the
  cast-site and `is_310p()` greps and routes to
  `../feature-enable-recipes/weight-nz-layout-by-soc.md`, and when neither
  `weight_nz_mode` nor `VLLM_ASCEND_ENABLE_NZ` is in the tree it says the layout is not
  a flag there instead of printing the mode escape.
- `scripts/toolcall-raw-output-audit.py <dir-or-file> [...] [--markers '<a>,<b>']`
  — read a case's request / rendered-prompt / tokenized-request / completion
  dumps and print the lane: no tool-call marker in the engine's own completion
  means the sampled sequence has no call (model-choice lane, a *rate* question),
  a marker present means the parser lane; it also prints `sampling_params`
  (a `temperature > 0` body explains a non-zero miss rate) and joins the dumps by
  prompt-token count, warning when the request body and the completion come from
  different requests (#12658). Exit 1 = no completion dump to judge.
- `scripts/tool-schema-type-audit.py <request-body.json|->` — per tool argument,
  the declared JSON-Schema type, flagging the arguments a streaming parser will
  emit **unquoted** (`anyOf: [string, null]` and other schemas with no top-level
  `"type": "string"`); run it on the client's own body before reading a parser
  (#12261).
- `torchrun --nproc_per_node=<N> scripts/npu-gather-probe.py [--group] [--tokens N]
  [--cols N] [--dtype {float16,bfloat16,float32}]` — runs torch_npu's `gather`
  **emulation** outside vLLM (#1579): HCCL has no `gather`, so `_gather` broadcasts a
  `recv_size_list` with `dist.broadcast_object_list` and then all-gathers. `--group`
  takes the explicit-group branch (`distributed_c10d.py:126`, the frame in that case's
  tracebacks), the default takes `:110`; the tensor is 2-D fp16, `--cols` divided by the
  world size, so it matches a sharded logits shape. Exit 0 = every rank gathered and (on
  the destination) the per-rank sums match, 1 = the primitive failed (quote
  `type: message`), 2 = it could not import torch on this host. Run it inside the
  reporter's image on the failing node; `EOFError: Ran out of input` here puts the owner
  in the image's object collective, a clean run puts the logits path back in play.
  Page: `../startup-failures/profile-run-logits-gather-eof.md`.

- `scripts/cann-error-lookup.sh <CODE|NUMBER> [<CODE> …] [--root DIR]` — the vendor's
  own meaning for a CANN/HCCL code, in one call. An `E`-code (`EJ0001`, `EI0006`,
  `EL0004`, `EI0017` …) prints whichever table the snapshot exports —
  `cann/metadef/error_manager/error_code.json` and
  `cann/hcomm/src/common/error_code/error_code.cc` — with `errClass`, `errTitle`,
  `ErrMessage` and the **`Possible Cause` / `Solution` text**; a number prints the
  `HcclResult` enum name from any exported `hccl_types.h` (`7` → `HCCL_E_UNAVAIL
  resource unavailable`). `--root` defaults to `/workspace/repos`; scope it to
  `/workspace/repos/<issue>`. Exit 0 when every argument resolved, 1 otherwise, 2 on bad
  usage. When no table is exported it says so instead of implying absence (the
  v0.9.1-era `cann/` export holds only `catlass` — #3172). Quote the printed Solution as
  the vendor's statement, then keep the cause open until the code is matched to the line
  that produced it: `EJ0001` ("wait for 10s after killing the last training process") is
  the sleep-mode/second-engine lane (`../runtime-memory/sleep-mode-multi-instance.md`),
  and `EI0006`'s solution names `HCCL_CONNECT_TIMEOUT` before any link theory
  (`../hccl-and-hardware/link-establish-timeout.md`).

- `scripts/dep-range-check.sh <package> <reported-version> <tree-root>...` — one
  dependency line from an env dump against the ranges the pinned trees declare:
  prints every `path:line` that constrains the package (exact name, so
  `sentence-transformers` is not `transformers`), then a per-file verdict
  (`INSIDE`/`OUTSIDE`) with an exit code. Use it when a version string looks wrong
  but no install ever failed — a floor-only downstream pin beside an upstream cap
  (`transformers>=4.57.4` in vllm-ascend vs `>= 4.56.0, < 5` in vllm) installs
  cleanly and then breaks the layer that trusts the newer major (#7876). The torch /
  torch-npu / torchvision trio and the release compatibility matrix stay in
  `pip-pairing-check.sh`; this one is for any other package.
