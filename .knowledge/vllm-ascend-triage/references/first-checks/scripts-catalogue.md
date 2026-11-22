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
- `scripts/additional-config-key-audit.py [--json '<additional-config JSON>'] [--doc <tutorial.md>
  [--line N]] [--tree <vllm-ascend root> | --config-file <.../vllm_ascend/ascend_config.py>]`
  — grades every `--additional-config` key in one call against the pinned
  `AscendConfig` (#18130: the v0.27.1rc1 GLM5.3 tutorial prints `fuse_muls_add` flat).
  Prints the **schema generation** first — `STRICT` (`@config` pydantic dataclass,
  `extra="forbid"`) or `PERMISSIVE` (plain `__init__` + `additional_config.get`, the
  pre-`@config` era) — then one row per key: `OK` / `NEST-REQUIRED` (fatal, names the
  parent key to nest under) / `WARN-STRIPPED` (`_NON_USER_INPUT_KEYS`) / `UNKNOWN`
  (fatal) on a strict pin; `FLAT-READ` / `NEST-ONLY` / `IGNORED`, **none fatal**, on a
  permissive one, where a pydantic `unexpected_keyword_argument` therefore cannot come
  from this module. `--doc` audits every `--additional-config '{...}'` line of a model
  tutorial, i.e. grades the reporter's whole command without waiting for a paste, and
  the run ends with a corrected `--additional-config` string. Flat `fuse_muls_add` is
  `NEST-ONLY` at v0.26.0rc1 `f2f74a1` and `NEST-REQUIRED` at v0.27.1rc1 `3b31886`.
  Exit 0; 2 on a missing schema file or unparsable JSON. Step 1 of
  `../startup-failures/additional-config-schema.md`.
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
- `scripts/benchmark-table-normalize.py --run 'NAME OUT_TOK_S WALL_S REQS [CONC]'
  [--run …] [--total-tok-s NAME=V] [--ttft NAME=SECONDS] [--mean-itl-ms NAME=MS]
  [--mean-tpot-ms NAME=MS] [--median-itl-ms NAME=MS] [--arrival-rate NAME=RPS]
  [--compare A B]` — the two-run ratio in one call (#2833 Ascend-vs-H800 EvalScope
  screenshots, #14291 v0.23→v0.24 PD "12s vs 4s"). The summary block's only
  OCR-stable fields are the wall time and the two throughput fields; this derives
  per-request output/input tokens, the decode step (`concurrency / output tok/s`),
  the per-sequence rate, the prefill cost (`TTFT / input tokens`), a wall-clock
  closure check (`wall / (waves × tokens × step)`, `>>1` = queueing/warmup/fixed
  per-request cost) and, with `--compare`, the cost ratio per axis plus which axis
  moved more and on which run. **Omit CONC** for an arrival-rate driver
  (`vllm bench serve --request-rate …`): the in-flight sequence count is then
  derived from the mean latencies (`output tokens × mean ITL / wall`, or
  `out tok/s × mean TPOT`), the run is priced against the offered rate
  (`served ÷ offered req/s`), `mean TTFT / wall` is read as a FIFO-drain test
  (`≈0.5` ⇒ TTFT is queue time and the prefill figure is suppressed), and
  `mean / median ITL` flags the long gaps that emit no decode tokens (#3112:
  26 in flight, mean/median 13.5×, 410× oversubscribed). Prints the derived
  averages to quote instead of the table's own transposable columns. Exit 0; 2 on
  non-numeric or zero-throughput input. Rows with no `--total-tok-s` keep only the
  decode figures. Step 1 of
  `../performance-throughput/cross-run-ratio-normalization.md`; second half of
  `../performance-throughput/prefill-queue-and-throughput.md`.
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
  `161002`), candidate-first: `START HERE` lists the `_fp32` companions that are
  `CREATED BUT NEVER READ` with the launch line still passing the non-fp32 twin (the
  v0.17.0rc1 linear site, `w8a8_dynamic.py:103` vs `:86`), every quant-method launch
  site whose scale is not float32, and the sites taking the op's output dtype from a
  scale tensor (`moe_mlp.py:144`, `device_op.py:142` — the gate's second operand);
  the full inventory and companion list follow. It resolves **one alias hop**, the
  only way the MoE twin appears (`fused_experts(w2_scale=w2_scale)` `:251` ←
  `[layer.w2_weight_scale]` `:249`, #7443). Companions are a **same-file** fact, so
  `no _fp32 companion in this file` must be read against the allocation (static
  `deq_scale` is `float32` for bf16 and `int64` for fp16, `w8a8_static.py:67-69`) or
  the delegate before it is cited. Run it whole through
  `uv run --no-project python …` (~114 lines at v0.17.0rc1 — a `head -60` keeps the
  verdict and drops the sweep); per-token/activation scales are excluded by design.
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
- `scripts/modelscope-mirror-audit.sh <vllm-ascend-root> [<vllm-root>]` — grade a
  "国内用户下载模型失败 / Quick Start 应强调 ModelScope" report in one call (#14502).
  Prints, against the pinned checkouts: every `VLLM_USE_MODELSCOPE` site in
  `docs/` (locale excluded, so `quick_start.md:91,96` and `installation.md:287` sit
  beside 20 tutorial pages), the China framing that only the *image* mirror carries
  (`faqs.md:32,240`), the entry pages' cross-links **in both directions**,
  `N of M` tutorial pages already exporting the flag, the `pip install modelscope`
  line in every `Dockerfile*` (which tiers the snippet by install path), upstream's
  default (`envs.py:18,558-559`) and the bare
  `from modelscope.hub.snapshot_download import …` under
  `if envs.VLLM_USE_MODELSCOPE` (`weight_utils.py:184,188`), the entry page's example
  model ids, and whether each `tests/e2e/doctest…` path the page's own comments
  name exists. Exit 0; 2 on a missing checkout. Step 1 of
  `../tutorial-verification/missing-guidance-and-contributor-docs.md` §*The weights
  mirror is the other half*.
- `scripts/reference-locate.py TOKEN [TOKEN …] [--root DIR] [--max N] [--min-grade N]
  [--regex] [--case] [--quiet] [--issue N]` — step 2b's page-location grade in one call.
  Pass `--issue <the report's own id>` and a `[case arrows]` block above the grades marks
  that entry `[YOUR CASE]` with its `sed` range — the own entry is an earlier run of the
  case and beats the topical page beside it. One token
  per argument, taken from the paste: the error sentence, the attribute/function/op/
  kernel name, a numeric code, a flag, a symptom phrase. Prints each token's file count
  (rarest first), then every page carrying **all** tokens with that page's own
  `# heading` beside the path — that grade uncapped while it is a decision list
  (≤ 25 members), because `[case]` pages sort first and the topical owner is often
  its last line (recorded #6972 lost
  `startup-failures/graph-mode-not-applied.md` into `+5 more` of a capped 8 of 13) —
  then the `N-1`/`N-2` grades capped at `--max` (default 8) lines each, and the two shortest
  lists when no page carries two tokens. Pass a **flag as it stands**
  (`'--compilation-config'`): a dashed token is accepted instead of rejected as an
  unknown option, and searched with and without its leading dashes. A token also
  matches a file **name** — the owning page rarely repeats its own slug while every
  sibling links to it as a path, so a content-only grep ranks the owner below the pages
  that point at it — with `-`, `_` and spaces folded onto `-` on both sides, because
  slugs are dash-separated while a caller guesses underscores: `one_boundary_per_chunk`
  then reaches `tool-call-streaming/one-boundary-per-chunk.md` at the all-token grade
  (recorded #5596 graded it 0 and fell back to the family-word guard; fixed here).
  `[case]` = a `recorded-cases*` page (open it first), and its line
  names the entries inside that file that share most of the query tokens —
  `→ #12985:488 (3/3)` is id, record-heading line and matched-token count — so a hit
  names the case to read, not just the file; when no entry matches, the line carries the
  file's number of `**#<id>**` entries instead, so a 30-80-case page is greped for the
  report's id rather than read whole. `[index]` = a topic index (a route, not an owner),
  `[dispatch]` = a wording table, `[row]` = a `first-checks/signature-routes/` page —
  printed right after the `[case]` lines, and when an all-token grade carries both the
  script says to read that row's line with the `[case]` arrow, because several recorded
  cases can share three or four tokens and the arrow then names a neighbour in another
  lane (recorded #2984: arrow → #3494, a `…-w8a8` export, while the row owned the report). **A family-word guard prints before the grades** when
  no page carries all tokens and the widest token alone hits 40+ pages: the grade then
  cannot name the owner, and the moves that do are, in order, a **re-grade of the two or
  three shortened tokens** (the owner is usually already on the near-miss lines the first
  call printed, and nothing absent from the printout is opened — a page recalled by
  *name* from the `skill_view` `linked_files` inventory is periphery: recorded #2105
  opened `startup-failures/profile-run-logits-gather-eof.md` that way, its name matching
  the paste's `profile_run` frame while it is the V0 `EOFError` sibling and graded on no
  list, and never opened `hccl-and-hardware/collective-error-chain.md`, which the same
  grade listed at 3-of-5 holding the case entry), the shape's row in
  `references/dispatch/classified-shapes.md` and the `[case]` file opened by the report's
  own wording — a token-less paste (two benchmark tables, two screenshots, "X is slower
  than Y") is that case by construction (recorded #5605's five family tokens graded
  `4-of-5: none` with `mtp` on 183 pages and never listed the owning page). Default
  matching is case-insensitive fixed-string: `--regex` for an alternation
  (`preprocess_mamba|get_temporal_copy_spec`), `--case` to disable `-i`. Exit 0 always —
  a thin or empty grade means the tokens are family words (shorten them to the
  attribute, flag or code the paste names) or spelled differently in the library, not
  that no page exists. The two recorded sessions that ran the loop by hand (#6509 four
  tokens → 3.4 KB, #9003 five tokens → 19.8 KB of `rg -l` output, the intersection
  picked by eye across four lists) resolve to the same one or two pages this prints as
  the all-token grade, with the heading that makes the choice without opening them.
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
  prints only matching lines with their **raw-file** line numbers (pull the traceback,
  the serve command, or the `Encoder cache will be initialized …` line); `-n` caps the
  matched lines as well, so a long body needs `-n 0` and a broad `-g '.'` still stops at
  80, and because the printed stream is collapsed those `-g` numbers are not `sed` ranges
  into it (recorded #4766: `history-body.sh 3513 | sed -n '190,320p'` printed nothing,
  and the next call was a `wc -l`); `--env` prints the dump
  instead (the artefact when the torch_npu/CANN pin, card count or host RAM is the
  question); `--all` prints everything. Several ids per call. `--root` picks one case
  directory when several share an id, and the header flags that case (`x2`). Use it
  instead of `sed -n '1,60p'`: the bug heading sits below the dump (#3963 line 148,
  #4663 line 168), so a prefix read costs a call and returns no symptom.
- `scripts/310p-rope-branch.py <config.json | model dir>` — which 310P rope
  implementation a checkpoint lands in, from its own `config.json`: multimodal or
  not, `uses_mrope` (`mrope_section` in `rope_parameters`), `head_size` /
  `rotary_dim`, the predicted `_rope_forward_oot` branch (`:121-186` at v0.23.0rc1;
  a = head 128 + full cache, b = partial rotary 64, c = no globals), whether that
  branch's cos/sin globals are populated for a multimodal config, and which
  recorded symptom that predicts — a `profile_run` crash with `Check cos != nullptr`
  (#13522, `../soc-platform-paths/310p-vl-rope-cos-nullptr-startup.md`) or wrong
  text out of a serve that came up (#11936,
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
- `scripts/triton-placeholder-probe.py VLLM_ROOT [PATH ...]` — does an **absent**
  triton break this pin's import? Executes the pin's own
  `TritonLanguagePlaceholder`/`TritonPlaceholder` (no vllm import, no device) and
  prints its attribute set beside every module-level `tl.*`/`tldevice.*`/`triton.*`
  read under `vllm/model_executor/layers/fla/ops` (default), tagging each
  `present`/`MISSING` with the guarding branch. The list is version-dependent: at
  vllm v0.11.0 it has no `exp`, so the import-time `tl.exp` at `fla/ops/op.py:26` kills
  `import vllm_ascend.models.qwen3_next` (recorded #3325/#4206 inspection crash);
  at v0.16.0 it has `exp/log/log2` and `gather` is `hasattr`-guarded, so the same
  import survives. Run it before repeating either answer from memory — an absent and
  a mispaired triton raise the identical `AttributeError`
  (`../startup-failures/model-type-not-recognized.md`, inspection-crash section).
  Exits 1 when a read is MISSING (read the branch label — a guarded one is not
  evaluated), 0 when every read is provided.
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
  [--stage-gmu] [--stage-devices] [--stage-params]` — weights-per-card vs budget
  and the gmu sum over processes sharing a device (#9335; the `--stage-*` half is
  the 3-stage omni layout of #8068). `P` is **billions** of parameters (`671`, not
  `671e9`); the script exits 2 on a raw count rather than printing arithmetic that
  inverts the verdict, and it has no default per-stage gmu — those values come
  from the reporter's stage yaml, because a 2-card deploy can pass once `devices`
  are re-pinned (`../deploy-sizing/card-budget-and-device-visibility.md`).
  `--stage-params` is the per-stage footprint; without it each stage is charged a
  full copy of `--params` and the printed per-device `weights` column over-states a
  one-checkpoint pipeline (vllm-omni's talker/code2wav), which the script says in a
  note line — the **gmu sum** is the decision either way.
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
  --aiv --max-num-seqs --sizes --draft-layers --era 0.11|0.12|0.18|0.20]` — the
  capture-size budget: maximum captureable sizes and the sampled subset (94
  layers, TP=16 gives 6 sizes, #3389). `--era` follows the paste's tree, not the
  newest one: 0.11 and 0.12 share one formula (draft term included) and differ
  only in the default size list (67 vs 51 sizes at cap 512), 0.18/0.20 add the
  `-100`-per-CP subtraction; a v0.12.0rc1 paste reproduces #5557 as `72 / 51
  sizes / no adjustment`.
- `scripts/seq-cliff-capture-budget.sh --max-num-seqs N [--num-spec K]
  [--max-num-batched-tokens T] [--running 256,258,264] [--observed-cliff M]
  [--max-cudagraph-capture-size V] [--tp --dp] [--enforce-eager]` — the
  concurrency cliff: the Ascend default capture ceiling
  (`min(max_num_seqs × (1+num_spec), 512)`), the 8/16-step grid floor that makes
  the effective ceiling, the running-sequence count where dispatch flips to
  `NONE`, the padded size per step, the per-engine vs dp-aggregate boundary, the
  truncation warning a run must carry, and the `max_cudagraph_capture_size` A/B
  value. `--observed-cliff` reconciles the reporter's own step index with the
  arithmetic and, on a mismatch, lists the per-engine `--max-num-seqs` values
  that would produce it (#14440: tp8 `264 × 2 → floor 512 → 256`, dp2/tp4
  `224 × 2 → 448 → 224`, MTP off `264 × 1 → floor 256`).
- `scripts/rope-cos-sin-slice-check.sh --num-tokens N --size M [--paste <log|->]
  [--width W | --head-size H --partial P] [--capture-sizes 1,4,8,16] [--num-spec K]
  [--max-num-seqs S]` — the rope cos/sin reshape in one call: the row count
  `M / width` against `num_tokens`, the two width candidates the allocation can take
  (`head_size`, or `head_size × partial_rotary_factor` when the checkpoint carries that
  attribute) with which one raises, the `pad_for_cudagraph` map the launch resolves to,
  and a per-request-count table marking every count that raises and every count that
  returns **wrong rope with no exception** — then the two next calls
  (`rg -n 'update_cos_sin' vllm_ascend/`, the ablation ladder) (#6449 v0.14.0rc1).
- `scripts/qknorm-mrope-exposure.sh <vllm-ascend-root> [--kv-heads N]` — the fused
  qkv-split + RMSNorm + RoPE kernels in one call: the commit and the `triton-ascend`
  pin (`==3.2.1` at v0.23.0rc1, `==3.2.2` at v0.23.0 — read it, the kernel source does
  not move), every per-head `tl.sum(…, axis=1)` reduction site, which kernel hands a
  slice of the *normalized* tile to a cos/sin multiply (the consumer shape the
  zeroed-scale miscompile needs), every caller of the op with the FX pass's
  `head_size != 128` / bf16-only gates that a direct `torch.ops` call skips, whether
  any env switch could turn the fused op off, and the nightly tests' parameter tuples
  with the kv-head counts CI runs (`--kv-heads 4` answers whether the reported model's
  count is covered) plus the K variance chain to quote (#12642; the alignment sibling
  is `rope-cos-sin-slice-check.sh` for the rope reshape, not this).
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
  data-plane, flags the canned `Check memory and store capacity` text, and reads the
  master's `SSD Storage: X / Y` as its DFS/file layer (`--root_fs_dir` +
  `--global_file_segment_size`) rather than a client offload switch (#11454).
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
  all_reduce), and the four-row ablation (#3128). `--capture-sizes` also takes the
  launch's pasted `--compilation-config` value, `'{"cudagraph_capture_sizes":
  [8,1]}'`, or the bare list as a positional; a launch with no
  `--compilation-config` has no default list to assume and the script says so
  instead of inventing one. Invoke `bash <skill_root>/scripts/dp-single-request-bucket.sh …`.
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
  (`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` Step 1b). Leave
  `--num-spec` out when the launch carries no `--speculative-config`: the script then
  assumes `num_speculative_tokens = 0` and **prints that assumption** above the decode
  ceiling, because a `{"num_speculative_tokens": N}` in the serve line raises the axis
  to `max_num_seqs × (1 + N)`. It also runs
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
  the vllm pin) (#1796, #4737). Its **first finding** now restates the
  `torch-npu==X` beside `torch>=X` pair in `[build-system] requires`: that pair is only
  jointly satisfiable through a `+cpu`-labelled torch, so it prints the pip text to expect
  (`Cannot install torch-npu==X and torch>=X … torch-npu X depends on torch==X+cpu`), the
  rung (`--no-build-isolation` / `--extra-index-url …/whl/cpu`), and the tree's own x86
  CPU-index line — which refutes "no wheel for my arch" (#1411). It also prints the
  **release's own contract** in the
  same call — the `docs/source/community/versioning_policy.md` matrix row(s) and the
  `conf.py` pip-pair substitutions — plus a reminder that `installation.md` prints
  `|pip_vllm_version|` tokens, so a version grep there comes back empty by design;
  that block is what answers a pin-bump report (`dependency-pin-bump-request.md`,
  #3266), where there is no pip error to read.
- `scripts/pep440-local-check.py '<spec>' '<spec>' [--candidates v1,v2]` — PEP 440 local-label
  matching for a `Cannot install X and Y … conflicting dependencies` paste: prints which concrete
  versions satisfy every spec, so `==2.9.0` vs `==2.9.0+cpu` is read as compatible (a missing
  index) instead of unsatisfiable. Falls back to `pip._vendor.packaging` when `packaging` is
  absent and the network is closed (#7089, #4621). Prereleases follow packaging's own
  policy — a specifier that names a prerelease admits them — so a dev-vs-`rc` **build
  floor** reads right (`torch_npu >= 2.5.1rc1` rejects `2.5.1.dev20250320`, #446);
  `--prereleases` / `--no-prereleases` force the policy either way and the header prints
  which one was used.
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
  (`../startup-failures/weight-scale-keyerror-quant-variant.md`). It also prints the
  description's `version` / `group_size` with the two booleans the W4A8 MoE loader
  derives from them (`new_quant_version = version == "1.0.0"`,
  `is_per_channel_weight = group_size == 0`, absent `group_size` = 256), and the
  `version: "1.0.0"` + `group_size: 0` pair adds the prediction of the load-phase
  `RuntimeError: The expanded size of the tensor (16 // N) must match the existing
  size (16)` (`../startup-failures/quant-aux-param-tp-shape-mismatch.md`). `--dtypes` switches
  it to the **weight-dtype** census instead: the header `dtype` histogram over every
  shard, the module families carrying `F8_E4M3`/`F8_E5M2` *weight* tensors (scale-like
  suffixes counted apart), and `quant_model_description.json` / `config.json` side by
  side; exit 1 = FP8 weight tensors, i.e. the upstream DeepSeek-V3/R1 release that the
  910 does not execute — the answer `config.json` cannot give, since the conversion
  guide has converters delete those fields first
  (`../startup-failures/quantization-method-gate.md` §E).
- `scripts/quant-description-census.py <model-dir|quant_model_description*.json|->
  [--json <path>] [--config <path>]` — the **scheme** census of a modelslim artifact,
  which the scale/dtype modes of the script above do not answer: the value histogram
  over `quant_model_description.json` (`W8A8` static per-tensor vs `W8A8_DYNAMIC`
  per-token vs `FLOAT` = layer skipped back to an unquantized linear), any top-level
  `fa_quant_type` / `kv_quant_type` / `kv_cache_type`, the `FLOAT`-only module families,
  and — with a `config.json` — per-family parameter counts, the unquantized share `u`
  and the decode ceiling `R = 2/(1+u)`. It warns when the only file is the misnamed
  `quant_model_description_<scheme>.json` the reader ignores, and skips the ceiling on
  a MoE checkpoint (the expert budget is EP-dependent). One call for
  `Qwen2.5-7B-w8a8量化配置，没有性能收益…精度同样降低` (#10604): the histogram decides the
  accuracy half, `R` prices the speed half before any engine cause is ranked
  (`../performance-throughput/quantized-dense-no-gain.md`).
- `scripts/checkpoint-header-scan.py <model-dir|*.safetensors>` — is each shard a
  safetensors *container* and does it hold the tensor data its header declares: per
  file the size, whether the first 8 bytes are a Git-LFS pointer / zeros / HTML, the
  decoded header length, the tensor count, the declared data region against the file
  size (a short/interrupted copy is `BAD` even with a parseable header) and a `BAD`
  verdict, plus `model.safetensors.index.json` vs disk; exit 1 names the file behind a
  `HeaderTooLarge` / `HeaderTooSmall` / `invalid JSON` report (#955) or behind a serve
  that loaded and then answered one repeated token (`../accuracy-benchmarks/
  constant-token-degeneracy.md` Step 3b, #6136). Structural, not a hash — compare sizes
  or `sha256sum` with the hub as the second step.
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
  escape. Three version boundaries it now reports instead of guessing: when **no**
  gate function exists at the root it prints the cast-site and `is_310p()` greps and
  routes to `../feature-enable-recipes/weight-nz-layout-by-soc.md`; when neither
  `weight_nz_mode` nor `VLLM_ASCEND_ENABLE_NZ` is in the tree it checks
  `ascend_config.py` for the pre-v0.10 spelling and, if present, names
  `--additional-config '{"enable_weight_nz_layout": true}'` with its **`False` default**
  (v0.9.1rc2 `b9f715d6`, where int8 weights therefore run ND) — the speed-direction
  lever on that line; and only when none of the three exists does it say the layout is
  not a flag there at all — with the v0.11.0rc0 caveat that a launch template under
  `examples/` can still carry the old spelling (`examples/external_online_dp/run_dp_template.sh:46`),
  which is not a reader.
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
  own meaning for a CANN/HCCL/DCMI code, in one call. An `E`-code (`EJ0001`, `EI0006`,
  `EL0004`, `EI0017` …) prints whichever table the snapshot exports —
  `cann/metadef/error_manager/error_code.json` and
  `cann/hcomm/src/common/error_code/error_code.cc` — with `errClass`, `errTitle`,
  `ErrMessage` and the **`Possible Cause` / `Solution` text**; a number prints the
  `HcclResult` enum name from any exported `hccl_types.h` (`7` → `HCCL_E_UNAVAIL
  resource unavailable`). A 6-digit **runtime** code, which no `HcclResult` enum holds,
  now prints the `ACL_ERROR_RT_*` define with the header's own comment (one canonical
  copy, not the five that ride along in the exports) plus the vendor row:
  `507010` → `ACL_ERROR_RT_LOST_HEARTBEAT // ts lost heartbeat` with 「task scheduler
  丢失心跳」 and the vendor's 联系技术支持 action, `107030` →
  `ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT` (`cann/runtime/include/external/acl/
  error_codes/rt_error_codes.h`, `cann/runtime/docs/api_docs/aclError.md`). 5070xx names
  the TS/aicore/device owner, i.e. below vllm-ascend
  (`../hccl-and-hardware/device-health-checks.md`); 107030 points back into graph-mode
  code. Until this branch existed the script answered a `507010` report with
  "no HCCL_E_* = 507010" (#11171). `--root` defaults to `/workspace/repos`; scope it to
  `/workspace/repos/<issue>`. A full scan of that root takes ≈2 s, past the 1 s
  foreground cap, so the call comes back as a background job — pass a longer timeout
  (a `timeout_ms` of ~30 s) instead of paying a second `job_output` round trip to
  collect it, as recorded #16581 did. Exit 0 when every argument resolved, 1 otherwise, 2 on bad
  usage. When no table is exported it says so instead of implying absence (the
  v0.9.1-era `cann/` export holds only `catlass` — #3172). Quote the printed Solution as
  the vendor's statement, then keep the cause open until the code is matched to the line
  that produced it: `EJ0001` ("wait for 10s after killing the last training process") is
  the sleep-mode/second-engine lane (`../runtime-memory/sleep-mode-multi-instance.md`),
  and `EI0006`'s solution names `HCCL_CONNECT_TIMEOUT` before any link theory
  (`../hccl-and-hardware/link-establish-timeout.md`). A **`-8xxx`** argument is the DCMI
  return code a container's `npu-smi` prints: `-8005` → `DCMI_ERR_CODE_INNER_ERR
  (BASE - 5)` from `cann/driver/src/custom/include/dcmi_interface_api.h:1682` (base at
  `:1677`) plus the vendor row 「内部错误。请提供日志…」, `-8020` →
  `DCMI_ERR_CODE_RESOURCE_OCCUPIED` (vendor action 「请排查设备是否被其他容器、进程等占用」,
  i.e. the another-holder lane), and the neighbouring rows (`-8002` not-permitted,
  `-8007` invalid-device-id, `-8008` device-not-exist, `-8013`
  not-supported-in-container, `-8012`/`-8017` a device-state wait) expanding the
  discriminating window. **Run this branch with the default root even when the reported
  issue's own snapshot exports no `cann/driver`** (v0.8.4/v0.9-era, e.g. #632): the
  return-code table is an ABI, so any other pin's export answers the name and only the
  *holder* stays open — the miss message says so; when
  `src/custom/lqdrv/kernel/ioctl_comm_def.h` is exported its `LQ_DCMI_*` collision is
  printed under a `SEPARATE TABLE, do not cross-apply` heading, because that numbering
  says `-8005 = IOCTL_FAIL` and mis-quoting it cost the #8809 run a posted correction
  (`../startup-failures/no-npu-in-container.md` §*cheapest classifier*).

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

- `scripts/env-knob-liveness.py [NAME …] --root <vllm-ascend-root> [--vllm-root <vllm-root>]
  [--from-paste <launch.sh>] [--max-hits N]` — is each export in a pasted launch block a
  lever **at this pin**? Prints one verdict per name — `LIVE` (a code path reads it via
  `os.environ.get`/`getenv`/`environ[…]`), `LIVE-UPSTREAM` (read only in vllm, so a
  downstream `patch/**` may bypass it), `SET-ONLY` (set in docs/tests/CI but read by no
  Python here — the consumer is CANN/HCCL/another component, or the spelling is stale),
  `CONSTANT` (only a module constant shares the name), `COMMENTED` (only inside comments),
  `ABSENT` — plus up to `--max-hits` `path:line` sites each. `--from-paste` harvests the
  names from a script (`export X=…`, `- X: …`, `"X": …`). Originating case #12534
  (v0.23.0rc1 `f4a08bbd`): of five exports in the reporter's 310P block,
  `USE_MULTI_GROUPS_KV_CACHE` is `CONSTANT` (`patch_kv_cache_coordinator.py:35`) and
  `USE_MULTI_BLOCK_POOL` is `COMMENTED` (`GLM5.2.md:316`) while `HCCL_BUFFSIZE` /
  `ACL_OP_INIT_MODE` are `SET-ONLY`, so only `HCCL_OP_EXPANSION_MODE` and the allocator
  value could move anything. Step 3 of
  `../runtime-memory/allocator-expandable-segments-unmap-crash.md` §*If the same reporter
  returns …*; the rule also feeds
  `../feature-enable-recipes/per-soc-recipe-and-quantized-weights.md` §*Prune the flags
  that provably do nothing*. It reports sites, not causes: a `LIVE` verdict only says the
  knob is reachable.

- `scripts/cpu-bind-plan-audit.py <serve.log …> | -` — inverts every
  `The CPU allocation plan is as follows:` / `NPU<n>: main=[…]  acl=[…]  release=[…]`
  block of one or more serve logs and prints, per NPU, the pool `cpu_binding.py`
  actually built (`allocate()` trims it: `main = pool[2:-2]`, `pool[0..1]` reserved for
  the sq/cq IRQs, `acl = pool[-2]`, `release = pool[-1]`), the IRQ cores, the
  `[migrate]` NUMA anchor and a `disjoint` / `overlap N cpus` / `IDENTICAL pool` verdict
  per pair of plans — so "two instances bind the same cores" is one call instead of
  eyeballing two 44-number lists. Reads the `[cpu_bind_mode]` line too, and reports
  `Skip binding cpu` as a finding (no plan line at all ⇒ binding never ran). Originating
  case #8600 (`../performance-throughput/cpu-binding-multi-instance-overlap.md`, pin
  v0.18.0rc1 `99e1ea0`, `cpu_binding.py:352-372`). Exit 0 when a plan was parsed, 2 when
  nothing was (say which lines you searched for; do not report "no overlap").
