# Symptoms: failure families, first suspects, worked cases

## Scope

From the observed failure stage to the module that owns it, with the recurring vLLM-Ascend failure
classes and the reports that established them. Load when the report gives a symptom but no
mechanism, or to check a candidate mechanism against a class that has already been analysed.
Method lives in `../workflow/index.md`; paths and flag histories in `../vllm-ascend/index.md`.

## Common rules

- **An operator that cannot get a resource is not a Python bug, and neither is an op that rejects its
  argument.** `EI0007`/"Resources are exhausted" is HCCP QP allocation, `EZ1001` is an aclnn parameter
  check; the deliverable is the vllm-ascend decision that selected that op and its inputs, plus the
  CANN component that owns the pool/contract — not a `vllm_ascend/` line that "raises" it
  (`moe-ep-dispatch.md` §A, `quantization-load.md` §5).
- **A tracing failure is not a device failure.** `attempted to run this operator with Meta` means no
  code reached the NPU: never route it to the DFX/CANN pages, and never read the neighbouring
  `Failed to import from vllm._C` warnings as the cause (`graph-capture-and-custom-ops.md`).
- **Search the framework's symbol, not the model's marketing name.** Hybrid linear-attention layers are
  `MambaSpec`/`mamba_cache_mode` in the transfer code (never `gdn`, `linear_attn` or `state_cache`), and
  a custom op is found through its registration name — an empty grep for the wrong word reads exactly
  like an absent feature (`hybrid-gdn-pd.md`).
- **Choose the suspect from the failure stage, not from the model family.** The last worker log
  line, the deepest `vllm_ascend/` traceback frame, or the accuracy delta picks the page; the model
  name does not.
- **Decide first whether the run stalled, errored, or ran with wrong output.** Each has a different
  evidence source: last worker log line (stall), traceback (error), benchmark delta plus code path
  (accuracy).
- Every entry below is a *starting hypothesis with pointers*, not a lookup table. Re-read the class's
  code at the reported ref — line numbers and even modules move — and confirm with a second signal
  (later fix, unit test, related issue, doc line) before asserting.
- **A maintainer statement is evidence of the workaround, not of the mechanism.** A stale-closed
  issue with a one-line answer is a workaround to relay at best.
- **A silent hang has no traceback; the CANN side does.** For a mid-run freeze the plog/HCCL heartbeat
  (`status[STUCK]`, identical operator count on every rank, the communicator and collective named) is
  the strongest evidence available — read it before any vLLM log line or code path.
- **A message that names a library may be the framework's own.** `ValueError: As of transformers
  v4.44, default chat template is no longer allowed …` is raised by vLLM's `apply_hf_chat_template`,
  not by `transformers`; a grep for it in the named dependency returning nothing is the answer, not a
  dead end. Before routing a message to the project it names, grep for it in the repo that serves the
  request (`tokenizer-and-chat-template.md`). The converse also holds: `... has model type <X> but
  Transformers does not recognize this architecture` names transformers and really is the installed
  transformers version — the config table is missing there, so no `vllm_ascend/` line can own it
  (`model-type-support-floor.md`).
- **A refusal raised while the engine config is built, naming a model architecture or a feature, is the
  framework's own selector — not the plugin and not the model.** `NotImplementedError: VLLM_USE_V1=1 is not
  supported with ['<Arch>']` comes from upstream vLLM's V1 oracle, and its three-value `VLLM_USE_V1`
  semantics mean an **explicitly exported** value raises where an unset one only warns and falls back to V0:
  a report whose own command exports it carries its own workaround. Read the gate (`is_v1_compatible` =
  `not supports_v0_only`), never the `--task`/request flags — `v1-engine-selector.md`.
- **A `site-packages/` frame is evidence about the installed wheel, not about the repo.** Place the
  version from the traceback's own `file:line` and argument list, compare it with the pin in the paired
  vLLM tag's `requirements/common.txt`, and answer with a version/image statement —
  `grammar-bitmask-and-dep-skew.md`.
- **A metadata field named in a traceback is a producer/consumer question, not a shape bug.** Find the
  assignment site and compare its guard with the reader's branch (`pcp-dcp-and-mtp.md` §B/§C).
- **A crash inside `VllmConfig.__post_init__`, before any weight load, is config construction.** The
  failing branch is in `platform.py`'s `check_and_update_config`, and the discriminator is the *shape of
  the model's HF config* (a wrapper such as `Gemma3Config` keeps transformer fields in `text_config`),
  not the model family — `aclgraph-mode-and-capture.md` §C.
- **A memory number in the log is a shape until you divide it out.** "Tried to allocate N GiB",
  "OOM at KV-cache init", "mask too large": invert the arithmetic (`n² × itemsize`,
  `rows × cols × itemsize`, `n × hidden × 2 B`) before choosing a class; a mismatch means another
  class.
- **A `[Doc]` report is answered against the docs tree, not a release tag, and may have no code window
  at all.** "The latest guide" means `docs/source/…` on the default branch (the site's
  `/en/latest/`); a versioned docs URL means that tag's tree. The evidence is the page at the tag the
  image names — flags per hardware tab, the capacity row in §3.1, the §Limitations disclaimers — and
  the defect is a value in a published serve block that the page's own limitations, a sibling page, the
  support matrix or the release notes contradict: cite those lines, never a `vllm_ascend/` frame
  (`doc-guide-vs-capability.md`). The half you cannot verify (device memory state, why a restart after
  an OOM fails) goes into the reply as `hypothesis` with its falsifier, not into another batch. A doc
  report with **no symptom at all** — a `[Community Test]` row ending `测试通过` with zero comments, or a
  `[Misc]` question about duplicated `docker run` steps or a stale `export IMAGE=…:v0.1x.0rcN` pin — is a
  page read, not a fault hunt; enumerating what "might" break is how one recorded run spent its whole
  budget and shipped nothing (`doc-guide-vs-capability.md`, "Nothing failed").
- **"The final release is worse than its rc" is a range question, not a code-window question.** No
  traceback, no bad log line: the answer is the range listing once, the commits that *remove* or
  *re-default* behaviour (`-i --grep='revert\|rollback'`, a diff of `envs.py`/`ascend_config.py`), and the
  ancestry of the PRs the thread names — then the reply, because which commit moved the score needs the
  reporter's benchmark (`../workflow/fix-discovery.md`, "When the report is a same-line regression").
- When the report gives a driver/HDK/CANN version, quote it: several classes are platform-scoped
  (HDK 25.1; "HDK 0.26.0 or earlier"; CANN 9.1.0.BETA1), and a CANN generation decides which branch
  of several gates runs.
- **A `torch._dynamo` error is a tracing failure with no owning frame.** The `from user code:` tail stops
  at the last framework call before the side effect, so the search key is the skipped callable in the
  `Explanation:` line (`posix.putenv` → an `os.environ[...] = …` in the traced region), not the frame
  (`dynamo-graph-break.md`).

## Routes

- `community-test-task-sheets.md` — `[Community Test]` / 众测 task-sheet issues: a sheet copied from
  `推理众测任务书-vllm.xlsx`, a versioned tutorial URL, 交付件/验收标准, and **no failure at all** (all 14 in
  the repo are open with zero comments). The deliverable is a deployment recipe plus the tutorial lines
  that disagree with each other or with the sheet's 蓝区镜像 tag — a hardcoded `v0.11.0rc1` image pin on
  the `v0.18.0` page, an "8×A2 NPUs" requirement beside a TP4 command (#11737).
- `310p-capability.md` — 310P hardware (Atlas 300I DUO / 200I Pro / 310P3): a `[Usage]` "when will the
  310 series support graph mode / quantization" question, a start that dies at `capture_end` with
  `Insufficient_Event_Resources(EL0008)` / `207007` / `507903` on a TP>1 run (the hardware event-id
  ceiling the tag's own 310P page warns about, which `acl_graph.py`'s `207008` marker does not match), and
  a W8A8SC checkpoint directory that asserts in `vocab_parallel_embedding.py:463` because its per-rank
  `model-rank-N-part-0.safetensors` files were loaded without `--load_format sharded_state`. Carries the
  page-path moves (`single_node_300i.md` → `hardwares/310p.md`, both gone/rewritten by `v0.30.0`), the
  doc's per-model capture-size lists (stated per TP) and the explicit `--max-model-len` rule; the
  first-tag support timeline is `../vllm-ascend/hardware-support-by-release.md` and the loader half is
  `quantization-load.md` §8.
- `symptom-index.md` — the routing table: symptom in the report → first places to look → phase of
  failure, plus the hardware/version caveats that decide whether it is a new bug at all. Its
  `symptom-index-stages.md` companion carries the same routing by failure **stage** (what was happening
  when it failed) for reports that say where they died rather than what the message was.
- `../hccl/index.md` — the comm layer as its own owner when the engine never serves and every message is
  a transport one: `Getting socket times out`, `EJ0002 … The network port is down`, `ra rdma init failed,
  devid[..] ip[..]`, `listen on ip[..] port[..] fail`, `wait socket establish timeout`, `CreateCommPlane
  failed, tag[…_MC2MultiServer]`, `Task run failed`/`Notify record failed`, `Load graph failed`. Those are
  HCCL/HCCP with their own EI codes (EI0006/EI0019/EI0001/EI0002), documented in the `cann/hccl` clone,
  and no `vllm_ascend/` line owns them. Says which plane failed (the device NIC's RoCE IP is not the host
  `HCCL_IF_IP`), why `--enable-expert-parallel` on one 16-device node builds a multi-server communicator,
  the two-question environment-vs-plugin test, the levers, and the verified TorchAir removal at
  `v0.12.0rc1` (`init-failures.md`, #1955); its `env-support-matrix.md` decides whether the launch script
  can support the reported topology (`HCCL_INTRA_PCIE_ENABLE` is intra-server only,
  `HCCL_OP_EXPANSION_MODE=AIV` has per-product limits — #18315).
- `pooling-model-startup.md` — an embed/pooling model that never serves: the two start-up classes
  (`VLLM_USE_V1=1 is not supported with [...]`, upstream vLLM's oracle; the no-KV-cache `AscendScheduler`
  `IndexError`) and how to tell them apart. The mechanisms are `v1-engine-selector.md` and
  `ascend-scheduler.md`.
- `quantization-load.md` — load-time failures, by cause: dense bf16 load of a quantised checkpoint
  (startup OOM, §1); the quant-description `KeyError` (§2) and its FP8/compressed-tensors twin whose
  missing key starts with `visual.` (§2b — the HF `quantization_config` dict read as if it were a
  modelslim description, #4313); a quantised checkpoint loaded into an unquantised model
  (`KeyError: ... .weight_scale` for an FP8 artefact, `... .deq_scale` for a W8A8 one, §3 — where the
  parameter name identifies the linear method that ran, the modelslim W8A8 hub directory's `config.json`
  carries no `quantization_config`, and the page shows how to prove that the `--quantization` flag never
  reached the process); an incomplete weight shard
  (`SafetensorError ... file not fully covered`, §4 — the deepest callee is a third-party file reader,
  so the file is wrong, not the code); the operator-rejected weight layout (§5: int4 converted to
  FRACTAL_NZ with `group_size 256`, which aclnn accepts only at 64/128); and the non-failure where one
  projection of a "W8A8" model runs `aten::matmul` because the description marks it `"FLOAT"` (§6, #979
  — no flag can fix it); the same KeyError on a `≤ v0.7.3rc2` build, where `get_config_filenames()`
  declares no description filename at all, so the file is never read and the dict indexed comes from the
  HF config (§2c, #569 — fixed by #743 from `v0.7.3`); and an *accuracy* report for a quantised **variant
  the ref's tutorial does not list** (§7, GLM-5 w8a8 at `v0.17.0rc1`, #7507) — a validation/pin answer
  read off the model page's Model Weight list before any quant kernel; and a **shard-format** mismatch
  (§8, #11406): a pre-sharded `model-rank-<r>-part-*.safetensors` 310P `w8a8sc` artefact served with
  `load_format=auto` dies on `assert loaded_weight.shape[output_dim] == self.org_vocab_size` at
  `vocab_parallel_embedding.py:463`, and the lever is the doc's own `--load_format sharded_state`, which the
  reporter's argv omitted. Also **§9: a scheme that no longer exists** — a body comparing two tags by
  line count or class name is a declared removal, not a regression (`W4A8_DYNAMIC` linear: last
  registered at `v0.26.0rc2`, removed by `efda0446d` from `v0.27.1rc1`, announced in the `v0.26.0rc1`
  release notes; replacement W4A4 FlatQuant/LAOS) — answer with `git grep register_scheme` per tag and
  the `NotImplementedError` the missing pair now raises, never with a load-path hunt.
- `multimodal-input-not-applied.md` — a VLM on a healthy engine answers as if the image were not
  there (wrong caption, generic answer, or symbol soup right after `<image>`): the three members in
  the order worth checking — the request dict's `multi_modal_data` key (the only spelling vLLM reads,
  and the rest of the dict is never validated, so a misspelling silently selects the text-only
  branch), the placeholder token the checkpoint expects against the one the prompt carries, and the
  `--limit-mm-per-prompt` / encoder-cache budget. The evidence is `usage.prompt_tokens` growth plus
  the encoder-cache line, never the echoed prompt; the worked case is #2040 (one letter,
  `multi_model_data`).
- `tokenizer-and-chat-template.md` — the chat endpoint 500s with `ValueError: As of transformers
  v4.44, default chat template is no longer allowed ...` while raw completion works: vLLM's own guard
  after four template priorities failed, the tokenizer files of the served directory are the source, and
  `--chat-template` is the version-independent lever (#6345, duplicate #5353). The neighbouring
  `Tokenizer class <X> does not exist or is not currently imported` is not this page — it is the
  tokenizer twin in `model-type-support-floor.md`.
- `reasoning-parser-output.md` — the thinking is empty, mislabelled or lost while the service is healthy:
  `reasoning_content` is not the response field from vLLM `v0.16.0` on (`reasoning` is; an echoed
  `reasoning_content` in the request history is silently ignored until `v0.22.0rc1`), a parser *moves* the
  thinking out of `content` rather than deleting it, the prompt state decides whether anything was generated
  (`is_reasoning_end`), `include_reasoning`/`reasoning_effort: none` can discard a correct parse, a
  `KeyError: Reasoning parser '<name>' not found` means a tool-parser name was passed, and a turn-2 loss in
  a tool-call loop is the prompt (only the assistant message's `reasoning` key survives,
  `chat_utils.py:1511`; the interleaved-thinking table lists `kimi_k2`/`minimax_m2` only). Includes the
  apparent-TTFT measurement that replaces an engine hunt. `scripts/reasoning_output_probe.sh` prints all of
  it for a tag pair (#8022, #9659, #9355).
- `v1-engine-selector.md` — startup refused at engine-config time with `NotImplementedError:
  VLLM_USE_V1=1 is not supported with ['<Arch>']` (or naming a feature flag), deepest frame in
  `vllm/engine/arg_utils.py`: upstream's V1 oracle, its three-value `VLLM_USE_V1` semantics (explicit →
  raise, unset → warn + V0 fallback, `=0` → V0), the `is_v1_compatible` = `not supports_v0_only` gate and
  the architecture→class registry map at vLLM `v0.9.2`, the upgrade that removes the limit (`v0.10.1rc1`,
  #21270), and the version range in which this message exists at all (≤ `v0.11.0`; gone at `v0.11.1`,
  #28116) — #2169.
- `aclgraph-mode-and-capture.md` — failures decided by `compilation_config` graph settings rather than
  model math, members **A**–**F**: **A** `FULL_DECODE_ONLY` killing the engine at request time on a model
  whose backend has no working full-graph decode path at that release (Qwen3 / DSv3 at `v0.11.0rc1`;
  rewritten in `v0.12.0rc1` by #3970, #4706), with the `UNIFORM_SINGLE_TOKEN_DECODE` fact that the
  runner only downgrades the *mixed* mode, so a broken decode path stays reachable (#4326); **B** the
  FULL-decode `query_start_loc` padding writing `num_input_tokens − num_reqs` values into a
  `max_num_seqs + 1` buffer because `num_input_tokens` is the DP-wide maximum
  (`RuntimeError: The expanded size of the tensor (63) must match the existing size (6151)`, #5076,
  fixed in `v0.13.0rc1` by #4963/#5164); **C** the derived capture sizes reading a nested multimodal
  `hf_config` (`AttributeError: 'Gemma3Config' object has no attribute 'num_hidden_layers'` at config
  validation, `--enforce-eager`, fixed by `d1c640841` from `v0.10.0rc1`, refined for MTP drafts by
  #7064; #1183/#359); **D** garbled decode only in graph mode on a DP rank with DSA-family attention,
  where the dummy batch writes through released KV slots (§D, #17593, three shas); **E** the legacy
  `full_cuda_graph: true` boolean of `≤ v0.9.x` only — a post-startup freeze whose py-spy frame is
  `graph_task_update_end`, the per-layer task-update loop the flag alone reaches (drop the flag or
  `--enforce-eager`; gone from the fork at `v0.10.0rc1`, §E, #4180); **F** `cudagraph_mode: "FULL"` on a
  hybrid linear-attention model (Qwen3.5/Qwen3-Next GDN) with no crash and no wrong output — upstream's
  mode resolver rewrites mixed `FULL` to decode-only when the backend's `cudagraph_support` is below
  `ALWAYS`, and the fork's empty `splitting_ops` makes that fallback decode-only, so prefill stays eager
  and the profiler gaps are the downgrade rather than a defect (§F, #11947); **G** a worker death inside
  `execute_dummy_batch`/`_dummy_run` on a 2-node PD mixed deployment (`FULL_DECODE_ONLY`, GLM5-class,
  aivec `DDR address of the MTE instruction is out of range` + `TransposeKvCacheByBlock` plog errors) —
  three padded counts must agree with their source lists, instances #7290 and #7460 (§G, #7401) — while
  the *same* aivec line in an `--enforce-eager`, no-PD run is not §G at all (`device-kernel-faults.md`,
  the aivec sibling section); **H** a
  310P/300I DUO *start* that exhausts capture **events** at `capture_end` (`EL0008`/`207007`, then
  `507903`) while the tag's own handler and FAQ know only the stream code `207008`/`EL0009` (§H, #12989).
- `startup-and-stall.md` — the startup stage as a whole: read the stall signature, "a hang is not an
  error", the KV-cache-allocation OOM right after weight load (#3308), the "each card looks full /
  the whole model on every card" accounting half (§5, #1316 — `usage ≈ util × total` by construction,
  or one process per device), and two non-worker shapes: §6 a multi-node DP launch whose per-node
  `--data-parallel-size-local` values do not add up to `--data-parallel-size` (silent head-node
  handshake, a DEBUG-only `wait_for_engine_startup` poll, so nothing prints; #5942), and §7 an
  `init engine … took N seconds` that regressed between two tags (#3045 — the timer wraps
  `compile_or_warm_up_model`, and a deleted "ACL Graph does not support <family>" block is what puts the
  model on the capture path); the page otherwise routes into the startup-stage classes. A *serving* hang
  that begins at a prompt-length threshold is not this page: `kv-admission-cliff.md`.
- `sleep-mode-and-allocator.md` — wrong output (or a replay `AttributeError`) on the **first** call only,
  with `enable_sleep_mode=True` and no `sleep()`/`wake_up()` ever called: the flag selects an allocator, not
  a lifecycle, so the offline `LLM` path enters it on every run. Read the reporter's
  `enable_sleep_mode` × graph mode × TP matrix first — sleep mode is the necessary trigger and the graph
  mode only decides crash-vs-garbage; the code boundary is `v0.18.0` (`camem.py` only,
  `is_sleep_mode_available() → True`, `PYTORCH_NPU_ALLOC_CONF` skipped) versus `v0.23.0`+
  (`SleepWakeupManager`, #9585); what the allocator promises (tags, `sleep` discards untagged memory,
  `sleep_persistent_tag`) and the documented sleep/wake contract that "enabled but never slept" sits
  outside of; the async-H2D candidate #12079 is already in `v0.23.0`'s tree, so "absent fix" is not the
  mechanism (#15560). Lever `enable_sleep_mode=False`.
- `kv-admission-cliff.md` — no traceback and no stall signature, yet prompts above a token count never
  return and every later request (even 7 tokens) is never answered until restart: a scheduler
  **admission** cliff, where `allocate_slots` fails, the request stays in `waiting` and head-of-line
  blocks the queue. Compute the cliff (`num_blocks × block_size` of the smallest-page group) against the
  `max_admission_blocks_per_request` cap, note that vllm-ascend overrides upstream's manager factory, and
  use `--max-num-batched-tokens` as both the lever and the confirmation test; a startup
  `Maximum concurrency … 1.10x` line is the byte-ratio estimate and does not refute the class
  (#16365 / #15984; the kpool tail-cache variant is `v0.28.0.rc1`+).
- `deepseek-v4-thinking-and-effort.md` — a DeepSeek-V4/V4.1 benchmark below the published number on a
  reasoning dataset (GPQA, AIME) while the service is healthy: the frontend patch
  `patch/platform/patch_deepseek_v4_thinking.py` decides `thinking_mode` and `reasoning_effort` for every
  request (`:90-93` at `v0.20.2rc1`, no key = chat mode; any effort string → `"high"`; imported
  unconditionally), so read the eval config before the kernels. The score signature (MMLU-Pro matching
  while GPQA lags), the `thinking: True` / `reasoning_effort: high|max` workaround and its output-length
  cost, and the `#13519`/`#14317` lineage that makes the no-flag case thinking (first tag `v0.25.1rc1`);
  an eval client passing `thinking: False` is config, not a defect (#10413).
- `benchmark-accuracy.md` — the **score** is wrong while the service is healthy: two harnesses disagree on
  one deployment (prompt shape, default sampling, metric, `max_tokens` — in that order; harness vs the
  model's official number is the only comparison that decides anything, #10068), and an accuracy collapse
  that tracks DP>1 with `--enable-expert-parallel` on, where `platform.py` has already replaced
  `additional_config.expert_tensor_parallel_size` and `ep_size` spans the DP replicas (#1791 → #1289).
  Metadata the other class pages cannot supply: the sampler values the client never sent (vLLM's
  `--generation-config auto` fills them from the checkpoint's `generation_config.json`), the harness's
  meta file at its own version (evalscope `mmlu_pro.json` requires a terminal `ANSWER: [LETTER]`, so a
  truncated CoT scores a correct model wrong), the raw-generation request that separates prompting from
  repetition, and how to read a foreign harness file through the `github` tool. §4 is the other wrong-score
  shape that is not a score at all: an image prompt answered with garbage because the caller's dict key is
  not `multi_modal_data`, which vLLM drops without validation (#2040). The DeepSeek-V4 thinking default is
  a different page (`deepseek-v4-thinking-and-effort.md`).
- `attention-mask/` — the quadratic attention mask as its own memory suspect: sizing rule,
  rows-vs-columns distinction, the runtime gate that picks the dense path. Pages:
  `attention-mask/prefill-dense-mask-a2-a3.md` (A2/A3, OOM mid-prefill, #3357) and
  `attention-mask/310p-dense-mask.md` (310P, stall at startup, #10872).
- `tool-call-streaming.md` — the OpenAI tool-call *shape* is wrong while the service is healthy: a
  streaming `tool_calls` entry with no `id`, `function.arguments` an empty string or a one-character
  tail, the last fragment in the `finish_reason: "tool_calls"` chunk, a zero-argument call that never
  streams, or the inverse — an extra whitespace-only `content` delta (`"\n\n"`) before the first call
  (§4b, the bundled upstream `qwen3_xml` parser, #15911). Decide delta-vs-whole-call first
  (`merge_tool_deltas.py` on the raw stream; a `DeltaToolCall` needs only `index`, a non-streaming
  `ToolCall` always has an `id`), then the three
  owning layers — `vllm_ascend/patch/platform/patch_*tool_call*`, the upstream `vllm/tool_parsers/*`,
  and `chat_completion/serving.py`'s terminal remaining-args delta — with the patch tag map and the
  worked cases #11788 (terminal empty delta) and #7682 (tail-only streaming); §4b is the third shape — a
  **whitespace-only `content` delta** before the first tool call, emitted by the bundled upstream
  `qwen3xml_tool_parser.py`, which is a vLLM-tag file the plugin does not own (#15911). Two further
  members: internal markup (`</think>`, `<tool_call>`, `<arg_key>`/`<arg_value>`) left in `content` or
  inside `function.name`, which is **parser selection** — the serve command is the first input, not a code
  window (§4c, #12398) — and an agent client's `JSON Parse error: Unterminated string` whose differential
  is the spec-token count (§3/§7, #7580 and its `v0.17.0rc1` twin #7682).
- `grammar-bitmask-and-dep-skew.md` — failures inside guided decoding / tool calling where the frame is
  in `site-packages/xgrammar` (or `outlines_core`, `llguidance`): the Ascend CPU round-trip that makes a
  library's CPU branch NPU-only, the tensor-vs-list `indices` mismatch and its traceback fingerprint, and
  the rule that a `site-packages` frame is evidence about the image, not the repo (#5894). §D is the
  other owner: `AssertionError: (-1, '<req>', {'<req>': [-1, -1, -1]})` / `Failed to advance FSM … for
  tokens -1` out of `vllm/v1/structured_output/` is a **scheduling placeholder** reaching the grammar
  under MTP plus structured output — filled by the Ascend recompute scheduler
  (`additional_config.recompute_scheduler_enable` is the lever) or by the upstream async-scheduler
  placeholder at vLLM `v0.13.0`, skipped by construction from `v0.14.0` (#6421).
- `metrics-counters.md` — the serve process dying in the metrics writer
  (`ValueError: Counters can only be incremented by non-negative amounts`): upstream `PromptTokenStats`
  accounting, where a connector's external-computed tokens can exceed the request's cached snapshot; the
  clamp and `PrefillStats` lineage, the upgrade/disable-pool ladder, and why the cited PR's title looks
  unrelated (#8664).
- `startup-port-conflicts.md` — `zmq.error.ZMQError: Address already in use` at multi-node worker
  startup: vLLM core's probe-then-bind port race, not a KV connector or an HCCL setting.
- `graph-capture-and-custom-ops.md` — failures while a graph is traced, captured or replayed, three
  shapes: a missing meta/fake kernel (`NotImplementedError: _C::<op> ... with Meta`, #1895, fixed by
  `c0f0b7081`/#2113 from `v0.10.1rc1`); an op the SoC's op store lacks (ATB `_npu_rotary_embedding` on
  310P, `EZ3002 Optype [TransData]`, #1383; and §B2's `_warm_up_atb` — `LinearOperation CreateOperation
  failed!` on every 310P start `≤ v0.12.0rc1`, skipped from `v0.14.0rc1`, #5170); and **§C** a host sync or
  copy issued *inside* the
  ACL-graph capture window (`107041` / `ACL_ERROR_RT_CAPTURE_MODE_BLOCK_ASYNC`, `EE1016`,
  `rtMemcpy … the current capture mode does not support this operation`), whose CANN rules live in
  `cann/runtime docs/zh/FAQ/ACLGraph_restrictions.md`; §C also fixes where a `torch.npu.synchronize()`
  frame sits (the replay branch, not the capture) and separates it from a same-log `507015`
  (#6529). A mode-level support gap — `{"cudagraph_mode": "FULL_DECODE_ONLY"}` on a family the
  tag cannot capture — is not this page: `aclgraph-mode-and-capture.md` §A, and a `torch._dynamo` error
  whose `Explanation:` names a CPython builtin is `dynamo-graph-break.md`.
- `dynamo-graph-break.md` — the *other* tracing failure: `torch._dynamo.exc.Unsupported: Attempted to
  call function marked as skipped` during the dummy/profile forward, because code under `torch.compile`
  performs a CPython side effect — `os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"` inside
  `AscendMRotaryEmbedding.forward_triton` (`ops/rotary_embedding.py:497-500` at `v0.18.0rc1`). The
  traceback names no owner: read the skipped callable (`posix.putenv`) and grep for the write. Three
  levers, one per condition of the guard (pre-set the env var, keep the per-forward token budget under
  65535, or `--enforce-eager`); present from `v0.15.0rc1` (#6301) through `HEAD` (#8224).
- `missing-downstream-feature.md` — a feature upstream vLLM already supports that vllm-ascend's copy of
  the same file has not plumbed: the first request raises a bare `TypeError`/`AttributeError` on a field
  (`prompt_embeds`: `TypeError: object of type 'NoneType' has no len()` in
  `CachedRequestState.__post_init__`, #4505). The field plus the `num_prompt_tokens` derivation differ
  between `vllm/v1/worker/gpu_input_batch.py` and `vllm_ascend/worker/npu_input_batch.py`, the API forbids
  passing token ids beside embeds, and the support lands in `v0.12.0rc1` (#3026, `--enable-prompt-embeds`,
  which force-disables prefix caching). Also the path-move trap: the same field lives in
  `worker/v1/model_runner.py` at HEAD. Status is answered by `git log --reverse -S'<feature>'` +
  `git tag --contains`, not by reading `HEAD`.
- `startup-and-stall.md` §3b — a copied tutorial block whose `--max-model-len` cannot fit the KV budget
  on the reporter's hardware: upstream `kv_cache_utils._check_enough_kv_cache_memory` prints the
  estimated maximum length, so the doc `file:line` and the corrected block are the deliverable, and the
  post-OOM recovery half is operator-owned (#18158); `doc-guide-vs-capability.md` has the doc lines
  and the matrix row.
- `fa3-backend.md` — the run selected the **FA3** backend (`attention_config={"backend":
  "FLASH_ATTN"}` **and** `VLLM_BATCH_INVARIANT=1`, a non-MLA/SFA model, `flash_attn_npu_v3` importable; the
  startup log's *"FA3 will not be enabled when not in training-inference consistency scenario"* is the
  discriminator) and dies with `507015` on a batch that mixes decode with prefill: FA3 slices token-indexed
  `query` and request-indexed `block_tables`/`query_start_loc`/`seq_lens` in the same call, so
  `num_speculative_tokens > 0` makes `num_decode_tokens != num_decodes` and the prefill call gets rows and
  offsets that belong to no request (worked case #17991, `fa3_v1.py:132` at `v0.23.0rc1`, fix #18068 on
  `main` only — no tag through `v0.30.0`). The frame is a stream sync, not the faulty call; the page carries
  the three-owner split of `507015`, the eager-vs-capture discriminator (so a device-looking crash under FA3
  is not sent to `device-kernel-faults.md`) and the three-hunk patch.
- `device-kernel-faults.md` — device DFX kernel name/hash plus a runtime error (e.g. `507015`,
  "L0C read/write conflict"): reading order, the 310P W8A8 NZ layout that selects the failing kernel,
  which tag first offers the 310P model-runner V2, why the fix is a CANN release, and two siblings of
  the same owning layer — `not found in this op store` (the op-class is missing for the SoC) and
  `call aclnn<Op> failed … AclOpKernelInit failed opType` (the image's CANN base is another SoC's build,
  e.g. a `910b` base on a 910C/A3 card, #1696; `Dockerfile`/`Dockerfile.a3`/`.310p`/`.a5` decide it, and
  §1 of `../vllm-ascend/install-and-image.md` carries the per-ref inventory). Its message-only sibling is
  the bare aivec `DDR address of the MTE instruction is out of range`: a family of 57 fork issues, where
  the `fault kernel_name=…` line from the plog is the only localiser, and a report without it (an eager
  run, a screenshot, a maintainer "use the latest release" close — #3911) ends at the plog request plus
  that attributed lever, not at a code read.
- `pd-collective-hang.md` — the mid-run silent freeze under PD/multi-DP: HCCL `STUCK` at one operator
  count, the skipped DP all-reduce, and the per-rank MoE collective choice (`select_moe_comm_method`)
  that turns a token-count imbalance into a mismatched collective; plus the neighbouring shape that is not
  a hang — the `≤ v0.11` `llmdatadist` producer holding its blocks until
  `VLLM_LLMDD_ABORT_REQUEST_TIMEOUT` (mean TTFT in minutes, #3062), whose mechanism lives in the page
  below.
- `pd-llmdatadist-timeout.md` — PD on a build whose `--kv-transfer-config` names
  **`LLMDataDistCMgrConnector`**/`llmdatadist` (the 0.9–0.11 connector): the prefill logs
  `Some requests in prefill node fail to receive KV Cache transfer done signal in 120s` at
  `llmdatadist_c_mgr_connector.py:886` and the user-facing symptom is a very slow TTFT, not a crash. The
  prefill parks the request's computed blocks in `_reqs_need_send` with a
  `VLLM_LLMDD_ABORT_REQUEST_TIMEOUT` deadline (`envs.py:140-141`, default 120 s) and only a decode-side
  `ReqForFinished` — one distinct `tp_rank` per decode TP rank — releases it, so the timeout sits in the
  request's critical path. The module lives in `distributed/`, **not** `distributed/kv_transfer/`, and is
  gone from `v0.12.0rc1` (#4780), so the durable lever is the connector choice. #3062.
- `mooncake-transfer-errors.md` — the PD KV-transfer *failure* (as opposed to the hangs elsewhere): PD up,
  both roles ready, the **first request** failing in the connector with
  `Failed to connect to target: <engine-name>, status: 103900` (`adxl_compat.h` `PARAM_INVALID`, ADXL
  connect stage) and/or `Sync batch data transfer timeout after <N>ns` with `WAITING_FOR_REMOTE_KVS`
  repeats (data stage). `1039xx` is the ADXL table, not CANN/HCCL; the printed name is each side's
  registered transport-engine name, and `use_ascend_direct` is read twice with opposite defaults at
  `v0.11.0` (`config_data.py:441` False, `mooncake_connector.py:906` True), so one side advertises a name
  the peer does not know. Maintainer lever is `"use_ascend_direct": true` in `kv_connector_extra_config`
  plus `/etc/hccn.conf` and the `ASCEND_BASE_PORT` range; advice only, mechanism `hypothesis` (#5635).
  A third, easily merged shape is the **store child** of a `MultiConnector`: `503900`/`103902` and
  `Failed to put key …, res:[-800]` from `kv_pool/ascend_store/backend/mooncake_backend.py:85` while
  requests still answer `200` — a failed put that revokes the block, not a worker death (#7401).
  A `[Usage]` question about the Mooncake **store** backend (support, and whether
  `-DUSE_ASCEND_DIRECT=ON` can be skipped) is not this page's transport story: it is the pool backend's
  `protocol` gate, `kv-pool-store-hang.md` §D (#5299).
  The same page's **"The silent sibling"** is the accuracy half — a P/D + MTP deployment that serves and
  answers fluent garbage: the producer ships every computed block id while decode allocates only the
  prompt's, so the extra MTP block misaligns the positional mapping at block-boundary prompt lengths
  (`mooncake_connector.py:1065` at `v0.18.0`; cut by #8908 `015af05d6`, and for the hybrid child by #9810
  `31860b0d7`, first tag `v0.20.2rc1`; #9430, reporter-confirmed 「v0.20的版本问题解决了」).
- `prefix-cache-granularity.md` — "caching is on and identical requests never hit" as a unit mismatch:
  the DeepSeek-V4 compressed KV group (`block_size × compress_ratio`, write path vs lookup path, the
  `--block-size 32/64/128` change and its measured hit-rate table), SWA-layer retention, and the PD
  fan-out question. The CP multiplier stays in `pcp-dcp-and-mtp.md` §A; this page is the other members.
- `c8-and-kv-layout.md` — the two C8 sub-families. **Sparse** C8 (`enable_sparse_*_c8`, DSA models):
  `store_kv_block` hangs and garbled first output under PD pooling, plus the separate E8M0 K-scale
  corruption class (`#19064`: NaNs and repeated-exclamation output that survives the reshape-optim
  switch). **Dense** C8 (`enable_c8_quant` from the checkpoint, GQA models): with EAGLE/MTP the captured
  decode path hands FIA a per-request KV-length array against a per-token q batch and the operator
  refuses (`561002` / `EZ9999 ... actual sequence lengths size[N] ... q batch[M]`), because the BNSD
  conversion in `full_graph_fia` lacks the `[:batch_size]` trim the eager path has (#12956,
  maintainer-declared unsupported).
- `kernel-launch-limits.md` — `coreDim=<N> can't be greater than UINT16_MAX` /
  `rtKernelLaunch ... [invalid value]` from a Python-level Triton kernel: launch geometry that grows
  with the scheduled tokens, why the operator name and the traceback frame disagree, the vendored
  `MAX_CORES` fix, and the tokens-per-step lever.
- `pooling-and-embedding.md` — pooling/embedding models with wrong vectors while the service is
  healthy: the pooling branch that bypasses the KV cache, the prefix-caching interaction, the
  reporter-confirmed `--no-enable-prefix-caching` workaround and the fix's tag containment.
- `ascend-scheduler.md` — the fork's **own scheduler subclasses**, all config-selected and all off by
  default. (a) `AscendScheduler` (`ascend_scheduler_config.enabled`, `v0.10.x`–`v0.11.x`, dropped before
  `v0.12.0rc1`): the marker warning `Using configured V1 scheduler class
  vllm_ascend.core.scheduler.AscendScheduler`, the pooling-model crash `IndexError: tuple index out of
  range` at `core/scheduler.py:209` (zero KV-cache groups → `create_empty_block_list()` is an empty tuple →
  unconditional `.blocks[0]`), the `max_num_batched_tokens < max_model_len` `ValueError` (#3462). (b)
  `SchedulerDynamicBatch` (0.19–0.23, `SLO_limits_for_dynamic_batch != -1`):
  `TypeError: SchedulerDynamicBatch.__init__() got an unexpected keyword argument 'hash_block_size'`,
  because the subclass never took upstream's new keyword while `scheduler_profiling_chunk.py` at the same
  tag did; removed by #11933 in `v0.24.0rc1` (#18688). (c) `BalanceScheduler`
  (`patch/platform/patch_balance_schedule.py`, env `VLLM_ASCEND_BALANCE_SCHEDULING=1`, monkeypatched over
  `vllm.v1.core.sched.scheduler.Scheduler`): **no error** — a guided/structured-output request is born in
  upstream's `skipped_waiting` queue (`Request.__init__` sets `WAITING_FOR_FSM`) and the copied
  `schedule()` only reads `self.waiting`, so the stream never emits a first token; fixed by #10509/#10559/
  #10699, first tags `v0.22.1rc1`/`v0.23.0rc1` (#8328). The pooling *accuracy* class stays in
  `pooling-and-embedding.md`; this page is the crash.
- `flashcomm1-and-moe-reduce.md` — the collective-reduction plumbing: an FC1/SP warm-up or
  cudagraph-capture crash whose `src`/`dst` row ratio equals the TP size, a MoE TP all-reduce applied
  twice so a healthy server answers fluent nonsense, and a deployment whose own official config
  enables FC1 while the reporter does not (#10503, mechanism open).
- `lora-and-tp.md` — LoRA on Ascend with TP>1: which kernel family the run uses (`--max-lora-rank` ≥ 128
  or 310P switches to the upstream torch ops; the module path itself moves —
  `lora/punica_wrapper/punica_npu.py` at `v0.9.1rc1`, `lora/punica_npu.py` later), the `hidden in should
  be smaller than hidden out` expand contract and its `lora_rank = x.size(1)` consequence, why the
  default MCP path gathers `TP × rank`, the workaround ladder, why the nearest Qwen3.5 LoRA-TP commit is
  a lead and not the fix (#16586), and the cheap shape where the adapter is simply slow: at
  `v0.9.1rc1` the wrapper *is* the upstream torch implementation, so 2–3 tok/s is expected until the
  custom kernels (#2325) and their aclgraph capture (#2113) land in `v0.10.1rc1` (#1629).
- `hybrid-gdn-pd.md` — hybrid linear-attention (Qwen3-Next/Qwen3.5/Qwen3.8) under PD or a KV pool:
  the `MambaSpec` naming trap (`mamba`, never `gdn`), which connector carries the linear-layer state per
  tag, and #8920's closed-without-diagnosis thread.
- `layer-name-keyerror.md` — `KeyError: 'model.layers.<n>.self_attn.attn'` out of upstream
  `get_layers_from_vllm_config` before any forward: route by the *caller's* frame — the attention-backend
  init in `model_runner_v1.py` (a rank/stage scope mismatch, PP reports, open) or a connector's
  `register_kv_caches` (a layer-class filter; goes to `kv-connector-registration.md`). Also carries the
  vLLM lookup body itself and the `1483`/`1486` line fingerprint.
- `kv-connector-registration.md` — a connector that dies **before any forward**, at KV-cache
  registration: `KeyError: '<layer>'` (upstream looks layers up by `Attention`, which MLA layers are
  not; fixed by `ca1954d58c`/`#37090`, first tag `v0.17.2rc0`), an assert on a layer/backend map, or an
  unsupported cross-layer KV layout. Includes which connectors vllm-ascend actually replaces, the MLA
  vs GQA discriminator, the reading order, and the maintainer answer that points at
  `AscendStoreConnector` instead (#7971). Its last section separates the same layer-name `KeyError`
  when **no connector is configured**: `initialize_attn_backend` → `get_layers_from_vllm_config` indexing
  `static_forward_context` with a layer list this PP rank did not build (#6678, open).
- `multi-connector-composition.md` — a `--kv-transfer-config` of `MultiConnector` pairing a **layerwise P/D**
  connector with a **KV-pool** connector (`MooncakeLayerwiseConnector` + `AscendStoreConnector` and
  siblings): `AssertionError: len of local trans block ids : 0 not equal to the len of remote trans block
  ids : 2` at `mooncake_layerwise_connector.py:758` (the fn name is spelled `add_tranfer_task`). Upstream
  `MultiConnector.update_state_after_alloc` gives every non-chosen connector `blocks.new_empty()` and `0`,
  which starves a transport connector; the fork only owns this composition from the release whose
  registration binds `MultiConnector` to `AscendMultiConnector` (#7032 `675387f1f`, first tag
  `v0.16.0rc1`), and the user's config is unchanged by the upgrade.
- `kv-pool-store-hang.md` — connector stalls and deaths, split by which half died: **nothing is ever PUT**
  (store master `Keys: 0` / `PutStart=0/0` while the node's KV cache climbs, `--speculative-config` set →
  the runner skipped `finalize_kv_connector`, #8808); **the load never completes** (the repeating
  `kvpool hit tokens / need to load` line, `distributed/kvpool/` →
  `distributed/kv_transfer/kv_pool/ascend_store/` path move, #6512); and **the worker dies during the
  load** — one `KV pool load spec created … load_async=False` line, then
  `Worker proc … died unexpectedly` with no traceback and no core, because `load_async=False` means
  `start_load_kv` calls the backend `get` inline on the runner thread (§C, #16342). Read `load_async` on
  the reported line before anything else: `False` is a death, `True` is a stall.
  §D is the **config** half: the store backend's `MooncakeBackend` drives KV bytes through ADXL/HIXL and
  refuses any `mooncake.json` `protocol` but `ascend` — `UnboundLocalError` on `ret` at `v0.12.0rc1`
  (`distributed/kvpool/backend/mooncake_backend.py:32-48`), an explicit `NotImplementedError` from
  `v0.19.1rc1` (#8514), with `-DUSE_ASCEND_DIRECT=ON` therefore mandatory rather than an optimisation
  (#5299). §E holds two shapes. (a) The **no-fault** half, which is not a bug at all: hits are positive,
  `need to load` is `0`, throughput is below the no-connector baseline, and the clamp
  (`need_to_allocate = 0` when the external hit is already in the local prefix cache) is upstream vLLM's own
  logic — the lever is a build with the lookup-cost optimisation (#12783 `9e45da191`, first tag `v0.23.0`)
  or dropping that child (#13356). (b) **The store that never initialises**: a bare `AssertionError` at
  `backend/memcache_backend.py:95 assert res == 0` after `store.init(self.local_rank, init_bm=…)` returned
  nonzero from the vendor `memcache_hybrid` — no message in the assert, so the answer is the vendor versions
  and a `pp=1` A/B, and any cited fix PR needs `merge-base --is-ancestor` first (#13418: three of the four
  named PRs are already in `v0.23.0`).
- `model-type-support-floor.md` — serve or `LLM(...)` start dies before any weight load with
  `ValueError: The checkpoint you are trying to load has model type `<x>` but Transformers does not
  recognize this architecture` (or a `CONFIG_MAPPING` `KeyError`), frames through
  `vllm/renderers/hf.py` → `vllm/tokenizers/hf.py` → transformers `AutoConfig.from_pretrained`: a
  **dependency floor**, not a code path. The tokenizer twin
  (`ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported`, #8213)
  is the same floor on the checkpoint's `tokenizer_class` string. Find the registration commit in the
  transformers clone (the name table moved from `configuration_auto.py` to `auto_mappings.py` at v5.6.0,
  so a one-file grep reads as absence), its first release tag, then the three-pin check (image, repo pin,
  paired vLLM's `requirements/common.txt` cap) at the reporter's tag (#8315: `>=4.57.4` shipped 4.57.6,
  predating `qwen3_5` by three weeks while vLLM still capped `< 5`). The **processor** twin is a `TypeError`
  rather than a `ValueError` — `Glm4vProcessor.__init__() got multiple values for argument 'tokenizer'` from
  transformers `v4.53.0`'s `from_args_and_dict` (positional args removed by name, so `video_processor` slides
  into the `tokenizer` slot) while vLLM passes `tokenizer=` because its guard compares the version string to
  exactly `4.53.0` (`vllm/inputs/registry.py:133-141` at `v0.9.2`); fixed in `v4.53.1` (`a10c2bf840`,
  #39125), reporter-confirmed by a transformers upgrade (#1768).
- `pd-proxy-examples.md` — PD is up but the client-facing proxy misbehaves (P engine shuts down after
  the first request, requests never return): the two examples in `examples/disaggregated_prefill_v1/`
  implement *inverted* flows — prefiller-first (`do_remote_decode`, `remote_*: None`) for
  `MooncakeConnector`, decoder-first with a proxy `/v1/metaserver` callback for
  `MooncakeLayerwiseConnector`. Pick by connector, not by hardware; a layerwise producer indexes
  `params["remote_cached_tokens"]` (`mooncake_layerwise_connector.py:990` at `v0.23.0`), a key only the
  layerwise proxy sends, so the plain script `KeyError`s in the P scheduler step. The Qwen3.5 guide at
  `v0.23.0` pairs them backwards and cites a `proxy.sh` absent from the tag (#15389).

- `../vllm-ascend/dockerfile-env.md` — reports where the "bug" is the image's own build: an env var it
  only writes to `~/.bashrc` (missing on non-interactive startup, §A), which NPUs the container can see
  (§B), or a package it never installed — `libjemalloc2` absent at `v0.12.0rc1`, so
  `from mooncake.store import …` aborts with `corrupted size vs. prev_size` (§C, #5609).
- `../vllm-ascend/npu-device-visibility.md` — the container cannot reach the cards it was given (works
  on an idle host, fails while another container runs; only some ranks die): the
  `ASCEND_VISIBLE_DEVICES` (docker runtime) vs `ASCEND_RT_VISIBLE_DEVICES` (in-process, and the one
  variable vLLM's rank→physical mapping uses) distinction, the verified anchors, the reporter-confirmed
  env pair from #6942, and why this class has no `vllm_ascend/` line to hunt.
- `pcp-dcp-and-mtp.md` — the PCP/DCP-only bookkeeping failures that are routinely mis-merged: a prefix
  cache that **never hits** because CP multiplies the scheduler block size (`block_size × dcp × pcp`,
  1024 tokens at `--block-size 128` + dcp 8, plus `eagle_drop`/alignment), a worker death on inputs shorter
  than `num_speculative_tokens + 1` (`'PCPManager' object has no attribute 'mtp_slot_pad'`, an
  `AssertionError` on later tags), the `NoneType` assignment at `cad.seq_lens_cpu` when
  `--async-scheduling` blanks the CPU mirror, and §E's
  `ValueError: could not broadcast input array from shape (29,) into shape (31,)` in
  `update_tokens_for_pcp` when MTP meets a mixed prefill+decode step (#13750, #8578/#8402, #8926, #11231).
- `spec-decode-and-parallelism.md` — MTP acceptance collapse (placeholder token), **§1b's stall**: a P/D
  + MTP engine that stops logging after ~1 h and is dead hours later on `0.18.0rc1`, whose accepted-token
  mirror was copied with no synchronize until `9a0b786f2` (#8764, first tag `v0.18.0`) — the reusable half
  is that the fork's `_build_attention_metadata` **overrides** the upstream method, so a paired-upstream
  fix (vLLM #36013, already in `v0.18.0rc0`) does not reach it — and accuracy loss that appears only with
  a parallel/KV feature (PCP without EP, PD + DSpark + prefix caching, and hybrid GDN/Mamba models under
  PD — see `hybrid-gdn-pd.md`), plus the **resolved** device-fault case: the `507034` + `507035` death of
  DeepSeek-V4-Flash/Pro on A3 PD under DSpark, fixed in `v0.26.0rc2` (#16712 empty DSpark context, #16714
  token-id validation; case #15985). Proposer-metadata crashes and the cache-unit miss are in
  `pcp-dcp-and-mtp.md`.
- `moe-ep-dispatch.md` — the expert-parallel MoE all-to-all path, three members: the
  `aclnnMoeDistributeDispatchV4` `561000` with `EI0007` "Failed to allocate resource qp … Resources are
  exhausted" (HCCP QP pool; read `select_moe_comm_method` to say which collective ran, the GLM-series EP
  gap #10944 → #11394, and why "EP off" also forces FlashComm1 off); the sibling collective
  disagreement / missing shared buffer (#5295, with the silent version in `pd-collective-hang.md`); and
  the capture-time `107027` "Not allow to synchronize captured-stream" that a data-dependent op
  (`torch.bincount` in `init_routing_quant`) produces inside an ACL graph (#2567, fixed by #2469 from
  `v0.10.1rc1`); plus §C, `VLLM_ASCEND_ENABLE_FUSED_MC2=1` on a **BF16** checkpoint — the A3 selector
  requires `quant_type == "w8a8_dynamic"` through `v0.16.x` and the tag's own Qwen3-235B tutorial calls the
  feature W8A8-only, while a patched BF16 run stalls inside `dispatch_ffn_combine_bf16` above M≈2048 and
  surfaces only as vLLM's 60 s `shm_broadcast` warning (#7154).
- `memory-sizing-and-parallelism.md` — "the whole model is loaded on every card", "TP does not split
  it", "0.9 of the card is used by a 7B model", with no command and no log: the KV cache is sized to
  `gpu_memory_utilization × device total` on every rank (`worker.py determine_num_available_blocks`),
  so the memory reading is expected; separate `tp=1`, one-process-per-card replicas, and a real TP
  split by the pid table and the `# NPU blocks` / `Maximum concurrency` lines (#1316).
- `post-startup-oom.md` — the service **started healthy** and died of `NPU out of memory.
  NPUWorkspaceAllocator tried to allocate …` + `current working operator name is <op>` + `EL0004` minutes to
  hours later, with an `npu-smi` timeline that plateaus first. The failing pool is torch_npu's per-stream
  op-workspace cache (`NPUWorkspaceAllocator.cpp`: one block per `aclrtStream`, grow-only, freed and
  re-allocated with `ACL_MEM_MALLOC_HUGE_ONLY` on every growth, plus a device synchronise), so "295 MiB free
  and a 216 MiB request fails" is a huge-page refusal the caching allocator's `expandable_segments` cannot
  reach; the operator name comes from the task queue (`NPUQueue.cpp:178-188`), not from the owner of the
  memory. Why `GraphParams.workspaces` is *not* an unbounded per-seq-len ladder (seeded from the capture
  sizes, unguarded `events[num_tokens]`), the `memory_summary()` evidence to request, and the levers that
  move the envelope (#12692) — including lever 7, the **inverse** of the `expandable_segments` advice: a
  user `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:…` suppresses the fork's own default
  (`v0.23.0rc1:platform.py:761-777`, #16022), and a report whose timeline is a **plateau** (`44.5 → 46.0 GB`
  then flat, KV back to 0 %) has already answered the "leak" question.
- `doc-guide-vs-capability.md` — `[Doc]` reports where the guide's serve block exceeds the hardware:
  which ref is governed (default-branch docs vs a versioned URL), the four lines that decide it (the
  serve block, the page's own limitation, the sibling page's warning, the release-notes known issue),
  and why the "unrecoverable after OOM" half is the runner's `Error in memory profiling` guard plus
  residual processes (#18158). Second shape: a page that performs the **same step twice** (two
  `docker run` blocks under Installation and Deployment, with an internal cross-reference between them) —
  a `[Misc]` question answered from the two headings and their parameter deltas, fixed later only on the
  default-branch docs (#9988 → `908915982`/#10728, first tag `v0.23.0`). A third shape has **no symptom
  at all** (`[Community Test]` rows, stale image pins): `community-test-task-sheets.md`,
  "Nothing failed" on this page. A **contributor dev-env** `[Doc]` report (`testing.md`'s nightly
  container, `contribution/index.md` "Run CI locally") belongs here too, but its root cause is the image's
  editable install plus the build-generated `_build_info.py`, so the mechanism lives in
  `../vllm-ascend/install-and-image.md` §4 (#17718).
- `issue-ledger/` — the worked reports: `index.md` (scope, which report is where, recheck rules),
  `reports-early.md` and `reports-later.md` (per report: versions, thread state, best-supported
  conclusion and the verified facts already established), and `lessons-across-reports.md` for what the
  rows show in aggregate.
