---
name: vllm-ascend-bug-report-triage
description: "vLLM-Ascend NPU bug reports: workaround + root cause."
---

# vLLM-Ascend bug report triage

## The write gate: the reply is the deliverable

Own output — `reasoning_content`, tool-call arguments and `content` — is charged to the same
run-level allowance, and the reply you actually write is the only artifact of the run. A tool
*result* is free. The failure that loses a report is therefore not weak analysis but **no `content`
at all**, or a `content` that starts too late to finish: a run can end mid-reasoning, or inside an
uncalled tool call, after several good tool results.

- **The reply is `content`, and its message carries no reasoning at all.** `reasoning_content` is
  emitted *before* `content`, so "answer first, derive after" is not something a message can do. The
  reply-slot message also has no tool, so nothing written there can fetch evidence: every sentence of
  reasoning in it is a draft paid for at reply prices and never read. Open it with `symptom →` and stop
  thinking. A sentence that already carries **both** the workaround and the status *is* the reply, not a
  note on the way to one; written into `reasoning_content` it is followed by a "quick confirmation read"
  the class page already answered. **The reply's first three lines are one sentence each** (`symptom →
  mechanism`, `workaround`, `root cause path:line`), because a run stops inside the fourth line more
  often than not: a recorded FA3 reply opened with a five-line mechanism paragraph and its `content` ended
  at the words `root cause`, shipping no location at all. The derivation belongs further down, under
  `evidence`.
- **Three tool-calling messages, then the reply: the message after the third is `content`, whatever is
  settled.** A *work message* fetches evidence about the report; **loading the skill or a reference page
  (`skill_view`) is instruction reading and never counts as one of the three work batches** — a recorded
  run counted its skill load as batch 1, concluded its third read had already overrun the budget, and spent
  the window re-counting instead of answering. The message count is still charged, though: a message
  carrying only `skill_view`, `skills_list` or an `ls /workspace/repos` **spends** one of the three slots,
  so the skill load belongs in the same message as the thread read and `open_answer.sh` (the script path is
  stable under `/workspace/.agents/skills/<name>/`). Three of the four runs in one recorded batch (a 910B4
  memory climb, a W4A8 dense "regression", a 310P "runs but no output") each opened with such a preamble
  message and then took a fourth or fifth read; all three shipped nothing. Count **messages**, not calls or
  results: work batch 1 is the thread *plus* `open_answer.sh` *plus* the ledger grep in one message (three
  results back, still one batch), work batch 2 the one code window or class page the symptom needs, work
  batch 3 only when batches 1–2 returned neither a lever nor a `path:line` — and then it is the last tool
  message. The slot is fixed by that count, not by whether the mechanism feels settled: a result that names the lever (a
  maintainer's config delta, a fixing PR, the failing branch) is answered in the next message, and an open
  mechanism ships as `hypothesis` with its falsifier. "Not settled yet" is a `limits` line — never a
  answered. **Fanning out candidates is what spends the window** — "maybe the fabric, maybe the port,
  maybe the graph mode, maybe the model" after batch 2 is already an answer: take the owner the log's own
  stage names (which layer printed it, which EI code, which group), label the rest `hypothesis`, and stop
  enumerating. When the reporter's own A/B already names the flag, the reply is that A/B plus the document
  that limits it, not a search for the code that "really" caused it — and a document or code site that
  **contradicts** the thread's answer (the reported tag's docs still promoting a key the maintainers call
  unsupported; the key still wired into a comm path) is a finding *for the reply*, quoted with its
  `path:line`, not a reason to keep reading. That contradiction is often the most useful line in the reply,
  and a run that files it as "unfinished business" ships nothing.
- **A reply-slot message with no `content` is a lost run, and it is the commonest failure.** Four recent
  reports in one group (a thread whose only comment was "network card issue", a doc-conformance env pair,
  a P/D accuracy alternation, a packaging report) — and three of the four in the next — each ended inside a
  4 K–12 K-character reasoning block at the reply slot and shipped nothing; in each the analysis was
  already complete — "Confirmed everything:", "Key facts now:", a drafted verdict — and the message
  holding it had no `content`. When a sentence starts to look like the answer, that sentence is the
  answer's first line. The same batch shows the two shapes the loss takes once the analysis is in hand: a
  reply-slot message that opens with a plan for one more read (`Batch 3: …`, `let me just confirm the exact
  …`) and a reply-slot message that drafts the whole answer into `reasoning_content` and is clipped before
  `content` starts. In both the answer existed in the trace and the user received nothing.
- **Thread thin, body rich: the body is the evidence.** A thread with one line ("It looks like a network
  card issue.", 「已解决」, `@X PTAL`) or none does not mean nothing is known — the reporter's `vllm serve`
  argv, traceback and pasted log/plog are evidence already in hand. Classify from the symptom, name **one**
  ref-pinned fact, label the mechanism `hypothesis` with the print that would settle it, and reply. The
  fabric report in that group answered itself from the first line of its own plog
  (`ra rdma init failed … devid[12] ip[192.168.20.160]`); its run spent the allowance deriving that and
  shipped nothing.
- **The report may already be analysed — grep the ledger in batch 1.** Add
  `grep -n -m3 '#<N>' <skill_root>/references/symptoms/issue-ledger/index.md` to the first `bash`
  call (`skill_view` prints `skill_root`; the number is in the report's own title/body). A hit means
  the thread was read, the frames pinned, the candidates ruled out and the open question written:
  batch 2 is *that row's* detail page and the reply follows it — no re-verification, no second code
  read. `references/symptoms/layer-name-keyerror.md` is the cautionary page: it settles workaround,
  status and an `[inference]` mechanism, and a run that re-derived its four-key arithmetic instead of
  replying shipped nothing.
- **An open mechanism in a class page is a `limits` line, not a work item.** When the page says
  "status: open / `[inference]` / the discriminating print is X", quote that and hand the print to
  the reporter. Arithmetic no traceback can settle — which rank holds which stage's list, which
  commit introduced a line, which of several candidate fixes is the twin — is not the reply; a run
  that starts it has already spent the answer.
- **Do not measure yourself.** Re-counting your own output, restating this gate, promising to comply
  or planning the reply are all output spent on the thing they measure, and self-estimates are off
  by ~2× anyway. The trigger is countable rather than felt: *the third work result has arrived, so
  the next message is `content`*.
- **One batch per message, not one call**: 2–3 independent calls, each with one question and short
  argument strings. A single call per message wastes the batched shape; a call in the closing
  message is cut by the framework and executed by nobody.
- **The reply is compact: aim ~1.2–2 K characters.** Symptom → mechanism, workaround with its cost,
  root cause `path:line`, why-this-symptom, status/limits, falsifier, evidence lines last so a clip
  still ships the lever. If the text does not fit, cut the alternatives, never the reply. A CJK
  reply costs more of the allowance per character than an English one.
- **A miss is the reply trigger, not a retry.** `fatal: path … exists on disk, but not in '<tag>'`,
  an empty grep, an unexpected file, an unexplained fact inside evidence already in hand: the reply
  goes out with the thread's lever and a labelled `hypothesis`, and the unexplained fact becomes a
  `limits` line with its falsifier.
- **Provenance is not the lever.** A fix's first tag, `git merge-base --is-ancestor`,
  `git show --stat` of a PR the thread named, whether a symbol survives at `main` — none of them
  changes the workaround or the mechanism. They are `status and limits` lines written from evidence
  already in hand. The expensive shape is a **tag sweep** (`for t in v0.23.0 v0.24.0rc1 …; do git show
  $t:<path> | grep …; done`): it costs a batch per attempt and answers nothing the reply needs. One
  `git tag --contains <fix-sha>` — empty means "on `main` only" — plus one grep on the newest tag is the
  whole answer, and when every tag in the clone predates the merge, *"fixed on main, in no tag here"*
  **is** that status line. A run that then walks the tag list to watch the line move ships a clipped
  reply.
- **Never derive line numbers after the read.** Print them with the window:
  `git show <ref>:<path> | nl -ba | sed -n 'A,Bp'`, or `git grep -n`. Reconciling offsets in the
  closing block costs the reply. A `file:line` you did not print is a recheck line —
  "`platform.py:761-779` at the page's ref, recheck at yours" — never a fourth batch.
- **A closing tool call is cut by the framework, not executed** — a message whose tool-call JSON is
  truncated leaves neither a result nor a reply (`Model output was truncated during generation; the
  framework did not invoke this tool.`). Keep the last call small, and prefer none.
- **A string already in the report is quoted, never recalled.** The closing trap that reads as diligence:
  "let me confirm the exact wording — I recall `long_wait_time_msg` returns …" for a log line that is in
  the issue's own title, or re-deriving a message the thread already printed. Copy it from the body, mark
  it `as pasted`, and if its owner is not already named, spend the remaining room on `limits` instead.
  Recalling a string is never a fact and never worth the reply.
- **Stop triggers fire inside `reasoning_content` too.** "Let me structure the reply", "Now write
  reply", "I have enough", "Key facts now:", "Facts settled:", "Confirmed everything:", "Another
  idea", "One more check is worth it", "let me just confirm the fix PR", "check whether that branch
  is already there" all mean the message ends now, in `content`. The full word list:
  `references/workflow/answer-and-claims.md` ("Stop triggers").
- **The first message cannot be planned, but it can be shared.** This gate lives inside the file being
  loaded, so no message before the load can apply it; the load itself is instruction reading (not one of
  the three work batches) yet it is still one of the three tool-calling messages, so pair it with the
  thread read and `open_answer.sh` instead of spending a message on it alone. A repo listing,
  `skills_list` or a tag dump beside it buys nothing — the listing never changes a decision. `skills_list`
  is also **name**, not search: a phrase such as `{"query": "vllm-ascend bug triage"}` returns `[]`, and a
  run that opened with it lost three messages finding the name again — the exact name is in the prompt's
  `<available_skills>` and in `skill_view(name="vllm-ascend-bug-report-triage")`.

| assistant message | what it contains |
| --- | --- |
| 1 | batch 1: `skill_view` (if it is not already loaded), the `github` thread read, `open_answer.sh` and the ledger grep — **one** message |
| 2 | batch 2: the code window the deepest frame names, or one class page plus one ancestry query — **or the reply**, which is the right message-2 content whenever batch 1 already delivered a lever, a class page or an open question |
| 3 | batch 3 (optional): the single read the reply's `path:line` still needs — skip it when batch 1's thread already named the lever |
| 4 | **the reply**: `content` only, no reasoning, no tool call |
| — | any work after the result that names the lever: the reply is what gets lost |
| — | a message with no tool call and no `content`: nothing is left to learn, so the whole message is a draft |

Reply shape and the verbatim stop-trigger list: `references/workflow/answer-and-claims.md`.

## When to use

Use when a filed bug report against the Ascend LLM-serving stack (issue body, start command, logs,
`collect_env`) must become (a) at least one effective workaround and (b) a root cause concrete enough to
file — `path:line` at the reporter's version plus the mechanism. Targets: `vllm-project/vllm-ascend`,
upstream `vllm-project/vllm`, torch_npu, CANN, and the repos a frame or connector name lands in
(`Ascend/torchair`, `Ascend/memcache`, `kvcache-ai/Mooncake`); deployments cover single node, PD,
PP/TP/DP/EP/PCP/DCP/spec-decoding on A2/A3/A5/950 or 310P hardware.

Symptoms: startup stall or profile OOM; a load-time crash (truncated shard, a `NotImplementedError: …
doesn't support quant_type=… for layer_type=…` on a checkpoint an older tag served — a scheme retired on
purpose, not a regression — quant-description `KeyError`,
a `deq_scale`/`weight_scale` `KeyError` from a checkpoint served with the wrong quant method, FP8 or
unquantised checkpoint, or an `AssertionError` on `org_vocab_size` in
`vocab_parallel_embedding.weight_loader` when a pre-sharded `TP<n>/` artefact is served with
`load_format=auto`); a crash owned by a quantiser, custom op, device kernel, CANN resource check
or captured-stream sync (`107041` beside `507015`); a bare aivec `DDR address of the MTE instruction is
out of range` with `--enforce-eager` and no graph or PD (a plog-shaped report, not a code one); a Triton
launch limit (`coreDim … UINT16_MAX`); a
`507034` + `507035` pair that kills a DeepSeek-V4-Flash/Pro A3 PD engine under DSpark; an assistant
`content` delta that is whitespace only just before the first tool call on a `0.23.x` image; a hang
after weight load, a PD/multi-DP freeze (HCCL `STUCK`, AICore 0 %), a startup or graph load that dies in
the comm layer (`Getting socket times out`, `wait socket establish timeout`, `listen on ip[..] and specific
port[16666] fail`, `ra rdma init failed`, `Task run failed`, graph-mode `Load graph failed` beside
`EJ0002` — route to `references/hccl/index.md`, no `vllm_ascend/` line owns it), a multi-node DP launch that hangs
right after the engine banner (or after `Started DP Coordinator process`) because the per-node
`--data-parallel-size-local` values add up to less than `--data-parallel-size`, a prompt-length admission
cliff, or an `init engine (profile, create kv cache, warmup model) took N seconds` that is many times an
earlier tag's with nothing failing;
garbage or accuracy loss only with a parallel/KV feature, only in graph mode, or only on the first call; a
KV-pool store with no PUT, a pool load that kills the worker (`load_async=False`), a pool lookup that
hits while every request still loads 0 tokens (throughput *below* the no-connector baseline), or a prefix
cache that never hits; a malformed response body (a tool-call payload that loses `id`/`arguments`, or the inverse: a
whitespace-only `content` delta before the first tool call); a `--reasoning-parser` run whose `reasoning_content` is always null,
whose thinking left `content` or vanished from turn 2 of a tool-call loop, or whose first token got "slow"
with the parser on (`reasoning` field, `--reasoning-parser` "not found" list); a chat-template `ValueError`
while `/v1/completions` works; a VLM on a healthy engine answering as if the image were absent (wrong
caption, generic answer, symbol soup right after `<image>` — the request key, the placeholder token or the
encoder budget); a `Tokenizer class … does not exist or is not currently imported` at renderer
construction (a v5 tokenizer class on a 4.x transformers), or a processor-constructor
`TypeError: <X>Processor.__init__() got multiple values for argument 'tokenizer'` on a multimodal
checkpoint; an embed/pooling model that never serves; the
graph-mode and `torch._dynamo` tracing classes; a `cudagraph_mode: FULL` run whose decode is graphed but
whose **prefill is not** (gaps between operators, no error); a `TypeError: <Subclass>.__init__() got an
unexpected keyword argument …` from an opt-in `vllm_ascend/core/` scheduler; two connectors in one
`MultiConnector` whose block ids disagree (`len of local trans block ids : 0 not equal to …`); a slow or
non-starting LoRA adapter under V1 graphs; a
benchmark below the published number; a first request dying on `prompt_embeds` or answering with empty
content and one completion token; a first `llm.generate()` that repeats one token id while the second call
is correct (`enable_sleep_mode`); a build with no `libvllm_ascend_kernels.so`, a `Failed to import
vllm_ascend_C`, or an `AclOpKernelInit failed opType` on a card the image's CANN base does not cover; a
post-startup freeze under the legacy `full_cuda_graph: true` (`graph_task_update_end` frame); a
`TypeError: SchedulerDynamicBatch.__init__() got an unexpected keyword argument 'hash_block_size'` under
`SLO_limits_for_dynamic_batch`, or a `TypeError: <X>Processor.__init__() got multiple values for argument
'tokenizer'` out of a source-installed transformers; a usage question asking whether a Mooncake **store**
backend can be built without `-DUSE_ASCEND_DIRECT=ON`; a `from
mooncake.store import …` abort (`corrupted size vs. prev_size`) inside a released image; an OOM or refusal
after copying a documented serve block; "the whole model on every card"; a V1-oracle `NotImplementedError`
for an architecture; a `[Doc]` guide whose block exceeds the hardware it names, repeats a step, or leaves a
contributor's dev/test-env setup implicit (a mounted checkout, `set_env.sh`, an `_build_info` import error);
a streamed tool call that ends in `JSON Parse error: Unterminated string`, or whose text carries
`</think>`/`<tool_call>`/`<arg_key>` markup inside `content` or `function.name`; an `AssertionError` whose
payload is `-1` under MTP plus structured output; a 310P start dying in `_warm_up_atb` with
`RuntimeError: LinearOperation CreateOperation failed!`; a 310P/300I DUO TP>1 start dying during decode
graph capture with `Insufficient_Event_Resources(EL0008)` / `rtEventRecord … 207007` / `capture_end …
507903` (the hardware event-id ceiling, and the *event*-resource sibling of the documented stream-resource
failure); a 310P W8A8SC start asserting at `vocab_parallel_embedding.py:463` because a per-rank `TP<N>/`
directory was served without `--load_format sharded_state`; a `ModuleNotFoundError` for a whole package tree after a
non-editable `pip install .`/`pip wheel .` (a directory missing `__init__.py` is dropped by
`find_packages`, and the tree exists only on the reporter's branch build — `install-and-image.md` §5); a `[Usage]` "when will the 310 series support
graph mode / quantization" question, answered by first tags; a worker death in `execute_dummy_batch` under
`FULL_DECODE_ONLY` on a 2-node PD mixed deployment; a startup that prints `Available memory: 0, total
memory: <N>` on every rank and then `ValueError: No available memory for the cache blocks`, where the
*profile* pass overshot `total × gpu-memory-utilization` so no utilization flag helps
(`startup-and-stall.md` §3c); a GLM4.7 turn
that logs `Failed to extract tool call spec … TypeError: expected string or buffer` and answers with
native `<tool_call>` markup in `content`; an engine that never starts, whose only error is
`WorkerProc initialization failed due to an exception in a background process` with **no child
traceback** beside `[shutdown] … sending SIGTERM count=N` and `TBE Subprocess … main process
disappeared!` (the child was killed, not raised — `startup-and-stall.md` §8); a crash inside the first
captured graph on an `-rc` image whose `torch-npu` `.post` build contradicts the tag's
`pyproject.toml` pin; a P/D deployment on MTP answering fluent garbage with no error (the connector's
block-id cut — `mooncake-transfer-errors.md`, "The silent sibling"); a PD build on the retired `llmdatadist` connector whose
prefill warns `fail to receive KV Cache transfer done signal in 120s` while the complaint is TTFT; decode
throughput that collapses as running sequences per engine cross a power of two; a report
where **nothing fails**; a `RuntimeError: cancelled` from `patch_shm_broadcast.py` as the deepest frame of
an engine death; a `ValueError: could not broadcast input array from shape (N,) into shape (M,)` out of
`worker/pcp_utils.py` `update_tokens_for_pcp` on the first real request once PCP and MTP are both on and the
model is a hybrid-attention one; a structured-output **stream that never emits its first token** (a
`guided_choice`/`regex`/`grammar` request, ordinary traffic serving, engine up) in a deployment whose argv
matches a working one and whose only delta is a block of vendor `VLLM_ASCEND_*`/`HCCL_*`/`ATB_*` env vars —
whose live member is `VLLM_ASCEND_BALANCE_SCHEDULING`; a PD connector
that cannot connect (ADXL `status: 1039xx`) or whose transfer times out; two
connectors in one `MultiConnector` (a layerwise P/D child beside a KV-pool child) whose block ids disagree
(`len of local trans block ids : 0 not equal to …`); an `IndexError` in `vllm_ascend/core/scheduler.py`
`.blocks[0]` on a pooling model, or a `TypeError` from an opt-in `vllm_ascend/core/` scheduler
(`TypeError: SchedulerDynamicBatch.__init__() got an unexpected keyword argument 'hash_block_size'`); a
`cudagraph_mode: FULL` run whose decode is graphed but whose **prefill is not** (gaps between operators, no
error); a service that **started healthy** and OOMs minutes-to-hours later with `NPU out of memory.
NPUWorkspaceAllocator tried to allocate …` plus `current working operator name is <op>` and `EL0004`; a
`[Usage]` question about whether a release *ships* a connector (`does vllm-ascend support the Mooncake store
connector`) or whether a third-party library must be built with a vendor flag (`-DUSE_ASCEND_DIRECT=ON`),
which is a gate in the fork, not a config. Verbatim anchors and routing:
`references/symptoms/symptom-index.md` (with its stage-based companion `symptom-index-stages.md`),
`references/workflow/log-signatures.md`.

The decisive constraint: the workspace clones are usually on `main`, months ahead of the reporter's
release. **The reporter's version decides what code ran**; reading `main` and reasoning about it produces a
plausible but wrong answer.

Not for: CI pipelines, release choreography, model onboarding, tuning a healthy service, or implementing
the fix. An **installation or image** report *is* in scope when it asks what a shipped artefact contains
(a missing `libvllm_ascend_kernels.so` after a local editable install, "does image X support 910C") — the
answer is a `setup.py` / `CMakeLists.txt` / `Dockerfile` line at that tag, not a service behaviour
(`references/vllm-ascend/install-and-image.md`). A *cross-release* performance difference is triageable from
the commits and defaults in the tag range, with the measurement left to the reporter. All of it is static
analysis — no NPU, nothing reproduced.

## Deliverables (both are required)

**Workaround** — the exact flag/config/env/version change that works on *their* version, what it trades
away, and whether they confirmed it or you propose it. **Root cause** — owning path at the reported ref
(`file:line`), the gate, the mechanism and why it gives this symptom. **Status and limits** — fixed-later /
open / known issue plus what you did not verify. Partial answers ship only as labelled `hypothesis`.

## Batches: what goes in each message

The gate above carries the message table; this is the operational shape of each batch.

**Batch 1** — one `github` call for `/issues/N/comments` (`params: {"per_page": 100}` with `fields` and
`max_chars: 40000`, the tool's ceiling, so that a 20 K thread arrives whole instead of costing a paging
round; never `paginate` with a hand-set `json_pointer` — `references/workflow/evidence-sources.md`) plus

```bash
bash scripts/open_answer.sh "<repo>#<N> — <symptom in one line>" <version> [repo-path-or-name]
```

which resolves the reporter's spelling (`0.18.0`, `v0.21.0rc1-310p`) to an existing tag, writes
`/workspace/scratch/answer.md` and prints the ref facts whole. Append one cheap command to the same
`bash` call — `grep -n -m3 "#<N>" <skill_root>/references/symptoms/issue-ledger/index.md`: a hit means
the report is already analysed and batch 2 is that row's page, not a fresh code read (the gate's
ledger bullet); exit 2 = nothing resolved, exit 3 = the
argument names no clone, **not** evidence a tag or path is absent. Pass the repo that owns the failing
path (`vllm` for anything under `/vllm-workspace/vllm/…`): the snapshot keys its sections on whether
`vllm_ascend/` exists at the ref. The file is **scratch**: read it, never rewrite it, never point the
reply at it. If the last comment or the **tail of the report body** names the lever ("use the latest
release", a reporter's `export` pair, 「升级 transformers 之后可以正常启动」), the reply follows directly —
at most one confirmation read, and the mechanism you still want is a labelled line in `status and limits`,
not another batch (the class pages under `references/symptoms/` usually carry it). A body whose
`### 任务问题和需求` says `测试通过` with an empty thread has **no symptom to localise**: one read of the
page it names, then the reply (`references/symptoms/community-test-task-sheets.md`). A `任务问题和需求`
that names a *tool* (「容器内无法使用postman」) is the same shape: the page's own `docker run`/`vllm serve`
block answers it from the host, and the reply is that request path plus the page lines that disagree.

Five tool-argument traps cost a batch each, and all are mechanical: the `bash` tool **requires
`description`** — a call without it is rejected (`'description' is a required property`) and never runs,
leaving batch 1 with the thread and no ref facts; `git log … --grep -iE 'a|b'` is read as a revision
(`fatal: ambiguous argument`), so write `git log -i --grep='a\|b' <range> -- <path>`; `view='text'` on a
`.json` `github` read returns `view='text' requires base64 file/blob content` — read a foreign repo's file
with `view='full'` plus `json_pointer` (or the raw `Accept` type), and prefer the `github` contents API to
`web_fetch` on `raw.githubusercontent.com`, where a recorded call died after a 120 s timeout — the same
timeout hit `huggingface.co/<org>/<repo>/raw/main/config.json` (#3398), while `curl -sSL -m 30` in `bash`
returned that JSON and `curl -sSL https://huggingface.co/api/models/<org>/<repo>` lists a checkpoint's
files (that is how a `draft/config.json`'s `architectures` gets settled without the reporter); a plain
(non-`paginate`) request whose endpoint returns a bare array takes **no** `json_pointer` —
`/issues/N/comments` with `json_pointer: "/items"` answers `json_pointer does not identify a value in this
response` (`response_shape: {"type": "array"}`) and the batch is gone (#5188), so leave the
pointer unset and let `fields` project each item; and
`open_answer.sh "<repo>#N — …" <version> <clone>` resolves paths only for the clone it is given — passing
`vllm` for a report filed in vllm-ascend prints every `vllm_ascend/` line as `no`, which is not evidence
that a path is absent. And a tag trap, which is not an argument trap: **never type a release tag as a
range endpoint before listing the tags.** The 0.x lines are rc-suffixed (`v0.19.1rc1`, `v0.20.2rc1`,
`v0.23.0`, `v0.26.0rc2`) and there is no `v0.20.0`, so `git log v0.18.0..v0.20.0` answers
`fatal: ambiguous argument` — one `git tag --list 'v0.2*'` (or `origin/main` as the upper bound)
settles it, and a thread's 「v0.20」 means the 0.20 line, here `v0.20.2rc1`
(`references/workflow/version-pinning.md`).

**Batch 2** — the code window the deepest frame, log line or doc line names (`git show <ref>:<path> |
sed -n 'A,Bp'`, `git grep -n -A`), one ref-pinned probe (`scripts/ref_snapshot.sh <ref>`) and one history
query (`git log -S'<expression>'`, `git tag --contains <sha>`); one class page at most — read it with
`skill_view` and `file_path: references/<topic>/<page>.md`, or from the `skill_root` the load prints, in a
`bash` call you were making anyway. When the thread names the fixing release, that query is a **diff**
(`git diff <old-tag> <new-tag> -- <path>`). Compute dimensions and ancestry with a command.

**What not to spend the budget on**: prose arithmetic, a body you have not read, candidate ranking
("could be #A, #B or #C"), a framework function re-derived from memory, or `git log --grep='<symptom word>'`
over 3 500 commits. The cue is the wording — "I recall", "upstream probably" — and one grep replaces the
paragraph it was about to become. An unplaceable log line is one `git grep -n '<verbatim message>' <ref>`
(`log-signatures.md`) — but a line carrying a `[<file>.py:<line>]` suffix is exact even in a corrupted
paste, so read the file at that line at the tag; a body whose frames mix `1`/`l` is an OCR paste, not a
fork. Re-reading an ambiguous quoted sample is the same waste: the *class* of the symptom localises it,
so quote it as pasted and put "the sample cannot be reproduced from the log" in `limits`.

Reply shape — essentials first, evidence last, so a clip still ships the lever:

```text
symptom → mechanism, one line, version-scoped          [verified|reported|hypothesis]
workaround: exact flag/env/version + cost + reporter-confirmed | proposed
root cause: path:line at the ref, the gating condition, what propagates
why this symptom / status and limits / falsifier
evidence: one line per claim — last, so a clip still ships the lever
```

`templates/finding-report.md` is the long form.

## Classify before reading code

Five facts change the owning module: **release** (tag or release branch + pinned vLLM/torch_npu/CANN;
reconcile prose against the log banner); **hardware class** (310P vs A2/A3/A5/950 — 310P has `vllm_ascend/_310p/`; a *support*
  question for that hardware, "does it have graph mode / quantization", is answered from its tutorial
  page at the ref — `references/vllm-ascend/hardware-support-by-release.md`);
**parallelism axes** (TP, DP, EP, PP, SP, PCP, DCP, DSA-CP); **quantisation** (W8A8/W4A8/W4A4, MXFP8, "C8" =
fp8/int8 KV); **feature switches** (`--additional-config`, `VLLM_ASCEND_*`).

## Hard rules

- **Analyse the ref that matches the reported version, not HEAD.** Paths, symbols and subsystems move:
  never transplant the reporter's traceback line onto HEAD, and never conclude "the code is missing" from
  a failed read at HEAD. Name the ref beside every `file:line`, and say where a fact holds: the reported
  ref, a release branch, or HEAD only.
- **A promoted shell call is not a finished shell call.** `git log -S` and tree greps return
  `"kind": "promoted"` with a `jobId`: collect it with `job_output(wait: true)` before concluding "no
  matches" — an empty `"output"` may mean "not collected yet".
- **Prove release inclusion, never infer it.** A merged PR, a closed issue or a release-note bullet is not
  in the reporter's build: `git merge-base --is-ancestor <sha> <tag>`, `git tag --contains <sha> --list
  'v0.*' | sort -V` (the `vllm` clone also carries six `proto-v0.*` tags, which sort before every release
  tag, so the bare form reports `proto-v0.1.0` as "the first tag"). A spoken minor is not a ref either: a
  thread's 「v0.20」 or "on 0.24" may be `v0.20.2rc1` and there is no `v0.20.0`, so enumerate
  (`git tag --list 'v0.2*'`) and name the tag the range actually used. A false
  ancestry result is not absence — release branches carry cherry-picks under new shas, so read the tag's
  tree too (`references/workflow/fix-discovery.md`). Check the gate the code actually tests. A
  maintainer's or bot's "resolved in the latest version, use the latest release/rc" is not a tag: one
  `git show <candidate-tag>:<path>` per candidate answers it, and a thread that closed that way can still
  show the failing call at the next two rc tags with the skip arriving two releases later. A backport PR
  named in a maintainer's "merge these" comment is not a fix until `/pulls/N` says `merged: true`
  **and** the commit is an ancestor of a tag: #13349 is a closed, **unmerged** PR against
  `releases/v0.23.0`, while #12762/#13028/#12643 are already inside the `v0.23.0` tag — so that list
  explains nothing a reporter on the release lacks (`references/workflow/fix-discovery.md`).
- **A `[Doc]` report is answered against the docs tree, not the reporter's tag, and the copied block can be
  the defect itself.** "The latest guide" is the default branch's `docs/source/…`; a versioned docs URL is
  that tag's tree. Diff the page **at the reporter's tag** against the code *and against its own other
  blocks*: prose, command line and connector disagree, a `>=` floor ships an image that cannot load the
  model, a page repeats a step. Quote those lines, never a `vllm_ascend/` frame
  (`doc-guide-vs-capability.md`, `pd-proxy-examples.md`, `model-type-support-floor.md`). A copied
  `--max-model-len` beyond the KV budget is a startup refusal with no code window (`startup-and-stall.md`
  §3b). A **contributor dev/test-env** report (`testing.md`'s container, `contribution/index.md` "Run CI
  locally") is a build-artefact gap: the image's package is editable at `/vllm-workspace/vllm-ascend`, so a
  mounted checkout loses the build-generated `vllm_ascend/_build_info.py`, and `bash format.sh ci` runs no
  tests (`references/vllm-ascend/install-and-image.md` §4, `doc-guide-vs-capability.md`).
- **A switch is not evidence that the switch did something.** Keys are silently ignored when renamed or
  removed, later code rejects unknown keys, and the platform can *rewrite* a value
  (`max_num_batched_tokens` → `max_model_len` at `v0.11.0rc0`) — read the flag's derivation at the
  reporter's ref first (`references/vllm-ascend/config-flag-lifetimes.md`, Trap D). A rewritten
  `max_num_batched_tokens` also sizes `profile_run`, so it can be the whole reason a start refuses with
  `Available memory: 0` (`startup-and-stall.md` §3c). The clearer case is `VLLM_ASCEND_ENABLE_FUSED_MC2`:
  through `v0.16.x` the A3 selector also requires `quant_type == "w8a8_dynamic"`
  (`ascend_forward_context.py:252` at `v0.15.0rc1`), so on a BF16 checkpoint the var changes nothing, and
  the tag's own tutorial says the feature is W8A8-only — while a locally patched BF16 run stalls inside
  `dispatch_ffn_combine_bf16` above M≈2048 and shows only vLLM's 60 s `shm_broadcast` warning (`moe-ep-dispatch.md` §C).
- **The issue number is the report's own, and N±1 is a PR.** A "duplicate of #NNNN" names the twin; a
  neighbouring number returns a pull request whose comments are codecov reports and review pings — a
  recorded run lost a work batch to `#1630` for `#1629`. Verify the fetched `title` before reading comments
  (`references/workflow/evidence-sources.md`).
- **Locate the package at the ref before `git show <ref>:<path>`.** Feature modules move, and `git show` on
  a `main`-only path dies with `fatal: path … exists on disk, but not in '<tag>'` — a wasted batch. One
  `git ls-tree -r --name-only <ref>:vllm_ascend/<pkg>` answers the layout first (LoRA table:
  `lora-and-tp.md`). Never read a **line-capped** listing as absence: `git ls-tree -r --name-only <ref> |
  grep -i <name>` sorts `docs/` < `tests/` < `vllm_ascend/`, so a ten-line cap drops exactly the
  `vllm_ascend/` hits — one delivered answer reported a patch file as gone from `main` that still exists
  there (`references/workflow/version-pinning.md`, "Read-only clone traps"). If the **reported** path is
  missing at the reported tag, it is either newer/vendor code (`git log --diff-filter=A -- <path>` +
  `git tag --contains`: `models/glm5next/` first ships in `v0.28.0.rc1`) or a name no clone ever carried,
  which is a substitution through an analogue, labelled as such — never a `path:line` transplanted from
  `main` (`references/workflow/version-pinning.md`).
- **Ref-pinned reads only.** The clones' working tree is the default branch, so a bare `sed`, `grep -rn`,
  `wc -l` or `cat .github/vllm-release-tag.commit` answers `main` (that file says `v0.32.0` there and
  `v0.23.0` at the tag). Use `git show <ref>:<path> | nl -ba | sed -n 'A,Bp'`, `git grep -n <pattern>
  <ref>` or `scripts/export_ref.sh`, and never read the version off `git describe`
  (`references/workflow/version-pinning.md`). Keep the `nl -ba`: an unnumbered window makes the closing
  block count offsets instead of writing the reply. Grep **3–4 words**, never a sentence, and check the
  sibling repo before reading an empty result as absence (`tokenizer-and-chat-template.md`,
  `model-type-support-floor.md`).
- **A frame in `vllm/…` is answered by the `vllm` clone at its own tag, never from memory.** These APIs
  move (`get_layers_from_vllm_config` indexes `static_forward_context` at v0.14, `.get(...)` from v0.22.0);
  one `git show <vllm-ref>:<path> | sed -n '/def <symbol>/,+25p'` settles it, while remembered variants
  cost a run its whole allowance with no answer (`references/workflow/version-pinning.md`).
- **Read the whole error, deepest `vllm_ascend/` frame first**: `EngineCore/WorkerProc failed to start` are
  wrappers; dedupe the per-rank tracebacks and keep the last project frame with the exception — but a
  shutdown-gated raise is teardown, not the fault.
- **`WorkerProc initialization failed due to an exception in a background process` with *no* child
  traceback means the child was killed, not raising.** At vLLM `v0.23.0` `wait_for_ready` catches only
  `EOFError` and re-raises the generic message with `__suppress_context__` set
  (`vllm/v1/executor/multiproc_executor.py:760-762`), so the "root cause" it points at is text the child
  should have printed: a child that raises during init logs `WorkerProc failed to start.` with its
  traceback first (`:887-888`), and a child that was killed (host OOM, native abort) printed nothing. The
  parent's grace-period `SIGTERM/SIGKILL` and the dead worker's own `TBE Subprocess … main process
  disappeared!` lines are consequences. So the reply's first item is the grep for
  `WorkerProc failed to start` in every rank's log plus the out-of-vLLM evidence to request
  (`dmesg`/`journalctl -k`), never a mechanism nobody can see (`startup-and-stall.md` §8).
- **The thread's lever ships even when your mechanism is unproven.** "Pls take a look" and a stale-bot
  close are not fixes, but six shapes are the deliverable's workaround line as it stands, *attributed*:
  a reporter's confirmed delta (`pip install transformers==5.5.3` inside the image, 「把权重换成w4a8正常」),
  a maintainer's "expected to be resolved, verify on the latest main / PR #N", a closing
  「已经解决」 with the step that resolved it, a reporter's own 【原因定位】/【修复建议】 block (quote its
  A/B and its delete-the-key / use-this-env pair), a **community tester's** newer-rc reproduction that
  reports numbers (`44.5 → 46.0 GB` then stable, KV usage back to 0 % — a plateau, so the "leak" half is
  answered), and a **deployment-config** delta that leaves no code defect
  at all (a per-node `--data-parallel-size-local` that did not add up, a `--gpu-memory-utilization` the
  reporter had to raise). Read to the last comment first. What you add on top is **one**
  ref-pinned fact — the requirements floor, the class absent at that tag, the weight list in the model's own
  tutorial, the single gate that accepted the bad value — plus the mechanism as `hypothesis` with its
  falsifier. You do not owe the thread a solved mechanism: one recent set of reports contained all of these
  shapes, and the three runs that kept hunting the code-level cause after reading them shipped nothing.
  **A config lever has no failing `path:line` to pin.** Once the thread confirms the flag change, the reply
  is the next message and its one extra fact is one grep for the gate that let the value through — not a
  walk through the sibling clone. Two recorded runs did the opposite: one hunted vLLM's
  `wait_for_engine_startup` loop across six one-call messages for a rank-count mistake the reporter had
  already fixed, and one ran five further batches through the MoE comm selector for a
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` that the model's own tutorial FAQ states; both shipped nothing.
- **Verify a cited fix by its hunk, not its subject** — a fix rides inside a PR named after something
  else: `git show --format= <sha> -- <the failing file>`, `git log -S'<expression>'`
  (`fix-discovery.md`).
- **A missing or empty `reasoning` block is a field-name question before it is a parser bug.** vLLM's
  chat message and stream delta carry `reasoning` only from `v0.16.0` on (`reasoning_content` was removed;
  it survives in the Responses API and as a deprecated request key), and a parser configured with
  `--reasoning-parser` *moves* the thinking out of `content` rather than deleting it — so a client that
  prints `delta.reasoning_content` reports a failure the server is not making. One
  `git grep -n "reasoning: str" <vllm-ref> -- 'vllm/entrypoints/openai/*protocol.py'` plus the family's
  `vllm/reasoning/<name>_reasoning_parser.py` settles the class; a turn-2 loss is the prompt (only the
  assistant message's `reasoning` key reaches the template, `chat_utils.py:1511`), and Qwen3.5 is in no
  version of vLLM's interleaved-thinking table — `reasoning-parser-output.md`.
- **A wrong response body is a report about `vllm_ascend/patch/platform` first — but only after checking
  whether the parser is bundled at all.** Those files monkeypatch the parsers and `OpenAIServingChat` for
  every model, and a malformed `id`/`arguments` can come from the terminal chunk of a *correct* stream;
  yet a legacy parser such as `qwen3_xml` is **upstream at the image's `VLLM_TAG`**, and a whitespace-only
  `content` delta before the first tool call is its text-buffer flush, not a plugin line —
  `git grep -ln '<parser name>' <ascend-ref> -- vllm_ascend` empty routes the report to the vLLM tag, and
  the file may be gone from the next release (`tool-call-streaming.md` §4b, #15911). Two neighbouring
  symptoms are configuration, not code: `</think>`/`<tool_call>`/`<arg_key>` **inside `content`** or inside
  `function.name` means the reasoning/tool parser was not configured or not matched (native GLM tool text in
  `content` is the tell — `tool-call-streaming.md` §4c), and `JSON Parse error: Unterminated string` with
  `num_speculative_tokens: 3` is the terminal-`arguments` loss class at `v0.17.0rc1` (§3/§7). A **dropped**
  tool call with `Failed to extract tool call spec` / `TypeError: expected string or buffer` in the log is
  the upstream `glm4_moe` parser on a call shape its arg regex cannot consume: the exception is caught,
  logged and returned as `tools_called=False, content=model_output`, unguarded through vLLM `v0.14.1`,
  guard `a52d1396a7` (#32321) from `v0.15.0rc0`, fork backports #7710/#8832/#8833. The frame's `file:line`
  dates the bundled vLLM (`:104` is `v0.13.0`'s file, `:118` at `v0.14.x`), so quote that tag and do not
  derive how the argument becomes `None` (`tool-call-streaming.md` §4a).
- **A lower score is a request-path question before it is a kernel question, and tool-vs-tool is not a
  measurement.** On a DeepSeek-V4-family model read the eval config and
  `patch/platform/patch_deepseek_v4_thinking.py` at the ref, where `thinking`/`enable_thinking` default
  `False` (`deepseek-v4-thinking-and-effort.md`). Otherwise the evidence is the endpoint and template, the
  sampler values the client never sent (vLLM fills unset fields from the checkpoint's
  `generation_config.json` while `--generation-config auto` is the default — `vllm/config/model.py:267`/
  `:1391` at `v0.19.1`), the harness's own meta file at its version (evalscope's `mmlu_pro.json` wants a
  terminal `ANSWER: [LETTER]` line, so a `max_tokens` that truncates the CoT scores a correct model wrong)
  and the postprocessor — then the raw generations. Harness-vs-the-model's-official-number is the only
  comparison that decides anything (`benchmark-accuracy.md` §1); a 5-point delta localises nothing in the
  repo. When the delta tracks DP>1 instead, read `platform.py`'s parallelism algebra first: with
  `--enable-expert-parallel` the `additional_config.expert_tensor_parallel_size` is overwritten and
  `ep_size` spans the DP replicas (`benchmark-accuracy.md` §2, #1791 → #1289; `platform.py:136-149` at
  `v0.9.2rc1`, key gone from `v0.10.0rc1`), and a tenfold gsm8k drop whose raw output repeats tokens is
  that class, not a kernel.
- **A `site-packages/` frame is evidence about the installed wheel, not the repo**: place the version from
  its `file:line`, then answer with an image/pin statement plus the upstream fix
  (`grammar-bitmask-and-dep-skew.md`). The plugin's own image can carry the wrong wheel too: a crash
  inside the **first captured graph** with no weight-load error, on a fresh `-rc` image, is a `torch-npu`
  `.post` build the tag's `pyproject.toml` never asked for — one
  `git show <tag>:pyproject.toml | grep -n torch-npu` beside `git show <tag>:Dockerfile | grep -n
  'torch-npu'` decides it (`.post` installed over the pin at `v0.13.0rc2`, aligned from `v0.13.0rc3` by
  #6287; `references/vllm-ascend/dockerfile-env.md` §D). A prefix cache that never hits is a unit
  mismatch (`prefix-cache-granularity.md`).
- **An `AssertionError` carrying `-1` is a scheduling placeholder, not model output.** `(-1, '<req>',
  {'<req>': [-1, -1, -1]})` from `vllm/v1/structured_output/__init__.py` is the grammar step meeting a
  scheduled spec token that is still `PLACEHOLDER_TOKEN_ID`. At vLLM `v0.13.0` the loop accepts every
  non-`None` token; from `v0.14.0` `-1` is its own end sentinel, so padding is skipped. Two `verified`
  fillers: the Ascend recompute scheduler (`additional_config.recompute_scheduler_enable` →
  `core/recompute_scheduler.py:101-111`, MTP + PD KV consumer) and upstream's async-scheduler placeholder
  (`async_scheduler.py:37`). Ask for the decode node's effective `async_scheduling` (the plugin disables it
  whenever a `speculative_config` is set), `num_speculative_tokens` and the recompute key; the lever is P/D
  config parity or no MTP. Do not look for a tokenizer, sampler or model bug
  (`grammar-bitmask-and-dep-skew.md` §D).
- **An OOM *after* a healthy start names an allocator, not a leak.** `NPUWorkspaceAllocator tried to
  allocate …` is torch_npu's **per-stream op-workspace cache** (`NPUWorkspaceAllocator.cpp`: one block per
  `aclrtStream`, grown and never shrunk, allocated with `ACL_MEM_MALLOC_HUGE_ONLY`, freed+reallocated under
  `AclrtSynchronizeDeviceWithTimeout`), and `current working operator name is …` comes from the task queue
  (`NPUQueue.cpp:178-188`) — the op that was running, not the owner of the memory. So "295 MiB free and a
  216 MiB request fails" is a huge-page refusal, `expandable_segments` cannot reach that pool, and the
  question to answer is *who filled the card after `profile_run`* (`post-startup-oom.md`). Do **not** read
  `GraphParams.workspaces` as an unbounded per-seq-len ladder: it is seeded from `cudagraph_capture_sizes`
  and the sibling `events[num_tokens]` index is unguarded, so a new rung would raise `KeyError`, not grow.
- **A third-party build flag that a `vllm_ascend/` gate requires is a code requirement, not a tuning knob.**
  The KV-pool `MooncakeBackend` accepts `protocol: "ascend"` only — `UnboundLocalError` on anything else
  before `v0.20.2rc1`, an explicit `NotImplementedError` after — and its transfer engine initialises ADXL
  with a hard-coded `"ascend"`, so `-DUSE_ASCEND_DIRECT=ON` is mandatory for that connector while the
  maintainers' host-RDMA/TCP fallback describes Mooncake's own transports (`mooncake-transfer-errors.md`,
  "The store backend is ADXL-only").
- **A claim in the report is a claim, not evidence**: a "preliminary diagnosis", a summary naming closed or
  unmerged PRs, or "same error" for a sibling — grep the quoted symbol and check each PR
  (`version-pinning.md`). That rule is about *proposed causes*, not about **a config A/B the reporter ran
  themselves**: 「在 v0.17.0rc1 上正常，升级到 v0.18.0 后出现」「删除该参数后正常」 is an experiment, and it
  outranks the reporter's own stated reason — when their reason contradicts the tag's docs, the
  contradiction is a `limits` line, not a licence to search for the code that "really" did it (#12278: the
  user's 【原因定位】 named both the lever and the fix, while the tag's docs still carry the key — quote
  both). The same holds for what the report *omits*: the model directory's own `config.json`,
  `quant_model_description.json` or serve block can be fetched (`curl` the ModelScope/HF repo file list
  and the one JSON) instead of asking the reporter — a JSON is prompt-free evidence, and
  `quantization-load.md` §3 reads a W8A8 directory that way.
- **A `507034`/`507035` pair that kills a DeepSeek-V4 A3 PD engine under DSpark has a named fix: `v0.26.0rc2`.**
  The `507035` half is unvalidated client `stop_token_ids`/`allowed_token_ids` trapping CANN's `IndexCheck`
  kernel; the `507034` half is a **discarded** speculative step making
  `ops/triton/spec_decode/utils.py:120` read the previous request's position (`has_valid_context` clamp in
  `155f68974`, #16712). Those release-branch commits carry tag prefixes and, on other rc lines, different
  PR numbers — find them by PR *title* with `git log --all --grep`
  (`spec-decode-and-parallelism.md` case #15985, `references/workflow/fix-discovery.md`).
- **A device kernel name/hash is not a Python bug, and the same `507015` without one is not a kernel
  fault.** A DFX/plog `kernel hash` names a CANN AICore kernel; `107041` or a sync on the ACL-graph wrapper
  is a host sync inside a captured stream (`device-kernel-faults.md`, `graph-capture-and-custom-ops.md`
  §C). A bare `507015` on an **eager** run is neither, when the argv pins the FA3 backend
  (`attention_config={"backend": "FLASH_ATTN"}`): `fa3_v1.py` slices token-indexed `query` against
  request-indexed `block_tables`/`query_start_loc`/`seq_lens`, so any speculative decoding makes the prefill
  sub-batch read another request's rows and the device faults — `fa3-backend.md`, #17991, fix #18068 on
  `main` only, so no released tag has it. An op that cannot *initialise* (`AclOpKernelInit failed opType`,
  `Parse dynamic kernel config fail`) is the **image's SoC build**: read that tag's `Dockerfile*` `FROM` line (910b = A2, `-a3` =
  A3/910C, 310p, a5) before touching Python (`references/vllm-ascend/install-and-image.md` §1).
  A bare aivec `DDR address of the MTE instruction is out of range` belongs to that owning layer too, and
  it is a *family*, not a class: with `FULL_DECODE_ONLY` or a PD decode node it is §G's padded dummy batch,
  but an `--enforce-eager` run with no PD executes no capture, so nothing in the fork explains it and only
  the plog's `fault kernel_name=…` line names the owner. A thread whose whole content is "use the latest
  release/rc" plus a stale close (#3911) ships that lever, attributed, plus the plog ask — never a code
  window (`device-kernel-faults.md`, the aivec sibling section).
- **A 310P start that dies in `_warm_up_atb` is a warm-up call to a sample-only torch_npu op, and the fix is
  a no-op override.** `RuntimeError: LinearOperation CreateOperation failed!` under
  `compile_or_warm_up_model` → `_warm_up_atb` → `torch_npu._npu_matmul_add_fp32` is ATB's example API
  (torch_npu: "just a sample, not intended for external use, and it currently supports no device model"), so
  every 310P start on `≤ v0.12.0rc1` dies before serving — the call is unconditional there. #5776
  (`1ffca8673`, 2026-01-17) adds `vllm_ascend/_310p/worker_310p.py:NPUWorker310._warm_up_atb`, which only
  logs `Skip warm-up atb ops for 310P device`; first tag `v0.14.0rc1`. Workaround on the reporter's build:
  make `_warm_up_atb` return immediately (it pre-warms `ReshapeAndCache` latency only) or move to a `-310p`
  image of a release carrying the override. Read the method at the **reporter's** tag: it lives in
  `worker/worker_v1.py` at `v0.11.0` and in `worker/worker.py` from `v0.13.0`
  (`graph-capture-and-custom-ops.md` §B2).
- **A 310P/300I DUO start that dies at `capture_end` with `EL0008` has run out of *capture events*, and the
  fork's own guidance does not fire for it.** `Insufficient_Event_Resources(EL0008)` +
  `Create capture event failed … event_id=[-1->640]` + `rtEventRecord … 207007` come first; the
  `507903` / `an error has invalidated the stream capture sequence` /
  `NPUModelRunner failed, error is capture_end …` stack after it is the *consequence*. The handler at
  `vllm_ascend/compilation/acl_graph.py` keys on `207008` / "stream resources are insufficient" only, and
  the tag's FAQ documents that sibling (`docs/source/faqs.md` §17, `EL0009`), so this variant reaches the
  reporter bare. Levers, in order: `--enforce-eager` (reporter-confirmed on TP2), fewer
  `cudagraph_capture_sizes` (the list is entirely the user's — `platform.py` injects no default and prunes
  nothing for 310P once it is set, and the old `update_aclgraph_sizes()` shrink was removed), or a newer
  HDK/CANN. It is not the padded-dummy-batch class and not a host sync in the capture window —
  `aclgraph-mode-and-capture.md` §H.
- **A pool lookup that hits while `need to load` stays 0 is upstream semantics, not a connector bug.** The
  `num_external_hit_tokens < num_computed_tokens → need_to_allocate = 0` clamp exists in upstream vLLM's own
  Mooncake store scheduler at the paired tag and the fork's `pool_scheduler.py` is a copy of it, so a
  workload whose shared prefix already sits in the local prefix cache pays the lookup RPC and loads
  nothing. The lever is a build carrying the lookup-cost optimisation (#12783 `9e45da191`, first tag
  `v0.23.0`, not in `v0.23.0rc1`) or dropping that child — `kv-pool-store-hang.md` §E1.
- **A 310P report is answered from the tag's own 310P page, and that page's serve block is the deployment
  contract.** The page moves (`single_node_300i.md` along the 0.9–0.14 line → `tutorials/hardwares/310p.md`
  from `v0.15.0rc1` → the `getting_started/` includes from `v0.30.0`), and its wording moves with it
  (eager-only through `v0.19.1rc1`; the event-id note first in `v0.18.0`, continuous from `v0.20.2rc1`).
  Read it before any code: it carries the per-model `cudagraph_capture_sizes` lists (stated for the TP they
  are shown with, so a TP>1 run needs *fewer* sizes than the TP1 block), the mandatory explicit
  `--max-model-len`, and `--load_format sharded_state` for the Eco-Tech W8A8SC `TP<N>/` directories, whose
  `model-rank-N-part-0.safetensors` carry `[vocab/N, hidden]` embeddings. A `[Usage]` "when will 310 support
  X" question is a `git log --diff-filter=A` + `git tag --contains` lookup, never a promise; the whole
  capture-size lists, the `--max-model-len` rule and the sharded-checkpoint gate are
  `references/symptoms/310p-capability.md`; the first-tag timeline and the per-release support statement
  are `references/vllm-ascend/hardware-support-by-release.md`.
- **A worker death inside `execute_dummy_batch` under `FULL_DECODE_ONLY` on a 2-node PD mixed deployment is
  padded dummy-batch metadata, not the model.** aivec `DDR address of the MTE instruction is out of range` /
  `EL0004` / `TransposeKvCacheByBlock` plog errors under graph replay mean one of the three counts the dummy
  run keeps (`num_reqs`, the descriptor's padded count, `_pad_query_start_loc_for_fia`'s) disagrees with its
  source list, so the replay indexes a `seq_lens`/block slot that was never written. Two verified instances:
  #7290 (the drafter's `seq_lens_list` one short) and #7460 (`cbf46fad3`, which adds `and
  self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL` to the `num_reqs_padded = num_reqs` branch at
  `worker/model_runner_v1.py:567-570`, first tag `v0.18.0rc1`). The `v0.17.0` release notes record the class
  as a known issue ("… 2-node PD mixed deployment … may hang when concurrency exceeds 8") — quote that line,
  keep the residual `hypothesis`, and treat a recurrence on `0.18.0rc1`/`0.20.2rc1` as an open report
  (`aclgraph-mode-and-capture.md` §G).
- **A prefill that repeats `fail to receive KV Cache transfer done signal` is holding blocks, and on
  `≤ v0.11` the owner is the legacy `llmdatadist` C-mgr connector.** `envs.VLLM_LLMDD_ABORT_REQUEST_TIMEOUT`
  (default `120`, `vllm_ascend/envs.py:140-141` at `v0.9.1`) only *releases* the producer's blocks late, so
  the log's own advice (`export VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=600`) lengthens the stall instead of fixing
  it: `distributed/llmdatadist_c_mgr_connector.py:286-287` stamps the expiry in `request_finished` and
  `:882-892` prints the warning, while only a decode-side `ReqForFinished` per announced decode TP rank
  releases the block. The file is **not** under `distributed/kv_transfer/` at those tags and the whole
  connector was offlined in #4780 (`a77045f35`, first `v0.12.0rc1`) — `pd-llmdatadist-timeout.md`.
- **Read the effective graph mode from the log, not the flag**: `platform.py` prints `… compilation
  enabled on NPU` above the traceback and rewrites `FULL_AND_PIECEWISE`→`PIECEWISE`, `NONE`→
  `NO_COMPILATION`. Expanded-size `RuntimeError` at a `query_start_loc` write = buffer arithmetic, config-time
  `AttributeError` on a wrapper config = derived capture sizes, graph-mode death without a worker traceback =
  support gap, garbage only in graph mode = the dummy-batch slots, a freeze under `full_cuda_graph: true`
  with a `graph_task_update_end` frame = the 0.9-only flag (`aclgraph-mode-and-capture.md` §A–§F). A
  `torch._dynamo.exc.Unsupported … posix.putenv` is a **host write inside the traced body**: the skipped
  callable, not the frame, is the search key (`dynamo-graph-break.md`; levers: pre-set the variable, keep
  per-forward tokens under 65 535, or `--enforce-eager`).
- **A Triton launch error is host-side, not a device fault**: `coreDim=<N> can't be greater than
  UINT16_MAX` on a snake_case `*_kernel` is launch geometry growing with scheduled tokens
  (`kernel-launch-limits.md`).
- **Keep three evidence classes apart** — *verified* (code at the ref, ancestry, patch, test, doc line),
  *reported* (issue text, maintainer comment, log), *hypothesis* (inference plus falsifier) — and make the
  **symptom follow from the mechanism**: one predicting an exception where the report shows a stall is not
  yet the cause.
- **A mechanism that no clone can show ends the search, not the answer**: CANN/HCCL resource arithmetic,
  an unopened installed-wheel behaviour, a metric explained only by a fix history. One batch pins what the
  ref does, then the reply carries the workaround and the mechanism as `hypothesis` with its falsifier
  (selector line here, owner there: `moe-ep-dispatch.md`, `quantization-load.md` §5,
  `graph-capture-and-custom-ops.md` §C, `references/vllm-ascend/npu-device-visibility.md`). The same rule
  covers a package manager's or engine's internals behind a repo line — state the gap instead.
- **A reduction fault can present as fluent nonsense**: a `copy_` mismatch whose row ratio equals
  `tensor_parallel_size`, or output scaled by `tp` (`flashcomm1-and-moe-reduce.md`). 乱码 that appears only
  with `--enable-expert-parallel` and disappears without it is the same family on an EP-enabled MoE model:
  the comm type — not the model — differs between the two runs (`select_moe_comm_method`,
  `vllm_ascend/ascend_forward_context.py:199` at `v0.17.0rc1`), and the reporter-confirmed lever is
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` **with** EP kept on (#8223, `flashcomm1-and-moe-reduce.md` §B).
- **A P/D deployment on MTP that answers fluent garbage has one owner: the connector's block-id cut.** The
  P side ships every computed block id while decode allocates only the prompt's blocks, so with MTP the one
  extra speculative block misaligns the positional remote↔local mapping whenever the prompt length lands on
  a block boundary — stray tokens inside otherwise normal sentences, no error, plog clean.
  `mooncake_connector.py:1065` at `v0.18.0` sends the unstripped list (`num_prompt_blocks` computed at
  `:1060`); `015af05d6` (#8908) cuts it at `:1121`, and the hybrid child gets `_compute_transfer_block_ids`
  in `31860b0d7` (#9810) — **both first tag `v0.20.2rc1`**. A connector file the tag's tree does not
  contain (no `mooncake_hybrid_connector.py` at `v0.18.0`) means a named image, not the release: say so
  instead of reading `main` (`mooncake-transfer-errors.md`, "The silent sibling"). The EP/FlashComm1 twin
  is the rule above; garbage at every prompt length is not this class.
- **No hardware, no extrapolation; declare gaps.** Their `npu-smi`/`collect_env` is the only hardware
  evidence; a UT-level check is not hardware validation, and a missing ref or pinned dependency is a gap to
  state, not paper over with current code.
- **A deepest frame that is a downstream patch's cancellation raise is teardown, not cause.**
  `RuntimeError: cancelled` out of `patch_shm_broadcast.py` fires only after `self.shutting_down` (vLLM
  #45224 / vllm-ascend #14181): the worker died first, and its log/plog holds the real error
  (`log-signatures.md`).
- **A `[Community Test]` / 众测 task sheet is a deployment question, not a bug.** The thread is empty by
  construction (14/14 open, zero comments); the evidence is the linked tutorial **at the sheet's own 蓝区镜像
  tag**, read against itself (image pin, hardware, serve block, matrix row). Its `HEAD` may be a
  deprecation stub — main-only, never the reporter's (`community-test-task-sheets.md`).
- **A crash, a `TypeError`, or a request that never runs while an Ascend scheduler subclass is selected,
  means that subclass, not upstream's scheduler.** (a) `AscendScheduler` (`ascend_scheduler_config.enabled`, default false but *auto-enabled*
  for a non-MLA model on the `v0.11` line — `platform.py:139-163` at `v0.11.0rc0`; gone after
  `v0.11.x`; marker `Using configured V1 scheduler class …`): its `.blocks[0]` `IndexError` is a zero-KV-group
  (pooling) model meeting an unconditional index — drop the block, not the pooling attention, and do not
  infer from the reporter's command that it was off. (b)
  `SchedulerDynamicBatch` (`additional_config.SLO_limits_for_dynamic_batch != -1`, gate `platform.py:713` at
  `v0.23.0`, 0.19–0.23): a `TypeError` on `hash_block_size` is the subclass never taking an upstream keyword,
  not a config error — the sibling `scheduler_profiling_chunk.py` at the same tag already accepts and forwards
  it. Drop the block (the default scheduler starts) or upgrade past the removal (#11933, `v0.24.0rc1`)
  (`ascend-scheduler.md` §B). (c) `BalanceScheduler` (`patch/platform/patch_balance_schedule.py`, env
  `VLLM_ASCEND_BALANCE_SCHEDULING=1`, installed by a module-level `Scheduler = BalanceScheduler` rebind):
  **no error at all** — its copied `schedule()` loops on `self.waiting` only, while upstream's `add_request`
  parks every structured-output request in `skipped_waiting` (`Request.__init__` sets `WAITING_FOR_FSM`), so
  a streamed guided request never emits a first token; unset the var or take a build with #10509/#10699/
  #10559 (`ascend-scheduler.md` §C).
- **A `TypeError` naming a `transformers` processor argument is version skew, not an NPU or model bug.**
  `Glm4vProcessor.__init__() got multiple values for argument 'tokenizer'` is transformers `4.53.0`'s
  `from_args_and_dict` deleting positional args by name while vLLM passes `tokenizer=` — and vLLM suppresses
  that kwarg only when `transformers.__version__` is *exactly* `4.53.0`, so a source/`dev0` install bypasses
  the guard. Identify the library by its line numbers (`processing_utils.py:1105`+`:1304` = 4.53.0); the lever
  is `transformers>=4.53.1` (`model-type-support-floor.md`, "processor twin").
- **"Does the Mooncake store backend work without `-DUSE_ASCEND_DIRECT=ON`?" is answered by the backend's own
  gate, not by Mooncake's transports**: `MooncakeBackend` drives KV bytes through ADXL/HIXL and accepts only
  `protocol: "ascend"` — `UnboundLocalError` on `ret` at `v0.12.0rc1`, `NotImplementedError` from
  `v0.19.1rc1` (`kv-pool-store-hang.md` §D; #5299).
- **A source install that dies with no `vllm_ascend/` frame is the build script, and from `v0.12.0rc1` that
  script needs a git submodule.** `csrc/build_aclnn.sh` requires `csrc/third_party/catlass/include` and, when
  it is absent, runs `git submodule update --init --recursive` against `https://gitcode.com/cann/catlass.git`
  — `exit 1` on failure, which `setup.py`'s `build_aclnn` command turns into `SystemExit` and the whole
  `pip install -e .` aborts before cmake. 310P exits before that check; `main` additionally checks out a
  pinned commit. Lever: pre-seed the directory or redirect git to a mirror
  (`references/vllm-ascend/install-and-image.md` §3, #8788).
- **Two unrelated symptoms in one body get two route rows, not one shared mechanism.** A body that pairs a
  PD decode-node hang with an OOM on another host, and a thread whose only answer is "known issue, we'll fix
  it", has no unified cause to find: answer each symptom with its own page's lever and label the shared story
  `hypothesis`. A recorded run spent four batches looking for one mechanism spanning a hang and an OOM and
  shipped nothing (#10963 → `pd-collective-hang.md`, `post-startup-oom.md`).
- **A multi-node DP launch that stalls right after the engine banner is rank arithmetic before it is a
  collective fault.** The DP ranks actually started must sum to `--data-parallel-size` — that is the sum of
  `--data-parallel-size-local`, with `--data-parallel-start-rank` the running total of the earlier nodes —
  and only `local <= size` is validated (`vllm/config/parallel.py:286` at vLLM `v0.13.0`). `dp_rank 0`
  handshakes with every DP engine (`vllm/v1/engine/utils.py:841-848`) and waits in a poll loop whose only
  progress line is `logger.debug` (`:941-952`), so a short launch hangs **silently and forever**: no worker
  log, no HCCL, no traceback. Worked case #5942 (DP4 TP4 on two A2 nodes, `--data-parallel-size-local 1` on
  each): the reporter's own fix is `--data-parallel-size-local 2` on both nodes plus
  `--data-parallel-start-rank 2` on the headless one, with `--gpu-memory-utilization > 0.95` answering the
  sibling OOM. `references/symptoms/startup-and-stall.md` §6; `VLLM_LOGGING_LEVEL=DEBUG` prints the
  shortfall.
- **The reply is the deliverable — obey the write gate above.** An answer still being polished when the
  run ends delivers nothing.

## Procedure

`REPO` is the clone (`/workspace/repos/<owner>/<repo>`), `TAG` the reported ref; the clone set is fixed
(`vllm`, `vllm-ascend`, `huggingface/transformers`, `Mooncake`, `Ascend/{torchair,memcache,pytorch,cann}`,
mirrors). The eight-step order (thread, pin the ref, harvest the log, read the failing path at that ref,
release notes and tutorial config, the fix and its inclusion, the class, the answer) is an **order, not
eight turns**: batch 1 is 1–2, batch 2 covers whichever of 3–6 the symptom needs, 7–8 are the reply, and a
route row may be answered right after batch 1 (`references/workflow/index.md`,
`references/vllm-ascend/repo-map.md`); stop once the mechanism is confirmed at the ref and one independent
signal agrees.

## Reference map

- `references/workflow/index.md` — the method: version pinning, evidence sources, log-signature routing, fix
  discovery, answer shape, order of work.
- `references/vllm-ascend/index.md` — repo map (incl. which clone owns a `/vllm-workspace/vllm/…` path),
  layout drift, version pairing (incl. the rule that the tag's own `Dockerfile` + `installation.md`
  already pin the CANN version a report names, so a "CANN skew" needs the `collect_env` to disagree, and
  that a tag's `Dockerfile` can pip-install a `torch-npu` *post* build that contradicts its own
  `pyproject.toml` pin — `v0.13.0rc2` ships `post1` over a `2.8.0` pin),
  `additional_config` flag
  lifetimes (incl. §6's *vendor-env-bundle liveness* table and §7's `VLLM_ASCEND_BALANCE_SCHEDULING` patch
  with its stale `schedule()`/`run_engine_core` copies), the Dockerfile/`~/.bashrc` env class, container
  device visibility, and
  `install-and-image.md` (image → hardware class per ref, the `vllm_ascend_C` /
  `libvllm_ascend_kernels.so` build with its `COMPILE_CUSTOM_KERNELS` gate and install-prefix rule, and §4:
  the image's **editable** install under `/vllm-workspace/vllm-ascend`, so a mounted checkout loses the
  build-generated `vllm_ascend/_build_info.py` — the builder-facing half of a `[Doc]` dev-env report; and
  §5's wheel that silently omits a package tree, read from `setup.py`'s `packages=find_packages(...)` line
  plus the tree at the reporter's sha — one directory whose parent lacks `__init__.py` drops the whole
  subtree).
- `references/symptoms/index.md` — the router for every verified failure class, one page each with the
  worked report, plus the page detail behind each `symptom-index.md` / `symptom-index-stages.md` row. Its
  scope rules decide whether a
  report is a device fault, a tracing failure, a doc block or a config copy. Frequently missed:
  `graph-capture-and-custom-ops.md` §B2/§C/§C2 (§B2 is the 310P warm-up `LinearOperation`),
  `aclgraph-mode-and-capture.md` (§A–§H: `FULL_DECODE_ONLY`
  support by release, decode padding, capture sizes, graph-only garbage, the legacy `full_cuda_graph`
  freeze, §G's `num_reqs_padded` dummy-batch death on a 2-node PD deployment **and its repeated-token twin**
  (#9666: one token looping to `max_tokens` at high concurrency on `v0.17.0rc1`, gone on `v0.18.0`), and
  §H's 310P capture-**event** exhaustion at `capture_end`), `fa3-backend.md` (the FA3 backend a report
  selected: mixed prefill+spec-decode batches and the eager `507015` they cause, fixed on `main` only),
  `pd-proxy-examples.md`,
  `dynamo-graph-break.md`,
  `pooling-model-startup.md`, `v1-engine-selector.md`, `missing-downstream-feature.md`,
  `memory-sizing-and-parallelism.md`, `doc-guide-vs-capability.md`,
  `model-type-support-floor.md` (all three members of the dependency-floor class — a `model_type` the
  installed transformers lacks, a `tokenizer_class` such as `TokenizersBackend` it never had, and a
  processor constructor that rejects the way vLLM calls it),
  `benchmark-accuracy.md` (a score below the model's own number or two harnesses disagreeing; an
  accuracy collapse that tracks DP>1, where `platform.py` has already replaced the etp the reporter set;
  and §4's image prompt that answers garbage because the caller's dict key is not `multi_modal_data`, which
  vLLM drops without validation),
  `reasoning-parser-output.md` (`reasoning` vs `reasoning_content`, the two prompt states, the parser/tool
  registry split, a turn-2 loss that is the prompt, the apparent-TTFT measurement),
  `kv-pool-store-hang.md` §C for a rank that dies silently during a `load_async=False` pool reload, §D for
  the store backend's single `protocol: "ascend"` (and the `-DUSE_ASCEND_DIRECT=ON` requirement), §E for a
  lookup that hits while `need to load` stays 0 (upstream's `need_to_allocate` clamp),
  `quantization-load.md` §7 for a quantised artefact the model's own tutorial never lists, §2d for a nested
  `llm.` prefix the loader does not strip, §3 for a `deq_scale`/`weight_scale` `KeyError` that means the
  model was built dense (and how to tell that the `--quantization` flag never reached the process), §9 for a
  scheme a newer release no longer registers — `NotImplementedError … doesn't support quant_type=… for
  layer_type=…`, answered with the last tag that carried it and the replacement, not a regression hunt,
  `tool-call-streaming.md` for every tool-call payload shape incl. §4a's swallowed
  `TypeError: expected string or buffer` out of upstream `glm4_moe_tool_parser` (fix #32321, first tag
  `v0.15.0rc0`), §4b's whitespace-only `content` delta
  from a parser bundled out of upstream vLLM and §4c's internal markup left in `content`,
  `grammar-bitmask-and-dep-skew.md` §D for the `-1` placeholder assert under MTP plus structured output,
  `pd-llmdatadist-timeout.md` (PD on an `LLMDataDistCMgrConnector` build: the prefill holds blocks for
  `VLLM_LLMDD_ABORT_REQUEST_TIMEOUT`, and that connector is gone from `v0.12.0rc1`),
  `mooncake-transfer-errors.md` (`status: 103900` is the ADXL
  `PARAM_INVALID`, an engine-name-form mismatch; "The silent sibling" is the page's P/D + MTP accuracy
  half — fluent garbage from an uncut block-id list, both fixes first tag `v0.20.2rc1`; "The TP-inequality
  shape" is the *alternating* one — the KV reformat ran before all TP-offset pulls arrived, fixed per
  branch `86217325b`/`v0.23.0`, mainline `1db626510`/`v0.27.1rc1`),
  `moe-ep-dispatch.md` (§A `EI0007` on the all-to-all, §B `107027` inside capture, §C
  `VLLM_ASCEND_ENABLE_FUSED_MC2=1` on a BF16 checkpoint — the gate needs `w8a8_dynamic`, and the forced op
  hangs above an M of 2048 with only upstream's 60 s `shm_broadcast` line to show for it),
  `multimodal-input-not-applied.md` (a VLM answering as if the image were absent: the
  `multi_modal_data` key is the only spelling vLLM reads and the rest of the prompt dict is unvalidated,
  then the placeholder token, then the `--limit-mm-per-prompt`/encoder budget — evidence is
  `usage.prompt_tokens` growth, not the echoed prompt),
  `post-startup-oom.md` (an OOM after a healthy start: the
  per-stream workspace cache and the per-`num_tokens` workspace tensors), `kv-connector-registration.md` (a
  pre-forward layer-name `KeyError` is a lookup-scope defect), `multi-connector-composition.md` (a
  `MultiConnector` starving its layerwise child; legal only from `v0.16.0rc1`),
  `310p-capability.md` (the 310P product line: the tag's own 310P page and its serve block, the doc's
  per-model capture-size lists stated per TP, the explicit `--max-model-len` rule, and the W8A8SC
  per-rank `TP<N>/` checkpoint directory with its `--load_format sharded_state` gate; the first-tag
  support timeline is `references/vllm-ascend/hardware-support-by-release.md`),
  `community-test-task-sheets.md` (incl. a sheet whose 任务问题和需求 is a *tooling* complaint and whose
  cited tutorial path does not exist at its own tag), `ascend-scheduler.md` (any selected Ascend scheduler
  subclass: `blocks[0]` on a pooling model, a `TypeError` from signature drift, or §C's env-gated
  `BalanceScheduler` whose copied `schedule()` never reads upstream's `skipped_waiting` queue so a guided
  request stalls silently),
  `pcp-dcp-and-mtp.md` (the PCP/DCP members: the inflated prefix-cache unit, `mtp_slot_pad`, the blanked
  `seq_lens_cpu`, the §D throughput cliff, and §E's `num_decode_tokens`-as-index broadcast under MTP),
  `deepseek-v4-thinking-and-effort.md` (its own class: the DeepSeek-V4 thinking/effort default, not the
  harness measurement question) and `sleep-mode-and-allocator.md`. `startup-and-stall.md` §3b
  covers `max_model_len` past the KV budget, §3c a per-rank `Available memory: 0` whose `profile_run` was
  blown up by the platform's `max_num_batched_tokens → max_model_len` rewrite (and the log line that dates
  the image before that rewrite was removed),
  §6 a multi-node DP launch whose per-node local ranks do not
  add up to `--data-parallel-size` (silent coordinator wait, debug-level retries), §7 an
  `init engine … took N seconds` that regressed between two tags (what newly enters compile/capture), §8
  the `WorkerProc initialization failed … background process` wrapper whose child was killed rather than
  raising;
  `kv-admission-cliff.md` a serving hang that starts at a
  prompt-length threshold; `lora-and-tp.md` §"It serves and is slow" the 2–3 tokens/s adapter.
- `references/hccl/index.md` — the comm layer as an owner: HCCL/HCCP plog keywords (`Getting socket times
  out`, `EJ0002 … The network port is down`, `ra rdma init failed`, `listen on ip[..] port[..] fail`,
  `wait socket establish timeout`, `CreateCommPlane failed …_MC2MultiServer`, `Task run failed`) mapped to
  the vendor's EI codes and stage pages (`cann/hccl` `docs/*/user_guide/fault_diagnosis/`), the
  failing-rank-pair rule, the graph-mode relabel (`LoadGraph failed` + `EJ0002`), and
  `env-support-matrix.md` for what `HCCL_INTRA_PCIE_ENABLE`/`HCCL_INTRA_ROCE_ENABLE` (intra-server
  **only**), `HCCL_OP_EXPANSION_MODE` (AIV's per-product limits),
  `HCCL_CONNECT_TIMEOUT`/`HCCL_EXEC_TIMEOUT`/`HCCL_NPU_SOCKET_PORT_RANGE` actually select. Read the plog
  top-down — the first error names the plane, a `192.168.*` device IP is not `HCCL_IF_IP`, EP makes
  `ep_size = world_size_across_dp` — and treat the launch script/doc block as the artefact that decides it.
- `references/symptoms/issue-ledger/` — the analysed reports with their settled facts (`index.md`: which
  report is where; `reports-early.md`/`reports-later.md`/`reports-newest.md`: the rows;
  `lessons-across-reports.md`).

## Scripts and template

Shell with `bash`, Python with `uv run --no-project python`; every script is read-only against the clones,
writes only under `/workspace/scratch` and takes a clone name or path (`scripts/repo_root.sh` resolves it;
exit 3 = no such clone). On the gate:
`scripts/open_answer.sh` (batch 1), `scripts/ref_snapshot.sh <ref>` (batch 2); for a raw tool-call stream,
`scripts/merge_tool_deltas.py` (`tool-call-streaming.md`); for a reasoning/thinking report,
`scripts/reasoning_output_probe.sh <vllm-ascend-ref> [vllm-ref]` prints the response field name, the
history field the renderer reads, both parser registries and the ascend-side patches for that tag pair
(`reasoning-parser-output.md`). Every argument, exit code and
`templates/finding-report.md`: `references/workflow/index.md` ("Support").
