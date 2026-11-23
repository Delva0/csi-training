# The graph lane, read off the serve line — and the opposite rungs it implies

## Scope

A **2×N 910B-era MoE serve** (GLM-4.5/4.6, Kimi-K2, Qwen3-MoE, DeepSeek-V3.x) on
vllm-ascend **v0.10.x**, where the pasted evidence is the *serve command* — often with
no log at all — and the symptom is either **throttled decode** ("输出 token 间时延
400 ms 以上") or **corruption/degeneration** ("简单问题正常，复杂问题回答混乱"). One
flag in that command decides which graph exists, and the two symptoms need that flag
moved in *opposite* directions. Classify the lane before ranking any mechanism: the
model+layout match that leads here is shared by both symptoms, so a page indexed by
model and card count will send half its readers the wrong rung.

Not this page: the flag is *missing or asymmetric* in a DP×TP command
(`moe-comm-path-when-ep-off.md` — its own scope excludes a command that carries
`--enable-expert-parallel` on both nodes), a graph-mode **start/capture crash**
(`../startup-failures/captured-stream-host-sync.md`,
`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`), a "graph mode is
slower than eager" report with the reporter's own ablation
(`graph-mode-slower-than-eager.md`), or a **pre-v0.10** engine where the platform
forces eager (`pre-graph-mode-eager-era.md`).

## Step 1 — four branches, one grep of the command and one of the config

At vllm-ascend **v0.10.0rc1** (`4604882a3e2c`) `check_and_update_config` reduces the
compilation level in this order:

| what the reporter's command carries | level | the line it prints | anchor |
|---|---|---|---|
| `--enforce-eager`, or `--compilation-config '{"level":0}'` | `NO_COMPILATION` | `Compilation disabled, using eager mode by default` | `vllm_ascend/platform.py:143-145` |
| `additional_config` `torchair_graph_config.enabled` | `NO_COMPILATION` | `Torchair compilation enabled on NPU. Setting level to NO_COMPILATION` — but on a model outside `TORCHAIR_MODEL_LIST` it raises first | `:151-156`, `vllm_ascend/ascend_config.py:20,23-27,165-168` |
| `--distributed-executor-backend ray` | `NO_COMPILATION` | `Ray distributed executor backend is not compatible with ACL Graph mode` | `:157-161` |
| nothing (the default) | `PIECEWISE` → **ACL graph on** | `PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode` | `:162-168` |

The default row is upstream's: V1 sets `level = CompilationLevel.PIECEWISE` whenever
`model_config.enforce_eager` is false (`vllm/config.py:4587-4591` at vllm v0.10.0), and at
that ref `PIECEWISE` is **3**, not 2 (`:3926-3931`) — so `"level": 0` in a
`--compilation-config` is the same lever as the flag and any other value is downgraded
with `NPU does not support … compilation level` (`platform.py:146-150`).

`TORCHAIR_MODEL_LIST` at this pin is `["deepseek", "pangu", "kimi_k2"]`
(`ascend_config.py:20`, lower-cased substring test at `:23-27`), so **`glm4_moe` cannot
take the torchair route** — its only graph is the "experimental" aclgraph branch, which
prints `ACL Graph is currently experimental … only qwen model is well tested`
(`ascend_config.py:171,181`) and does **not** refuse. (At v0.10.1rc1 the list had already
grown a `"qwen"` entry — re-grep both files per ref; the branch order above is what to
check, not the line numbers.)

So the question "is the graph on?" is answerable from the command before any log is
requested — and the two answers below cost one launch each.

## Step 2a — throttled decode **with** `--enforce-eager`: the flag is the cause

A `v0.10.x` MoE serve at 300-500 ms/token where the command itself carries
`--enforce-eager` is running every MoE layer host-launched, and the reporter chose that.
Rung 1 is to remove the flag and re-measure the same body — labelled a **hypothesis**:
eager-vs-graph on a healthy serve is usually worth ~1.2-2×, not the ~10× a 400 ms/token
decode implies, so keep a second rung behind it (the model directory's own quantisation:
a bf16 355B checkpoint double-reads expert weights per step against the W8A8 artifact the
release note's recipe uses). What the tree does when the flag comes off, and what to ask
the reporter to read:

- capture sizes are clamped by model depth, not by `max_cudagraph_capture_size`
  (that key does not exist at this pin):
  `max_num_batch_sizes = floor(MAX_CAPTURE_SIZE / (num_hidden_layers + 1) / parallel_factor)`
  with `MAX_CAPTURE_SIZE = 1920` (`vllm_ascend/utils.py:47,309-361`) and
  `parallel_factor = 1 + [dp_local > 1] + [tp > 1]` (`:325-328`) — 2 for a DP2×TP8
  layout, so a ~90-layer MoE lands near 10 capture sizes;
- it logs which of the two branches ran: `Calculated maximum supported batch sizes for
  ACL graph: N` (`utils.py:333-334`), then either `Adjusted ACL graph batch sizes for
  <Arch> (layers: L): a → b sizes` (`:347-352`) or `No adjustment needed …` (`:358-361`).

Do **not** transplant `sequence-count-cliff-capture-size.md`'s
`min(max_num_seqs × (1+num_spec), 512)` rule to this ref: that default is a later one,
and the depth-divided `1920` is what bounds capture here. Also drop the new release's
`max_cudagraph_capture_size` flag from any advice at this pin (empty grep is the proof).

## Step 2b — corruption **without** the flag, on a DP+EP serve: the flag is the cure

When the command has `--data-parallel-size N` + `--enable-expert-parallel` and **no**
`--enforce-eager`, the PIECEWISE branch above is what actually ran, and the pinned
release itself documents that combination as broken:

> `docs/source/user_guide/release_notes.md:53` (v0.10.0rc1, Known Issues):
> "Aclgraph could not work with DP + EP currently, the mainly gap is the number of npu
> stream that Aclgraph needed to capture graph is not enough. [#2229]"

Consistency check, not a proof: the decode MC2 path (`ep_size ≥ 16`, the DP2×TP8 case)
resolves its shared experts on a second stream — `npu_stream_switch("moe_secondary", 0)`
in `vllm_ascend/ops/fused_moe.py:191,273,1377` — which is the kind of extra stream the
row describes. Keep the mechanism labelled a hypothesis; the row is the project's
statement at that ref, and the observation supports it.

The deliverable is then a **two-rung** answer, not a source walk:

1. restart the same serve with `--enforce-eager` (both nodes, one variable) and re-send
   the *same* body. Corruption gone ⇒ the lane is closed: aclgraph × DP+EP at this ref,
   and the report is the reporter's flags plus their release. Corruption unchanged ⇒
   the graph lane is spent and the ranking moves to the body/sampling and the EP path.
   The line that proves the restart did what you asked is
   `Compilation disabled, using eager mode by default`.
2. read the request body before ranking numerics. A pasted `curl` that sets only
   `model`/`messages`/`max_tokens`/`stream` runs at the **server default sampling** —
   ask for the same body with `temperature: 0` (and `return_token_ids: true`) so
   degeneration is separated from a greedy-only corruption. `finish_reason: "length"`
   at the token cap with a colon-separated word loop, and `prompt_tokens` printed beside
   it, is the cheapest reproducible artefact the reporter already has.

Also note the polarity rule in one line inside the answer: at this era
`--enforce-eager` is a **cost** on a healthy serve and a **cure** on a DP+EP corruption
report. Quoting `qwen3-next-decode-throughput.md`'s "drop the tutorial's
`--enforce-eager`" into the second shape inverts the advice.

## Recorded cases

Both bodies below arrived **without a command** and got it as a later comment ("部署命令和
测试结果如上"), so the first answer owes both polarities in one paragraph — "if your
serve line carries `--enforce-eager`, drop it and here is the line that proves the graph
came back; if it does not and you run EP+DP, add it and re-send the same body" — instead of
one hypothesis that the follow-up then confutes.

- **#2446** — `[Performance]: 在两台8卡910B上跑glm4.5吞吐极低`, vllm-ascend v0.10.1rc1,
  GLM-4.5, 单/8 并发 both 400 ms+/token. The index routed the run to
  `moe-comm-path-when-ep-off.md` because that row names the same model, card count and
  release; the follow-up command then showed `--tensor-parallel-size 8
  --data-parallel-size 2 --enable-expert-parallel` (so `ep_size = 16`, MC2 decode) plus
  `--enforce-eager` — the EP-off page's own "not this topic" line applies. The run had
  `platform.py:143-147` and `MAX_CAPTURE_SIZE = 1920` by its 9th tool call and still shipped **no
  answer text**: two message drafts were cut mid-sentence, at the 8 192-token item cap.
- **#2537** — `[Bug]: v0.10.0部署GLM-4.5模型精度存在问题`, same 2×8 910B layout, DP2×TP8,
  `--enable-expert-parallel`, `--max-num-batched-tokens 2048`, `--max-model-len 16384`,
  no `--quantization`, `--no-enable-prefix-caching`, **no** `--enforce-eager`. The two
  `curl` transcripts were the whole decisive artefact: good run `prompt_tokens: 8` with
  a `<think>` block, bad run `prompt_tokens: 60`, `finish_reason: "length"`,
  `completion_tokens: 300`, content a `major X: major Y:` word loop with underscores and
  no `<think>`; neither body sets `temperature`/`top_p`. The run re-derived the MoE
  gating op for 26 turns (upstream's sigmoid path is bf16 there too, so the
  precision-drift hypothesis refuted itself), emitted 53 characters of message text in
  total, and shipped nothing — while the release-notes row above and the two OCR'd
  bodies (under `/workspace/attachments/<issue>/*.ocr.txt`; plain
  `ls /workspace/attachments` prints nothing for this per-issue layout) had them.

## Pins and recheck

Verified at vllm-ascend **v0.10.0rc1** `4604882a3e2c` + vllm **v0.10.0** `6d8d0a24c02b`.
Re-grep `platform.py` (branch order), `utils.py:47` (`MAX_CAPTURE_SIZE`),
`ascend_config.py:20` (`TORCHAIR_MODEL_LIST`) and `release_notes.md` (does the #2229 row
still stand in the next rc?) at any other ref. From v0.11.x on, the aclgraph warnings,
the capture-size plumbing (`max_cudagraph_capture_size`) and the torchair list all
differ, and the DP+EP × graph combination eventually becomes the default path — the row
is scoped to the releases whose Known Issues still print it.
