# Auditing a tutorial page: twelve checks, in this order

Every item is one or two shell calls against the pinned tree; the examples are
what the checks returned for the Qwen3-Omni tutorial
(`docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md`) at vllm-ascend
v0.18.0 (`e18643f8a4d5`) + vllm v0.18.0 (`bcf2be9612`), and the same page at
v0.19.1rc1 (`da421afad7`). Tier the result: **breaks the command** / **stale
baseline** / **cosmetic**.

## 1. Does every flag on every command exist at the pinned vllm?

```bash
rg -n -- '--swap-space|swap_space' vllm/engine/arg_utils.py vllm/config/cache.py
rg -n -- '--enable-log-requests|enable_log_requests|disable_log_request' vllm/engine/arg_utils.py
```

Result at v0.18.0: `swap_space` is gone from both files — CPU/KV offloading is
configured elsewhere — and only `--enable-log-requests` exists
(`vllm/engine/arg_utils.py:2208,2221`; the gate table is
`../observability-and-logs/request-output-logging-flags.md`). The doc's
performance server line (`:288` at v0.18.0, `:294` at v0.19.1rc1) passes
`--swap-space 16 --disable-log-stats --disable-log-request`, so **that command
cannot start** at v0.18.0 even though `--disable-log-stats` and `--load-format`
on the same line are fine. Report the line and the corrected one.

## 2. Does the `vllm bench` subcommand use the argument names of that release?

```bash
rg -n -- '"--random-input|"--random-output|"--num-prompts"' vllm/benchmarks/datasets.py
```

At v0.18.0 the names are `--random-input-len` (`:1565`) and
`--random-output-len` (`:1571`). The doc says `--random-input 200` (`:293`
v0.18.0) — and that still parses, because `FlexibleArgumentParser`
(`vllm/utils/argparse_utils.py:49`) subclasses `argparse.ArgumentParser`
without setting `allow_abbrev=False` (no occurrence in `vllm/`), so argparse
accepts it as an unambiguous prefix. Tier it **cosmetic** and say why; calling
it a startup failure is the more common mistake, and a wrong severity costs the
report's credibility.

## 3. Does every relative path in the doc exist, with the image's own WORKDIR?

```bash
rg -n 'WORKDIR|COPY \. ' Dockerfile
```

At vllm-ascend v0.18.0: `Dockerfile:30` `WORKDIR /workspace`, `:32`
`COPY . /vllm-workspace/vllm-ascend/`. The doc's
`pip install -r vllm-ascend/benchmarks/requirements-bench.txt` therefore
resolves only from `/vllm-workspace`; from the container's default cwd it fails
`No such file or directory`. Tier **breaks the command** (usability), give the
absolute path, and note that the bench dependency install step itself has bitten
before (#4737: v0.11.0rc2/rc3 images shipped without the bench dependencies).
The same check covers every **helper script the doc tells you to run** — one
`rg --files -g '<name>.sh'` settles it. Recorded #15389: the Qwen3.5-397B-A17B
page's §5.6 A3 tab ends with `cd vllm-ascend/examples/disaggregated_prefill_v1/`
+ `bash proxy.sh` (`:836-838` at v0.23.0), and that directory holds only
`load_balance_proxy_server_example.py`,
`load_balance_proxy_layerwise_server_example.py` and
`mooncake_connector_deployment_guide.md` — no `proxy.sh` anywhere in the tree.
Tier **breaks the command**, and quote the three files that do exist.

## 4. Which defaults is the doc silently relying on?

Two greps decide the multimodal cases:

```bash
rg -n 'def get_limit_per_prompt' -A14 vllm/config/multimodal.py
rg -n '_assert_url_in_allowed_media_domains' -A10 vllm/multimodal/media/connector.py
```

At v0.18.0 an unspecified modality's `limit_per_prompt` is **999**
(`:253-265`), so a single curl carrying image + audio + video needs no
`--limit-mm-per-prompt`; and `allowed_media_domains=None` becomes `[]`, with the
domain check skipped when the list is empty (`:158-167`), so remote `http(s)`
media URLs are allowed by default. Both are "the doc is right for a reason the
doc does not state" — the useful output is the FAQ line (the node needs egress;
behind a proxy, use `--allowed-local-media-path`) not a defect.

## 5. Does the stated card count match the weights, and do the sibling docs agree?

```bash
bash <skill_root>/scripts/card-budget.sh --params 30.5 --tp 2 --gmu 0.9 --cards 2
rg -n 'tensor-parallel-size should be at least' docs/source/tutorials/models/*.md
```

The doc says `requires 2 NPU Cards (64G × 2)` (`:19`) and its serve line is
`--tensor-parallel-size 2 --enable_expert_parallel` (`:185` v0.18.0, `:191`
v0.19.1rc1). Both flags exist — `--enable-expert-parallel` at
`vllm/engine/arg_utils.py:901` at v0.18.0 — and the underscore spelling parses
because the parser rewrites `_` to `-` (`argparse_utils.py:239`), so this line is
**not** a defect. The card count is consistent with the weights: 30.5B params at
2 B/param is ~61 GB, and TP=2 leaves ~30.5 GB per card against a 57.6 GB budget
at `gpu_memory_utilization 0.9`. But the same page's boilerplate sentence says
"for an Atlas A2 with 64 GB … tensor-parallel-size should be at least 1" (`:177`
v0.18.0, `:183` v0.19.1rc1), while the sibling tutorials of the same 30B-A3B
class say **at least 2**
(`Qwen3-30B-A3B.md:48`, `Qwen3-Coder-30B-A3B.md:63` at v0.19.1rc1). Tier
**stale/wrong**, and cite the sibling line — that is the parity proof, not your
own arithmetic.

The second boilerplate species is the **enablement sentence**, and its parity
anchor is on the same page. `Qwen3-VL-30B-A3B-Instruct.md:18` (recorded #11741):
"Running this model requires 1 Atlas 800I A2 (64G × 8) node or 1 Atlas 800 A3
(64G × 16) node" — while that page's own container line maps exactly two devices
(`--device /dev/davinci0`, `:42`, `--device /dev/davinci1`, `:43`) and its serve
line is `--tensor-parallel-size 2` (`:87`), and the in-tree gate
`tests/e2e/models/configs/Qwen3-VL-30B-A3B-Instruct.yaml:2,10` is
`hardware: "Atlas A2 Series"` + `tensor_parallel_size: 2` (plus
`max_model_len: 128000`, `gpu_memory_utilization: 0.7`, `enable_expert_parallel:
True`, mmmu_val gate `0.58`). The same `requires 1 Atlas … node` wording is copied
across siblings (`Qwen3-Coder-30B-A3B.md:19`, `Qwen3.5-27B.md:21-22`), so treat it
as prose that drifted from the recipe, not as a sizing claim: 30.5 B at 2 B/param
is ~61 GB and TP=2 is ~30.5 GB per card. Tier **wrong requirement** (it tells a
2-card tester they cannot run it); the correction is the device-map count, and the
task book's 卡数 is the same number. `scripts/card-budget.sh` only confirms it.

## 6. Which release does the reference number come from?

Grep the page for the release tag next to its result blocks: both the accuracy
and the performance block of the Qwen3-Omni page say "in `vllm-ascend:0.13.0rc1`
for reference only" (`:255`, `:296` v0.18.0) while the task image is v0.18.0.
Tier **stale baseline**: keep it as a baseline to beat, never as the expected
result, and never re-publish it as a measurement. When one of its numbers looks
anomalous, check the precedent before blaming the doc
(e.g. `#5991: Qwen3-Omni-30B-A3B-Thinking, when max-concurrency >= 2, TPOT only
100ms` against the page's ~99 ms mean TPOT).
A **dead flag** whose owner is the release, not the command, is graded one way
further: the page's image tag is not its text but the `{{ vllm_ascend_version }}`
substitution from `mkdocs.yml extra.*` at build time, so compare that value with
the tag the reporter pasted, and grade `/en/latest/` as main — `docs-version-space-and-image-tag.md`
(recorded #16954: a main-only `--kv-cache-dtype int8` pair printed beside a stale
`0.23.0` tag; tier **breaks the command**, but the defect is the version line).

## 7. What should the FAQ contain?

One call, then read 2-3 bodies whose titles name a mechanism (SKILL.md rule 8):

```bash
bash <skill_root>/scripts/history-titles.sh 'Qwen3-Omni' 
```

For this model the titles alone already list the FAQ a reporter will need:
startup abort in graph capture (`#3963 … AclrtSynchronizeStreamWithTimeout
(copy_stream), error code is 107027`), a startup failure on the same recipe
(`#4663 … 模型服务拉起失败`), `HcclAllreduce` at startup on a newer image
(`#7239`), `no module named 'flash_attn.ops'` from a processor import (`#5563`),
audio transcription returning random/empty text
(`#6699`), occasional decode hang (`#8070`), and a tutorial/FAQ thread
(`#3974`). Titles are precedent, not proof: quote them as titles.

## 8. Do the support matrix, the CI config and the release notes agree with the page?

The e2e config is the project's own known-good command for the same model
family: `tests/e2e/models/configs/Qwen3-Omni-30B-A3B-Instruct.yaml` at v0.18.0
runs `tensor_parallel_size: 4`, `gpu_memory_utilization: 0.7`,
`max_model_len: 8192`, `enable_expert_parallel: true`, MMMU `0.60` — while the
page documents TP=2 for the Thinking variant. Neither is wrong; say which
variant, which card count and which context length each number belongs to, and
flag any CANN/HDK requirement stated in the release notes but missing from the
tutorial's environment section (that is a 完整性 item with a `path:line`).

## 9. Does the page install a package version the image's own pins forbid?

```bash
rg -n 'pip install|transformers|torch==' docs/source/tutorials/models/<page>.md
rg -n 'transformers' requirements.txt              # vllm-ascend
rg -n 'transformers' requirements/common.txt       # vllm
```

At v0.18.0 the Qwen3-32B-W4A4 page says `pip install transformers==4.51.0`
(`:51`, with a comment that the msmodelslim converter needs it), while vllm pins
`transformers >= 4.56.0, < 5` (`requirements/common.txt:10`) and vllm-ascend
`transformers>=4.57.4` (`requirements.txt:42`). A user who converts and then
serves in the same container downgrades the runtime below both pins. Tier
**breaks the environment**, and say which order triggers it: the converter may
genuinely need 4.51, so the correction is a separate conversion environment (or
restoring the pins before `vllm serve`), never "the page is wrong to pin it".
This check generalises to any `pip install <pkg>==<ver>` or
`apt`/`yum install` line in a deployment step.

## 10. Does every model-specific override string name this checkpoint's family?

```bash
rg -n 'hf_overrides|architectures' docs/source/tutorials/models/<page>.md
rg -n 'ForSequenceClassification' vllm/docs/models/pooling_models/scoring.md
```

Pooling/reranker pages are the ones that carry an `--hf_overrides` architecture
string, and text/VL siblings of one model family share nearly all prose, so the
override line is a copy-paste casualty. At vllm-ascend v0.23.0 + vllm v0.23.0 the
text page `Qwen3-Reranker.md` carries the **VL** string on three lines (`:141`
A2/A3 serve, `:156` 300I DUO serve, `:284` mteb offline), while the sibling
`Qwen3-VL-Reranker.md:172,190,288` is where that string belongs; upstream
`docs/models/pooling_models/scoring.md:52,88` (`Qwen3ForSequenceClassification`)
vs `:100,109` (`Qwen3VLForSequenceClassification`) names each family, and the
repository's own e2e job runs the text name
(`tests/e2e/pull_request/one_card/lora/test_qwen3_reranker_lora.py:29`). Tier
**breaks the command** — not cosmetic: the wrong prefix routes to a multimodal
model class and startup dies in the APIServer with
`TypeError: Invalid type of HuggingFace config. Expected type: <class
'…Qwen3VLConfig'>, but found type: <class '…Qwen3Config'>` (recorded #14817).
Mechanism: `../pooling-embedding/seq-cls-arch-routing.md`.

## 11. In a tab-set, does each tab's block match its own prose and its own connector?

The model pages duplicate a section per SoC (`::::{tab-item} A3` /
`Ascend 950DT` / `A2`, paired by `:sync:`), and the blocks inside a tab are
copy-paste siblings of the other tabs. Two one-line greps compare a tab against
itself:

```bash
rg -n 'tab-set|tab-item|:sync:' docs/source/tutorials/models/<page>.md
rg -n 'kv_connector"|proxy_server_example|proxy_layerwise' docs/source/tutorials/models/<page>.md
```

The connector configured in the tab's **own** `--kv-transfer-config` decides
which proxy/script that tab may name: `MooncakeLayerwiseConnector` pairs with
`load_balance_proxy_layerwise_server_example.py` (the layerwise proxy drives
decode-first and exposes `/v1/metaserver`), the p2p `MooncakeConnector` pairs
with `load_balance_proxy_server_example.py`. Recorded #15389 at vllm-ascend
v0.23.0 (`5cb98caa`), `Qwen3.5-397B-A17B.md`: the A3 tab says "Use the layerwise
proxy script" (`:817`) and then runs the **non-layerwise**
`load_balance_proxy_server_example.py` (`:823`), while A3's own P/D commands use
`MooncakeLayerwiseConnector` (`:521`, `:597`); the 950DT tab (`:844-850`) runs
the layerwise script although 5.5 configures the p2p `MooncakeConnector`
(`:703`, `:786`). The two tabs' blocks are **swapped**. Tier **breaks the
command**, and the mechanism is checkable in the tree: the p2p proxy's
`build_prefill_request` sends `kv_transfer_params` =
`{do_remote_decode, do_remote_prefill, remote_engine_id, remote_block_ids,
remote_host, remote_port}` (`load_balance_proxy_server_example.py:790-800`)
while `MooncakeLayerwiseConnector`'s prefill branch reads
`params["remote_cached_tokens"]` (`mooncake_layerwise_connector.py:990`) — a
`KeyError` in `update_state_after_alloc` on P. A wrong-proxy report therefore
shows up as "P dies / P shutdown, D fine", not as a proxy-side error; the
layerwise proxy's own wildcard-`--host` guard (`:278-282`,
"Wildcard Address … is not allowed for proxy") is the other pre-run trap, since
D must be able to reach `host:port/v1/metaserver`.

A tab that is internally consistent is **not** a defect: when the reporter ran the
*other* SoC's block on their own cards, the page is fine and the report belongs to
fault triage — say which tab their command matched (`scripts/recipe-tab-diff.py
--page <page> --section <N.N> --command <their script>`, one call) and route the
live question to `../image-dependency-faults/native-string-abort.md` step 0
(#11300: A3 host running the A2 block). Grading the page for the reporter's tab
choice wastes the report.

## 12. Was the printed request/response body captured on the pinned release?

The doc's sample bodies are the one place where field names, not flags, can be
wrong, and one grep pair decides it:

```bash
rg -n 'class ChatMessage' -A12 vllm/entrypoints/openai/chat_completion/protocol.py
rg -o '"[a-z_]+":' docs/source/tutorials/models/<page>.md | sort -u   # keys the sample prints
```

At vllm v0.18.0 `ChatMessage` declares `refusal`, `annotations`, `audio`,
`function_call`, `tool_calls` and `reasoning` (`protocol.py:54-64`) and the choice
declares `stop_reason` / `token_ids` (`:84-94`), so a sample that prints those
fields is consistent — do not report the body as wholesale stale. The single
foreign key is the finding: `Qwen3-VL-30B-A3B-Instruct.md:130` and `:192` print
`"reasoning_content": null` inside `message`, and that name exists only on the
**request** side at this pin (`vllm/entrypoints/chat_utils.py:329,1538`), never in
the response model. Tier **cosmetic** (the command still works) and say what it
means: the body was captured on a different build, so its token counts
(`completion_tokens: 15`) are not this image's reference either — keep them as a
baseline, not an expected result. The same check catches a missing
`reasoning`/`annotations` field, which dates a sample the other way.

## Second worked page: Qwen3-32B-W4A4 at v0.18.0 (recorded #11759)

The same tree (`e18643f8`, vllm `bcf2be9612`), page
`docs/source/tutorials/models/Qwen3-32B-W4A4.md` (143 lines). Verified findings,
ready to reuse, each one `path:line`:

| Tier | Finding | Evidence |
|---|---|---|
| breaks intent | the verification curl sends `"max_completion_tokens": 128` to `/v1/completions`, which has only `max_tokens: int \| None = 16`; `OpenAIBaseModel` is `extra="allow"`, so the field is silently dropped and the doc's own request returns 16 tokens | page `:111`; `vllm/entrypoints/openai/completion/protocol.py:57`; `vllm/entrypoints/openai/engine/protocol.py:26-28` |
| breaks the environment | `pip install transformers==4.51.0` against the image pins (check 9 above) | page `:51`; `requirements/common.txt:10`; `requirements.txt:42` |
| breaks the command (conversion path only) | the container maps `/dev/davinci0` alone while the conversion step exports `ASCEND_RT_VISIBLE_DEVICES=0,1` — card 1 does not exist inside that container | page `:21` vs `:59`; harmless when the pre-quantised ModelScope weight is used, so tier it accordingly |
| completeness | no card count, no accuracy section, no performance section, no CANN/HDK requirement, while the sibling `Qwen3-Dense.md:265-311` carries Accuracy and Performance/`vLLM Benchmark` sections for the same model family | page headings (Introduction … Offline Inference) |
| cosmetic | the image line carries the `m.daocloud.io/` mirror prefix while the task book's 蓝区 image is `quay.io/...`; users behind other networks must strip it | page `:17` |
| pass | one card is arithmetically right: 32B at 0.5 B/param = 16.4 GB of weights against a 57.6 GB per-card budget at `gpu_memory_utilization 0.9`; bf16 would be 65.6 GB and would **not** fit | `bash <skill_root>/scripts/card-budget.sh --params 32.8 --dtype w4a4 --tp 1 --gmu 0.9 --hbm 64 --cards 1` |

## Third worked page: Qwen3.5-397B-A17B §5.6 at v0.23.0 (recorded #15389)

Page `docs/source/tutorials/models/Qwen3.5-397B-A17B.md`, pinned vllm-ascend
v0.23.0 `5cb98caaadef` (+ vllm v0.23.0 `0fc695fc`). Reporter: "5.6 请求转发-仓库示例
脚本 load_balance_proxy_server_example.py 错误", proxy starts but the P instance
goes down; swapping in `load_balance_proxy_layerwise_server_example.py` works.
Verified findings, each `path:line`:

| Tier | Finding | Evidence |
|---|---|---|
| breaks the command | §5.6 A3 tab prose names the layerwise proxy, the block runs the p2p one | page `:817` vs `:823`; connector `:521,597` is `MooncakeLayerwiseConnector` |
| breaks the command | §5.6 950DT tab runs the layerwise proxy while §5.5 configures p2p `MooncakeConnector` — the two blocks are swapped | page `:850` vs `:703,786` |
| breaks the command | `bash proxy.sh` names a file that does not exist in the tree | page `:836-838`; `ls examples/disaggregated_prefill_v1/` = two `.py` + one `.md` |
| cosmetic / FAQ | the section's "Common Issues Tip" (`:862`) never states the binding "layerwise connector ⇄ layerwise proxy", which is the whole failure | page `:862`; payload/key pair above |
| pass | both proxy scripts parse the doc's flags (`--prefiller-hosts/--prefiller-ports/--decoder-hosts/--decoder-ports/--host/--port`); the reporter's own successful swap confirms it | `load_balance_proxy_layerwise_server_example.py:271-282` (argparse + host guard) |

Open item, state it as such: whether the reporter's P crash is that `KeyError`
is source-inferred; the one command that closes it is
`rg -n "remote_cached_tokens|EngineCore failed|EngineDeadError" <P node log>`
over the last lines before the exit.

## Fourth worked page: Qwen3.5-27B, community-test row 15, at v0.18.0 (recorded #11771)

Same tree as the first two worked pages (vllm-ascend `e18643f8`, vllm `bcf2be9612`).
The 任务书 names `docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md`; that path
does not exist at v0.18.0 (`rg --files -g '*Qwen3.6*'` → 0 hits; the page is
`Qwen3.5-27B.md`, 197 lines) — that absence is finding #1 and goes into the message.
The row asks for **dp2 tp4** on 8×Atlas 800I A2, AIME through aisbench, and a
2048/2048 `vllm bench` baseline. Verified findings, each `path:line`:

| Tier | Finding | Evidence |
|---|---|---|
| breaks intent (完整性) | the page documents DP1/TP2 only and carries no DP>1 recipe, while the task requires DP2×TP4 | page `:108-109`; siblings do carry it (`GLM4.x.md:150`, `GLM5.md:300`, `DeepSeek-R1.md:104`) |
| stale baseline | every image string and the accuracy reference are `v0.17.0rc1`, one release before the task image; the accuracy table is gsm8k only | page `:9,35,166,170` |
| breaks intent | the verification curl sends `"max_completion_tokens": 50` to `/v1/completions`, which has only `max_tokens: int \| None = 16`; `OpenAIBaseModel` is `extra="allow"`, so the field is silently dropped and the doc's own request returns 16 tokens | page `:153`; `vllm/entrypoints/openai/completion/protocol.py:57`; `vllm/entrypoints/openai/engine/protocol.py:28` |
| completeness | the Performance section has no result table at all, and its one `vllm bench` line is 200-in with `--request-rate 1`, so it cannot serve a 2048/2048 baseline; AIME exists only in the manual | page `:172-197`; `using_ais_bench.md:133-141` (dataset), `:219` (run command) |
| cosmetic | `--random-input 200` for `--random-input-len` — parses as an unambiguous argparse prefix (check 2), so give the right name and tier it cosmetic | page `:194`; `vllm/benchmarks/datasets.py:1565,1571` |
| pass | `--speculative_config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'`, `--async-scheduling`, `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` and `--additional-config '{"enable_cpu_binding":true}'` all resolve at the pin | `vllm/config/speculative.py:43`; `vllm/engine/arg_utils.py:1222`; `vllm/config/compilation.py:525`; `vllm_ascend/ascend_config.py:91` (the key defaults `True`, so the flag is a no-op, not a defect) |

Two things this page settles once, for the report body:

- **dp2 tp4 on one node needs no extra DP flags.** With the mp backend and
  `data_parallel_size_local` unset, vLLM infers it for internal DPLB
  (`vllm/engine/arg_utils.py:1618-1622`), so `--data-parallel-size 2
  --tensor-parallel-size 4` is 8 devices; `--data-parallel-address`, `-rpc-port` and
  `-size-local` belong to multi-node or external-LB layouts. State that the recipe is
  *derived* and that the page validates neither layout.
- **the in-tree gate beats the doc's stale baseline.**
  `tests/e2e/nightly/single_node/models/configs/Qwen3.5-27B-w8a8-A3.yaml` carries the
  CI-validated `server_cmd` (TP2/DP1, `--speculative_config` + `FULL_DECODE_ONLY` +
  `--async-scheduling`) and its own gates: `gsm8k-lite` accuracy baseline 95 with
  threshold 5 %, perf baseline 664.6449 at threshold 0.97. Quote those as the
  project's own gate; note TP4 appears only in
  `tests/e2e/multicard/4-cards/test_qwen3_5.py:24-36` (`max_model_len` 4096).

### The same page on a **4-card** row (recorded #11760 and #11770)

Two further rows of the same 任务书 name this identical page with `卡数 4`,
`Eco-Tech/Qwen3.5-27B-w8a8-mtp`, HDK 25.5.0.B078 and a 部署 + 功能验证
deliverable (no DP recipe, no AIME). Decide the transfer explicitly instead of
copying the row-15 table:

- **transfers**: the page-name absence, `max_completion_tokens` on
  `/v1/completions` (`:153`), the `v0.17.0rc1` baseline (`:9,35,166,170`), the
  missing performance table (`:172-197`), `--random-input` (`:194`), the
  `m.daocloud.io/` prefix on the image line (`:45`), and the
  `requires … (64G × 8)` sentence (`:21-22`) — that last one is check 5's
  enablement-sentence species, contradicted by the page's own TP2/DP1 line.
- **does not transfer**: "the page documents no DP>1 recipe". A row that asks
  only for 部署 + 功能验证 never needs DP; presenting dp2×tp4 as the required
  recipe mis-grades the page. Keep that finding for a row that asks for it.
- **4-card device map, tiered honestly**: the run block maps `/dev/davinci0..7`
  (`:54-61`), but its own first comment says "Update `--device` according to your
  device (Atlas A2: `/dev/davinci[0-7]`)" (`:41`), so the reader was told to
  adapt it — tier the device list **usability/完整性**, not "breaks the command".
  What does break for a 4-card tester is `:21-22`, which reads as "you cannot run
  this". Ship the trimmed `davinci0-3` block anyway.
- **which parallelism on 4 cards**: the doc's TP2/DP1 line is the only recipe the
  project gates, and its gate is the **A3** file
  `tests/e2e/nightly/single_node/models/configs/Qwen3.5-27B-w8a8-A3.yaml` (no A2
  variant in that directory; the TP4 coverage above is BF16-only). Offer TP2 as
  the run-first line, TP4 to use all four cards, and label TP4 as derived — on a
  `w8a8-mtp` + 133 K-context row it is not what the gate runs.
- **weights cache**: the block's comment says "download the weight to
  `/root/.cache` in advance" (`:43`) while the `docker run` mounts no
  `/root/.cache` (mounts end at `:70`, `-it $IMAGE bash` is `:71`), so a `--rm`
  restart re-downloads ~27 GB. Add `-v /root/.cache:/root/.cache`; tier 完整性.

## What a finding must carry

`path:line` + pinned commit/ref, the exact command, the error class the reporter
would see, the corrected line, and the tier. Anything you could not check
because the snapshot lacks the artefact (a ModelScope/HF `config.json`, a
processor package export) goes into the report as an open item with the one-line
command that closes it, not as an assumption.
