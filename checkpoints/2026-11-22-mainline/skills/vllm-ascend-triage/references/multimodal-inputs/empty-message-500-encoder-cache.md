# A 500 whose `message` is empty on a repeated / duplicate media request (V1 encoder cache)

Scope: the engine is up, other modalities answer, and one request returns **HTTP 500**
with an OpenAI-shaped body whose `message` is the empty string —

```
{"error":{"message":"","type":"Internal Server Error","param":null,"code":500}}
```

— while the next request on the same engine succeeds, and the paste carries **no
traceback**, only access-log lines. Typical report shape: a Chinese title like
「重复输入同一个视频，服务报内部错误」 with text/image/audio requests at `200` and the
video request at `500`, a `curlvllm.sh` transcript and a `collect_env` dump. Owner of
the class: the **V1 multimodal encoder cache / placeholder-range accounting**, not the
decoder, not a dependency, not a kernel.

Not this topic: a 500 that carries exception text (read that text — it names its own
lane), `EngineDeadError`/`Worker failed with error` (the engine died, so this is a
device or worker fault), and a VL answer that is merely slow or wrong with the request
completing (`../accuracy-benchmarks/vl-vision-tower-degeneracy.md`).

## Step 0 — read the body before any code

`type: "Internal Server Error"` is not an error class: it is
`HTTPStatus(500).phrase`, and `message` is the exception's detail string. At vllm
v0.9.1 that is the whole handler (`vllm/entrypoints/openai/api_server.py:1038-1043`:
`ErrorResponse(message=exc.detail, type=HTTPStatus(exc.status_code).phrase, …)`), and
every generic raise in that file feeds it `detail=str(e)` (`:466-468`, `:500-502`,
`:597-599`).

Two consequences, both cheap:

- `message: ""` means **`str(exc) == ""`** — a bare `assert` or a zero-argument raise.
  Every import/decode/dependency failure carries text (`Please install decord`,
  `librosa`, `KeyError('<x>')`, `IndexError: list index out of range`,
  `ValueError: …`), so those hypotheses **cannot** produce this body. Rank inside the
  assertion-bearing mm path and keep the dependency rows only as "if the traceback
  disagrees".
- **Do not hunt the formatter.** `rg -n 'Internal Server Error'` over the pinned upstream
  tree returns **nothing** at vllm v0.9.1 (verified); `type` is the HTTP phrase and the
  literal appears in no tree of this snapshot. That absence is not a finding — say the
  handler is upstream and move on.

The **packaging** differs from that handler: at vllm v0.9.1 `err.model_dump()`
serialises `ErrorResponse` flat (`object: "error"` plus the four fields,
`vllm/entrypoints/openai/protocol.py:66-71`), while the pasted body wraps them in an outer
`"error"` key and drops `object`. The reported pin is vllm 0.10.2, whose handler is not in
the snapshot, so treat the wrapper as that version's (or a fronting proxy's) business: the
two field meanings — `type` = the HTTP phrase, `message` = the detail string — are what the
diagnosis uses, and both are what v0.9.1's handler encodes.

## Step 1 — check the pairing before quoting upstream code

`SOURCES.json` can pin an upstream `vllm` that is **not** the version the
vllm-ascend pin pairs with. Recorded #2975: vllm-ascend `v0.10.2rc1` (`048bfd5`)
with upstream `vllm v0.9.1` (`b6553be`), while that tag's own
`docs/source/conf.py:68` says `'vllm_version': 'v0.10.2'` — so the upstream tree in
the case view is the **older, working** side (the reporter's own 「0.9.1没有问题」)
and the failing upstream mm code is not in the snapshot. One grep settles it:

```bash
rg -n "vllm_version|pip_vllm_version|ci_vllm_version" <root>/vllm-ascend/docs/source/conf.py
```

When they disagree, the failing-pin evidence is the **downstream copy** of the shared
machinery — `vllm_ascend/worker/model_runner_v1.py` and `vllm_ascend/core/scheduler.py`
carry their own `_execute_mm_encoder` / `_gather_mm_embeddings`, so read those at the
pin and use the upstream tree only as the comparison side. State the gap in the answer
instead of reconstructing the absent file from memory.

## Step 2 — the two versions differ exactly where "same media again" matters

| Site | Fact |
|---|---|
| vllm v0.9.1 `vllm/v1/worker/gpu_model_runner.py:996-998` | `self.encoder_cache[req_id][input_id] = scatter_mm_placeholders(...)` — a **nested, per-request** cache; freed per `(req_id, input_id)` from `scheduler_output.free_encoder_input_ids` (`vllm/v1/core/sched/output.py:145`, set at `v1/core/sched/scheduler.py:553`, consumed at `gpu_model_runner.py:356`) and popped whole on `finished_req_ids` (`:342`) |
| vllm-ascend v0.10.2rc1 `vllm_ascend/worker/model_runner_v1.py:257` | `self.encoder_cache: Dict[str, torch.Tensor] = {}` — **flat, keyed by `mm_hash`** (the media content hash), written at `:907`, freed by `scheduler_output.free_encoder_mm_hashes` (`:509-510`, produced at `vllm_ascend/core/scheduler.py:457`) |
| `model_runner_v1.py:945` | `assert start_idx < end_idx` — **bare**, so an `AssertionError` with an empty string |
| `model_runner_v1.py:947-949` | `assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."` — carries text, so it cannot produce the pasted body |
| vllm v0.9.1 `gpu_model_runner.py:1034,1036` | the same bare `assert start_idx < end_idx`, plus a bare `assert i in self.encoder_cache[req_id]` (that one *can* be empty at 0.9.1, which is why the pair matters) |

So one media item is now **one cache entry shared by every request that carries it**
(and by both items when the same video appears twice in one request), while at 0.9.1
each request had its own. The reported workload — resending the same video — is exactly
the workload this keying makes different, which is why the hash-keyed file, not the
model file, is where the investigation starts.

## Step 3 — discriminate inside the path

The bare assert fires only where the scheduled token window does not overlap the
placeholder range in a non-empty way: with `num_encoder_tokens = pos_info.length > 0`
the two guards above it (the `break` at `start_pos >= num_computed + num_scheduled` and
the `continue` at `start_pos + num_encoder_tokens <= num_computed`) already cover the
empty intersection, so what is left is a **zero-length placeholder range**
(`pos_info.length == 0`) sitting past the computed tokens. That is placeholder
bookkeeping — the count/offset arithmetic — not the video decoder.

- Cheapest confirmation, reporter-side: the APIServer-side traceback
  (`VLLM_LOGGING_LEVEL=DEBUG`; the #2975 reporter had exactly that line commented out
  in their own config file), then for the failing request the runner's own state —
  `[(p.offset, p.length) for p in req_state.mm_positions]` and `req_state.mm_hashes`.
  A `length == 0` entry names the owner outright.
- One ablation, no restart: the same body with a **different** video vs the same video
  twice, plus one text request on the same engine. Same-video-only failing ⇒ the
  shared hash-keyed entry; every video failing ⇒ the single-video placeholder
  accounting.
- If the log does print `Encoder cache miss for <hash>.`, the message is **not** empty
  and the paste does not match this page: that is the free/lifetime side
  (`free_encoder_mm_hashes` releasing an entry an in-flight request still needs), and
  it is reported with its own sentence.

## Step 4 — the deliverable this shape owes

A ranked short list plus one check each, not a mechanism: (1) placeholder/encoder-range
accounting at `model_runner_v1.py:945` — dump `mm_positions`; (2) the hash-keyed entry
shared by the resent request — the same/different-video ablation and the scheduler's
`free_encoder_mm_hashes` line; (3) decoder/dependency rows (`decord`, `librosa`,
`qwen-omni-utils`) explicitly refuted by the empty message but kept as
"if the traceback disagrees"; (4) vllm-ascend kernels refuted by the engine surviving
and the other modalities answering `200`. Then the asks: the full request body (video
URL form, `use_audio_in_video`, `--limit-mm-per-prompt`), the traceback, and
`pip show vllm vllm-ascend transformers torch-npu qwen-omni-utils librosa` — #2975's
paste gave only `0.10.2rc1` and the driver, with no request body and no `vllm`
version, so the pairing itself was an open question.

Say the version scope with it: the shipped support matrix lists Qwen2.5-Omni as
supported via #1760 (`docs/source/user_guide/support_matrix/supported_models.md:59` at
v0.10.2rc1) and `docs/source/faqs.md:196-197` answers only the `librosa` /
`qwen-omni-utils` import error — neither covers a video request, so "supported" is not
evidence that this path was validated.

**Delivery.** This shape has no traceback, so it is exactly the case where the answer
is a short ranked list and a request for artefacts — write it by the 8th assistant
turn. Recorded #2975 held both decisive facts (the hash-keyed cache, the bare assert)
inside its reasoning and shipped **no message text at all** across 46 messages, ending
`output_truncated`: the missing turn was the ranked list, not another read. The next
session on the case shipped that list on its third turn, in six tool calls (locate,
this page, one re-grep of the five anchors), so nothing past the anchor re-grep is
owed. Read this page whole in one call — `sed -n '1,200p'`, not `1,140p` then a second
window — because the Step-4 list, the asks and the pin caveat all sit in its final
quarter.

## Recheck conditions

Line numbers are vllm-ascend `v0.10.2rc1` (`048bfd5`) and vllm `v0.9.1`
(`b6553be`). Re-grep at the reported pin:
`rg -n 'assert start_idx < end_idx|Encoder cache miss for|free_encoder_mm_hashes|free_encoder_input_ids' <tree>`.
The flat hash-keyed cache exists only where the copy follows the newer upstream; if the
pin's file is still req-keyed, the duplicate-media reading does not apply and the page's
Step 2 table is the wrong pair. The 500 body's handler is upstream and moves between
releases — re-read `api_server.py`'s `HTTPException` handler at the reported vllm ref
before quoting `type`/`message` as the phrase/detail pair.
