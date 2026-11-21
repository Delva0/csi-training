# The serve line is the contract: request names, and the weight pre-download

Scope: two graded doc defects that recur on any model tutorial page when the
report's own words are "curl/`vllm bench` 报 404" or "服务启动加载模型 3 小时 /
高并发读取". Both are decided from the pinned page plus three upstream anchors,
with no cards and no engine:

- a sample request or `vllm bench serve` returns **404** (`The model X does not exist`) while the service is healthy — the *served-name contract*, not
  ModelScope, not the repo id's letter case;
- the first `vllm serve` "takes hours" — a **first-start download**, not a
  load-phase regression, because the page never says to pre-download.

Inherit `index.md` rules (grade with `path:line` at the pin, never invent a
measurement, separate severity). Tier each finding **breaks the command /
stale baseline / cosmetic**.

## Step 1 — one grep of the page decides the 404

```bash
rg -n 'vllm serve|served-model-name|vllm bench|--tokenizer|"model"' \
  docs/source/tutorials/models/<Page>.md
```

Read the page as **one contract**: the alias on its `vllm serve` line is the only
name `/v1/*` answers, so every other command in the same file that sends a
`model` field must use it. At vllm-ascend v0.18.0
(`Qwen3-235B-A22B.md`, `e18643f8a4d5`) that grep is a 2-vs-1 in one file:

| line | content |
|---|---|
| `:114` | serve line carries `--served-model-name qwen3` |
| `:270` | the curl body already sends `"model": "qwen3"` |
| `:370-371`, again `:598-599` | **working** bench form: `--model qwen3 --tokenizer vllm-ascend/Qwen3-235B-A22B-W8A8` |
| `:313` | the **broken** form: `vllm bench serve --model vllm-ascend/Qwen3-235B-A22B-w8a8 …`, no `--tokenizer`, no `--served-model-name` |

So the defect sentence is "the page contradicts itself and prints the working
recipe 250 lines below", and the repair is already in the file — say that instead
of re-deriving a fix.

Mechanism, verified at vllm v0.18.0 (`bcf2be961200`):

| `path:line` | fact |
|---|---|
| `vllm/benchmarks/serve.py:1511-1518` | `--served-model-name` defaults to `None`: "If not specified, the model name will be the same as the `--model` argument" |
| `serve.py:1676-1677` | `model_name = args.served_model_name`; `model_id = args.model` |
| `vllm/benchmarks/lib/endpoint_request_func.py:170-172` (same dict at `:299`, `:393`, `:538`, `:574`) | the request body's `"model"` is `model_name if model_name else model` |
| `vllm/entrypoints/openai/engine/serving.py:636-642` | `create_error_response(… err_type="NotFoundError", status_code=HTTPStatus.NOT_FOUND)` — the reporter's 404 |

Three working forms, cheapest first — **the cheapest one is not on the page**:

1. **Omit `--model` entirely** (v0.18.0 and later): `serve.py:1669-1675` fetches
   `GET {base_url}/v1/models` and takes `data[0]["id"]` as the name and
   `data[0]["root"]` as the id, and `:1684` sets
   `tokenizer_id = args.tokenizer if args.tokenizer is not None else model_id` —
   so both the request name and the tokenizer come from the running server.
   `vllm bench serve --dataset-name random …` then just works.
2. The page's own form: `--model qwen3 --tokenizer <hub id | local path>`.
3. Bench-side alias: `--served-model-name qwen3 --model <hub id>` (payload name
   becomes the alias, tokenizer stays the hub id).

Do not follow the report's own guess: `export VLLM_USE_MODELSCOPE=True` changes
model *resolution*, never the name in the request body, so it cannot fix a 404
from a live server. Repo-id letter case (`…-w8a8` vs `…-W8A8`, page `:22`) is
only a hypothesis while the **same string already works on the serve line** —
which the reporter's own "the service started (3 h)" sentence establishes.

Standalone form (the request is the reporter's own, no doc under test): the same
grep is `curl -s http://<host>:8000/v1/models` — the served names are that array,
and the answer is "ask for one of these, or pass the server's own name".

## Step 2 — "加载 3 小时 / 高并发读取" is the first start, not the load phase

Two checks, both runnable on the reporter's box:

1. **Is `--model` a hub id or a path that exists?** `vllm/transformers_utils/utils.py:116-124`
   `convert_model_repo_to_path()` returns the string unchanged only when
   `Path(model_repo).exists()`; otherwise it joins the ModelScope cache root. A
   hub id means "resolve/download first"; a path means "read from disk now".
2. **Watch the first minutes**: a downloading start prints modelscope progress and
   the cache directory keeps growing (`du -sh` twice, a minute apart); a
   reading-from-disk start does not.

Mechanism (`vllm/model_executor/model_loader/weight_utils.py:180-205` at v0.18.0):
under `VLLM_USE_MODELSCOPE` the pull runs inside
`with get_lock(model, download_dir)` (`:108` defines the lock) and only then calls
`snapshot_download(...)`. So the ranks do **not** each download — they serialize
behind the file lock while one process pulls the whole checkpoint (~230 GB for a
235B W8A8 set), and the other ranks wait. The page makes it worse in two ways: it
sets `VLLM_USE_MODELSCOPE=True` in the serve block (v0.18.0 `:98`) and recommends
the *shared multi-node* directory `/root/.cache/` (`:24`), which turns the 8-rank
concurrent read into network I/O when that path is a mount.

Fix ladder: `modelscope download --model <id> --local_dir <dir>` once on the host
→ node-local disk → pass the **local path** as `--model` (this also short-circuits
check 1, so `VLLM_USE_MODELSCOPE` can be dropped). Tier the missing pre-download
step **完整性/易用性**; the graded defect is that the page has no such step and
never says "local path".

Keep the timing claim open: the 3 h figure quantifies bandwidth, so say the wall
time needs splitting (download vs read vs profile) rather than calling it a
weight-load regression. A stall *after* `Starting to load model` with the
checkpoint already on local disk is a different page —
`../startup-failures/weight-load-stall-in-shard-loop.md`.

## Recorded run (#11738, Qwen3-235B-A22B-W8A8, image `v0.18.0`)

Turn 2 opened `index.md`, turn 3 found the page, and the 404 chain was complete
by turn 10 — page `:313` / `:114` / `:370-371` plus the four vllm anchors above —
with both defects graded (完整性 3/5 · 易用性 3/5 · 正确性 2/5), an 8.9 K
character report written in three appends, a 2 K delivery message, and the
performance table correctly left empty with the one command that fills each slot.
Cost: 33 messages against a 90 KB SKILL.md in context on every turn, and the
16 214-completion-token budget was spent to the end — which is why the two greps
above are worth taking before any source walk.
Two gaps worth closing next time: it never found form 1 (`--model` omitted), so its
recommended fix also required the maintainer to add `--tokenizer`; and it quoted
`--served-model-name` at `:113` where the absolute line is `:114` — quote page
anchors from an absolute `rg -n '<token>' <page>`, never by counting inside a
`sed -n 'A,Bp'` window, because in a doc-verification report the anchor is the
first thing a maintainer re-checks.
