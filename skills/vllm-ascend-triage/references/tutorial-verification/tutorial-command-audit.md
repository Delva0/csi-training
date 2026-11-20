# Auditing a tutorial page: eight checks, in this order

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

## 6. Which release does the reference number come from?

Grep the page for the release tag next to its result blocks: both the accuracy
and the performance block of the Qwen3-Omni page say "in `vllm-ascend:0.13.0rc1`
for reference only" (`:255`, `:296` v0.18.0) while the task image is v0.18.0.
Tier **stale baseline**: keep it as a baseline to beat, never as the expected
result, and never re-publish it as a measurement. When one of its numbers looks
anomalous, check the precedent before blaming the doc
(e.g. `#5991: Qwen3-Omni-30B-A3B-Thinking, when max-concurrency >= 2, TPOT only
100ms` against the page's ~99 ms mean TPOT).

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

## What a finding must carry

`path:line` + pinned commit/ref, the exact command, the error class the reporter
would see, the corrected line, and the tier. Anything you could not check
because the snapshot lacks the artefact (a ModelScope/HF `config.json`, a
processor package export) goes into the report as an open item with the one-line
command that closes it, not as an assumption.
