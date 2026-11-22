# DCP × SFA-C8 with garbled output and no fault: the matrix is two recipe rows

## Scope

A sparse-attention deployment (GLM-5.1 / 5.2 / 5.3 `w8a8c8` / `w4a8c8`, DeepSeek-V3.2)
that serves with `--decode-context-parallel-size` > 1 **and** `enable_sparse_sfa_c8`,
starts cleanly, and produces **garbled / wrong output with no error, no dump, no
plog fault**. The reporter usually hands you a *matrix* instead of a log — recorded
#16085 (v0.26.0rc1, GLM5.3-w8a8c8, A3): "dcp=8 + sfa_c8 true → 必定乱码; dcp=16 正常;
dcp=8 + sfa_c8 false 正常". Verified at v0.26.0rc1 `f2f74a1` (SOURCES `16085`;
upstream vllm v0.26.0 `568afb3a13`).

Not this page: the same flags dying with `MTE accesses an invalid GM address` /
`507015` / `507035` (`dcp-c8-kv-address-fault.md`); a prefill node that stops
answering or a first-request-only garble under `c8_enable_reshape_optim`
(`c8-flags-and-kv-store-path.md`, whose Rules 1-3 own the flag algebra and the pinned
recipes and are not repeated here); a KV-construction page-size assert
(`c8-page-size-reshape-assertion.md`).

## Rule 1 — `dcp == tp` is enforced, so "dcp=8 vs dcp=16" is not one variable

`vllm_ascend/utils.py:122-129`: `enable_sfa_dcp_replicated_indexer` =
`model_uses_sfa_sparse(model) and decode_context_parallel_size > 1`. For every GLM/DSv3.2
SFA model that gate is **on** as soon as DCP > 1, and `vllm_ascend/platform.py:723-731`
then refuses anything else:

```python
if sfa_dcp_replicated_indexer:
    if parallel_config.decode_context_parallel_size != parallel_config.tensor_parallel_size:
        raise AssertionError("DCP for SFA is only supported when dcp_size(...) == tp_size(...)")
    if get_ascend_device_type() == AscendDeviceType.A5:
        raise NotImplementedError("SFA DCP with replicated indexer is not supported on A5 yet.")
```

Three consequences to state in the first paragraph:

- the reporter's two runs **differ in TP as well as DCP** (TP8/DCP8 vs TP16/DCP16) —
  ask for both serve commands' parallel block before ranking a mechanism, and do not
  treat the dcp=16 run as a one-flag control;
- every validated pinned row has `tp == dcp` for the same reason: 1M single-node
  TP16/DCP16 vs 1M dual-node TP8 × DP4 (`docs/source/tutorials/models/GLM5.2.md`,
  `--decode-context-parallel-size 16` at `:1140`, `… 8` at `:1197`) — so a reporter on
  dcp=8 is running the **TP8 × DP4 recipe's regime**, not a scaled-down version of the
  16-card one, and the whole `--additional-config` lattice of that row applies
  (`c8-flags-and-kv-store-path.md` Rule 3);
- a report that says "I set `--decode-context-parallel-size 8` on the same 16-card
  command" cannot be true for an SFA model: that launch raises at engine start. Say so,
  and ask for the startup log if they claim it anyway.

## Rule 2 — the pin itself calls this combination not recommended

`docs/source/user_guide/release_notes.md:259-262` at this pin:

- `:260` — for P/D-disaggregated SFA models "DCP must be enabled on both prefiller and
  decoder nodes or disabled on both. Asymmetric DCP configuration can cause accuracy
  issues" (#14322).
- `:262` — "combining GLM-5.2 DCP with Sparse Flash Attention C8
  (`enable_sparse_sfa_c8`) has known issues, including performance degradation, and is
  **not recommended**" (#14011).

So the verdict is not "a new mystery": a GLM-family DCP × sfa_c8 accuracy report lands
on a documented known-issue lane. Precedents to quote as one line each: `#12905`:
v0.23.0rc1, GLM-5.2, DCP=8 + `enable_sparse_sfa_c8` → first curl 乱码 (cache-hit caveat
in `c8-flags-and-kv-store-path.md` row 2); `#14378`: GLM-5.2-W4A8C8 + sfa_c8 moves GPQA
~2-3 points run to run (`release_notes.md:270`) — the lane has a noise floor, so demand
a fixed-seed/KSample comparison before calling a small gap a bug; `#14320`/`#15906`: the
same replicated-indexer id space *faults* at DCP=8 (reporter-confirmed overflow,
`dcp-c8-kv-address-fault.md` Rule 4) — a silent wrong-row read is the same address
arithmetic one step short of out-of-range.

## Rule 3 — the interleave flag is unchecked on exactly this path

`--cp-kv-cache-interleave-size` defaults to **1** (upstream `vllm/config/parallel.py:360`),
and the guard that forces it to `block_size` is skipped for the replicated indexer
(`platform.py:745-757`: the `logger.warning_once` + override carries
`and not sfa_dcp_replicated_indexer`). Only the P/D + KV-pool combination *asserts*
equality (`platform.py:734-742`). Meanwhile the two index computations disagree by
construction:

| anchor | granularity it uses |
|---|---|
| `sfa_cp.py:105-112` | `replicated_view_block_size = kernel_block_size` (128 for SFA; block size must be divisible by it) |
| `sfa_cp.py:245-255` | the replicated slot mapping quantizes positions by `replicated_view_block_size`, never by the interleave flag |
| `sfa_cp.py:406`, `:534-544` | `_remap_sparse_indices` maps top-k ids with `self._dcp_interleave_size`, with a separate `interleave_size == 1` branch (`topk / dcp`) |

So one flag selects between two token→local-position conventions while the write side
is pinned at 128. Every validated tutorial row sets the interleave explicitly to
`block_size` (128). **Check the reporter's command for it** and treat a missing or
non-`block_size` value as the cheapest hypothesis — but state it as a check, not a
cause: which convention is correct for a given DCP layout is not decidable from this
snapshot alone. An explicit `128` **closes** this rung: say so and drop it from the ladder,
because two of this topic's config-level hypotheses can already be dead in one report.
Recorded #16976 (v0.26.0 rc1 *and* rc2, GLM-5.3-w8a8c8, A2 8-node P/D, `DCP=8 == TP=8`,
interleave `128 == --block-size 128`, `sfa_c8`+`li_c8` on, `c8_enable_reshape_optim` absent
so the fallback scatter write ran — `c8-flags-and-kv-store-path.md` Rule 1): its first rung
was `enable_sparse_li_c8: false`, not either of the two flags above.

## Rule 4 — coverage: state what the pin actually exercises

Do not generalise from one file. At `f2f74a1`:

- the **only** in-tree accuracy test combining DCP with `enable_sparse_sfa_c8` is
  `tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py:121-146`, case
  `dsv3_2_sfa_dcp_replicated_indexer`, on `vllm-ascend/DeepSeek-V3.2-W8A8-Pruning`,
  `decode_context_parallel_size: 2`, both C8 flags `True` — i.e. DCP=2 on one model
  family, never DCP=8/16 on GLM;
- the GLM `w8a8c8` nightly configs use `enable_dsa_cp` and carry **no**
  `--decode-context-parallel-size`, so they do not cover this path at all — and their
  C8 rows differ file by file: `.../GLM-5.2-W8A8C8-A3_128k_90_50.yaml:49` is
  `sfa_c8: false` + `li_c8: true`, while `.../GLM-5.2-W8A8C8-A3_198k_function.yaml:48`
  is `sfa_c8: true` + `li_c8: true`. "The GLM nightlies run sfa_c8 off" is false: grep
  all of `tests/e2e/nightly/**` before writing a coverage sentence, never a
  `head -3` file list.

## Rule 5 — ablation ladder, one variable per launch

1. Re-send the same prompt with a nonce (or `--no-enable-prefix-caching`): only if the
   garble is not reproducible per request does the cache-hit story of #12905 apply.
   The reporter's "必定" is itself the answer to this rung — keep it as the reason the
   ladder starts at 2, not as a "confirmed".
2. `c8_enable_reshape_optim: false` — changes only the indexer-cache write op
   (`c8-flags-and-kv-store-path.md` Rule 2), keeps LI C8 on.
3. `enable_sparse_li_c8: false` with `enable_sparse_sfa_c8: true`: separates the SFA
   packed-C8 attention kernel from the indexer-cache write path. The mirror row
   (`sfa_c8: false`, `li_c8: true`) is the reporter's own known-good calibration.
4. `--decode-context-parallel-size 1` (drop the replicated-indexer gate entirely) and,
   separately, the *same* TP with `dcp == tp`: the two pinned rows, so the pair
   TP8/DP4 vs TP16 tells you whether the difference is the row or the DCP size.
5. Confirm `--cp-kv-cache-interleave-size == --block-size` (Rule 3), and `--block-size`
   divisible by 128 (`sfa_cp.py:105-112` raises only on indivisibility).
6. Version lane, last but keep it: DCP>1 + replicated indexer + C8 is **new in v0.26**
   (`release_notes.md:12`, a v0.26.0rc1 highlight: "Added SFA DCP with a replicated
   indexer, compact KV gather, and C8 support"), so a later rc may carry a fix the snapshot
   cannot show. The reporter is on the first rc of the feature; naming the next rc as
   a rung is honest, promising that it works is not.

## What to send back

Both serve commands whole (the parallel block is the evidence, not a summary), the
`--additional-config` lattice, `--block-size` / `--cp-kv-cache-interleave-size` /
`--max-model-len`, one raw request/response pair showing the garble, and whether it is
P/D-disaggregated (asymmetric DCP is its own documented accuracy cause, `:260`).

## Scope and recheck conditions

- Anchors verified at v0.26.0rc1 `f2f74a1`. Recheck with
  `rg -n 'DCP for SFA is only supported|not sfa_dcp_replicated_indexer' vllm_ascend/platform.py`
  and `rg -n 'replicated_view_block_size|_dcp_interleave_size' vllm_ascend/attention/context_parallel/sfa_cp.py`.
- Claim classes: the `dcp == tp` gate, the skipped interleave override, the defaults and
  the coverage rows are source-verified at the pin; the release-note lines are the
  project's own position; **no mechanism for #16085 is confirmed** — the reporter never
  posted a resolution in this snapshot, so rank the C8-write path and the derived id
  space as hypotheses with the checks above and say the cause is open.
- Delivery: the recorded #16085 session held this matrix by its 4th tool turn and spent
  25 turns / ~57 K characters of reasoning on the derivation, shipping **zero** message
  text before the 16 384-token cap. The matrix already is the deliverable skeleton —
  one verdict line per row, then the ladder (`../evidence-workspace/delivery-discipline.md`).
