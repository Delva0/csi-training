# C8 KV cache: the sparse `store_kv_block` path, and dense C8 under spec decode

Two C8 sub-families share the name and nothing else. **Sparse C8** (`enable_sparse_sfa_c8` /
`enable_sparse_li_c8`, DSA models: GLM-5.x, DeepSeek V3.2/V4) is covered by the first half of this
page: P-node hangs, garbled first output, `store_kv_block`. **Dense C8** (`enable_c8_quant` off the
checkpoint's quantisation config, GQA models such as Qwen3) is the second half: with speculative
decoding the FIA operator refuses the KV-length array during capture, `561002` / `EZ9999`. Check which
one the report has before reading further — the flags are different and neither key belongs to the
other family.

Family (sparse half): sparse-attention models (GLM-5.x, DeepSeek V4.x) served PD-disaggregated with
MemCache/AscendStore and sparse KV-cache quantisation ("C8"). Requests to the prefill node hang with
no response, or the first response is garbled. The flag lifetimes and per-tag states are in
`../vllm-ascend/config-flag-lifetimes.md`; this page is the symptom→mechanism→workaround view.

## Terminology

**C8** = fp8/int8 KV cache. On the sparse/DSA side it is `enable_sparse_sfa_c8`, the LightningIndexer
cache (`enable_sparse_li_c8`), or the pre-split `enable_sparse_c8`. On the dense/GQA side it is
`enable_c8_quant` on the quantisation config (a property of the checkpoint, see the dense section at
the end). These are KV-cache quantisations,
often derived from `cache_config.cache_dtype` / `attention_config.indexer_kv_dtype` rather than
directly from `additional_config`; newer releases unify the dtype via `--kv-quant-dtype` (#15514).

## Symptom

- A single `curl` to the P node never returns; `TimeoutError: RPC call to sample_tokens timed out`;
  shm-broadcast timeouts.
- Or the run answers but the first response is garbled when SFA C8 is on.
- Setting `c8_enable_reshape_optim: false` restores service (reported in #12533 and #15927).
- A second, independent C8 accuracy class: output that degenerates into **repeated exclamation
  marks** / NaN text on a single `curl` with no PD and no pool, fixed by the K-scale write path
  (`#19064`, below). Distinguish it from the reshape-optim hang by the switch test: here the
  reshape-optim key changes nothing.

## Code

- `vllm_ascend/ascend_config.py` (`v0.26.0rc1:288`):
  `self.c8_enable_reshape_optim = self.enable_sparse_li_c8 and additional_config.get("c8_enable_reshape_optim", False)`;
  the sibling keys are `enable_sparse_sfa_c8` and `enable_sparse_li_c8`, both gated on
  `model_uses_sfa_sparse(model_config)`.
- `vllm_ascend/attention/sfa_v1.py` at `v0.26.0rc1`: `:460` stores KV-block metadata when the flag
  is on, `:1547` `_use_li_c8_reshape_optim()` (requires `enable_sparse_li_c8` **and** the flag),
  `:1983+` `store_kv_block` on the forward path. From `v0.28.0.rc1` the same op is driven from
  `vllm_ascend/attention/indexer.py` through `_c8_reshape_optim_enabled` = flag AND
  `enable_sparse_li_c8` AND prefill node (`ascend_config.py:786` at `v0.29.0.rc1`, `:933` at
  `v0.30.0rc2`).
- `v0.23.0rc1` `attention/sfa_v1.py`: with the flag, indexer cache writes use
  `torch.ops._C_ascend.store_kv_block(...)` fed by
  `store_kv_block_metadata(slot_mapping, group_len, group_key_idx, group_key_cache_idx, block_size)`
  computed in the metadata builder; without it they use `torch_npu.npu_scatter_nd_update_`. The
  branch also splits `fused_kv_no_split` differently per `enable_sparse_sfa_c8` /
  `enable_sparse_li_c8` and indexes `kv_cache` at different positions, so cache-layout assumptions
  differ per combination.
- The ops live in `csrc/attention/store_kv_block*/` with bindings in `csrc/torch_binding_meta.cpp`.

## Mechanism

The "reshape optim" stores the quantised KV cache in a block-aligned layout via `store_kv_block`, so
block transfers (PP handoff, MemCache/AscendStore moves) can be contiguous copies. Anything that
reads the cache expecting the unreshaped layout — or a metadata path that groups slot mapping
differently — sees mismatched geometry; the reported hang and the garbled first output are consistent
with that class, but the traces did not localise a single line. For the hang, the stronger statement
is platform-level and documented: the custom AICPU tiling operator is launched in the same process as
the pooling machinery, which is the conflict described in issue #15896 and scoped by #16952 to "HDK
0.26.0 or earlier". `[thread]` + `[code]`; mark the exact line as not localised.

### Second mechanism: C8 K-scale corruption on a cache write (verified, #19064)

`e6c0f4b7d` — "[BugFix][Attention] Fix C8 K-scale corruption during KV transfers (#19064)",
2026-10-17. Commit text, verbatim: "C8 K-scale advanced-index assignment on NPU can materialize a
packet-strided cache view and write it back through `IndexPutV2` / `ViewCopy`, overwriting a
concurrent KV transfer to an unselected page. Invalid E8M0 scales can then produce NaNs and
repeated-exclamation output." The fix replaces the assignment with a stride-aware Triton byte
scatter that writes only the addressed token/head/group elements (production CPU fallback removed;
empty batches return before touching cache storage; `num_elements` stays a runtime scalar with
`do_not_specialize`).

Consequences for triage: `!!!!`-style output with a C8 KV cache is an E8M0/K-scale corruption
signature, not "sparse attention picked the wrong top-k". `[code]` from the commit: the fix presupposes
a stale page scale and a concurrent write, which is a PD/KV-transfer shape; `[lead]` the same bad
write-back would also corrupt a scale on any path that rewrites it with advanced indexing, so
single-node reports can share the class — name the check rather than asserting it. Only builds
**after** 2026-10-17 carry the fix (`git merge-base --is-ancestor e6c0f4b7d <ref>`); for any report
whose build contains it, this is a candidate — but check that the code it rewrites exists at that ref
first: for #13983 it does not (see that report below). Not hardware-verified, and no thread confirmed
it.

## Workaround

- On the releases that still expose the key (0.23–0.27, 0.29, 0.30.0rc*):
  `--additional-config '{"c8_enable_reshape_optim": false}'` together with whichever C8 option is
  enabled. The flag is ANDed with `enable_sparse_li_c8` on every line that has it, so it is already
  off when LI C8 is off — a report where turning it off changes behaviour proves LI C8 is on.
- On `v0.28.0.rc1` the user key does not exist at all, so the only lever is turning
  `enable_sparse_li_c8` off — an "opt-out did nothing" report against 0.28 is explained by the
  removed key, not by a broken config (`../vllm-ascend/config-flag-lifetimes.md` §4).
- On versions where #16952 applies, the switch defaults to **true**, so "it started hanging without a
  config change" is expected: the *default* changed.
- Otherwise drop the sparse-C8 keys entirely, or upgrade past #18128 (v0.30.0+) where the path no
  longer exists. Disabling a feature changes precision/perf baselines: state the cost.
- Before insisting on a code defect, check the three documented sparse-C8 accuracy caveats
  (`../vllm-ascend/config-flag-lifetimes.md` §6).

## Reported cases

**[thread]** #12533 — `0.23.0rc1`, GLM-5.2 W4A8C8, MemCache on a 910-class A3 P/D deployment, MTP 1
token: a single curl to P hangs once `enable_sparse_sfa_c8` / `enable_sparse_li_c8` /
`c8_enable_reshape_optim` are enabled, in a layout that works with C8 off. Closed 2026-08-14 by the
stale bot with no fix; comments are "Pls take a look", a driver-version request and `/wait-feedback` —
evidence of no resolution, not of a fix.

**[thread]** #15927 — `0.23.0`/`0.26.0rc1`, GLM-5.2, MemCache, 4-machine PD using the official docs
config, DSA-CP + `enable_sparse_sfa_c8`/`li_c8`: curl to P gets no response;
`c8_enable_reshape_optim:false` restores service; garbled first response with sparse C8. The docs at
the reported tags enabled the key (`GLM5.2.md:683/:752` in `v0.23.0rc1`, `:505` in `v0.26.0rc2`), it
launches a custom AICPU operator next to pooling (#15896), #16952 documents the conflict and the
disable-as-workaround, and #18128/#18301 remove the path. Workaround: switch it off, or upgrade past
`v0.30.0`.

**[thread]** #13983 — `0.23.0` build of 2026-08-10 (so between `v0.23.0rc1` (2026-07-19) and
`v0.23.0` (2026-08-17), i.e. on `origin/releases/v0.23.0`; there is no nightly ref — say so, and note
that a bare `releases/v0.23.0` is not a local branch), GLM-5.2 **w4a8c8**, A2/A3-class, a single
`curl` returns 乱码 (garbled text, screenshot only — the body carries no command, no log and no
attachment beyond an image) once `enable_sparse_sfa_c8` is on; the reporter first says w8a8c8 weights
are fine, then updates on 2026-09-02 that the deployed **w8a8c8** weights garble too. Read that update
as the important fact: it kills "the W4A8 weight quantisation is at fault" and leaves the C8 KV path.
Thread state when read (2026-11-17): open, labels `bug`/`glm5`/`llm-model`, and three comments that
are only an automated request for `collect_env` + the serve command + a log, `/wait-feedback` and a
stale-bot note. No technical reply, so the mechanism is yours to supply.

**Two attractive October fixes do not apply here — keep the negative result.** `e6c0f4b7d` (#19064,
"C8 K-scale corruption during KV transfers") rewrites `scatter_mxfp_k_scale_cache`, a function that
does not exist at `v0.23.0` or `v0.24.0rc1` (the MXFP8 C8 cache arrives later, #18485), and `99d96c3da`
(#17594) repairs a truncation introduced by #17135 in September, i.e. a regression newer than the
reporter's build. Both post-date the reported line, so citing either as this report's fix is wrong;
they remain the right check only for reports whose build contains them
(`git merge-base --is-ancestor e6c0f4b7d <ref>`).

What the report still gives you: the flag is read as
`self.enable_sparse_sfa_c8 = additional_config.get("enable_sparse_sfa_c8", False) and use_sparse`
(`v0.23.0` `ascend_config.py:280`), so it is inert unless the model is sparse-attention; the same-class
ledger row is #15927 (PD + `enable_sparse_sfa_c8`, garbled first response); and documented sparse-C8
accuracy caveats (`../vllm-ascend/config-flag-lifetimes.md` §6: #14011 GLM-5.2 DCP + SFA C8, #14378
±2–3 GPQA) were filed within days of this build, so quote them before calling it a new defect.
Workaround to offer: turn `enable_sparse_sfa_c8` off (and `enable_sparse_li_c8` with it if the indexer
cache is also fp8), or move to the released `v0.23.0` tag instead of the nightly. Cost: C8 off changes
KV-cache precision and the memory/throughput baseline, so accuracy comparisons must be re-based.
Falsifier for "it is the C8 KV path": reproduce with C8 off and the same weights; if the garble
persists it is the dense/MLA path. Next evidence to ask for: the exact launch command, whether P and D
both have C8 on, and one request id's first-token logits/`npu-smi` state.

## Dense C8 + spec decode: the FIA operator refuses the KV-length array (`561002` / `EZ9999`)

Symptom of the second family, from #12956 (Qwen3-32B-w8a8 with a C8 KV entry, EAGLE3, decode graphs
captured on `v0.23.0rc1`):

```text
RuntimeError: ... aclnnFusedInferAttentionScoreV3 failed, error code is 561002
EZ9999: the key/value's actual sequence lengths size[256] should be greater than q batch[512] or equal to 1.
        [FUNC:ProcessActualSeqLen][FILE:incre_flash_attention_tiling.cpp][LINE:977]
  at vllm_ascend/attention/attention_v1.py:893 full_graph_fia
     -> torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)
```

Read the two numbers before anything else: `256` = decode **requests**, `512` = decode **tokens** =
`requests × (num_speculative_tokens + 1)`. With no speculation the two are equal and the same build
serves fine, so the report is about the C8 decode path under EAGLE/MTP, not about the model.

### Where the C8 lever actually lives (verified, `v0.23.0rc1`)

`enable_c8_quant` is **not** a CLI flag and **not** an `additional_config` key: it is read off the
quantisation config (`vllm_config.quant_config.enable_c8_quant`, `attention/attention_v1.py:434-435`)
and comes from the checkpoint's quantisation description — `self.enable_c8_quant = kv_quant_type ==
"C8"` (`quantization/modelslim_config.py:942` at `v0.23.0rc1`, `configs/modelslim_config.py:1185` at
HEAD), feeding `quantization/methods/kv_c8.py` (`AscendC8KVCacheAttentionMethod`, docstring "C8 INT8 KV
cache quantization for dense-attention models (e.g. Qwen3)"). `grep` for an env var or `additional_config`
key that clears it returns nothing at either ref. So the workaround is a *different checkpoint* (a W8A8
directory whose description has no `kv_quant_type: C8`), not a flag. The sparse keys are inert here: `enable_sparse_sfa_c8`/`enable_sparse_li_c8` are
gated on `model_uses_sfa_sparse(model_config)` (`ascend_config.py:265-267`), which is false for a dense
Qwen3 — adding or removing them cannot change this error
(`../vllm-ascend/config-flag-lifetimes.md`, Trap D).

### Mechanism (verified; NOT fixed after the report)

The backend has two decode paths and they disagree about what a "batch" is:

- **eager** — `_forward_c8_decode` (`:1756`) converts to BNSD with
  `query[:batch_size].unsqueeze(2)` (`:1772`), `batch_size = len(attn_metadata.seq_lens_list)`, i.e. the
  request count, and passes the per-request `actual_seq_lengths_kv`. Batch and KV-length array agree, so
  the op runs — and the spec-token queries past `batch_size` rows are silently not computed.
- **captured** — `forward` takes the C8 branch (`:1595`, `if _EXTRA_CTX.capturing and attn_state ==
  DecodeOnly`) into `full_graph_fia`, whose C8 block sets `input_layout = "BNSD"` and
  `query = query.unsqueeze(2)` (`:856-857`) with **no trim**: `q batch` becomes `num_tokens` = 512 while
  `actual_seq_lengths_kv` is still the 256-entry per-request list built by `_get_fia_params` (`:1174`,
  DecodeOnly branch returns `attn_metadata.seq_lens_list`). CANN's IFA tiling wants that array to be
  length `1` or the q batch, hence the refusal.

The defect is therefore a missing `query[:batch_size]` (and the matching output slice) in the
graph-capture branch; the eager branch is the reference implementation. Still present at HEAD
(`query = query.unsqueeze(2)` with no trim, `attention_v1.py:693`), so a `v0.23.0rc1..HEAD` search for
a fix returns nothing — say "open", not "fixed later".

Two consequences worth stating rather than guessing at:

- The eager path *works* with C8 + speculation while dropping the draft-token queries (read off
  `query[:batch_size]` plus `output[:batch_size] = …`; not run), so a build that keeps decode out of
  full graphs can emit plausible-but-wrong tokens instead of crashing. "Use `--enforce-eager`" is a
  diagnostic with an accuracy asterisk, not a fix.
- `enforce_eager: true` inside `--speculative-config` only covers the draft model; the capture that
  fails is the target model's, which is why the reporter's setting did not help.

### Workaround (maintainer-attributed, #12956)

`earthmanylf`: 「C8 quantization with EAGLE3 is currently unsupported」, later 「vllm-ascend框架限制…当前暂时无法给出确切计划」
— a declared framework limitation, not a bug with a fix in flight. Relay:

- drop C8 (serve non-C8 W8A8 weights, KV cache back to bf16): costs ~2× KV bytes, so a shorter
  `--max-model-len` or lower concurrency;
- or drop speculation and keep C8: costs the accept-rate gain.
- Check the code, not the version, before calling it fixed: the check is whether the C8 block in
  `full_graph_fia` still lacks the trim (command below), and whether `_forward_c8_decode` is still
  reached with a full-length query.

Falsifier for "the C8 BNSD conversion is the cause": run the same C8 checkpoint with speculation on and
`cudagraph_mode` piecewise so decode does not take `full_graph_fia`; if the numbers become wrong rather
than the op refusing, the mechanism is confirmed (the eager branch is computing only the first
`batch_size` rows). If instead the ref is absent at the reported tag, the report belongs to the sparse
half above.

## Follow-up questions worth asking (`[lead]`)

Does the metadata (`group_*` buffers, block size) match the cache layout when DSA-CP and MTP drafts
are active? Is the stall at the first `store_kv_block`/`store_kv_block_metadata` call? Does the
problem reproduce with `c8_enable_reshape_optim=false` but C8 on? Does the reporter's release still
contain the op at all?

## Check before reuse

For the dense half:
`git show <ref>:vllm_ascend/attention/attention_v1.py | sed -n '/enable_c8_quant and layer is not None/,+20p'`
— no `[:batch_size]` trim inside that block means the captured C8 decode is still unbounded by the
request count; `git grep -n 'query\[:batch_size\]' <ref> -- vllm_ascend/attention/` shows the eager
path that has it.

For the sparse half:
`bash scripts/ref_grep.sh <repo> c8_enable_reshape_optim <ref>` to see whether the ref has the flag
and which of the three call sites exist; then decide whether the report is about the layout, the
metadata build, or the transfer path.
