# `When layout is TND, not support Page Attention`: the toolkit release, not the caller's arguments

## Scope

A lane-C paste whose deepest frame is `npu_fused_infer_attention_score` / a
`torch_npu` wrapper, with

```text
RuntimeError: npu_fused_infer_attention_score_symint … NPU function error:
  call aclnnFusedInferAttentionScoreV3 failed, error code is 561002
[ERROR] … ERR00100 PTA call acl api failed.  E89999: Inner Error!
E89999: [PID: …] <ts> When layout is TND, not support Page Attention
  [FUNC:RunBigKernelTilingWithParams][FILE:prompt_flash_attention_tiling.cpp][LINE:3400]
        FusedInferAttentionScore do tiling failed, ret is -1.
        Check NnopbaseExecutorDoTiling(executor) failed   (… TilingAndUpdateBinInfo / MatchCache / RunForWorkspace)
```

Nothing launched, so there is no device dump to read. The tell is the wording: the
sentence says **not support** a *combination* (layout × cache kind). It does not
compare two numbers and does not name a tensor's dim, so it is a capability question
about the installed toolkit's tiling — not an argument invariant.

| Sibling with the same op and a different owner | Wording that separates it | Page |
|---|---|---|
| padding bookkeeping disagrees | `queryT(N) must be equal to the last element of actualSequenceLengthQ(M)` | `../spec-decode-mtp/fia-tnd-seqlen-mismatch.md` |
| caller's tensor dims must agree | `cos or sin shape is invalid`, `<tensor> shape is invalid` (`GetShapeAttrsInfo`) | `tiling-attribute-rejection.md` |
| caller-built mask is the wrong rank/shape | `atten mask dim should be 2 or 4, but got 3`, `get unsupported atten_mask shape` (`AnalyzeOptionalInput`) | `fa-tiling-argument-refusals.md` |

`RunBigKernelTilingWithParams` in `prompt_flash_attention_tiling.cpp` is the
*prompt / big-kernel* tiling path, so the refused object is a mode combination that
only that path builds (here `sparse_mode`/TND with a paged KV cache) — a check the
`GetShapeAttrsInfo` family never reaches.

## What to do: one version comparison, in the first message

1. **Get the reporter's toolkit version** — the environment block usually carries only
   vllm lines (recorded #4373's `collect_env` block is exactly `vllm 0.11.0+empty` +
   `vllm_ascend 0.11.0rc1`), so this is the missing fact, not a detail:
   `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` plus `pip show torch_npu`.
2. **Compare against the pin's own documented floor**, not against a recalled default:
   `docs/source/installation.md:15` at vllm-ascend **v0.11.0rc1** (`c5fe179`) reads
   `| CANN | >= 8.3.RC1 |`, and `:82-93` is the full `8.3.RC1`
   toolkit / `Ascend-cann-kernels-910b` / `nnal` install recipe. A toolkit below that
   floor is the finding: the paged-TND prompt path is not in its tiling.
3. **Rung 1 is the upgrade**, stated as the answer with its evidence class.
   Reporter-confirmed on #4373 (2025-11-24, original reporter): *"updating cann version
   to 8.3 can solve this."* Which build added the path is **not** verifiable locally —
   this case's exports hold no `cann/ge`/`cann/opbase` (only `cann/catlass`), so label
   the mechanism reporter-confirmed and the floor source-verified.

## Why a source walk of the caller cannot produce this answer

At that pin `_forward_v1_style` (`vllm_ascend/attention/attention_v1.py:540-554`) passes
`block_table=attn_metadata.block_tables` **together with** `input_layout="TND"` and
`sparse_mode=3`, and its own comment says the FIA op is intended to be used more widely
in later versions. So the pair is what the plugin *means* to do, and the pin documents a
toolkit whose automation accepts it. The reachability question (chunked prefill, or an
EAGLE/EAGLE3/MTP draft, puts the batch in `ChunkedPrefill`, the only state that reaches
`_forward_v1_style`) decides *whether you hit the wall* — it does not rename the wall.
Its rung, `--max-num-batched-tokens` ≥ prompt length so the batch takes
`PrefillNoCache` → unpaged `_npu_flash_attention`, is a **sidestep** worth one line as an
unblock, never as the cause.

Recorded cost of getting that backwards: #4373's session verified the launch pair
carefully, ranked it cause 1, and left "your CANN may be below 8.3.RC1" at rung 3 marked
*not verifiable locally* — while the floor was in the pinned `installation.md` and the
reporter's fix was one upgrade.

## Rules

- Classify on the **verb**, before any source read: `not support <mode/cache kind>` ⇒
  toolkit release; `must be equal` ⇒ bookkeeping; `… is invalid` ⇒ caller tensor.
  `561002` / `ERR00100` / `E89999` are shared across all three and name no lane.
- Do not offer `--enforce-eager` or `ASCEND_LAUNCH_BLOCKING=1` as the mechanism: nothing
  launched, so error attribution is already complete and both are inert here. Eager is
  still a legitimate *unblock* only if it changes which attention call runs — check the
  tree before claiming it does.
- If the reporter's toolkit already meets the pin's floor, it is a genuine toolkit
  defect: ask for toolkit + driver + `npu-smi info` and a framework-free
  `npu_fused_infer_attention_score` repro with the same `input_layout`/`block_table`/
  `sparse_mode`, so the op owner gets shapes instead of a serving log. State that the
  refusal cannot be repaired from the serve line.
- Version scope: the floor, the launch site and the wording are verified at vllm-ascend
  v0.11.0rc1; the fix is reporter-confirmed for CANN 8.3 on that pair. Re-read
  `_forward_v1_style` before quoting lines at another ref — the FIA call moves
  (`:789-795` trimming, `fia-tnd-seqlen-mismatch.md` Lane B helpers at v0.19).
