# Corruption that follows a hybrid KV-cache knob (`--block-size`, prefix caching)

Scope: 乱码 / repeated / degenerate output on a **hybrid linear-attention** model
(GDN/Mamba: Qwen3.5, Qwen3.6, and other hybrid attention+mamba checkpoints)
where the reporter's "works" and "fails" commands differ by a *scheduling* knob — `--block-size`, `--enforce-eager`,
`--enable-prefix-caching`, `--mamba-cache-mode` — rather than by weights, image or
topology. Process alive, no traceback, prompt short. Inherit the rules in this topic's
`index.md`; the sibling pages are
`short-prompt-corruption.md` (MoE/EP/quantisation suspects, long-ish prompt),
`garbled-long-context-output.md` (100K+, sparse c8/DCP/PD) and
`qwen3-next-gdn-numeric-path.md` (a short digit-heavy prompt on Qwen3-Next with
*no* scheduling-knob difference — the corruption follows the content there, not the
cache geometry).

## Step 0 — grade the confound before the mechanism

Recorded #13853 (vllm 0.20.2 + vllm-ascend v0.20.2rc1, Qwen3.5-0.8B, single
910B): the failing command is `--block-size 2048`, the "正常" command is
`--block-size 512 --enforce-eager`. **Two knobs changed.** Nothing is attributable
until they are separated, and the separation is two restarts, not a source walk:

| Row | Command | Corrupt ⇒ | Clean ⇒ |
|---|---|---|---|
| A | `--block-size 2048 --enforce-eager` | the block geometry owns it, not graph capture | `--enforce-eager` missing was the trigger |
| B | `--block-size 512` (same flags as the failure) | — | the block size is the trigger |

Ask for the response body of each row (`finish_reason`, `usage.prompt_tokens`,
the text). A report that shows only the command pair and the word 乱码 has not
shown the failing request.

## The geometry that makes this knob different on hybrid models

Verified at vllm-ascend v0.20.2rc1 (`367b8e6`), upstream vllm v0.20.2 (`bc150f5`):

- `vllm_ascend/utils.py:1326-1329` — `refresh_block_size()` **returns early when
  `model_config.is_hybrid`**, so the generic rule two lines below
  (`:1331-1335`: "Block size is set to 128 if prefix cache or chunked prefill is
  enabled") never runs for a Qwen3.5-class model. A user `--block-size` survives.
- `vllm_ascend/patch/platform/patch_mamba_config.py:37,72-88` — block size can be
  **raised**, never lowered: `attn_block_size` is computed so the attention page
  size equals the SSM state block page size, and `cache_config.block_size` is
  overwritten upward when the user's value is smaller. So on a hybrid model the
  effective cache block is *not* the usual 128, and it varies with tp (docs state
  the adjusted value reaches 2048 at tp 2 — `docs/source/tutorials/models/Qwen3.5-27B.md:136`,
  `Qwen3.5-397B-A17B.md:145`; also release-note Known Issue, v0.17.0rc1,
  `docs/source/user_guide/release_notes.md:418`).
- `vllm_ascend/attention/attention_v1.py:139-140` — the attention backend declares
  kernel block sizes `[128]`, so a cache block of 2048 is split 16:1.
  `vllm_ascend/worker/model_runner_v1.py:3467` (`use_hybrid_blocks = len(attn_groups) > 1`),
  `:4075-4083` (`select_common_block_size` per attention group),
  `:4084-4090` (a Mamba group gets `[0]`, "no splitting"), `:2994-2997`
  (per-group `block_table_tensor` swap).
- `vllm_ascend/worker/block_table.py:56-84` — physical→logical conversion:
  `blocks_per_phys_block = physical_block_size // 128` (16 at 2048) and
  `use_hybrid_blocks`; `:106-109` every appended block id is converted, so the
  *attention* table is in 128-token units while the mamba group keeps the physical
  block size. Upstream's align path divides by `kv_cache_spec.block_size`
  (`vllm/v1/attention/backends/utils.py:860-904`, `mamba_get_block_table_tensor`).
- `vllm_ascend/patch/platform/patch_mamba_config.py:92-96,103-106` —
  `mamba_page_size_padded = attn_page_size + conv_block_page_size`, and
  `mamba_block_size = block_size` only under `enable_prefix_caching` **and**
  `mamba_cache_mode == "align"`, else `max_model_len` (the same gate
  `../spec-decode-mtp/acceptance-collapse.md` records from the spec-decode side).

That is why a block-size change is not a tuning change here: it moves the
attention block table's units, the slot mapping, the mamba state block and the
padded mamba page at once. Whether the corruption comes from the logical-block
split or from the align-mode state block is **decided by the prefix-caching row
below**, not by reading more code.

## Matrix (one restart per row, same failing prompt, `temperature=0`)

| Row | Change | Corrupt here means | Clean here kills |
|---|---|---|---|
| block sweep 128 / 256 / 512 / 2048 | only `--block-size`, `--enforce-eager` fixed on | corruption only above 128 ⇒ the 16:1 logical split; or only at values that raise the mamba page ⇒ the padded-page path | "2048 is just too big for the card" |
| `--block-size 2048 --enforce-eager` | graph capture off | ACL graph capture of the split table | graph mode |
| `--block-size 2048 --no-enable-prefix-caching` | flips `mamba_cache_mode` off align, so `mamba_block_size = max_model_len` and the state block is whole-sequence | the align-mode state block/`mamba_block_size = block_size` path owns it | the attention-side logical split |
| `--block-size 512 --enable-prefix-caching --mamba-cache-mode align` | forces align at a small block | align path breaks at any block size | align is fine and only the size matters |
| same flags at `--tensor-parallel-size 2` | the documented hybrid regime (adjusted block size ≈ 2048) | reproduces the class on the configuration the docs and CI exercise | a tp=1-only artefact |

Rows 3 and 4 are the discriminators the reporter will not think of; state them as
such. Two release-note precedents show this geometry has broken silently before —
"hybrid attention plus mamba models … could start with an incorrect block size"
(#7528, v0.18.0rc1, `release_notes.md:355`) and "Fix kernel block size for computing
slot mapping" (#7019, v0.16.0rc1, `:487`). Quote them as history, not as the cause.

## Answer shape

1. Direct answer: on a hybrid model the cache block is *not* clamped to 128 and is
   only ever raised by the mamba/attention page-size constraint, so a hand-set
   `--block-size 2048` is outside what the project's own recipes and CI run; the
   cheapest action is to serve `--block-size 128` (or the value the log prints at
   startup) with identical flags.
2. Then the confound: their clean command also adds `--enforce-eager`, so rows A/B
   above decide whether the block size or the graph was the difference.
3. Then the matrix, with rows 3-4 named as the ones that identify the owning path.
4. What to send back: both serve commands, the startup log line that reports the
   adjusted block size, the two response bodies, and the model's `config.json`
   head dims/counts if a page-size assert was logged.
5. Scope: verified at v0.20.2rc1; the reporter is on vllm 0.20.2/0.23 — recheck
   `refresh_block_size` and `patch_mamba_config` before repeating the mechanism on
   another release.

## Stop rule (this is where the recorded run lost its session)

#13853 spent 22 assistant turns inside `patch_gdn_attn.py` chunk metadata,
`gdn_chunk_meta` and the block-table plumbing, reached the two-clamp story around
turn 10, and then ran out of output tokens **without ever writing an answer**. The
answer above was available from the first two code sites plus the row list. Once
rows A/B and the block sweep are specified, the deliverable is complete; the
owning line inside the split is a `git diff` for the maintainers, not a triage
result.
