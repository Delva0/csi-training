# Per-token KV on a DeepSeek-V4 / DSA serve: four cache kinds, three SoC gates

Scope: the report prices KV from the checkpoint's compressed latent layout ("≈7.4 KB/token,
computed from the fp8 KV layout") and then measures several times that ("27 KB/token/card"),
usually with a hypothesis attached ("compressed KV dtype is promoted to bf16 on non-A5
hardware, gated in `device/hardware_profile.py`"), **or** asks how far context can go on one
card in this release ("is there a path to ≥512K, or 1M, on A2?"). Deliverable: the
reconciliation — which cache kinds the serve actually allocates, each one's dtype gate at the
pin — plus the **per-SoC envelope from the pinned tutorial**, not a dtype-promotion theory and
not another flag.

Inherit `index.md` (arithmetic first, quantify the options, state that a fit check is a
non-exclusion) and `../runtime-memory/index.md` when host RAM or the pool-vs-device split is
also being claimed. Verified at vllm-ascend **v0.23.0.post1** (`1cdb8c4db6e5`) + vllm
**v0.23.0** (`0fc695fc6d1d`); re-grep the dtype lines per release.

## Procedure

1. **Take the reporter's own division, do not build a new one.** `bytes/token/card` is
   `Available KV cache memory` ÷ the engine's `GPU KV cache size` token count (both printed at
   startup), or available KV ÷ (`num_blocks × --block-size`). A reporter who already did this
   has given you two measured numbers; the work left is the term list, not the total.
2. **Enumerate the kinds from the model file, per pin.** On a DSV4 serve there is no single KV
   spec: `rg -n 'def get_kv_cache_spec' vllm_ascend/models/deepseek_v4.py` returns
   `AscendDeepseekV4IndexerCache.get_kv_cache_spec` (`:152-169`, an
   `AscendMLAAttentionSpec`), `AscendDeepseekV4SWACache.get_kv_cache_spec` (`:191-206`),
   `AscendCompressorStateCache.get_kv_cache_spec` (`:121-133`), and the
   same file's per-layer `compress_ratio` variants (`:568`, `:854`). Each contributes its own
   per-token (or window-bounded) bytes.
3. **Read the dtype gates in those files, not in a `device/` module.** At this pin
   `vllm_ascend/device/` has no `hardware_profile.py` and `DSV4_COMPRESSED_CACHE` occurs
   nowhere — a report that names them is describing a newer tree (state that). The gates are:
   - `core/kv_cache_interface.py:20,24` — `_get_c8_k_cache_dtype()` = `float8_e4m3fn` on A5
     else `int8`; `_get_c8_k_scale_cache_dtype()` = `float32` on A5 else `float16`.
   - `models/deepseek_v4.py:154,193` — inside both `get_kv_cache_spec` bodies, A5 overwrites
     `self.dtype = torch.float8_e4m3fn` (and `cache_config.cache_dtype`).
   - `models/deepseek_v4.py:568` `k_dtype = float8_e4m3fn if A5 else int8`;
     `:854` `k_dtype = float8_e4m3fn if A5 else bfloat16` — the SWA cache's dtype on a **non-A5**
     card is bf16, so "the fp8 layout" arithmetic under-counts that kind by 2× before any
     defect is involved. The A5 branch also widens the page (`cached_head_size = head_dim + 128`,
     `:196-197`), so the two SoCs are not one formula with a dtype swapped.
   - `AscendMLAAttentionSpec` (`core/kv_cache_interface.py:28-110`) is itself 2-4 tensors:
     `kv_cache[0]` kv_lora, `[1]` k_rope, `[2]` the indexer K (int8/fp8), `[3]` the indexer
     scale (fp16/fp32, only with LI C8), and `page_size_bytes` sums them. A "latent only"
     calculation omits the indexer terms entirely.
4. **Multiply the indexer terms by the DCP replication factor.**
   `core/kv_cache_interface.py:64` `sfa_dcp_replicated_indexer_size: int = 1`, set to
   `self.dcp_size` when `enable_sfa_dcp_replicated_indexer()` (`worker/model_runner_v1.py:455-457`).
   So DSA-CP/DCP > 1 replicates the indexer cache per rank — a context-parallel row in the
   reporter's matrix can therefore *raise* per-card KV, which is the opposite of the usual
   "CP spreads memory" intuition and belongs in the answer.
5. **Price the kinds with the existing script rather than by eye.**
   `bash <skill_root>/scripts/kv-page-size-arithmetic.py --layout v023 --soc A3 --dcp 1
   --k-head-dim <kv_lora_rank> --v-head-dim <qk_rope_head_dim> --index-head-dim <index_head_dim>
   --block-size 128` prints `page_size_bytes`; ÷ `--block-size` is the per-token cost of that
   kind. The sliding-window and compressor kinds carry their own padded page sizes from
   `models/layer/attention/layer.py:34-50` (`_DSV4_BLOCK_SIZES`: keys **32/64/128 only**, and
   the A5 table differs — e.g. block 128 is `[16640, 131072]` on non-A5 vs `[16896, 81920]` on
   A5), so `--block-size` is a per-kind mapping, not one constant. Their `sliding_window=` field
   means those terms saturate at the window instead of growing with `--max-model-len`: a single
   KB/token figure mixes linear and bounded terms, so present it as "the serve's" number, never
   as the model's property.
6. **Answer the context question from the pinned envelope, per SoC.** In the DSV4-Flash tutorial
   the A2 single-node block is `--max-model-len 133120` (`:201`) and the A2 PD nodes use
   `135000` (`:784`, `:862`), while **every** 1M row of the recommended-configuration tables is
   A3 (8 NPUs) or Ascend 950DT (`:1176-1180`). CI agrees: the DSV4-Flash single-node nightly row
   is `linux-aarch64-nightly-a3-16` (`.github/workflows/configs/nightly_config.yaml:165-167`)
   and the A5 row is `linux-aarch64-a5-8` (`:269-271`) — there is no A2 single-node DSV4 nightly
   entry at this pin. So "the context row you are already serving (512K) is inside what your
   measured footprint supports, while 1M is a documented A3 / 950DT row and not an A2
   configuration at this pin" is the answer; the levers are the A3/950DT row, fewer
   bytes/token (steps 3-5), or more cards — in that order.
7. **Do not offer `index_topk_freq` as a memory lever.** `use_index_cache` /
   `index_topk_freq` / `index_topk_pattern` decide whether a layer **reuses another layer's
   top-k** (`models/deepseek_v4.py:840-852` `skip_topk`, `patch/worker/patch_deepseek_v2.py:186-215`
   `_skip_topk`); the pinned configs that set it are throughput rows
   (`--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'`,
   `tests/e2e/nightly/multi_node/internal_dp/config/GLM-5.1-W8A8C8-A3_128k_90_50.yaml:32`). No
   line in that path shrinks the indexer cache, so it is not a "get to 1M" knob.

## What to send back

`Available KV cache memory` and `GPU KV cache size` (the reporter's division is only reusable
with both), `--block-size`, `gpu_memory_utilization`, `config.json`'s `kv_lora_rank` /
`qk_rope_head_dim` / `index_head_dim` / `index_topk` / `compress_ratios`, and whether the
`--additional-config` carried `enable_dsa_cp` on the failing row. With those, steps 3-5 are
arithmetic and need no launch.
