# PD decode-node corruption and scale faults: diff the recipe contract first

Scope: the P/D deployment is up and (a) the decode node answers with a
**degenerate/repeated** continuation (the same token or phrase to the model cap,
`finish_reason: length`) or otherwise wrong text, or (b) a device fault
(`ACL stream synchronize failed`, `AclrtSynchronizeStreamWithTimeout`, `EZ9999`)
appears only after the batch knobs are raised. Inherit `index.md` and the SKILL.md
rules. In both shapes the decisive cheap action is the same and comes **before** any
`kv_p2p/` source walk: diff the two serve commands against
`docs/source/tutorials/models/<model>.md` at the pinned ref. Two recorded cases:
#13381 (GLM-5.2 560T W4A8C8, DSA/MLA + MTP, "1 + 1 = 2." ×2048 on the decode node)
and #9578 (Qwen3.5-35B-A3B 8P8D, stream-sync fault only at raised scale, no log).

A model family named in the report is not a connector verdict. GLM-5/5.2 are **DSA
sparse-attention MoE** models and DeepSeek-V3.x is MLA — neither is a GDN/Mamba
hybrid; the hybrid connector is the DeepSeek-V4/Qwen3.5-family recipe
(`hybrid-state-transfer.md`).

## The four invariants to check (each one grep of the paste)

| # | Invariant at v0.23.0rc1 (`f4a08bd`) | Where the recipe states it |
|---|---|---|
| 1 | Connector class matches the family | GLM-5.2 PD: `MooncakeConnectorV1` with `use_ascend_direct` (`GLM5.2.md:694,763,831,899,1371,1442`); the A2 8-node variant uses `MooncakeConnector` (`:1040,1121`). Qwen3.5-397B: `MooncakeLayerwiseConnector` (same tree, `Qwen3.5-397B-A17B.md:361,442,524`). DSv4-Pro/Flash: `MooncakeHybridConnector` (`hybrid-state-transfer.md`) |
| 2 | `--speculative-config` is present on **both** sides | GLM-5.2 PD prefill `{"num_speculative_tokens":1,"method":"deepseek_mtp","enforce_eager":true}` (`:679,748,1064`); decode `num_speculative_tokens` 5 (`:818-820,886-888`) or 3 (`:1109,1147`) |
| 3 | `--no-disable-hybrid-kv-cache-manager` only on a hybrid family | Present in DeepSeek-V4-Pro (5 sites), V4-Flash (4) and Qwen3.5-397B-A17B (3) tutorials; **nowhere** in `GLM5.2.md`, which never says "hybrid" |
| 4 | DSA sparse-gate **parity and the per-SoC value** | GLM-5.2's **A3** PD blocks set both C8 flags `true` on both sides (`:683,752,821,889`; co-located adds `c8_enable_reshape_optim`), the **1M PD** block does too, prefill `:1365` and decode `:1436` (`enable_flashcomm1`/`enable_dsa_cp` true on P and false on D, `recompute_scheduler_enable` false on P and true on D — the exact lattice recorded #14085's two nodes carried) — but the **8-node Atlas 800 A2 (910B)** PD block sets both `false` (`:1059-1060` P, `:1142-1143` D) and uses no DCP at all (`:984`; `--decode-context-parallel-size` appears only in the 1M and single-node sections, `:1251,1306,1362,1433`). So parity is necessary and not sufficient: compare the reporter's `--additional-config` key lattice against **each** block. A lattice match to the 1M block on a 200 K 910B deployment is a cross-copy, and its C8/DCP rows are the deviation — say which block they copied, not just which flag differs. Flag algebra: `../dsa-sparse-attention/c8-flags-and-kv-store-path.md` |

Why invariant 2 is not cosmetic: the connector derives its transfer geometry per
side from that config, so an absent `--speculative-config` on one side is two
different layer windows and block indices over the same request — the "silently
wrong tokens, no error" shape. Current-pin sites (`mooncake_connector.py:533-543`
`num_draft_layers`, `:798` window, `:842` block index) and the method-string trap are
in `hybrid-state-transfer.md` §3; do not re-derive them. Rank the asymmetry above any
connector theory that needs a code defect.

Why a non-hybrid model on `MooncakeHybridConnector` stays a hypothesis:
`mooncake_hybrid_connector.py` is a separate implementation whose accounting branches
are SWA/Mamba-typed (`:1274` group-count assert, `:1282-1287` "Returns N-1 for Mamba
models", `:1523-1547` `MambaSpec` sizes), and it carries **no MTP accounting at all** —
`grep -c mtp` is 0 at this pin, against the generic connector's sites above. Use that
as the ranked mechanism, and name the value that decides whether the wrong branch is
taken (`use_hybrid` / `need_truncate` on a model with no Mamba group) instead of
asserting it.

## Do not rank these (checked in #13381, not the mechanism)

- Decode `--max-num-batched-tokens`: the recipe's `164` (`GLM5.2.md:818,886,1109`)
  against a reporter's `128` with `--max-num-seqs 8` and 3 draft tokens still fits
  (8 × 4 = 32 tokens), so the value is config hygiene, not the corruption.
- `--block-size 128` and `--enable-prefix-caching` appear in this model's own recipes
  (`:459,519` co-located; `:1033,1115` A2) — a reporter using them is inside the tree.
- `enforce_eager` inside `--speculative-config` only affects the **drafter**. The
  top-level `--enforce-eager` is the ablation that removes torch.compile/ACL graph
  (SKILL.md rule 15; `../startup-failures/compile-backend-banner-then-silence.md`).
- A hand-written `cudagraph_capture_sizes` list whose entries are **not** multiples of
  `num_speculative_tokens + 1` (a `298` beside `256`/`320` with `n=3`). Under full
  decode graphs with spec decode, vLLM round-ups every entry to that multiple
  (`adjust_cudagraph_sizes_for_spec_decode`, `vllm/config/compilation.py:1432-1443`
  and `:1473-1516` at vllm v0.23.0) and only raises when nothing valid survives — one
  wasted bucket, not a wrong token. Recorded #14085 spent four calls re-deriving this
  lane; state it as closed and move on.

## Ablation ladder to hand back (one launch per row, same fixed prompt, `temperature=0`)

1. Drop `--no-disable-hybrid-kv-cache-manager` on both sides (cheapest, one flag).
2. Switch to the tutorial's connector class, keeping the reporter's dp/tp block.
3. Add `--speculative-config` to the side that lacks it (try `1`, then match the peer).
4. Remove `--speculative-config` from the decode node entirely — if the repetition
   goes, the drafter/verify path owns it.
5. Top-level `--enforce-eager` on the decode node only.
6. Decode node alone, no PD, all other flags unchanged — separates transfer from a
   local decode defect.
Send the same request twice: an identical 2048-token repeat is a state/geometry
symptom, a different tail is a sampling symptom. **A reported rate changes the
sample size** — "偶现" with a count (recorded #14085: 29 repeats in 193 runs of one
fixed prompt) needs `N ≈ 3/p` serial reruns (≈20 at 15 %) before a clean run means
anything; two or three clean runs are not evidence, and a rate on one prompt retires
the "this prompt always loops" row while leaving per-request state and sampling both
open (`../accuracy-benchmarks/long-output-repetition.md`).

## Hybrid families: the P/D lattice is asymmetric by design (#12792)

On a **hybrid** family the two sides of the documented recipe are *meant* to differ, so
"P and D are not identical" is not a finding. DeepSeek-V4-Pro-w4a8-mtp, Atlas 800 A2
(64G × 8), 1P1D = 4 P + 4 D (`docs/source/tutorials/models/DeepSeek-V4-Pro.md:26,141`,
§5.2.2 at `:846`, P block header `:956`, D block header `:1033`), verified at v0.23.0rc1
(`f4a08bd`):

| Flag / value | Prefill (`:956-1030`) | Decode (`:1033-1110`) |
|---|---|---|
| `--kv-transfer-config` | `MooncakeHybridConnector`, `kv_producer`, `kv_port 30000`, `engine_id 0` (`:1017-1019`) | `kv_consumer`, `30100`, `1` (`:1094-1096`) |
| `kv_connector_extra_config` | `prefill{dp 4, tp 8}` / `decode{dp 8, tp 4}`, the same block on both sides (`:1020-1029`, `:1097-1106`) | idem |
| `--no-disable-hybrid-kv-cache-manager` | `:1001` | `:1079` |
| `--no-enable-prefix-caching` | `:1012` | `:1089` |
| `--speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}'` | `:1013` | `:1090` |
| top-level `--enforce-eager` | `:1011` | **absent** |
| graph / scheduling | — | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` `:1091`, `--async-scheduling` `:1077`, `--block-size 128` `:1078` |
| `--additional-config` | `enable_dsa_cp` true (`:1014`) | `recompute_scheduler_enable` true (`:1108`; `:1177` — "enable this only on decode nodes") |
| `--max-num-batched-tokens` / `--max-num-seqs` | `4096` / `16` (`:999`, `:1000`) | `120` / `60` (`:1075`, `:1076`) |

Do not report the decode `--max-num-batched-tokens 120` or its missing `--enforce-eager`
as deviations: both are the recipe, and the decode node is the side that keeps graph
capture. The asymmetric 4/8 versus 8/4 geometry is legal because the TP-equality assert
sits **inside** the `use_mamba` branch —
`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py:1515-1530`
(`use_hybrid` = hybrid manager enabled ∧ a non-`FullAttentionSpec` group ∧ >1 group, then
`assert self._prefill_tp_size == self._decode_tp_size, "… different TP size with Mamba."`),
sizes assigned at `:1587-1598`; the only unconditional geometry gate is
`assert self.pcp_size * self.dcp_size == 1` (`:1193`, `:1505`). So "different TP with
Mamba" is not a contradiction of the doc unless a Mamba group exists.

Coverage at that pin: `rg -l MooncakeHybridConnector tests/` returns only
`tests/ut/kv_offload/test_mooncake_hybrid_connector.py` — no e2e row, so the A2 4P4D
hybrid recipe is doc-only. Quote it as coverage, never as the cause.

Recorded #12792 (A2 8-machine DSv4-Pro 1P1D, "请求回答乱码", body carrying no log and no
serve command): the reporter later pasted both serve blocks and they match the table
above flag for flag — including the P/D asymmetry, the 4/8 ↔ 8/4 extra-config and
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` on the prefill side — so the **recipe lane closed on
the reporter's own evidence**. The next artefacts are the decode node's log
(`num_blocks: <N>` and `Initializing Mooncake Scheduler <engine_id>`,
`mooncake_hybrid_connector.py:1610`, `:1187`) plus one raw request/response pair, not
another flag diff.

## The variant with no log at all: fault only at raised scale (#9578)

"Reduce `max-num-batched-tokens` / `max-model-len` / `max-num-seqs` / capture sizes and
it works" puts the owner in the **capture path** (`../startup-failures/acl-graph-capture.md`;
the decode-node recipe is capture sizes `n × (mtp+1)`, `n ≤ max-num-seqs`,
`Qwen3.5-397B-A17B.md:407,518` and `max-num-batched-tokens 128` there, `4096` on
prefill, both read in the v0.19.1rc1 tree) or in the **chunked-prefill compute path**
(`../hccl-and-hardware/collective-error-chain.md`; plog `fault kernel_name=`), never in
KV transport: the title's sync call is the reporting point, and the code after it
(`507015`, `507035`, `107027`) names the family. Two verified non-answers at v0.19.1rc1
(`da421af`) that otherwise cost a session each:

- **The GDN chunk-meta buffers cannot overflow from those knobs.**
  `patch/worker/patch_gdn_attn.py:339-375` allocates rows `(max_num_batched_tokens, 2)`
  and `(max_num_seqs + 1,)`, while the required counts are
  `Σ ceil(len_i/64) ≤ Σ len_i ≤ max_num_batched_tokens` and `num_seqs + 1 ≤ max_num_seqs + 1`:
  the invariant holds by construction, so "a bigger batch overruns the metadata buffer"
  is refuted before reading the slicing code.
- **#7863's mechanism is no longer on the 910 path at this pin.** Its fault was the
  masked assign `initial_state[~has_initial_state, ...] = 0` (a data-dependent
  `Nonzero` under capture); at v0.19.1rc1 the clearing is the Triton `clear_ssm_states`
  (`ops/triton/fla/utils.py:98`), and the masked assign survives only on 310P
  (`_310p/ops/fla/gdn_310.py:195`). Cite #7863 for symptom and owner, never for the line.
- State the scope defect in the first paragraph: a variant absent from the support
  matrix at that pin (`Qwen3.5-35B-A3B` is not listed; only 397B-A17B and 27B,
  `docs/source/user_guide/support_matrix/supported_models.md:82-83`) is an unvalidated
  model on a validated architecture.
