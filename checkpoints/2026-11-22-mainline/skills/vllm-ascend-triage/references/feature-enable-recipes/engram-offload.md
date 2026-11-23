# DeepSeek-V4.x Engram offload: the key set and its four recorded rows

Scope: a launcher carrying an **Engram** key — `enable_engram`, `engram_storage`,
`enable_engram_ple_offload` — or the host-memory pair the table needs
(`dp_shared_memory`, `enable_cpu_binding`), or a log naming `…engram.embed.weight`.
Entry shapes: "DS-V4.1-Flash 如何开启 engram", a `[Doc]: Feedback for
…/DeepSeek-V4.1-Flash.html` startup failure, an A3 start that hangs **after**
warm-up/graph capture, and `507011` / invalid-GM at concurrency with PLE offload on.
Read it before quoting a context or KV envelope for a DS-V4.x serve as well: engram
moves both the per-rank weight footprint and the KV capacity by the measured amounts
in row 3, so the tutorial's `--max-model-len` row is not the budget of an engram
launch (`../deploy-sizing/dsv4-kv-footprint-terms.md` step 6).

**Evidence class: corpus issue bodies only.** No engram code is readable here — the
v0.18.0 vllm-ascend tree this case family pins has **zero** `deepseek.?v4` hits and
no `engram` string at all — and comments/PRs are not readable, so nothing below is
source-verified at a pin. Each row names the issue that carries it: re-read
that body before quoting a version, and keep "reported failure" separate from
"confirmed fix".

## The keys, as a working launch spells them

Minimal form, from the reporter-confirmed fix in `#17236` (A3 dual-node,
DeepSeek-**V4.1**-flash **W8A8**, started by following the tutorial's own serve line):

```bash
--additional-config '{"enable_engram":true,"engram_storage":"int8"}'
```

Their words: add this to the serve command and keep everything else the same as the
official page (其他的保持官网一样) — after that the service started. The full form a
4-node run uses is the recipe below (`enable_engram_ple_offload`,
`enable_cpu_binding`, and the `ascend_compilation_config` pair on top).

## The four rows

| # | The token to grep | What it is | One action |
|---|---|---|---|
| 1 | `engram.embed.weight: expected BF16 source for bf16` | during weight loading: the serve line never enabled engram, so the embed table is converted through the default bf16 path | add the minimal `--additional-config` above; `#17236` |
| 2 | `migratepages` helpers alive >15 min, `No available shared memory broadcast block found` / DP coordinator READY, ports never open, worker kill leaves the helpers | CPU binding runs after warm-up and migrates the whole address space while the registered HOST_UVA table (~206 GiB) is shared across DP ranks | workaround `enable_cpu_binding=false`; `#17286` |
| 3 | `507011` / invalid GM address at concurrency 24-32, plus decode regression | the PLE offload implementation itself, only with `enable_engram_ple_offload=true`; controlled A/B clean with it `false` | the key off, or a build past that PR revision; `#16828` |
| 4 | answers carry only reasoning content / `content` absent, no `think` tag | parser/template contract on a `--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41` launch, not the offload | `../accuracy-benchmarks/reasoning-output-routing.md`; `#16877` |

**Row 1** (`#17236`) is a *documentation* defect on the DS-V4.1-Flash tutorial page —
the page's serve command is incomplete — so it is filed as `[Doc]: Feedback for
<url>`; grade it there (`../tutorial-verification/index.md`). The issue carries **no
environment block**: no release, no image tag, so its version scope is unknown. Ask
which image before quoting it as "the" fix.

**Row 2** (`#17286`: A3, CANN 9.1, DS-V4.1-Flash W8A8, D node TP2/DP8 on 16 NPUs,
`dp_shared_memory=true`, `enable_cpu_binding=true`, DSpark 5 speculative tokens,
`cudagraph_mode=FULL_DECODE_ONLY`, `async_scheduling=true`) is the shape to read
before blaming the runtime: weight load and graph capture **complete**, then
`migratepages <pid> 0,1,2,3 <target_numa>` stays alive and the DP coordinator never
signals READY. The Engram shared table was already created and registered with
`aclrtHostRegisterV2(… ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)` when the CPU-binding
path migrated the process. The **trap** is that the same
`No available shared memory broadcast block found` line is
`../startup-failures/mm-encoder-profile-stall.md`'s signature for a multimodal
*encoder profile* stall: discriminate on the launch, not the line — engram/HOST_UVA
keys plus a phase past graph capture ⇒ here, a VL serve with
`--limit-mm-per-prompt` / `--max-num-batched-tokens` ⇒ there.

**Row 3** (`#16828`: 2×A3, DP4/TP8/EP32, vllm 0.28.0.dev0 `84030bbe`, vllm-ascend
`0.1.dev16544` at `b72810b`, the PR #16544 Engram host-offload implementation) is the
only row with a **measured** memory effect: with the key on, ≈12.87 GiB less
model-weight memory per rank and ≈1.97× KV capacity, then engine death at concurrency
24-32; the same image/model/topology/workload is stable with it `false`. PR #16544
was force-pushed afterwards to a different host-registration/UVA path, so this row
does **not** transfer to the current head — ask which revision the reporter's build
contains (`vllm_ascend.__version__`, image tag) instead of asserting the bug is
present.

## A field recipe: DS-V4.1-Flash on 4×8 A2, colocated (no PD)

`#16877` is the only corpus launcher for this topology — 4× A2 nodes × 8 NPUs,
DP4 (`--data-parallel-size-local 1`), TP8, EP32, rank 0 exposes the API and ranks 1-3
run `--headless`, image `quay.io/ascend/vllm-ascend:deepseek-v4.1-flash-engram`,
weights `/data/models/DeepSeek-V4.1-Flash-w8a8`. It reached serving (its defect was
row 4), which certifies "the launch comes up" and nothing more:

```bash
exec vllm serve /data/models/DeepSeek-V4.1-Flash-w8a8 \
  --data-parallel-address <rank0-ip> --data-parallel-rpc-port 13399 \
  --data-parallel-size 4 --data-parallel-size-local 1 --data-parallel-start-rank $NODE_RANK \
  $HEADLESS_ARG --tensor-parallel-size 8 --enable-expert-parallel \
  --served-model-name DeepSeek-V4.1-Flash \
  --max-model-len 262144 --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.95 --block-size 128 \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --trust-remote-code \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy --quantization ascend \
  --additional-config '{"enable_engram":true,"enable_engram_ple_offload":true,"engram_storage":"int8","enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Its env block is part of the recipe — per-NIC `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`
/ `TP_SOCKET_IFNAME` + `HCCL_IF_IP`, `HCCL_OP_EXPANSION_MODE=AIV`,
`HCCL_CONNECT_TIMEOUT=1800`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`,
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `OMP_PROC_BIND=false`, jemalloc
preload — take it from the body, not from memory.

Say two things when handing it over: `262144` is a **field** value on four colocated
A2 nodes, not a documented A2 row (the tutorial's A2 rows are 133120/135000 and its
1M rows are A3/950DT — `../deploy-sizing/dsv4-kv-footprint-terms.md` step 6), and the
engram keys are part of what makes that context fit, so dropping them changes the
budget the same `--max-model-len` needs.

## What to send back

- the whole `--additional-config`, plus whether `dp_shared_memory` / `enable_cpu_binding`
  are in it, the image tag and `vllm_ascend.__version__`;
- where the start stops: inside weight loading, after `Loading model weights took X GB`,
  or after graph capture — with the last 20 lines of the rank-0 log;
- per-node `npu-smi info`, and whether `migratepages` helpers are alive (row 2);
- for row 3, the concurrency at which it dies and whether the key off is clean on the
  same script.

## Recheck conditions

These are new keys, so names and nesting move between releases; on a strict
`AscendConfig` generation an unknown key is **fatal** rather than ignored, so run
`uv run --no-project python <skill_root>/scripts/additional-config-key-audit.py
--tree <checkout> --json '<the JSON>'` before telling anyone a key is misspelled or
removed (`../startup-failures/additional-config-schema.md`). If the pinned tree carries
no engram key at all, say the path does not exist at that pin and ask for the image tag
— do not read a newer tree as if it were theirs.
