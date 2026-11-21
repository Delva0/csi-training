# Check `max_model_len` before anything else at load time

Reports that read "the model could not load / the process crashed during startup"
with an unclear or missing worker error are frequently a context-length budget
problem, and the budget is invisible because the flag was simply omitted.

## How the default is produced

vllm derives `max_model_len` from the HF config when `--max-model-len` is absent:
it takes the smallest of `max_position_embeddings`, `n_positions`,
`max_seq_len`, `seq_length`, `model_max_length`, `max_target_positions`,
`max_sequence_length`, `max_seq_length`, `seq_len`, then the tokenizer's
`model_max_length` (`_get_and_verify_max_len` in `vllm/config/model.py`,
v0.11.0rc3, lines 1766+). It logs `Using max model len N`
(`vllm/config/model.py:1510`).

For long-context VL/MoE checkpoints that default is the model's advertised
context — hundreds of thousands of tokens — which is far beyond what a 64 GB
card can hold together with weights for a given `--gpu-memory-utilization`.

## Reporter-confirmed case (vllm-ascend issue #3543)

- Environment: image `quay.io/ascend/vllm-ascend:v0.11.0rc0` (vllm-ascend
  v0.11.0rc0 + vllm v0.11.0rc3), 910B3, `--tensor-parallel-size 4`,
  `--max-num-batched-tokens 4096`, `--gpu-memory-utilization 0.85`, **no
  `--max-model-len`**, model explicitly named `Qwen3-VL-30B-A3B-Instruct`.
- Symptom: rank assignment is logged, then a long silence, then the parent-only
  failure lane (`WorkerProc initialization failed ...`, no worker traceback) plus
  `TBE Subprocess ... main process disappeared`. See `silent-worker-exit.md`.
- Reporter's own finding: on the 910B3 machine `max_model_len` took the model's
  default maximum, which prevented the model from loading and crashed the
  process; after lowering `max_model_len` the model loaded and the service
  started normally. (Reporter-confirmed, not derived by us.)
- Corroboration available in the same release's docs (pinned vllm-ascend
  source): `docs/source/tutorials/multi_node_qwen3vl.md` passes `--max-model-len 32768`
  in both node commands for `Qwen3-VL-235B-A22B-Instruct`. The failing command
  passed none. Diffing a reported command against the shipped tutorial/CI command
  is a cheap, high-yield step.

## What to do

1. Ask for / read the `Using max model len N` line, or compute the default from
   `config.json` (`max_position_embeddings`) — one file read, no guessing.
2. Have the reporter re-run with an explicit modest budget, e.g.
   `--max-model-len 32768` (keep `--max-num-batched-tokens` at or below it). If
   the service starts, the default was the problem; no code defect needs filing.
3. Keep the same explicitness in any suggested command you publish, and say why
   (the tutorials already do it).

## Raising the budget past the checkpoint has a runtime signature (recorded #2349)

`--max-model-len` larger than the derived maximum needs
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, and that env var turns the guard into a
warning ("This may lead to incorrect model outputs or CUDA errors",
`vllm/config.py:3507-3523` at v0.9.2). One of those errors is concrete: the
RoPE cache is built from the **checkpoint's** `max_position_embeddings`
(`vllm/model_executor/models/qwen3.py:178` →
`vllm/model_executor/layers/rotary_embedding.py:132`), is not re-sized by the flag, and on
Ascend is indexed by `positions`
(`torch_npu._npu_rotary_embedding(positions, …, cos_sin_cache, …)`,
`vllm_ascend/ops/rotary_embedding.py:78-85`, installed at `:286`). A request
whose position passes that value trips a device-side gather assert —
`gather_v3_base.h:137 … Index N out of range[0 M)` — which reads like a card
fault and is not one. Two commands separate it from every other out-of-range
report: `jq .max_position_embeddings <model>/config.json` against the serve
line's `--max-model-len`. Page:
`../custom-ops-runtime-faults/gather-index-out-of-range.md`.

## The same default can kill the **host**, not the card (recorded #3314)

On a long-context checkpoint the omitted flag has a second failure signature that
is not an HBM OOM and not a card problem: every worker builds an eager
`max_model_len²` attention mask on the **host** in `NPUModelRunner.__init__`, so
the box thrashes or the kernel OOM-kills the ranks before `Starting to load model`
(structured prompt, then 1.5 TiB of RAM at 100 %, `WorkerProc initialization
failed` with no worker traceback). #3314: Qwen3-30B-A3B-2507 (ceiling 262144),
no `--max-model-len`, TP=4/8 → worked at 65536, and the reporter's follow-up
confirmed the derived 256K and that lowering it is the circumvention.

So when the reporter asks *"how large a `max_model_len` can I use?"*, the answer is
never a number seen in another report: it is `sqrt(headroom · host_RAM / (9 ·
ranks))` per `scripts/attn-mask-budget.sh`, minus what the weights and KV setup
need. `../runtime-memory/attention-mask-host-allocation.md` carries the arithmetic,
the version/CANN gate that changes it, and the two rungs that do not help
(`--enforce-eager`, `--max-num-batched-tokens`).

## Limits of this check

- It is a configuration/budget effect, not a proven defect in vllm-ascend: the
  same maintainer with the same vllm/vllm-ascend versions did not reproduce on
  their own hardware. Per-card HBM and model variant decide whether the default
  fits.
- It does not explain collective failures: the recorded CANN/HCCL case
  (vllm-ascend issue #7477) already passed `--max-model-len 32768` and still
  failed inside `HcclAllreduce` — that route is
  `../hccl-and-hardware/index.md`.
