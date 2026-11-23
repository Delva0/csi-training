# A degenerate answer whose fix is the dtype (fp16 → bf16, `INF_NAN_MODE_*`)

Scope: a serve whose answers are one repeated token — all `!` / 感叹号, `，，，`, a
single word — or fluent garbage, with **no traceback**, where the reporter's own
controls say the bytes are fine: the same checkpoint answers correctly through
transformers (or the vendor fine-tuning tool) and on a GPU box. That control is what
ranks this page **above** the checkpoint/SFT row and above TP/shape theories, because
those predict the working control failing too (SKILL rule 15). It applies to a
**vision** request (`vl-vision-tower-degeneracy.md`) and to a **text-only** serve
(`constant-token-degeneracy.md`) — the lever is the same, so it lives here once.

Label discipline: the **fix is reporter-confirmed** (#562 below). The mechanism —
fp16 overflow producing `inf`/`nan` that a greedy decode then collapses onto one id —
is a **hypothesis**: the report carries no logits, and no snapshot can show the
overflowing op.

## Step 0 — the check is a two-line diff on the dtype axis, not a theory

The pinned recipe for the model in the report very often already says bf16, and the
reporter's launch usually does not: at vllm-ascend **v0.7.3rc2** (`00459aea`),
`docs/source/tutorials/single_npu_multimodal.md:147` is
`vllm serve Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384`
— bf16 spelled out, in the only VL recipe that pin validates.

Nothing downstream picks the weight dtype for you at that pin: `vllm_ascend/` only
reads it back for the KV cache and a few internal tensors (`platform.py:152-153` sets
the quant KV cache dtype, `worker/model_runner_v1.py:82-88` and `:393`), so under the
default `--dtype auto` the engine takes the **checkpoint's**
`config.json:torch_dtype`. Two reads settle which lane the reporter is in:

1. `python -c "import json;print(json.load(open('<model>/config.json'))['torch_dtype'])"`
   on their exported directory (a re-exported / merged / quantised checkpoint carries
   whatever dtype the export tool wrote, not the base model's).
2. the engine's own startup line `… dtype=float16 …` in their log — that is the
   authority; the flag is not, when the flag is absent.

Then one restart, same body and image: add `--dtype bfloat16`.

## Step 1 — bf16 is never "unsupported" on this hardware

`IsBF16Supported()` is `GetSocVersion() >= SocVersion::Ascend910B1`
(torch_npu `torch_npu/csrc/core/npu/NpuVariables.cpp:79-82`, Ascend/pytorch
`v6.0.0.alpha001-pytorch2.5.1`), and the enum makes that literal: `Ascend910B1 = 220`,
`Ascend910B4 = 221` (`torch_npu/csrc/core/npu/NpuVariables.h:16-21`), so every 910B SKU
accepts `--dtype bfloat16` even when the checkpoint is fp16. Do not accept "bf16 cannot be used here" as a
reason to stay in fp16; the only real cost is a weight cast at load.

## Step 2 — the `INF_NAN_MODE_*` levers, and which stack accepts which

These are **isolation** levers, not recommendations: they change overflow semantics
(`inf`/`nan` versus saturation) and therefore move or hide the overflowing op, while
the dtype change removes its cause. Never spend a restart on them before `--dtype
bfloat16`. Two variable names exist in the wild, and the fact that decides which one
the reporter's install reads is the **build's own source**, not the `torch_npu==`
version in the dump:

| build (source-verified) | Fact |
|---|---|
| Ascend/pytorch `v6.0.0.alpha001-pytorch2.5.1` (`a3b75fc0`, 2024-11) | `INF_NAN_MODE_ENABLE` is the only variable in the tree: parsed with default **1**, rejected unless 0/1 (`torch_npu/csrc/core/npu/register/OptionsManager.cpp:56-65`). Setting it to `0` makes `IsSupportInfNan()` false (`NpuVariables.cpp:62-72`), and stream creation then enables the overflow switch, `AclrtSetStreamOverflowSwitch(stream, 1)` (`torch_npu/csrc/core/npu/interface/AclInterface.cpp:151-153`). `INF_NAN_MODE_FORCE_DISABLE` has no reader anywhere. |
| Ascend/pytorch `v7.1.RC1.alpha002-pytorch2.7.1` (`d33df559`) — the first tree in this snapshot that carries both — and the 2.8.x tree a later note cited (`v7.2.0.1-pytorch2.8.0`) | Both names are parsed, and on any device whose **default** is inf/nan-capable — every 910B SKU is (`>= Ascend910B1 && < Ascend310B1`, `NpuVariables.cpp:69-71`) — `INF_NAN_MODE_ENABLE=0` **aborts**: `AT_ERROR("INF_NAN_MODE_ENABLE shouldn't be set to 0 on the current device. If you want to disable inf-nan mode, please export INF_NAN_MODE_FORCE_DISABLE=1")` (`NpuVariables.cpp:67-77`, gated on `CheckInfNanModeForceDisable()`); `INF_NAN_MODE_FORCE_DISABLE=1` on its own returns false (`:78-80`, parsed at `OptionsManager.cpp:82-93`). |

So the split is a **build** split, not a version-number split, and the version line of
the env dump does not decide it: #562's reporter ran `torch-npu 2.5.1.dev20250320` — a
2025 dev build of the `2.5.1` line, which the table above would put on the "old" side —
and in that thread (2025-04-22) a maintainer answered a MindIE-derived
`export INF_NAN_MODE_ENABLE=0` suggestion with "INF_NAN_MODE_ENABLE shouldn't be set to
0 on the current device … please export INF_NAN_MODE_FORCE_DISABLE=1", reported as a
correction of the advice. Their exact dev build is not in the snapshot, so read that as
the maintainer's statement about the build in front of them, not as a pinned line: what
it settles is that a dev build of an older minor can already carry the guard a later
minor is credited with. Before naming either variable, grep the reporter's install
(`torch_npu/csrc/core/npu/{NpuVariables,register/OptionsManager}.cpp`) or read the
paste — the abort text **is** the discriminator. If it appears, the lever is
`INF_NAN_MODE_FORCE_DISABLE=1`; if the process starts normally with `ENABLE=0`, that
build has no guard and there is no second switch. Give the one you verified with its
`<ref>`, not the pair as "either one".

## Recorded case — #562 (reporter-confirmed)

vllm-ascend `v0.7.3rc2` ↔ vllm `v0.7.3`, 8×910B4, torch-npu `2.5.1.dev20250320`,
CANN 8.0.0; Qwen2.5-VL-7B after a LLaMA-Factory SFT answers **全是感叹号**, while
"使用llamafactory推理是正常的，在v100上也是正常的". Resolution, posted by the reporter
(2025-04-23): 「已解决，将精度改为 bfloat16 就可以了」.

The SFT checkpoint was **never shared**: asked for the post-SFT weight on 2025-04-21 the
reporter answered "Sorry, this model weight is not open to the public". The two clean
controls therefore exist only as the reporter's own statement — quote them as the
report's controls, not as something a re-run here can confirm, and keep "the weight" off
the send-back list. The same thread also shows the order the levers were tried in: on
2025-04-22 the reporter was handed `INF_NAN_MODE_ENABLE=0` and answered "Okay, thanks.
I'll try it", the maintainer corrected that variable name the same day (Step 2), and the
one that closed the issue a day later was still the dtype.

Two facts from that session are worth carrying into the next one:

- At that pin there is **no downstream Qwen2.5-VL code at all** —
  `vllm_ascend/models/__init__.py` registers only `Qwen2VLForConditionalGeneration`,
  `DeepseekV2ForCausalLM` and `DeepseekV3ForCausalLM`, and the directory holds
  `deepseek_v2.py` + `qwen2_vl.py` only; `Qwen2_5_VLForConditionalGeneration` resolves
  to upstream (`vllm/model_executor/models/registry.py:180`). So on a 0.7.3rc2 report
  the head-dim padding family (`MIN_PAD_SIZE`) cannot be the cause and
  `USE_OPTIMIZED_MODEL` does not exist yet — say that instead of ranking it.
- `tests/` at that pin is `conftest.py`, `model_utils.py`, `ops/`,
  `test_offline_inference.py`: no VL test, so the single-NPU tutorial command is the
  whole validated envelope — and its dtype is the axis the reporter's launch differed
  on.

Episode detail, for the cost of skipping this step: the recorded session had the
report's two working controls in hand, spent 16 turns on TP>1 and checkpoint/SFT
rows (plus a `rg -rn` rewrite it caught), ranked the dtype axis **last**, and was cut
at the output cap with no answer text — while the reporter closed the issue with the
one-flag fix.

## What the first message still owes

The ranked verdict with `--dtype bfloat16` as rung 1, then the two reads above
(`config.json:torch_dtype`, the log's `dtype=`), then — only if the bf16 re-run still
degenerates — the checkpoint row (`vl-vision-tower-degeneracy.md` rows 2-6) and the
bf16 run's own `finish_reason` / `usage.prompt_tokens` compared with the tutorial
sample, so the next answer can tell "same degeneration" from "a different one".
