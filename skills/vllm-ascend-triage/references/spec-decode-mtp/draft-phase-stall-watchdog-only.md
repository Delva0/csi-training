# Draft-phase stall where the only line is the ringbuffer watchdog

Verified at vllm-ascend **v0.11.0rc0** (`00ba07102212`) with vllm **v0.11.0rc3**
(`8ce5d3198d00`); recorded for **#3398** (Baichuan-M2-32B + its shipped `draft/`
subdirectory, `method: eagle3`, `num_speculative_tokens: 1`, `max_model_len: 4096`,
`tensor_parallel_size=2`, `distributed_executor_backend="mp"`, no `enforce_eager`;
reporter's env block says vllm v0.11.0 + vllm-ascend v0.11.0rc0, 910B).

Signature: the paste is a launch plus **one** log line, and that line is

```text
INFO 10-10 15:15:11 [shm_broadcast.py:466] No available shared memory broadcast block
found in 60 seconds. This typically happens when some processes are hanging or doing some
time-consuming work (e.g. compilation).
```

with no traceback, no rank prefix, and the reporter supplying the phase themselves
(「在运行草稿模型阶段卡住」). Treat that attribution as a **claim**: the line names no
phase, and the reporter's own wording is inferred from where their log stopped.

## The line is a consequence, and the side is all it gives

`../hccl-and-hardware/rpc-watchdog-timeouts.md` owns the reading: the string is printed by
the ring-buffer's `acquire_read` **and** `acquire_write` wait loops with identical text
(two sites in one file at vllm v0.11.0rc3 — `acquire_write` `shm_broadcast.py:373`, emits
at `:397-402`; `acquire_read` `:431`, emits at `:466-471`), so "which side waited"
decides the sentence and the line number does not port across builds. It proves no reply arrived inside 60 s; it does not prove a rank, an operator, an
OOM or a link fault. Do not open a `/dev/shm` sizing lane from it (`--shm-size` is not a
rung) and do not quote the reporter's build's line number as the pin's.

## The pin prints its own phase markers — read them in order

At this ref `vllm_ascend/worker/model_runner_v1.py` walks the drafter through three
stages in one place, and each one is distinguishable in a per-rank log:

| Marker | Site | What a stop here means |
|---|---|---|
| `logger.info("Loading drafter model...")` | `:2644` | the drafter exists and the worker is about to construct/load the draft checkpoint |
| `self.drafter.load_model(self.model)` | `:2645` → `eagle_proposer.py:78-111` (`get_model(...)`, so the checkpoint *is* loaded) | a stall here is draft **load/construction**; nothing on device has run |
| `self.model.set_aux_hidden_state_layers(self.model.get_eagle3_aux_hidden_state_layers())` | `:2646-2648`, EAGLE3 only | the target class must implement `SupportsEagle3`; missing it is an **`AttributeError` in the log**, never a silent hang |
| `self.drafter.dummy_run(...)` / capture | `:2482-2483` | device work has started: a hang here is the first-step / capture lane |

So one artefact decides the phase: **the last line of the full per-rank log**, not the
paste's title. A paste that stops after `Loading drafter model...` is a load-phase
question; an `AttributeError` on `get_eagle3_aux_hidden_state_layers` is a target-family
question; a paste that passes both and then stops is a device/plog question. Ask for that
tail before ranking a mechanism.

## Which target families can reach this path at all

- `SupportsEagle3` at vllm v0.11.0rc3 is implemented by `llama`, `qwen2`, `qwen3`,
  `minicpm`, `gpt_oss` and `apertus` (`vllm/model_executor/models/*.py`, one
  `rg -n 'get_eagle3_aux_hidden_state_layers'`). `baichuan` is registered
  (`registry.py:55-58`) and implements none of it, so a `BaichuanForCausalLM` target
  cannot supply the three aux hidden states — read the target's `config.json`
  `architectures`/`model_type` from the reporter's disk rather than from the model name
  (a Qwen2-derived release has `qwen2` here and the path exists).
- Upstream's own gate is `eagle3_target_supported = ["llama", "qwen", "minicpm",
  "gpt_oss"]` as a **substring** match on `hf_text_config.model_type`
  (`vllm/config/speculative.py:545-552`). **It does not run downstream at this ref**:
  `vllm_ascend/patch/platform/patch_common/patch_config.py:85` defines its own
  `__post_init__` and `:312` executes `SpeculativeConfig.__post_init__ = __post_init__`,
  a wholesale replacement, and `rg -n 'eagle3_target_supported' vllm_ascend/` returns
  nothing. Quote that as "an unusual family is not refused here" — an unvalidated path
  proceeding is a coverage statement, never the cause of the stall.
- The draft checkpoint's own contract: downstream asserts the draft model is upstream's
  `Eagle3LlamaForCausalLM` (`eagle_proposer.py:14` import, `:412` the `assert
  isinstance`, `:413` `self.model.combine_hidden_states(...)`), whose projection is
  `fc = Linear(target_hidden_size * 3, hidden_size)` (`llama_eagle3.py:130-131`) — so a
  valid draft is a **Llama-based** eagle3 head carrying `target_hidden_size`, not a
  re-sharded copy of the target. Check `architectures`, `model_type`, `hidden_size`,
  `target_hidden_size`, `draft_vocab_size` in `<draft>/config.json`; a mismatch fails
  loudly at load, so it explains "cannot start", not "silently stops".

## What the release itself validates

- The only spec-decode e2e at this ref is
  `tests/e2e/singlecard/spec_decode_v1/test_v1_spec_decode.py` — **single card, TP=1**,
  `LLM-Research/Meta-Llama-3.1-8B-Instruct` as target (`:51-53`) with
  `vllm-ascend/EAGLE3-LLaMA3.1-Instruct-8B` as draft (`:60-61`), and for eagle3
  `num_speculative_tokens: 2`, `max_model_len: 128`, `enforce_eager=True` (`:126-132`).
- **The workflow that would run it is commented out**: `.github/workflows/_e2e_test.yaml:106-107`
  reads `# Fix me: OOM error` / `#pytest -sv tests/e2e/singlecard/spec_decode_v1/test_v1_spec_decode.py`.
  At this pin no spec-decode e2e runs in CI, so "eagle3 is supported" is a nominal
  statement — say that with the two anchors, and do not let it become "the model is
  unsupported".
- The feature is one release old on this line: `docs/source/user_guide/release_notes.md:25`
  under `## v0.11.0rc0` — "Eagle3 spec decode method is back now. [#2949]". A path
  restored in the reported release is the least-exercised one there.

## Ranked causes, each with the one check

1. **Stall inside draft construction/load on at least one rank** (the phase their own log
   stopped in). Check: the last line of *each* TP rank's log around the warning, then
   `py-spy dump --pid <each worker pid>` at the stall — Python frames in
   `load_model`/safetensors mmap name it; frames absent from both means the ranks are
   waiting on device or on each other.
2. **The first draft/verify step on device** (no Python frame, no error at all). Check:
   rerun with `ASCEND_LAUNCH_BLOCKING=1` and read the plog at that minute. Same-window
   precedent, #2667: the *validated* eagle3 recipe (Llama-3.1-8B +
   `EAGLE3-LLaMA3.1-Instruct-8B`, K=1, `max_model_len: 128`, `enforce_eager=True`) died
   with a full device dump — `fault kernel_name=
   paged_attention_mask_33_mix_aic`, `aicore exception`, `runtime result = 507053`,
   `EZ9999 Inner Error!`. That is what this path looks like when it **fails on device**:
   a dump, not silence — which is why a silent stop is a plog/blocking-mode question
   before it is a model-family one.
3. **The launch is outside everything the pin validates** (family, draft, TP>1,
   `enforce_eager=False`). That is the coverage verdict, not a mechanism: pair it with
   the control in the ladder, never with "so it cannot work".
4. **Version pairing.** `vllm-ascend v0.11.0rc0` was cut against vllm **v0.11.0rc3**;
   a final `vllm v0.11.0` on top of it is a pair no e2e at this pin exercises. Align the
   pair (or move to the image tag that carries both) before proposing a patch.
5. **Memory pressure during a 32B TP=2 load** (host swap shows as a slow, not stuck,
   rank). Check `free -g` and `dmesg` from the stall minute; a page-file thrash prints no
   NPU error.

## Ladder (cheapest first)

1. The same script with `speculative_config` **removed** — one launch, and it splits
   "target/model/TP/graph" from "draft path"; this is the rung both recorded sessions
   skipped.
2. `enforce_eager=True` (+ `max_num_seqs=1`), i.e. the only validated eagle3 shape at
   this pin — it also removes capture from the dump lane.
3. `ASCEND_LAUNCH_BLOCKING=1` to turn a silent stop into an attributed device error.
4. At the stall: `py-spy dump` of every worker pid, `npu-smi info` during the minute, and
   the plog for that minute.
5. The pin's validated pair as the environment control on the same box (Llama-3.1-8B +
   `vllm-ascend/EAGLE3-LLaMA3.1-Instruct-8B`, K=2, TP=1, `enforce_eager=True`): if it
   serves, the box and the install are exonerated and the case is the model × draft × TP
   combination.

## What to send back

The full per-rank logs of **both** TP ranks around the warning (not the filtered pair of
lines), `<draft>/config.json` and the target's `config.json` (`architectures`,
`model_type`, `hidden_size`, `target_hidden_size`), `npu-smi info` taken *during* the
stall, `free -g`, and the plog for that minute. Do not ask for the version block — it is
in the paste — and do not ask them to run `collect_env.py` again.

## Recorded cost, and what stays open

Two sessions on this very case ended without a usable answer: the first shipped **no
message text at all** after 62 messages / 30 tool results (its own last block decided
whether `EagleProposer` loads draft weights — a question that cannot produce a hang, and
answered wrong; `../upstream-downstream-drift/load-format-loader-contract-drift.md`
§*`get_model(...)` is the loader call*), and the second spent eight turns on repo
archaeology and was cut inside ranked cause 2 of the ladder it had just started writing.
Both had, within their first ten turns, the release-note line, the single-card coverage
and the watchdog's side — the missing piece was the phase marker, which is one `rg -n
'Loading drafter model' <ascend>/vllm_ascend/worker/` away.

Open: Baichuan-M2-32B's draft head architecture was never verified (no reporter reply in
the snapshot); whether the stall was load or first step is unknown; and which of #2949's
restored pieces a dense non-Llama target exercises is not established by any test at this
pin. A later release that re-enables `test_v1_spec_decode.py` in
`.github/workflows/_e2e_test.yaml`, or that restores `eagle3_target_supported` in the
downstream patch, invalidates the coverage half of this page — re-grep both before
reusing it.
