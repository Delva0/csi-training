# Omni/audio over the chat endpoint: a transcript that loops, or garbage

## Scope

A Qwen2.5-Omni / Qwen2-Audio / Qwen3-Omni serve where a request carrying audio —
an `audio_url` in `/v1/chat/completions`, a `file://` path under
`--allowed-local-media-path`, or a base64 data URL — comes back fluent for a clause
or two and then repeats one phrase to `finish_reason: "length"`, or returns random
or empty text, with no traceback. Verified pins: vllm-ascend **v0.11.0rc3
`b6d63bbd`**, vllm **v0.11.0 `b8b302cd`**, transformers v5.0.0rc0 partial export
(SOURCES `4962`).

Not this page: text-only degeneration (`short-prompt-corruption.md`), a **VL image**
request degenerating into one repeated token (`vl-vision-tower-degeneracy.md`), a
100K+ prompt (`garbled-long-context-output.md`), a start-up death on the composite
config (`../startup-failures/vl-layer-count-attribute.md`).

## Rule 1 — Split the pipeline before ranking causes

| stage | owner | anchor at the pin |
|---|---|---|
| request → chat template, `<\|AUDIO\|>` placeholders | upstream vLLM frontend | `vllm/entrypoints/chat_utils.py:69,77,816-904` |
| mel frontend + audio encoder | **transformers** (`Qwen2_5OmniAudioEncoder`, `WhisperFeatureExtractor`) | `vllm/model_executor/models/qwen2_5_omni_thinker.py:36,680,790` |
| thinker model body | upstream vLLM, except the Ascend **vision** tower | `vllm_ascend/models/qwen2_5_omni_thinker.py:38,49` (`self.visual = AscendQwen2_5_VisionTransformer`) |
| worker, graph, scheduling | vllm-ascend | `vllm_ascend/worker/model_runner_v1.py` |

Two consequences worth stating in the answer:

- A **pure-audio** request never touches `self.visual`, so the
  `USE_OPTIMIZED_MODEL` gate (default `1`, `vllm_ascend/envs.py:96-97`) — the first
  lever for *image* degeneration — is not a lever here. The downstream omni file
  can be excluded from the ranked list.
- The audio encoder lives in `transformers`, and this snapshot's transformers export
  is partial (v5.0.0rc0, 1478 files): only `docs/source/en/model_doc/qwen2_5_omni.md`
  matches `qwen2_5_omni`, there is no `modeling_qwen2_5_omni.py`. Say that the
  encoder's own numerics are a **stated gap** instead of walking a file that is not
  there (rule 9).

## Rule 2 — The online audio route is accepted but not covered at this pin

- The tutorial's online section says so in one sentence:
  `docs/source/tutorials/single_npu_audio.md:122` — "Currently, vLLM's
  OpenAI-Compatible server doesn't support audio inputs", citing vllm#19977 — and
  its worked example is the **offline** route `LLM()` with hand-written
  `<|audio_bos|><|AUDIO|><|audio_eos|>` placeholders (`:60-97`).
- Upstream vllm v0.11.0 nevertheless parses `audio_url`
  (`vllm/entrypoints/chat_utils.py:69,77,816-904`), so the endpoint accepts the body
  and the pinned doc sentence is stale as a statement about *parsing* — quote it as
  coverage, not as a refusal.
- Coverage at the pin: the only audio e2e is offline `VllmRunner` on
  **Qwen2-Audio-7B-Instruct** (`tests/e2e/singlecard/test_vlm.py:92-124`,
  `max_tokens=10`, asserts only "output is not empty"); there is **no**
  Qwen2.5-Omni e2e config under `tests/e2e/models/configs/`. The support matrix
  lists Qwen2.5-Omni ✅ against #1760
  (`docs/source/user_guide/support_matrix/supported_models.md:63`).
- So the correct framing is "the route is not validated at your ref, and the
  project's own online example does not exist" — then the reporter's control
  decides ownership. Do not answer "audio over the API is unsupported" and stop;
  the reporter's request demonstrably ran.

## Rule 3 — `usage.prompt_tokens` places the modality

The response body's prompt count is the placement test: ~775 prompt tokens with a
~20-token instruction means the audio features entered the prompt (~750 audio
tokens); a `prompt_tokens` equal to the text-only request means the modality was
dropped and the question is an endpoint/processor one, not a decode one. Put this
number in the verdict sentence — it is the cheapest fact in the paste (recorded
#4962 body: `prompt_tokens: 775`, `completion_tokens: 256`, `finish_reason:
"length"`).

## Rule 4 — The ablation ladder, one axis per launch

1. **Same wav through the tutorial's offline path** (`LLM()`, `<|AUDIO|>`
   placeholders, `limit_mm_per_prompt={"audio": 1}`, greedy). Clean ⇒ the server /
   processor side; same loop ⇒ the engine or the audio tower. This is the row that
   partitions Rule 1's table and costs one script.
2. **`--enforce-eager`** on the failing serve. Multimodal × ACL graph has a recorded
   functional fix (`docs/source/user_guide/release_notes.md:214`, #1803), and #1760's
   own workaround line for a Qwen2.5-Omni-7B serve on the v0.9.2rc1 image is
   `--enforce-eager` ("Workaround: Use eager mode with `--enforce-eager`").
3. Drop **`--async-scheduling`**.
4. `--data-parallel-size 1` (one die of the two the reporter exports) — per-rank
   engines, and the mm/mrope path is the new code there.
5. The **same instruction with no audio**. If the text-only request also loops, the
   audio condition is not required and the track is
   `short-prompt-corruption.md`, not this page.

Each row removes one phase; a row only eliminates something if its own log line or
output is in the paste (SKILL.md rule 15).

## Rule 5 — A loop at `temperature: 0` is a decode symptom, not proof about the encoder

The first clause being a correct transcription means the audio condition arrived and
early decoding worked; a phrase loop that starts later, under greedy, is what one
forked logit looks like on a long generation. So the ranked answer may not claim
"the audio encoder is broken" from the text alone — send it as the ladder in Rule 4
with the offline row first, and treat a `repetition_penalty` / `presence_penalty`
run as mitigation evidence only (labelled recalled, not verified in the pinned
tree). The open question is which element of the chain owns it, and the offline row
is what closes it.

## Delivery order

1. Verdict: a repetition loop in a route the release does not validate — the pinned
   tutorial says online audio input is unsupported (`:122`), the pin has no
   Qwen2.5-Omni e2e and only an offline 10-token non-empty assert for audio —
   so ownership goes to the offline-vs-online control, not to the model file.
2. Rule 1's ownership table with the anchors (what is *not* the owner: the Ascend
   vision tower, the `USE_OPTIMIZED_MODEL` gate, and the absent encoder source).
3. `prompt_tokens` arithmetic from Rule 3.
4. Ladder from Rule 4, cheapest first, then the artefacts to send back: the full
   response body, the exact serve line, the wav's duration, and the offline-path
   output for the same file.
5. Open items: whether the loop survives the offline path; whether the audio
   encoder numerics are reachable at all in this snapshot (they are not).
