# Script catalogue: the exact invocation of each one-call check

## Scope

The flag synopsis and the expected reading for every re-runnable script under
`scripts/`, i.e. the detail behind the compact name index in `SKILL.md`'s support
entry points and behind the "one call: `scripts/…`" pointers on the topical
pages. Open it when a page names a script and you need its arguments. Each
script's own header repeats its full flag list and is authoritative; the synopsis
here records the flags as written when the page was added, so a script changed
after that keeps the header, not this text.

## Common rules

- Inherit every rule in `SKILL.md`.
- A named script **is** the check: run it rather than re-deriving its arithmetic
  by eye or with an ad-hoc snippet (`cheap-checks-by-track.md`).
- `<skill_root>` is the path `skill_view` reports; every other link on this page
  is skill-root relative and every script is run from there.
- A script reproduces a calculation or extracts a set of sites. It does not
  decide the cause: quote its output as the check and keep the claim-strength
  label (`SKILL.md` rule 6).
- The issue id in brackets is where the arithmetic came from. Re-read the pinned
  source before trusting a derivation at another release.
- Several scripts answer per-track questions in one call; the track entry that
  names one is in `cheap-checks-by-track.md` or in its topical page.

## Entries

- `scripts/history-titles.sh [-i] '<regex>' [root] [limit]` — precedent index,
  one `<file> :: <title>` line per matching history issue (rule 8: titles first).
  Arguments are **positional**; `-i` (case-insensitive) is the only flag, and any
  other leading `-x` exits 2 with the usage instead of being taken for the regex
  (a `history-titles.sh -i 'reranker'` call once burned a turn on
  `no such history root: reranker`).
- `scripts/sources-summary.sh [SOURCES.json ...]` — the version contract, one line
  per repo with coverage and `runtime_verified`, plus the unavailable repos.
- `scripts/oom-budget-split.sh --line '<OOM line>' --gmu F [--params P] [--tp N]
  [--dtype D] [--hbm GB]` — splits a device-side OOM into inside vs outside the
  `gmu x total` budget, then prices weights/TP and the remainder (#6572).
- `scripts/card-budget.sh --params P [--dtype] [--tp] [--gmu] [--hbm] [--cards]
  [--stage-gmu] [--stage-devices]` — weights-per-card vs budget and the gmu sum
  over processes sharing a device (#9335). `P` is **billions** of parameters
  (`671`, not `671e9`); the script exits 2 on a raw count rather than printing
  arithmetic that inverts the verdict.
- `scripts/aclgraph-size-budget.sh --layers N [--tp --dp --pcp --dcp --ep --moe
  --aiv --max-num-seqs --sizes]` — the capture-size budget: maximum captureable
  sizes and the sampled subset (94 layers, TP=16 gives 6 sizes, #3389).
- `scripts/seq-cliff-capture-budget.sh --max-num-seqs N [--num-spec K]
  [--max-num-batched-tokens T] [--running 256,258,264]
  [--max-cudagraph-capture-size V] [--tp --dp] [--enforce-eager]` — the
  concurrency cliff: the Ascend default capture ceiling
  (`min(max_num_seqs × (1+num_spec), 512)`), the capture-size list near the
  boundary, the running-sequence count where dispatch flips to `NONE`, the
  padded size per step, and the `max_cudagraph_capture_size` A/B value with its
  truncation trap (#14440).
- `scripts/fia-tnd-seqlen-check.sh --query-t N --seqlen-last M [--paste <log|->]
  [--max-num-seqs S] [--num-spec K] [--capture-sizes 1,2,4,8,16,32] [--dp D]
  [--requests R] [--mode full|piecewise|eager]` — the FIA TND refusal in one
  call: the capture-size list that launch resolves to (upstream default
  `[1,2,4] + 8..min(max,255) step 8 + 256..max step 16`, max =
  `min(max_num_seqs × (1+K) × 2, 512)` clipped by `max_num_batched_tokens`),
  whether `queryT` is one of those buckets, every request/token decomposition of
  `last(actualSequenceLengthQ)`, the drafter-vs-target and DP-max readings, and
  the ordered ablations (#6598).
- `scripts/working-set-arithmetic.sh --growth 80GiB --days 7 --rps 20-40
  [--tokens N] | --samples <paste|->` — the container-memory lane: bytes/request
  and bytes/decode-step from the slope, or the rss-vs-cache split (#16563).
- `scripts/host-ram-check.sh [--total-gib N] <paste|-> [...]` — host RAM vs the
  per-rank RSS sum, with the VIRT and first-sample `%CPU` caveats (#3902).
- `scripts/attn-mask-budget.sh --max-model-len N [--ranks K] [--dtype] [--host-gib]
  [--log file]` — the eager `max_model_len²` host attention mask: peak (9·N²) and
  retained (2·N²) bytes per rank, the total across ranks, the share of host RAM,
  and the largest `--max-model-len` that fits (#2873, #3002).
- `scripts/kvpool-segment-check.sh <mooncake.json|serve-log|-> [...]` — the
  `global_segment_size` / `local_buffer_size` gate and the HDK band (#13439).
- `scripts/store-put-code-check.sh <paste|-> [...]` — the store **put** lane:
  maps `hccl_result = N` / `error_codes=[…]` to `hcclResult_t` and Mooncake's
  `ErrorCode` (`4` = generic INTERNAL, `-1` = generic store error, `-200` = out of
  space), reads the master's `PutStart`/`PutEnd`/`PutRevoke` as control-plane vs
  data-plane, and flags the canned `Check memory and store capacity` text (#11454).
- `scripts/ranktable-check.sh <rank_table.json|log> [...] | -` — `server_count` /
  `server_id` / duplicate `device_ip`, i.e. multi-server vs local HCCS (#14506).
- `scripts/hccl-link-pair-audit.sh <log|-> [--visible 0-3]` — the distinct failed
  `createLink` edges of a `Transport_Init_Error(EI0009)` / `DestLink` block, the
  `drvRetCode` / runtime-result codes, and same-card vs cross-card classification from
  the paste's `npu-smi` Bus-Id rows; it refuses to read equal address text as a duplicate
  `device_ip` (#4148).
- `scripts/plog-timeline.sh [--since YYYY-MM-DD] <log|-> [...]` — per-error-family
  timeline and the first in-window error of a plog (#9183); an older record it
  lists is a previous run **or** this process's own `init engine (...)` window, so
  read it against the startup lines before discarding it (#10700).
- `scripts/executor-timeout-check.sh <log|->` — vLLM's RPC/ringbuffer watchdog vs
  a real collective timeout; tolerates screenshot-OCR garble (#1865).
- `scripts/device-type-check.sh <paste|serve-log|-> | --soc-version N` — which side
  of the device-type assert is the card and which is the wheel (#12141).
- `scripts/cuda-only-path-audit.sh <vllm-root> [subpath] [vllm-ascend-root]` — the
  unguarded `torch.cuda` lines and whether the downstream tree replaces the class
  (#4309).
- `scripts/batch-split-audit.sh <vllm-ascend-root>` — every site that slices a
  request-indexed tensor by a token count (#17991).
- `scripts/lora-graph-descriptor-audit.sh <vllm-ascend-root> [vllm-root]` — the
  LoRA lane in one call: the punica binding (`torch_ops` vs the AscendC
  `lora_ops`) with its SoC / `max_lora_rank` predicate, every `has_lora`
  producer split into live-batch vs config-constant, whether the capture loop's
  per-case `activate_lora` is ever consumed, and upstream's guard / `lora_cases`
  / `cudagraph_specialize_lora` default (#6279, #1812).
- `scripts/pip-pairing-check.sh <vllm-ascend-root> [vllm-root]` — the install-time
  pin arithmetic behind "could not find a version that satisfies" / "unmatched torch
  version": reads both trees' `torch` / `torch-npu` / `torchvision` pins, flags the
  metadata-vs-tree split for `--pre` / `--extra-index-url`, and names each
  contradiction (a torch-npu line off the torch line, a torchvision cap that excludes
  the vllm pin) (#1796, #4737).
- `scripts/abi-symbol-check.sh <paste|->` — the `std::string`-ABI split in a
  bundled wheel (#1144).
- `scripts/image-arch-matrix.sh <vllm-ascend-root> [--soc 310p|a2|a3|a5]` — which
  CPU arch and SoC each published tag was built for, from the tree's CI: the
  `metadata-action` tag suffixes (→ SoC variant, with the bare/generic line
  separated), the `platforms:` / `arch:` matrix and its `push`-only condition, the
  runner legs, every `Dockerfile*` `FROM` base (an arm64 leg needs an arm64 base),
  the wheel-release matrix and the per-SoC `Dockerfile.buildwheel.*` list (#2178).
  It reads intent, not published manifests: the registry question is still
  `docker buildx imagetools inspect`.
- `scripts/checkpoint-key-prefix.py <model-dir|*.safetensors|*.index.json|->` — the
  checkpoint's root tensor-name prefixes from the safetensors header (no device, no
  weight load), plus `lm_head` presence, `config.json` `architectures` and any
  `config_sentence_transformers.json`; exit 1 = root-level names, which cannot load into a
  `…ForCausalLM` class (#5549).
- `scripts/sse-toolcall-audit.sh <raw-sse>` — the streamed `tool_calls` argument
  fragments, their assembly and a per-call verdict (`balanced` / `tail-lost` /
  `quote-gap` / `malformed`) (#17797, #12261).
- `scripts/seq-cls-arch-check.sh <vllm-root> <Arch>[,<Arch> ...] [--list '<frag>']`
  — resolve a `--hf_overrides architectures` string the way
  `ModelRegistry._normalize_arch` does: registered directly, else the first
  registered suffix rewrite, with the model family (text vs multimodal) that
  decides a `TypeError: Invalid type of HuggingFace config` report (#14817).
- `scripts/nz-format-gate-check.sh <paste|-> [vllm-ascend-root]` — which `EZ1001`
  family the paste is (aclnn registry vs dtype × layout), the phase of the first
  hit, and the NZ gate's fp32 policy at the checked-out ref (`is_enable_nz` vs
  `_should_trans_nz`), i.e. whether an fp32 pooling/score head is being cast to
  `FRACTAL_NZ` (#4679).
- `scripts/tool-schema-type-audit.py <request-body.json|->` — per tool argument,
  the declared JSON-Schema type, flagging the arguments a streaming parser will
  emit **unquoted** (`anyOf: [string, null]` and other schemas with no top-level
  `"type": "string"`); run it on the client's own body before reading a parser
  (#12261).

