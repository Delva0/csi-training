# The shard on disk is not a safetensors file (`HeaderTooLarge` and its siblings)

Scope: engine start dies at the **first weight group** with a Rust-library error
instead of a vLLM or device error — `safetensors_rust.SafetensorError: Error
while deserializing header: HeaderTooLarge` (also `HeaderTooSmall`,
`invalid JSON` / `expected value at line 1 column 1`), printed in the same
second as `Loading safetensors checkpoint shards: 0% … 0/N`, on **every** rank,
with `_pickle.PicklingError: Can't pickle <class
'safetensors_rust.SafetensorError'>` beside it on the non-driver ranks. The
report's title usually blames something else (the TP size, the card, the image
tag). It is none of them: the failure is a property of the file, one state below
configuration.

Verified at vllm-ascend v0.7.3 (`779eebb`) / vllm v0.7.3 (`ed6e907`); the
library constant and the loader path are version-independent, the `path:line`
in a reporter's paste are their own vLLM's.

## Answer shape

1. **Verdict first**: the named directory contains at least one file that is not
   a safetensors container, so nothing was loaded; re-download or re-copy that
   file. Not TP, not the card, not model support, not CANN/driver.
2. **One command for the reporter** (below), or `ls -l` of the directory plus
   `head -c 80` on the first shard.
3. Then the ranked contents of such a file (LFS pointer / partial copy / wrong
   file), each with the same check, and what to send back: `ls -l -h` sizes, how
   the weights were obtained (`hf download`, `git clone`, modelscope, rsync),
   `pip show safetensors`.

## Why the message is the whole diagnosis

`safe_open` is called on each shard by upstream's weight iterator —
`vllm/model_executor/model_loader/weight_utils.py:413` `safetensors_weights_iterator`
→ `:425` `with safe_open(st_file, framework="pt") as f:` at vllm v0.7.3 (the
reporter's 0.8.5 paste shows the same function at `:441`) — and parses the first
8 bytes as a little-endian u64 header length before reading any tensor. A length
of 0, or above the library's `MAX_HEADER_SIZE` (100 MB), is refused with exactly
this message. Consequences to state in the answer:

- The traceback's `vllm_ascend` frames are only the entrance
  (`vllm_ascend/worker/worker.py` `load_model` → `model_runner.py` `get_model`);
  below the loader every frame is upstream `vllm/**` or the Rust module. No
  `vllm_ascend` frame below the loader means no downstream code is involved.
- All ranks fail in the same second, **before** any collective, capture or
  device op — so the TP size is exonerated by construction and a `tp=1` re-run
  is an ablation for the reporter's peace of mind, not a fix.
- The `PicklingError` on ranks 1..N-1 is the executor failing to ship the
  exception object back (the Rust class is not importable in the child), so it
  masks the real line; read the driver rank's raw `SafeTensorError`.

## Ranked contents, cheapest check first

1. **Git-LFS pointer instead of weights** (most common — `git clone` on a host
   without `git-lfs`, or `GIT_LFS_SKIP_SMUDGE=1`). The shard is ~130 B and starts
   `version https://git-lfs.github.com/spec/v1`. Computed, not recalled: those
   first 8 bytes `b"version "` unpack to `2336927755350992246`, far above the
   100 MB cap → `HeaderTooLarge`. Check: `ls -l` (size) + `head -c 80`.
   Fix: `hf download <repo> --local-dir <dir>`, or the project's own ModelScope
   route (`VLLM_USE_MODELSCOPE=True`, `docs/source/quick_start.md:47-50;
   docs/source/tutorials/single_npu_multimodal.md:41` at v0.7.3).
2. **Truncated / garbage-prefixed / overwritten shard** (interrupted `rsync` or
   `cp`, a full disk, a netdisk or browser round-trip, a `dd`, two models copied
   into one directory). Note for the reporter: an interrupted `hf download`
   leaves `<file>.incomplete`, which vLLM's `*.safetensors` glob skips — a
   final-named file with the wrong content therefore points at the *copy* step,
   not at huggingface_hub. Check: compare each file's byte size with the hub
   entry, or `sha256sum` against a second copy.
3. **Directory/index mismatch**: `model.safetensors.index.json`'s `weight_map`
   names shards that are not on disk (renamed, an older conversion, a second
   model's files). The log's `0/N` is N from that map.

## One call

```
uv run --no-project python <skill_root>/scripts/checkpoint-header-scan.py <model dir>
```

It prints, per shard: size, whether the first 8 bytes are an LFS pointer / zeros
/ HTML, the header length, the tensor count and a `BAD` verdict, then the
index-vs-disk comparison, and exit 1 when a shard is not a container. On exit 0
the report's files are *not* the explanation: ask for the traceback's exact
shard path (a stale copy or another directory is still open).

## Ruled out, with the evidence that rules them out

- **TP=4** (the usual title claim): identical failure at TP=1; four ranks failing
  in one second on one shared read is a file, and the reader is rank-local.
- **Model support**: `docs/source/user_guide/supported_models.md:11` (v0.7.3)
  lists Qwen2.5-VL ✅; and the model never got constructed.
- **Image / CANN / driver / torch_npu**: no `call aclnn…`, `ERR00100`, `EZ9999`
  or `507*` text anywhere in the paste.
- **The version string the report disagrees with itself about** (title says image
  `v0.8.4rc2`, the env dump says `vllm-ascend 0.8.5rc1` + `vllm 0.8.5.post1`):
  worth one line for reproducibility, but no version of any of these can change
  a header parse.

## Recorded case: #955 (vllm-ascend image v0.8.4rc2, vllm 0.8.5.post1, Qwen2.5-VL-7B-Instruct, TP=4, 910B2C)

Body: "just start as `vllm serve` and set tp as 4, it shows HeadTooLarge", with
the full `collect_env.py` dump and the four-rank traceback. The snapshot's issue
corpus had **zero** `HeaderTooLarge` hits (lexical search and
`scripts/history-titles.sh`), so there is no product precedent and no vllm-ascend
defect to find — say so instead of widening the search. The delivered answer was
the file verdict + `ls -l`/`head -c 80` + the per-shard scan, with TP, model
support, image and the version mismatch explicitly retired.

Second-order item visible in the same paste, worth one line after the fix: the
command passes no `--max-model-len`, so the engine took the checkpoint's 128000
and warned; the pinned Qwen2.5-VL tutorial pins
`--max-model-len 16384 --max-num-batched-tokens 16384`
(`docs/source/tutorials/single_npu_multimodal.md:168,172` at v0.7.3, with the
`--max_model_len` rationale two lines below — the line numbers move per ref).

## Costs, if the lane is misread

Reading `qwen2_5_vl.py`'s `load_weights`, the registry override, the safetensors
version pin or the CANN pairing buys nothing: the exception is raised inside the
Rust library before the first tensor reaches a model. One scan, then answer.
