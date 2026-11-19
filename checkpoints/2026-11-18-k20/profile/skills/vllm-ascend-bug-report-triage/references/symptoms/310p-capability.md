# 310P hardware (Atlas 300I DUO / 200I Pro / 310P3): what these reports need from the tag's own page

## Scope

Any report whose hardware is 310P — `npu-smi` names `310P3`, the image tag ends `-310p`, the body says
300I DUO / 200I Pro / 300I Pro — where the question is decided by *what that release can do on this
device*, not by model math:

- **A capability question** (`[Usage]`): "when will the 310 series support graph mode / quantization",
  "which env var can make it faster", "how do I join the feature work". The answer is a tag-to-tag
  comparison plus the thread's lever, never a promise. The per-release support statement and its
  first-tag timeline are `../vllm-ascend/hardware-support-by-release.md`.
- **A start that dies during capture** at `capture_end` on a TP>1 run — the hardware event-id pool, not
  a Python bug (the class itself, with signature, mechanism and checklist: `aclgraph-mode-and-capture.md`
  §H).
- **A 310P MoE start that OOMs in the dispatcher** at `_310p/fused_moe/token_dispatcher.py` on a build
  before `v0.21.0rc1` — the platform's Python routing path, with `--enable-expert-parallel` unsupported
  there and rejected from `v0.21.0rc1` on (the rule below).
- **A W8A8SC start that asserts in the vocab loader** (`vocab_parallel_embedding.py:463`) because the
  checkpoint directory is *per-rank sharded* and the serve block's `--load_format` was dropped (the
  loader half: `quantization-load.md` §8).

Not this page: a 310P image whose CANN base carries no kernel for the selected op (`AclOpKernelInit
failed opType`, ATB `LinearOperation CreateOperation failed!`, `507015` + a DFX kernel hash) —
`device-kernel-faults.md` and `graph-capture-and-custom-ops.md` §B/§B2; the 310P dense-mask OOM
(`attention-mask/310p-dense-mask.md`); the `_310p/` tree layout itself (`../vllm-ascend/code-layout.md`).

## Common rules

- **Read the tag's own 310P page before any code, and find it first: the path moves.**
  `docs/source/tutorials/single_node_300i.md` (the 0.9–0.14 line; its note says the 310I series "only
  supports eager mode and the data type is float16") → `docs/source/tutorials/hardwares/310p.md` from
  `v0.15.0rc1` (`7d4833bce`, #6501 restructured the tutorials; `git cat-file -e <tag>:<path>` decides)
  → the `getting_started/` includes at `v0.30.0`. **The page's own wording changes under it, and that
  change is the capability answer:** at `v0.18.0rc1` and `v0.19.1rc1` it still says eager-only/float16
  with no graph-mode note; the event-id warning lands with `00ddacf4e` (#8639, first tag **`v0.18.0`** —
  the 0.18 release branch, *not* its rc, and `v0.19.1rc1` does not carry it) and is present continuously
  from `v0.20.2rc1`. One `git show <tag>:<path>` answers most of these reports: the page carries the
  graph-mode warning, the per-model capture-size lists, `--load_format sharded_state` and the
  `--max-model-len` rule. A page that is absent, or present without the section, is not evidence the
  feature is absent from the code — the tree at that tag is (`git ls-tree --name-only <tag>:vllm_ascend/_310p`).
- **The capture-size lists are stated for the TP they are shown with, and on TP>1 they are too many.**
  `[1,2,4,8,16,32]` (8B, TP1), `[1,2,4,8,16]` (14B, TP1), `[16,32]` (32B, TP4) at `v0.23.0rc1`: the long
  lists are TP=1 blocks, where nothing is shared. #12989 ran the 14B list at **TP2** — half the
  per-device event budget — and the capture died on the list's last rung (`4/5`, size 16). So the first
  lever on a TP>1 310P run is *fewer sizes than the TP1 block shows*, not the model's list verbatim;
  `--enforce-eager` is the reporter-confirmed total off (`aclgraph-mode-and-capture.md` §H has the
  event-id signature and why `acl_graph.py`'s guidance never fires for it). `cudagraph_capture_sizes` and
  `max_cudagraph_capture_size` must agree when both are given:
  `update_cudagraph_capture_sizes` raises `ValueError("customized max_cudagraph_capture_size(=…) should
  be consistent with the max value of cudagraph_capture_sizes(=…)")` (`vllm_ascend/utils.py:647-671` at
  `v0.23.0rc1`); with only the older key set it warns and truncates instead. Label "fewer sizes fixes it"
  `proposed` unless the reporter confirmed it — the maintainer's own line is that the ceiling needs a
  future driver release.
- **`--max-model-len` must be explicit on 310P.** The same page warns that auto detection may OOM
  because `AttentionMaskBuilder310` materialises an `O(max_model_len²)` float16 mask
  (`AscendAttentionMetadataBuilder310` → `AttentionMaskBuilder310`) and the 310P prefill path consumes
  it directly — `--max-model-len 16384` is the documented value. A copied block that omits it is a
  doc-copy defect, not a memory-sizing question (`doc-guide-vs-capability.md`).
- **`--load_format sharded_state` is the gate for an Eco-Tech W8A8SC checkpoint, and its absence is the
  assert.** The `TP<N>/…-tp<N>` directory of `Eco-Tech/Qwen3-8B|14B|32B-w8a8sc-310-vllm` is not one
  checkpoint split into files: it holds **one file per rank** (`model-rank-0-part-0.safetensors` …
  `model-rank-3-part-0.safetensors`), whose `model.embed_tokens.weight` is `[37984, 5120]` — exactly
  `vocab_size / 4` of the directory's own `config.json` (`151936`). Loaded by the ordinary loader, every
  rank reads a weight whose vocab dim is `vocab_size / TP` and vLLM's `VocabParallelEmbedding.weight_loader`
  requires the full count: `assert loaded_weight.shape[output_dim] == self.org_vocab_size`
  (`vllm/model_executor/layers/vocab_parallel_embedding.py:463` at vLLM `v0.21.0`, the tag
  `vllm-ascend v0.21.0rc1` pairs with), raising a bare `AssertionError` on every `Worker_TP*`. The tag's
  own 310P block serves the same directory with `--load_format sharded_state --quantization ascend
  --dtype float16 --max-model-len 20480` (`v0.21.0rc1:docs/source/tutorials/hardwares/310p.md:158-171`,
  the 8B/14B twins at `:114-148`; offline form at `:225-250`; same blocks at `v0.23.0rc1:225-296`), and
  `vllm_ascend/_310p/sharded_state_loader_310p.py` exists from **`v0.16.0rc1`** (`5899438a8`, #6878) —
  so on a build before that tag no flag helps and the answer is the upgrade. The loader-side mechanism,
  the "a TP4 directory must not be served at TP4" trap and the workaround's status (`hypothesis`,
  #11406 open) are `quantization-load.md` §8.
- **MoE on 310P: expert parallelism is not supported, and before `v0.21.0rc1` the routing path was Python.**
  A 310P MoE report (`Qwen3-30B-A3B`-class, `--enable-expert-parallel`) whose engine starts and then dies
  in the first forward at `vllm_ascend/_310p/fused_moe/token_dispatcher.py:113 in moe_init_routing` is the
  platform's own routing copy, not a model or allocator question: at `v0.18.0rc1` it builds dense index
  tensors of `num_tokens × top_k` (`torch.full((num_rows * k,), -1)` then
  `torch.arange(actual_expert_total_num)`, `:112-115`) instead of calling an NPU op, and
  `AscendFusedMoE310.__init__` took the EP branch there (`fused_moe.py:154-160`, building
  `global/local_expert_map` and slicing the expert range per rank). #8248 ran
  `--tensor-parallel-size 4 --enable-expert-parallel` on `v0.18.0rc1-310p-openeuler` and died with
  `RuntimeError: NPU out of memory. Tried to allocate 20.21 GiB (NPU 2; 43.24 GiB total capacity; 39.06 GiB
  already allocated; 2.14 GiB free)` at that line. #9105 (`2ce60baa1`, 2026-06-01, first tag
  **`v0.21.0rc1`**) deletes that function for `torch_npu.npu_moe_init_routing_v2`, converts `w13`/`w2` to NZ
  and **rejects EP outright**: `RuntimeError("Expert Parallel is not supported on 310P. Please remove
  --enable-expert-parallel.")`. Levers: drop `--enable-expert-parallel`, and/or move to `>= v0.21.0rc1`;
  leaving `--gpu-memory-utilization` at 0.95 on a 43 GiB card is a third, since the KV budget had already
  taken 39 GiB before the forward asked for more. The 20.21 GiB figure is **not** explained by
  `max_model_len × top_k` (20000 × 8 int32 = 640 KB), so name the frame and the unsupported path, and state
  the size as an open gap with `torch.npu.memory_summary()` as the probe — do not invent a shape for it.
- **Check the checkpoint with two read-only commands instead of asking the reporter.** The file list and
  the tensor shapes are prompt-free evidence (`quantization-load.md` §3 uses the same recipe):

  ```bash
  curl -s "https://www.modelscope.cn/api/v1/models/Eco-Tech/<model>/repo/files?Revision=master&Recursive=true" \
    | uv run --no-project python -c 'import json,sys; [print(f["Path"], f["Size"]) for f in json.load(sys.stdin)["Data"]["Files"]]'
  # then the header of one shard (first 2 MB is enough; the JSON header is at offset 0)
  curl -sL -r 0-2000000 "https://www.modelscope.cn/api/v1/models/Eco-Tech/<model>/repo?Revision=master&FilePath=TP4/<dir>/model-rank-0-part-0.safetensors"
  ```

  `struct.unpack('<Q', head[:8])` gives the header length, and the JSON after it carries `shape` per
  tensor. `model-rank-N…` names plus `[vocab/N, hidden]` embedding rows are the whole diagnosis.

## Checklist

```bash
R=/workspace/repos/vllm-project/vllm-ascend        # ref-pinned reads only
git -C $R ls-tree -r --name-only <tag>:docs/source | grep -iE '300i|310p'   # which page exists at the ref
git -C $R show <tag>:docs/source/tutorials/hardwares/310p.md | sed -n '1,40p'     # graph-mode note + limits
git -C $R show <tag>:docs/source/tutorials/hardwares/310p.md | grep -n 'load_format\|capture_sizes\|max-model-len'
git -C $R ls-tree --name-only <tag>:vllm_ascend/_310p                             # what the tag ships for 310P
```

## Recheck conditions

The page path, the capture-size lists and the event-id wording are documentation, so they move: re-read
the tag's 310P page for any report at or after `v0.23.0rc1`, and re-check `acl_graph.py`'s marker list —
a later release that adds `207007`/`Insufficient_Event_Resources` to `_STREAM_RESOURCE_ERROR_MARKERS`
changes the reply from "no guidance fires" to "the repo already names the lever"
(`aclgraph-mode-and-capture.md` §H). The `_310p/` timeline itself is
`../vllm-ascend/hardware-support-by-release.md`.
