# DeepSeek-V3.2 (sparse attention / SFA / MLAPO) reports

Scope: reports whose model is `DeepSeek-V3.2`, `-Exp` or `-W8A8` on the v0.11–v0.14 plugin
line (paired vLLM v0.12–v0.14, CANN 8.3–8.5). This model reaches the sparse-attention (SFA)
path — `vllm_ascend/ops/mla.py`, `attention/sfa_v1.py`, `ops/triton/rope.py`,
`models/layers/sfa.py` — so a start-up failure there may be a Triton/toolchain build failure
rather than a kernel bug. Verified by reading tag-pinned `vllm-project/vllm`,
`vllm-project/vllm-ascend` and `Ascend/pytorch` sources plus the recorded threads; nothing was
run on an NPU. Pin the tag before quoting any line. Other DeepSeek-V3.2 material is filed by
symptom rather than by model: a CP/PCP-topology report (TP=8, PCP=2, EP disabled) is in
[context-parallel-moe-paths.md](context-parallel-moe-paths.md), and an MTP/draft-metadata
failure is in [spec-decode-metadata-failures.md](spec-decode-metadata-failures.md).

## 1. Chat interface raises the transformers v4.44 chat-template ValueError

Symptom: the service starts, then the first `/v1/chat/completions` (curl or client) fails with
`ValueError: As of transformers v4.44, default chat template is no longer allowed, so you must
provide a chat template if the tokenizer does not define one.` — a request-time error, not a
start-up error. On a DeepSeek-V3.2 checkpoint that is the **tokenizer mode**, not the
checkpoint, the image or a quantisation fault.

Mechanism (upstream vLLM `v0.13.0`): `vllm/entrypoints/openai/serving_engine.py:1151` branches
`elif isinstance(tokenizer, DeepseekV32Tokenizer): tokenizer.apply_chat_template(...)` and
otherwise falls to `apply_hf_chat_template(...)` at `:1159`, which asks the HF tokenizer for a
template and raises when the checkpoint's `tokenizer_config.json` has none. The mode is chosen
by `--tokenizer-mode`: `TokenizerMode` gained `"deepseek_v32"` and
`vllm/tokenizers/registry.py:33` maps it to `("deepseek_v32", "DeepseekV32Tokenizer")`.
`vllm/tokenizers/deepseek_v32.py:46` (note the underscore — `deepseekv32.py` does not exist at
the tag; locate it with `git ls-tree -r --name-only <tag> | grep -i 'deepseek.*token'`)
implements `apply_chat_template` by generating the prompt
with `encode_messages(...)` (DeepSeek's own encoding; `thinking` kwarg selects chat vs thinking
mode, and historical reasoning is dropped when the last message is a user turn) — it never reads
a checkpoint Jinja template and ignores the `chat_template` kwarg. So the clean workaround is
the flag, not a template file:

```
--tokenizer-mode deepseek_v32 --tool-call-parser deepseek_v32 --enable-auto-tool-choice \
--reasoning-parser deepseek_v3
```

The current upstream recipes page `DeepSeek/DeepSeek-V3_2.md` ("Launching DeepSeek-V3.2")
states the same: the significant chat-template changes "are adapted through
`--tokenizer-mode deepseek_v32`". `--chat-template <file>` or a `chat_template` entry in the checkpoint also
unblocks the HF path, but then the prompt is rendered by whatever template that file holds
rather than by DeepSeek's `encode_messages` (thinking mode and historical-reasoning handling
then come from the template).

Boundaries and the doc trap:

- The `v0.12.0rc1` release-note section (`docs/source/user_guide/release_notes.md`, Known
  Issues, still present in the `v0.14.0rc1` file) says DSv3.2 "doesn't work with chat template ... vLLM
  v0.12.0 doesn't support it. We'll support in the next v0.13.0rc1". A report on 0.13-or-later
  is therefore *not* that gap; do not cite the v0.12 Known Issue to close a 0.13+ report.
- The plugin's own tutorial never passes the flag: `docs/source/tutorials/DeepSeek-V3.2.md`
  serve lines at `v0.14.0rc1` (single node `:144-160`, multi-node) and `v0.13.0rc2` contain no
  `--tokenizer-mode` at all. The same model's test configs do:
  `tests/e2e/nightly/single_node/models/test_deepseek_v3_2_w8a8.py:90`
  (`--reasoning-parser deepseek_v3 --tokenizer_mode deepseek_v32`) and
  `tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml:41,66`. A customer
  who copies the tutorial gets the HF path; the in-repo e2e config is the evidence that the
  model is verified with `deepseek_v32`.
- Thread shape: issue #6345 was closed as a duplicate of #5353 with the maintainer reply
  "specify the startup parameters as shown in the vLLM documentation ... or use the official
  DeepSeek-V3.2 client code" — that *is* the answer; the checkpoint's tokenizer files need not
  be inspected.

## 2. Start-up crash: Triton launcher-stub compile with `aclOpExecutor` undeclared

Signature (image `v0.13.0rc2-a3`, CANN 8.5.0, reporter `torch_npu==2.8.0.post1`):

```
RuntimeError: Failed to compile /tmp/tmp*/launcher_cxx11abi1.cxx, error:
  In file included from .../torch_npu/include/torch_npu/csrc/framework/OpCommand.h:5:
  In file included from .../torch_npu/csrc/framework/OpParamMaker.h:7:
  .../torch_npu/include/torch_npu/csrc/framework/interface/AclOpCompileInterface.h:146:55:
  error: use of undeclared identifier 'aclOpExecutor'; did you mean 'aclopExecute'?
  ... 2 errors generated.
RuntimeError: NPUModelRunner init failed, error is Failed to compile /tmp/.../launcher_cxx11abi1.cxx
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

The Python frames under it (`ops/mla.py:164 mla_forward` -> `attention/sfa_v1.py:771` ->
`indexer_select_pre_process` -> `ops/triton/rope.py:187 rope_forward_triton` ->
`triton/backends/ascend/driver.py:127 make_npu_launcher_stub` -> `utils.py:413 _build_npu_ext`)
only say **which Triton kernel compiled first** during `compile_or_warm_up_model ->
capture_model`; they do not name a faulty kernel. The failing text is the C++ compile.

Mechanism, verified at the wheel source (`Ascend/pytorch`; each tag's `version.txt` gives the
wheel version):

| tag | wheel | `AclOpCompileInterface.h` |
| --- | --- | --- |
| `v7.2.0-pytorch2.8.0` | 2.8.0 | no `aclOpExecutor` mention at all — compiles |
| `v7.2.0.1-pytorch2.8.0` | **2.8.0.post1** | declares `AclDestroyAclOpExecutor(aclOpExecutor *executor);` at `:146` with **no** `typedef`, and its vendored `third_party/acl/inc/acl/acl_op_compiler.h` never declares the type — every compile unit that includes this header fails |
| `v7.3.0-pytorch2.8.0` | **2.8.0.post2** | `typedef struct aclOpExecutor aclOpExecutor;` added at `:6` and the include switched to `<acl/acl_op_compiler.h>` (`42a721d353`, "update aclDestroyAclOpExecutor interface") — fixed |
| `v7.3.1-pytorch2.8.0` | 2.8.0.post3 | same as .post2 |

So the defect is inside the 2.8.0.post1 wheel, not in CANN 8.5.0, triton-ascend or the plugin.
The plugin reached it by force-installing the wheel: at `v0.13.0rc2` every `Dockerfile*`
installs `torch-npu==2.8.0.post1` (`Dockerfile.a3:75`) although `pyproject.toml:21` and
`requirements.txt:31` pin `torch-npu==2.8.0`. The tag's own release notes also recommend
".post1 ... install it by hand from [pypi mirror]" — the release note is the trap here, and the
image is the failing population (from-source installs pinned by `requirements.txt` get 2.8.0 and
do not fail).

Fix / workaround: install `torch-npu==2.8.0` or `>=2.8.0.post2` in the container, or use an
image built after the fix. `369ec5b30` (PR #6287, "[CI] Update pta to 2.8.0.post2", merged
2026-01-28 into `releases/v0.13.0`) deletes the forced install from all six Dockerfiles and
bumps both pins; it is contained in `v0.13.0rc3` and `v0.13.0` and **not** in `v0.13.0rc2`, so
"upgrade inside rc2" does not exist. Thread #6245 is closed with a maintainer redirect to
#6242 (same failure on the `-a3` image with a different model), which is where the PR number is.

Same class, different pair: the later "cann and pta header mixing bulid bugfix" family
(`10e2cc06ff`, 2026-08-28; `aclmdlRITask` undeclared when PTA ships headers from CANN 9.1 and
the build host has 9.0) shows the general rule — a C++ `use of undeclared identifier '<aclType>'`
while compiling against `torch_npu/...` headers is a header-version mismatch to resolve by
aligning torch-npu with the installed CANN, not by reading the Triton kernel.

## 3. Decode-node crash at a request: `rtMemcpy ... capture mode` / `507015`

Signature (`v0.13.0rc2`/`v0.14.0rc1`, A3 multi-node 1P1D, one curl to `/v1/completions`, both
decode nodes die):

```
RuntimeError: npuSynchronizeDevice: ... AclrtSynchronizeDeviceWithTimeout, error code is 507015
[Error]: The aicore execution is abnormal.  EE9999: Inner Error!
EE9999: rtMemcpy execution failed, reason=the current capture mode does not support this operation
```

with the reporter's own frame map `model_runner_v1.py:1905 (_generate_dummy_run_hidden_states) ->
model(...) -> compilation/acl_graph.py:199 (__call__) -> torch.npu.synchronize()`.

Pin the frame by statement, not line: at `v0.14.0rc1`
`def _generate_dummy_run_hidden_states` is at `:1902` with `self.model(...)` at `:1905`; the same
def is at `:2058` (call `:2061`) at `v0.13.0rc2`, so a report naming both versions is consistent
and `:1905` as a *statement* selects the newer tree.

Read `acl_graph.py:199` correctly before theorising: it sits in the **replay** branch, after
`logger.info_once("Replaying aclgraph")`, guarded by
`if self.runtime_mode != CUDAGraphMode.FULL or not forward_context.is_draft_model or not use_eagle:`
(`:198`), with the comment at `:186-192` explaining it orders `update_attn_params` against the
previous replay. A synchronize surfaces a device fault enqueued earlier; the failing statement is
whatever issued the `rtMemcpy` while the stream was in capture mode, and the Python stack does
not identify it.

What stays a candidate: which op enqueued the memcpy. Reachable candidates on this model are the
sparse-attention/indexer path (SFA/MLAPO), the MTP drafter's dummy run, or a host round-trip inside
the captured forward. Label it as a candidate and name the check (worker-side log ordering around
the sync; `--enforce-eager` or `VLLM_ASCEND_ENABLE_MLAPO=0` A/B on the decode node).

Verified configuration delta in the tag's own tutorial: the multi-node 1P1D section
(`A3-752T, 4 node`, `Prefill DP2+TP16 / Decode DP8+TP4`) runs the **prefill** tabs with
`--enforce-eager` and the **decode** tabs with
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,9,12]}'`,
identical at `v0.13.0rc2` (`:290`/`:365` vs `:438`/`:517`) and `v0.14.0rc1` (`:550`/`:626` vs
`:699`/`:776`). So a decode-node capture crash on a line copied from that doc leaves
`--enforce-eager` as the untested but documented-elsewhere workaround — at `v0.14.0rc1`
`worker.py:394-395` calls `capture_model()` only `if not self.model_config.enforce_eager`;
changing `cudagraph_capture_sizes` cannot help because every captured size replays the same
dummy forward through the same op.

Thread state: #6529 carries only boilerplate ("resolved in the latest version") plus the stale
bot (labels `bug`/`stale`/`wait-feedback`); there is no maintainer mechanism and no fix commit,
so the answer must be code-derived and must say so.

## 4. All-in-one `-deepseek-v3.2-exp` image build guide (#3278 class)

The issue body *is* the build guide for
`quay.io/ascend/vllm-ascend:v0.11.0rc0[-a3]-deepseek-v3.2-exp`; the "bug" to triage is a defect
in that procedure.

- At `v0.11.0rc0` `vllm_ascend/ascend_config.py:37-38` sets `use_sfa = self.is_deepseek_sfa` from
  the model config, with no env opt-out, so the SFA custom-ops package the guide installs is
  mandatory for DSv3.2-Exp on that line. `VLLM_ASCEND_MLA_PA` (default 0) and
  `VLLM_ASCEND_ENABLE_MLAPO` (default 0) are the only related switches in `envs.py:126-171`, and
  they do not bypass SFA. The later `additional_config.enable_mlapo` key has its own decode-only /
  W8A8-only / DSA-CP gates: [env-vars-and-additional-config.md](env-vars-and-additional-config.md). `use_sfa` is gone by `v0.13.0rc2` — do not quote it for later tags.
- The guide's artifacts are host-limited: `CANN-custom_ops-sfa-linux.aarch64.run` and
  `custom_ops-1.0-cp311-cp311-linux_aarch64.whl`. A thread tester on A2 reported
  `custom_ops-1.0-cp311-cp311-linux_aarch64.whl is not a supported wheel on this platform` and
  followed up asking for an x86 package. The wheel set is architecture-locked by filename, so
  an A2 host that is x86 rather than aarch64 cannot follow the guide as written; confirm the
  host's `uname -m` and Python tag before reproducing.
- It copies `libopsproto_rt2.0.so` into a hardcoded `/usr/local/Ascend/ascend-toolkit/8.2.RC1/`
  while the same steps export `.../ascend-toolkit/latest/...`; a base image on another CANN line
  makes that path wrong or absent. Check the image's CANN before believing a kernel is missing.
- The page the guide recommends (`tutorials/multi-node_dsv3.2.html`) is not in the tree at
  `v0.11.0rc0` (no `docs/source/tutorials/*dsv3.2*`); readthedocs `latest` tracks later/main
  docs, so cite the tag's own tutorial or say the page postdates the image.
