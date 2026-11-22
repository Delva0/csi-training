# `Illegal instruction (core dumped)` at `import torch`: the host CPU's ISA baseline

Scope: a `-310p`/other Ascend image whose **first import** dies with SIGILL, before any
engine, worker or device op. The owner is the **aarch64 wheel's instruction-set
baseline against the host CPU core**, not vllm-ascend, not the card, not CANN and not
the driver. Worked case #13936 (reporter-confirmed mechanism).

Reporter's report, verbatim (2026, `-310p` image, Phytium D2000 + 310P1 48 GB):

```text
进入容器后执行vllm就出现Illegal instruction (core dumped),排查到原因是由于import torch就出现这个问题,
torch版本是2.10.post2, 2.9的torch可以正常import, 但运行vllm有其他的问题
```

and the mechanism in the reporter's own later comment on the same issue:

```text
这个torch兼容armv8么，我看了是因为d2000只支持armv8，鲲鹏920和D3000可以支持armv8.2,
而torch.2.10默认使用armv8.2的指令集编译的
```

## Verdict shape

`Illegal instruction` here is a **CPU** fault on the host core, so answer "not this
project, and not the 310P card: the release's `torch` aarch64 build needs ARMv8.2 and
this core is ARMv8.0" and give the pre-bump image line. Two things the paste has
already settled before any source read: nothing in `vllm_ascend` has been imported
(the failure is inside `import torch`) and the reporter's own control is a *lower*
torch line that imports — so the variable is the wheel, not the box's mounts, the
HBM, the model or the driver pairing.

| Host core | ISA | What it means for a stock wheel |
|---|---|---|
| Phytium D2000 / FT-2000-class (FTC663) | ARMv8.0-A | no LSE atomics, no dotprod, no FP16 arithmetic — an ARMv8.2-baseline build dies at the first such instruction |
| Kunpeng 920, Phytium D3000 | ARMv8.2 | the same wheels run |

The table is the reporter's, not a vendor datasheet read at a pin; label it
reporter-confirmed and keep it to the cores that appear in a report.

## One round trip, in this order

1. **Get the offset and the `.so`.** On the host (not in the container):
   `dmesg -T | grep -i 'trap invalid opcode'` → `traps: python3[<pid>] trap invalid
   opcode ip:… sp:… error:0 in <lib>[<addr>]`. The tail names the module that faulted;
   by construction it is a bundled library (`libtorch_cpu.so` and the oneDNN/`libgomp`
   inside it are the usual candidates — inference, not verified) and never a
   vllm-ascend `.so`, because no vllm-ascend module has been imported yet.
2. **Read the host's own feature set.** `grep -m1 Features /proc/cpuinfo` and
   `LD_SHOW_AUXV=1 python3 -c pass`: `atomics` (LSE, v8.1), `asimddp` (dotprod, v8.2),
   `asimdhp` (FP16 arithmetic, v8.2), `sve`. Missing `asimddp`/`asimdhp`/`atomics` on
   an armv8.0 core is the whole answer.
3. **Decode the instruction only if the first two are ambiguous.**
   `gdb --batch -ex run -ex 'x/3i $pc' -ex bt --args python3 -c "import torch"`:
   `ldadd`/`cas` = LSE, `sdot`/`udot` = dotprod, `fmla vN.8h` = FP16 arithmetic,
   `pacia`/`autiasp` = PAuth (v8.3). `python -X faulthandler` adds nothing — SIGILL
   kills the process with no Python frame to dump.
4. **The sibling control, which is the cheapest refutation.** The same image on an
   armv8.2 host (Kunpeng 920, D3000, or x86) runs the same `import torch`. If the
   report does not establish that, ask for
   `python -c "import torch; print(torch.__version__)"` **on the failing box** and for
   the same command on any second host: that one line splits "wheel baseline" from "a
   broken mount/library", and it is the only control this shape needs.

## Version contract: which release line runs on an ARMv8.0 core

Pinned at v0.23.0rc1 (`f4a08bd`, the reported image's release):

| Fact | Anchor |
|---|---|
| The release pairs `torch==2.10.0` / `torch-npu==2.10.0.post2` (with `torchvision==0.25.0`, `torchaudio==2.10.0`, `triton-ascend==3.2.1`) | `requirements.txt:16-20` |
| The release's own dependency bullet says the same | `docs/source/user_guide/release_notes.md:56` |
| The 2.10 bump is **v0.20.2rc1** (2026.06.03): "Upgraded PyTorch and torch-npu to 2.10.0. #9128" | `release_notes.md:316` (section heading `:273`) |
| v0.21.0rc1 is unchanged, i.e. every later line is 2.10 too | `release_notes.md:246` |
| The **2.9** line immediately before it is v0.19.1rc1 (2026.04.30): `torch-npu` 2.9.0.post1+git4c901a4 | `release_notes.md:658` (heading `:458`); v0.18.0 at `:403`/`:337` |
| Both 310P recipes build on an aarch64 CANN base that carries the pair — the pin is the **image**, not a pip line | `Dockerfile.310p:18` and `Dockerfile.310p.openEuler:18` (`quay.io/ascend/cann:9.1.0-beta.1-310p-…`) |

So the rung is **the last `-310p` image tag published before v0.20.2rc1** (the
v0.19.1rc1 line), confirmed to exist for 310P first —
`docker buildx imagetools inspect quay.io/ascend/vllm-ascend:v0.19.1rc1-310p`, or the
tag inventory from `scripts/image-arch-matrix.sh <root> --soc 310p`
(`image-arch-and-tag-availability.md`). If no such tag was built, say so and stop
recommending it: a missing tag is an artefact gap, not a reason to hand-patch torch.

## Never pip-downgrade torch inside the newer image

`torch_npu` is version-locked to `torch`, and the image's pair is what the release
validated. `pip install torch==2.9.0` under `torch-npu==2.10.0.post2` produces exactly
the report's second sentence (`2.9 能 import，但运行 vllm 有其他的问题`): the import
succeeds and the stack fails later. Swap the whole image (or, for a manual build,
install the pair from that tag's `requirements.txt` and rebuild the vllm-ascend csrc
against it — the custom-op build links the same torch).

## What no export can answer, and what to say instead

- **The wheel's `-march`/`-mcpu` is not in any snapshot.** The case's
  `pytorch/pytorch` (v2.12.1) is a partial content export (`SOURCES.json`:
  `files: 1714` of `regular_files: 21642`, `complete_regular_files: false`) and even
  the exported subset carries no aarch64 baseline text — `rg -l
  'armv8|tsv110|asimddp|march=native'` over it returns nothing, and `.ci/pytorch/`
  holds only its README. torch_npu's own tree never picks an aarch64 baseline either:
  `setup.py:665-681` lists only manylinux `aarch64` platform tags, and the one
  `-march`-family branch in `torch_npu/_inductor/cpp_builder.py` is **ppc64le-gated**
  (`:174` `mcpu=native`, beside the riscv lines `:176,178`). Do not quote that line as
  "inductor follows the host on aarch64" (a recorded run did): on aarch64 that
  function adds nothing, so inductor codegen cannot be the SIGILL site and the
  prebuilt `libtorch_cpu.so` is.
- **The tree records no host-CPU baseline at all**: `rg -ni 'phytium|kunpeng|armv8'`
  over `docs/` at v0.23.0rc1 returns a contributor's handle only, and every NPU
  runner is `linux-aarch64-*` A2/A3
  (`.github/workflows/push_build_csrc_cache.yaml:19` "Only ARM64 is needed since NPU
  test runners are all linux-aarch64-*"), i.e. Kunpeng-920-class armv8.2 hardware. So
  the ISA claim is reporter-confirmed, not source-verified — say that, and put the ask
  on the wheel vendor (an armv8-a aarch64 build) or on a from-source `-march=armv8-a`
  rebuild of torch + torch_npu + the vllm-ascend csrc, priced as the expensive last
  rung.

## Sibling shapes this is not

- **`Illegal instruction, which is usually caused by unaligned UUB addresses`** in a
  plog / device dump, usually under load with `507011` and `fftsplus aivector error`:
  the aicore, a device-side alignment fault at runtime, unrelated
  (`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`). Same three words;
  the CPU shape has no `vllm_ascend` frame, no plog and no request.
- A glibc allocator abort at the same `import` (`corrupted size vs. prev_size`,
  `Aborted (core dumped)`) is allocation ownership, not ISA (`native-import-abort.md`);
  the pairing check there is which copy is loaded, here it is the core.
- An `ImportError: … undefined symbol` at import is the C++ ABI root
  (`abi-undefined-symbol.md`); all three are "the image's native code vs this box",
  and the message's own verb picks the page.
- `WARNING: The requested image's platform (linux/amd64) does not match the detected
  host platform (linux/arm64/v8)` is the *architecture* question
  (`image-arch-and-tag-availability.md`). This page needs the arch to be right and the
  ISA level still too low.

## Recorded run (#13936)

`v0.23.0rc1-310p-openeuler` on 飞腾 D2000 + 310P1; `import torch` →
`Illegal instruction (core dumped)`; reporter says `torch` 2.9 imports. The session
reached the right owner in 19 tool calls and shipped a ranked answer (host ISA first,
bundled `.so` second, half-upgraded pair third) with the `dmesg`/`gdb`/`AT_HWCAP`
trio and the "swap the whole image, don't pip-downgrade torch" rung. What it did not
have, and this page adds: the release line that last carried torch 2.9 (v0.19.1rc1,
`release_notes.md:658`), the note that the ppc64le `mcpu=native` line is not aarch64
evidence, and the warning that the `Illegal instruction` hits elsewhere in this skill
are the aicore's own message. The session's own guess list (`LSE` / `dotprod` / `FP16`
/ `PAuth`) was correct but unordered; the reporter's confirmed mechanism is the
**armv8.2 baseline**, so lead with that and keep PAuth as a footnote.
