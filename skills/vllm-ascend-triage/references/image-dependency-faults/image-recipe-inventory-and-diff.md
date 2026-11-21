# 「你们的镜像里装了什么依赖」: the image recipe at a pin, and the container-only diff

## When this page applies

A **hand-built** environment (self-installed CANN + conda, or a container the
reporter launched themselves) fails at engine start, and the follow-up is a
*sourcing* question rather than a stack: "0.10.1 是需要安装新的东西吗", "我用相同环境
把 vllm 从 0.10.1 改回 0.9.1 可以正常启动", "请问哪里可以看到你们镜像制作的依赖，我对比下".
Recorded on #2812 (v0.10.1rc1 @ `7e16b4a7cdb1`, Qwen3-8B TP2, CANN 8.2.RC1 + conda;
the reporter's second message carried both halves — the older release starts in the
same env, and they can only work **inside** the started container, no host access).
The deliverable is the **recipe list plus a diff the reporter can run**, not a
launcher and not an inventory of their machine.

## The four artefacts that *are* the dependency set

One call: `bash scripts/image-recipe-inventory.sh <vllm-ascend-root> --soc 910b
--image-tag <tag>` (`--soc a3|310p|openeuler|a3-openeuler` for the other variants).
It prints all of the following at the checkout you point it at. At v0.10.1rc1:

| what the image gets | where it is written |
|---|---|
| CANN **+ NNAL/ATB + kernel packages**, all inherited from the base image | `Dockerfile:18` `FROM quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11` (variants: `Dockerfile.a3:18` `-a3-`, `Dockerfile.310p:18` `-310p-`, `Dockerfile.openEuler:18` `openeuler24.03`) |
| the vllm half of the pair | `Dockerfile:40` `ARG VLLM_TAG=v0.10.1.1` (same line number in every variant) |
| NNAL is **sourced, never installed**, by the recipe | `Dockerfile:51` `source /usr/local/Ascend/nnal/atb/set_env.sh` — no `Ascend-cann-nnal*.run` line exists in any Dockerfile, so the component is already in the base image |
| `devlib` on `LD_LIBRARY_PATH` (for `libascend_hal.so`) | `Dockerfile:48` comment + `:52` append — **no doc reproduces this line**; a pip install on a CANN image that skips it differs from the image by one path |
| torch_npu pin and the index that can serve it | `requirements.txt:25-27` (`--pre`, `--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi`, `torch-npu==2.7.1.dev20250724`) |
| triton deliberately absent | `Dockerfile:44` `python3 -m pip uninstall -y triton` |
| extras | `Dockerfile:57` `modelscope 'ray>=2.47.1' 'protobuf>3.20.0'` |

The hand-built counterpart is the docs' own blocks: component table
`installation.md:12-18`, docker tab "No more extra step if you are using
`vllm-ascend` prebuilt docker image" `:104`, and the manual block
`Ascend-cann-toolkit` `.run --full` `:81-83`, kernels `.run` `:87-89`, NNAL
`:91-93` (needs `--header="Referer: https://www.hiascend.com/"`), then the
**separate** `source /usr/local/Ascend/nnal/atb/set_env.sh` `:95`. Quote the
`.run` and the second `source`; those are the two lines a "安装过程没报错" env is
usually missing (`nnal-atb-missing-library.md`).

## The version-swap control is a control only if the whole row moved

| release | CANN | torch / torch_npu | source at the pin |
|---|---|---|---|
| v0.10.1rc1 | 8.2.RC1 | 2.7.1 / 2.7.1.dev20250724 | `docs/source/community/versioning_policy.md:25` |
| v0.9.1 | 8.2.RC1 | **2.5.1 / 2.5.1.post1** | `versioning_policy.md:28` |

"用相同环境把 vllm 从 0.10.1 改回 0.9.1 就能起" therefore only separates the two
releases if `torch` and `torch_npu` moved with it; a downgrade of vllm-ascend alone
leaves the env on the 2.7.1 row and the working run is **out of matrix**, i.e. not a
control. Ask which of `vllm` / `vllm-ascend` / `torch` / `torch_npu` changed, and
read `pip show` for all four before ranking a mechanism (the #2812 paste had
`vllm [NETWORK_ADDRESS_1]` redacted, so the pair itself was never verified).

## Container-only reporters: drop the host rungs, keep the in-container ones

The driver and firmware are **host-side** and reach a container only through the
`-v` mounts of `installation.md:53-66` (`/usr/local/Ascend/driver/lib64/`,
`driver/version.info`, `/etc/ascend_install.info`, `/usr/local/dcmi`,
`/dev/davinci*`, `npu-smi`). A reporter who cannot touch the host cannot
reinstall or roll back the driver, and `dmesg` on their container is the
container's. So do not put "reinstall the driver", host `dmesg` or host
`npu-smi` in the ladder — replace them with:

```bash
pip list --format=freeze | sort > /tmp/mine.txt          # the pip half
ls -l <ASCEND_HOME>/nnal/atb/set_env.sh; echo $ATB_HOME_PATH
env | grep -Ei 'ascend|atb|nnal' | sort
cat <ASCEND_HOME>/ascend-toolkit/latest/*/ascend_toolkit_install.info | head -5
docker run --rm --entrypoint bash quay.io/ascend/vllm-ascend:<tag> -lc \
  'pip list --format=freeze | sort; ls -l /usr/local/Ascend/nnal/atb/set_env.sh'
diff /tmp/mine.txt <that output> | grep -E '^[<>]' | head -40
```

`<ASCEND_HOME>` for a conda env is commonly `$CONDA_PREFIX/Ascend`, not the doc's
absolute `/usr/local/Ascend` — the same blind spot that renders
`collect_env.py`'s `CANN: None` misleading (`nnal-atb-missing-library.md`,
distractor 1). Tags may need a SoC suffix (`-a3`, `-310p`):
`image-arch-and-tag-availability.md`. In the docs, `quay.io/ascend/cann:|cann_image_tag|`
(`installation.md:52`) is a build-time substitution, not a literal tag — do not
quote it as one.

## Answer shape

1. First line: does the release need a **new component** (yes/no + which one), and
   the one `.run`/`source`/pip line that supplies it.
2. The recipe pointers (`Dockerfile:18,40,44,51,52`, `requirements.txt:25-27`,
   `installation.md:91-95`) so the reporter can diff for themselves — that is the
   literal question.
3. The pair caveat from the table above, with the four-package `pip show` ask.
4. The in-container diff block, and — when the symptom was a lane A silent worker
   exit — the `--tensor-parallel-size 1` re-run that turns the wrapper into a real
   traceback (`../startup-failures/silent-worker-exit.md`), in the same message.

**Recorded case #2812**: the first answer classified lane A, said the paste carried
no root cause and gave the TP=1 ablation (correct); after the follow-up it named
NNAL/ATB from `installation.md:91-95` and `faqs.md` item 6 and offered the
`_register_atb_extensions` grep. The reporter's next message was the recipe
question, which `Dockerfile:18,40,48-52` + `requirements.txt:25-27` answer in one
call — so lead the NNAL answer with those paths and the diff instead of waiting for
it. Recheck conditions: every line number here is v0.10.1rc1 `7e16b4a7cdb1`; later
refs add a SoC gate around ATB registration (`nnal-atb-missing-library.md`) and
move the `installation.md` blocks.
