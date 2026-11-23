# ruff: noqa: INP001
"""Build the Windows data-only checkpoint metadata from the release's verified Git snapshots."""

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG = "2026-11-22-mainline"


def main():
    directory = Path(sys.argv[1])
    manifest = json.loads((ROOT / "checkpoints" / TAG / "manifest.json").read_bytes())
    with zipfile.ZipFile(directory / "csi-desktop-metadata.zip", "w") as bundle:
        for name, checkpoint in manifest["checkpoints"].items():
            commit, inventory = checkpoint["commit"], checkpoint["inventory_sha256"]
            raw = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":files.json"])
            if hashlib.sha256(raw).hexdigest() != inventory:
                raise ValueError("Changed snapshot inventory: " + name)
            roots = [r["path"] for r in json.loads(raw)
                     if r["path"].count("/") == 1 and r["path"].endswith("/SKILL.md")]
            roots.sort(key=lambda p: (not p.startswith("vllm-ascend"), p))
            body = ("---\nname: csi-vllm-ascend\ndescription: Triage vLLM-Ascend and Ascend NPU deployment, inference,"
                    " and runtime failures using logs, source and the selected CSI checkpoint.\n---\n\n"
                    f"# vLLM-Ascend 故障排查 · {name}\n\n"
                    "先用本机文件读取工具或 PowerShell 打开下面的主入口。路径相对于当前 SKILL.md 所在目录。\n\n"
                    "- 训练工具 `skill_view`、`skills_list` 对应直接读取这些 SKILL.md，"
                    "以及同目录下的 references/、scripts/。"
                    "用本机文件搜索或终端检索。\n"
                    "- 遇到 `/workspace/repos` 或 `SOURCES.json`，先查当前工作区的仓库与实际 git 版本，再选相关源码。"
                    "有 GitHub 连接器或可用 gh 时可查询公开项目。缺少对应代码或现场观测时向用户索取。\n"
                    "- `/workspace/scratch` 对应当前可写任务目录。运行随附脚本前检查解释器、参数和实际输入路径。"
                    "已有 WSL 时可在那里运行 Bash 脚本。\n\n")
            for i, p in enumerate(roots):
                label = "主入口" if i == 0 else "同版本的相关技能"
                body += f"- {label}：`../../../.csi-training/windows/snapshots/{commit}/skills/{p}`\n"
            for filename, value in (("commit.txt", commit+"\n"), ("inventory.txt", inventory+"\n"), ("SKILL.md", body)):
                info = zipfile.ZipInfo(name + "/" + filename, date_time=(2026, 11, 22, 0, 0, 0))
                bundle.writestr(info, value.encode(), compress_type=zipfile.ZIP_DEFLATED)
    template = (ROOT / "installer/Install-CSI.cmd.in").read_text()
    index = hashlib.sha256((ROOT / "checkpoints" / TAG / "manifest.json").read_bytes()).hexdigest()
    template = template.replace("@@INDEX@@", index)
    for key, filename in (("META", "csi-desktop-metadata.zip"), ("AMD64", "csi-install-linux-amd64"),
                          ("ARM64", "csi-install-linux-arm64")):
        template = template.replace("@@" + key + "@@", hashlib.sha256((directory/filename).read_bytes()).hexdigest())
    (directory / "Install-CSI.cmd").write_bytes(template.replace("\n", "\r\n").encode("ascii"))


if __name__ == "__main__":
    main()
