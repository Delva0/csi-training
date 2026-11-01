"""Query published vLLM Ascend image metadata without downloading filesystem layers.

Inputs are explicit Quay tags or sha256 digests and an OS/CPU platform, not source refs.
Dependencies are Python's standard library, curl and public HTTPS access to quay.io.
Invocation: uv run --no-project python scripts/query.py --output NEW_DIR TAG [TAG ...].
Outputs are JSON comparisons and optional digest-named raw responses; there is no cache.
Source: https://quay.io/v2/ascend/vllm-ascend, using the OCI distribution API.
Validation covers index selection, content digests, config/history extraction and real
public registry queries; installed package/source identity and NPU behavior are untested.
"""

# This is a standalone skill resource, copied and executed independently of any package.
# ruff: noqa: INP001

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REGISTRY = "https://quay.io/v2/ascend/vllm-ascend"
ACCEPT = ("application/vnd.oci.image.index.v1+json, "
          "application/vnd.docker.distribution.manifest.list.v2+json, "
          "application/vnd.oci.image.manifest.v1+json, "
          "application/vnd.docker.distribution.manifest.v2+json")
ARGUMENT = re.compile(r"\b((?:VLLM|TORCH|PYTORCH|TRITON|MOONCAKE|MEMFABRIC|MEMCACHE|FLA)_\w+"
                      r"|BUILD_TYPE|DAILY_DEPS_MODE|SOC_VERSION)=([^\s]*)")


def fetch(path, output, expected=None):
    """Retain exact response bytes and reject changed digest-addressed content."""
    url = f"{REGISTRY}/{path}"
    response = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                               "--connect-timeout", "10", "--max-time", "40", "--proto", "=https",
                               "-H", f"Accept: {ACCEPT}", url], capture_output=True, check=False, timeout=45)
    if response.returncode:
        raise ValueError(f"{url}: curl exit {response.returncode}: {response.stderr.decode().strip()}")
    digest = "sha256:" + hashlib.sha256(response.stdout).hexdigest()
    if output is not None:
        (output / (digest.replace(":", "-") + ".json")).write_bytes(response.stdout)
    if expected is not None and digest != expected:
        raise ValueError(f"{url}: content digest {digest} differs from {expected}")
    return json.loads(response.stdout), digest


def describe(config):
    """Keep declarations and build instructions distinct from installed runtime versions."""
    environment = dict(item.split("=", 1) for item in config.get("config", {}).get("Env", []) if "=" in item)
    arguments, steps, pins = {}, [], []
    for index, entry in enumerate(config.get("history", [])):
        instruction = entry.get("created_by", "")
        prefix, separator, command = instruction.partition(" -c ")
        for key, value in ARGUMENT.findall(prefix):
            arguments.setdefault(key, set()).add(value)
        if separator and (any(token in command for token in ("pip install", "install_daily_deps.sh"))
                          or re.search(r"\bgit(?:\s+-C\s+\S+)?\s+(?:clone|fetch|checkout)\b", command)):
            steps.append({"history_index": index, "command": command})
            pins.extend({"history_index": index, "requirement": requirement}
                        for requirement in re.findall(r"[\w.-]+==[\w.+%-]+(?=$|[\s'\";])", command))
    return {"created": config.get("created"),
            "config_environment": {key: value for key, value in environment.items()
                                   if re.search(r"VERSION|TAG|COMMIT|ASCEND_.*HOME|SOC_VERSION", key)},
            "cann_paths": sorted(set(re.findall(r"/usr/local/Ascend/cann-[\w.+-]+", " ".join(environment.values())))),
            "labels": config.get("config", {}).get("Labels") or {},
            "build_arguments": {key: sorted(values) for key, values in sorted(arguments.items())},
            "recorded_package_pins": pins,
            "build_steps": steps,
            "runtime_versions": {"status": "unknown", "reason": "Image filesystem and runtime were not inspected"}}


def query(reference, platform, output=None):
    """Resolve one exact reference and reject missing or ambiguous platform selection."""
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|sha256:[a-f0-9]{64}", reference):
        raise ValueError("Reference must be a complete Quay tag or sha256 digest")
    expected = reference if reference.startswith("sha256:") else None
    manifest, root_digest = fetch(f"manifests/{reference}", output, expected)
    manifest_digest = root_digest
    if "manifests" in manifest:
        candidates = [entry for entry in manifest["manifests"] if "/".join(
            entry.get("platform", {}).get(key, "") for key in ("os", "architecture", "variant")
        ).rstrip("/") == platform]
        if len(candidates) != 1:
            available = [entry.get("platform", {}) for entry in manifest["manifests"]]
            raise ValueError(f"Expected one {platform} manifest, found {len(candidates)}; available: {available}")
        manifest_digest = candidates[0]["digest"]
        manifest, _ = fetch(f"manifests/{manifest_digest}", output, manifest_digest)
    config_digest = manifest["config"]["digest"]
    config, _ = fetch(f"blobs/{config_digest}", output, config_digest)
    actual = "/".join(config.get(key, "") for key in ("os", "architecture", "variant")).rstrip("/")
    if actual != platform:
        raise ValueError(f"Config platform {actual!r} differs from requested {platform!r}")
    return {"image": f"quay.io/ascend/vllm-ascend@{manifest_digest}", "requested_reference": reference,
            "platform": platform, "root_digest": root_digest, "manifest_digest": manifest_digest,
            "config_digest": config_digest,
            "sources": {"manifest": f"{REGISTRY}/manifests/{manifest_digest}",
                        "config": f"{REGISTRY}/blobs/{config_digest}"}, **describe(config)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", nargs="+",
                        help="Complete tags or sha256 digests; never inferred from source refs")
    parser.add_argument("--platform", default="linux/amd64", help="Exact os/architecture[/variant]")
    parser.add_argument("--output", type=Path,
                        help="Retain raw response bytes and result.json; never overwrite a result")
    args = parser.parse_args()
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "result.json").exists():
            parser.error("Output already contains result.json; choose a new evidence directory")
    started = time.monotonic()
    result = {"observed_at": datetime.now(UTC).isoformat(), "registry": REGISTRY, "images": [], "errors": []}
    with ThreadPoolExecutor(max_workers=min(4, len(args.references))) as workers:
        pending = [(reference, workers.submit(query, reference, args.platform, args.output))
                   for reference in args.references]
        for reference, future in pending:
            try:
                result["images"].append(future.result())
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                result["errors"].append({"reference": reference, "error": str(exc)})
    result["elapsed_seconds"] = time.monotonic() - started
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        (args.output / "result.json").write_text(text)
    print(text, end="")
    return bool(result["errors"])


if __name__ == "__main__":
    sys.exit(main())
