#!/bin/sh
# Build matching native and WSL installers against the checked-out release manifest.
set -eu
cd "$(dirname "$0")"
out=${1:?Usage: build.sh OUTPUT_DIRECTORY [MANIFEST]}
manifest=${2:-../checkpoints/2026-11-22-mainline/manifest.json}
mkdir -p "$out"
out=$(cd "$out" && pwd)
index=$(sha256sum "$manifest" | cut -d' ' -f1)
build() {
    CGO_ENABLED=0 GOOS="$1" GOARCH="$2" go build -trimpath -buildvcs=false \
        -ldflags="-s -w -buildid= -X main.indexSHA=$index ${4-}" -o "$out/$3" .
}
for arch in amd64 arm64; do build linux "$arch" "csi-install-linux-$arch"; done
uv run --no-project python -B metadata.py "$out"
(cd "$out" && sha256sum csi-install-linux-amd64 csi-install-linux-arm64 Install-CSI.cmd csi-desktop-metadata.zip > SHA256SUMS)
