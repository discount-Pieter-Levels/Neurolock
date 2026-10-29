#!/usr/bin/env bash
# Build the registry program -> ./neurolock_registry.bin
#   ./build.sh          reproducible build in risc0's Docker image (Docker, or Podman via a `docker` shim)
#   ./build.sh --local  with the local risc0 toolchain (`rzup install rust`; not byte-for-byte reproducible)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [ "${1:-}" = "--local" ]; then
    (cd build-local && cargo run --release -q -- "$HERE")
    exit 0
fi
RISC0_DOCKER_CONTAINER_TAG="${RISC0_DOCKER_CONTAINER_TAG:-r0.1.91.1}" cargo risczero build --manifest-path Cargo.toml
BIN=$(find target -path '*riscv32im-risc0-zkvm-elf/docker/*' -name 'neurolock_registry.bin' | head -1)
[ -n "$BIN" ] || { echo "neurolock_registry.bin not found under target/" >&2; exit 1; }
cp "$BIN" "$HERE/neurolock_registry.bin"
ls -l "$HERE/neurolock_registry.bin"
