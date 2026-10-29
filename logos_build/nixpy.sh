#!/bin/bash
# Python whose glibc matches the Nix-built libwallet_ffi (needs GLIBC_2.39; system has 2.35).
source /etc/profile.d/nix.sh
exec nix --extra-experimental-features 'nix-command flakes' --option download-attempts 10 shell \
  --impure --expr 'with import (builtins.getFlake "nixpkgs") {}; python3.withPackages (p: with p; [cryptography numpy scipy opencv4 argon2-cffi galois scikit-learn])' \
  -c python3 "$@"
