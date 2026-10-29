# Logos facts for Neurolock Phase 3 (Step 0)

**Sources.** All read on 2026-10-27:
- hackyguru/logos-tutorials (an index; the code is in the 7 linked repos);
- logos-execution-zone at the commit the testnet runs (`411adc8`, v0.3.0 + r0 link fix);
- logos-delivery-module;
- logos-logoscore-cli.

Local copies are in `~/logos_ref/`.

| # | Question | Answer | Where |
|---|---|---|---|
| 1 | Language and SDK for LEZ programs | **Rust**, compiled for **risc0 zkVM 3.0.5** with crate `lee_core` pinned to `411adc8`. A program is `run_program(plan, apply)`: `plan` checks accounts and signers and emits one *effect* per account; `apply` turns an account's old bytes into new bytes. A panic rejects the whole transaction. Start from **lez-guestbook** (signed, write-once records in PDAs). | `lez-guestbook/guestbook-program/src/main.rs` |
| 2 | Deploy and call | **Deploy:** build the ELF with `cargo risczero build` (Docker), or with the local risc0 toolchain (`build-local`, as in p2p-mind-poker; no Docker). Upload with `wallet_ffi_program_loader_deploy`: 96 KiB segments + a header, and the header's account id is the program address. **Call:** `wallet_ffi_send_generic_public_transaction(accounts, borsh instruction, program, payer)`. **Read:** JSON-RPC `getAccount` on `https://testnet.lez.logos.co`; the program's data sits in `data.shards[program]`. Fees in LGO come from the public testnet faucet the tutorials use. | guestbook core, `verify.sh` |
| 3a | Signatures in programs | **Yes, two ways.** (1) **Native:** every account passed with `needs_sign` must sign the transaction (secp256k1 BIP340 Schnorr), and the program sees `is_authorized`. (2) **In-program Ed25519:** `ed25519-dalek` with risc0's accelerated curve25519 fork (p2p-mind-poker checks players' signatures on chain). **We use (1):** cheaper, and the chain itself rejects a forged signer. | `lee/state_machine/src/signature`, poker `Settle` |
| 3b | Time / blocks | `plan.block_window(from..to)` makes a transaction valid only for blocks in `[from, to)`; this is enforced by the chain. There is also a clock account `/LEZ/ClockProgramAccount/0000001` holding `(block_id, timestamp)` that programs can inspect. | `lee_core::program::Plan`, poker `ClockNear` |
| 3c | ZK proofs and private state | **Built into the protocol.** Private accounts exist: hidden commitments, a nullifier set, and client-side risc0 proofs for private execution (`wallet_ffi_send_generic_private_transaction`, `wallet account new private`). Programs don't change for this. **No Semaphore-style circuit or Groth16 verifier example exists**, so Mode B would mean running the delegation as a *private execution* from a private identity account. That's the roadmap item below, not built here. | LEZ README, `examples/program_deployment` |
| 4 | Python ↔ chain | **ctypes on `libwallet_ffi`** (C ABI, header `lez/wallet-ffi/wallet_ffi.h`, built by `nix build github:logos-blockchain/logos-execution-zone?rev=411adc8…#wallet`). Reads can also be plain HTTP JSON-RPC. | guestbook `tests/run.sh` |
| 4b | Python ↔ Basecamp modules | A **`logosctl` / `logoscore` headless daemon** runs any module. `logosctl call <module> <method> args…` calls a method, and `logosctl watch <module> --event <name> --json` streams events. Inside Basecamp, QML calls core modules with `logos.callModule(...)`. | logos-logoscore-cli `watch_command.cpp` |
| 5 | Messaging (Logos Delivery) | `delivery_module`: `createNode(json)`, `start`, `subscribe(contentTopic)`, `send(contentTopic, payload)`, and the event `messageReceived`. Topic format `/<app>/<version>/<subtopic>/<format>`. The `logos.dev` preset has stale bootstrap peers, so same-machine peers dial each other via `staticNodes` (shared-color). The `logos.test` preset needs an RLN membership funded with LEZ testnet native (~2e8). | logos-shared-color core, delivery `run-node.md` |
| 6 | Basecamp tab template | `ui_qml` module = `Main.qml` + `metadata.json` + `flake.nix` (`logos-module-builder`), built as `nix build .#lgx-portable` and installed via Modules → Install LGX. A `core` C++ Qt plugin exposes `Q_INVOKABLE`s to the UI. | logos-hello-world, logos-todo |

## Decisions
- **Mode A (signatures) only.** The user's on-chain identity is a LEZ public account whose secp256k1 key is
  derived from `S`. The chain enforces its signature. The program computes the nullifier itself as
  `H("neurolock/nullifier/v1" ‖ user_account ‖ app_id)`, so a client cannot submit a nullifier bound to someone else.
  - **Limitation:** all of one person's delegations are linkable through `user_account` (pseudonymous, not anonymous).
- **Mode B roadmap:** register writes an "enrolled" flag into the user's *private* account, and `delegate` runs as a
  private execution, so validators see a valid proof but not which enrolled account delegated. The nullifier PDA stays
  public so that "one active delegation per app" is still enforced. Needs `send_generic_private_transaction` from
  Python and client-side proving (30 s – 4 min per transaction per the LEZ docs).
- **Party window = `block_window`.** Registration is valid only for blocks in `[start, end)` (claimed in the
  instruction, checked against the config account). Delegation expiry also uses `block_window(now..expiry)`.
- **Messaging** goes through `logosctl` (subprocess) when it is installed, otherwise through a local socket with the
  identical JSON format (spec fallback).
- **Basecamp tab:** `ui_qml` + a small `core` plugin that runs `pop_cli.py` as a local process (`QProcess`).

## Blocked on this machine (needs the author's OK, see LOGOS_README "Setup")
- Rust + the risc0 toolchain (`rustup`, `rzup install`): to build and unit-test the program. Docker is absent; Podman is present.
- `nix build …#wallet`: `libwallet_ffi.so` for Python.
- `logosctl` + `delivery_module` (Nix build; the prebuilt Linux binaries need glibc 2.38, this machine has 2.35).

## Deployed program
Not deployed yet. After `python logos/deploy_registry.py`, the program id is written to `logos_config.toml` and
added here.
