# Neurolock on Logos: Phase 3 (LEZ registry + Delivery + Basecamp tab)

**Status: 28 Oct 2026**

**28 Oct update:**
- **Program tests:** `cargo test` passes, 14/14 (Rust from Nix).
- **Guest binary:** built reproducibly in `risczero/risc0-guest-builder:r0.1.91.1` via Podman
  (`logos_build/Dockerfile.guest`). This is needed because risc0-build asks the host `rzup` even in docker mode, and
  the rzup installer is blocked here.
  - Wrapped into `neurolock_registry.bin` with `registry-program/wrap-bin`.
  - Image id `eca89495e3ad49e016201ba15c076329f227feb40315ff6f3b38c91b67cc9015`.
- **Wallet library:** `libwallet_ffi.so` built with Nix. It needs glibc 2.39, so run the Logos scripts with
  `logos_build/nixpy.sh` instead of `../venv/bin/python`.
- **First deploy:** faucet funding and segment 0 were included (block 68452/68453). Segment 1 did not finalize
  because **the testnet stopped producing blocks at 68453** (no new block for 10+ min). Retry when it advances:
  `logos_build/nixpy.sh logos/deploy_registry.py deploy`.


| Part | Status |
|---|---|
| Registry program (Rust) | **Written, not yet built**: needs the Rust/risc0 toolchain on this machine, see Setup |
| Python `LogosLedger` | Written. Address derivations **checked against the live testnet** (guestbook header PDA, faucet account id) |
| Messaging (challenge/reply) | Working over the local-socket fallback (`results_logos_messaging.json`). The `logosctl` transport is written but not yet run |
| Basecamp tab (core + ui) | Written. The Python service behind it is tested with the local ledger. Nix packaging not built yet |
| Testnet deployment, `results_logos.json` | **Pending** the builds below |

Step 0 findings are in `LOGOS_NOTES.md`.

## Architecture: what lives where

```
 USER'S DEVICE (never leaves)                          ENROLMENT STATION / VERIFIER
 ─────────────────────────────                         ─────────────────────────────
 vein images ─► vein maps ─► fuzzy commitment ─► S     vein dedup (matcher, encrypted templates,
 EEG (SSVEP), finger video ─► pass / fail              deleted after the party)
 identity secret S ─► id key (Ed25519)                 station LEZ key (approved on chain)
                   └► LEZ user key (secp256k1)
          │                                                  │
          │ Logos Delivery, topic /neurolock/1/liveness/json │
          │ ◄── {challenge_id, nonce, flicker_sequence, issued_block}
          │ ──► {challenge_id, nonce, result: pass|fail, id_pub, sig}      (no signals)
          ▼                                                  ▼
 ┌──────────────────── LEZ testnet: neurolock_registry program ────────────────────┐
 │ config        admin, stations[], party window (blocks)                          │
 │ commitment C  {user LEZ account, id_pub}   write-once, station-signed,           │
 │               only inside the party window                                      │
 │ user u        C                            write-once                           │
 │ delegation    nullifier = H(u ‖ app) ─► {agent key, app, scope, expiry, active}  │
 │ agent index   H(agent ‖ app) ─► nullifier  (is_active(agent, app) = 2 reads)     │
 └─────────────────────────────────────────────────────────────────────────────────┘
          ▲
 Basecamp tab "Neurolock PoP" ─► neurolock_pop core module ─► 127.0.0.1 ─► pop_service.py
```

**On chain:** only `C = H(S)`, public keys, the nullifier, delegation metadata and transaction signatures.

**Never on chain, in messages or in the tab:** vein maps, vein codes, EEG features or raw signals. The on-chain
fields are listed in `results_logos.json → on_chain_payload_audit`, and the message fields are listed in
`results_logos_messaging.json → wire`.

## How the chain enforces each rule
| Rule | Mechanism |
|---|---|
| Only an approved station registers | The station's LEZ account must sign; the config's `stations` list must contain it |
| Only during the pseudonym party | `plan.block_window(start..end)`: the sequencer drops the transaction outside the window. The claimed window must equal the stored one |
| One registration per C and per user account | Write-once PDAs |
| One active agent per human per app | The program computes `nullifier = H("neurolock/nullifier/v1" ‖ signer ‖ app)` itself; a new delegation is refused while the stored one is active and not expired |
| Only enrolled humans delegate | The signer's user PDA must hold a commitment |
| Revocation | Only the same signer; sets `active = false` |
| Expiry | `expiry_block` is stored; `block_window(now..expiry)` stops late inclusion; `is_active` compares against the current block |

**Proof mode used: A (signatures).**
- The user's LEZ key is derived from `S`. It can only be produced after the veins unlock `S`, through
  `LogosLedger.device_unlock`.
- **Privacy limit:** every delegation of one person names the same user account, so apps can link them. That makes
  it **pseudonymous, not anonymous**. The registration links that account to `C` and the station, but not to a name
  or a biometric.
- **Mode B (roadmap):** LEZ has protocol-level private accounts with client-side ZK proofs. The register step would
  mark the user's *private* account, and the delegate step would run as a private execution. Validators would then
  verify "some enrolled account delegated" without learning which one, while the public nullifier slot still
  enforces one active agent per app. See `LOGOS_NOTES.md`.

## Setup (one time)
These builds fetch and compile outside code. Each needs the author's OK.
```bash
# 1. LEZ wallet C library for Python (Rust build; tens of minutes the first time)
source /etc/profile.d/nix.sh
nix --extra-experimental-features 'nix-command flakes' build -o logos_build/wallet-ffi \
  'github:logos-blockchain/logos-execution-zone?rev=411adc8fdb3f4c3af64354c4fc26d3e4a6a3d3f4#wallet'

# 2. Rust + risc0 toolchain (user-level, ~3 GB) to build and test the program
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
curl -L https://risczero.com/install | bash && ~/.risc0/bin/rzup install rust && ~/.risc0/bin/rzup install cargo-risczero 3.0.5
cd logos/registry-program && cargo test && ./build.sh --local && cd ../..

# 3. (Delivery transport) logoscore daemon + delivery_module, via Nix (prebuilt binaries need glibc 2.38)
nix build 'github:logos-co/logos-logoscore-cli' -o logos_build/logoscore
# then follow logos-delivery-module docs/pages/run-node.md (logos.dev preset + staticNodes for two local nodes)
```

## Run
```bash
python logos/deploy_registry.py deploy              # faucet LGO, deploy, Init(window), AddStation -> logos_config.toml
python logos/deploy_registry.py status
python run_pop_demo.py --ledger logos               # Asha/Bob story on the testnet -> results_logos_demo.json
python logos/evaluate_logos.py --people 10          # 10 people + chain attacks -> results_logos.json (~20-30 min)
python logos/challenge_demo.py                      # liveness messaging, local socket
python logos/challenge_demo.py --logosctl ~/.lc-verifier ~/.lc-device    # over Logos Delivery
python logos/pop_service.py --ledger logos          # then open the "Neurolock PoP" tab in Basecamp
```

**Basecamp tab:** build both modules (`nix build '.#lgx-portable'` in `logos/basecamp/neurolock-core`, then in
`neurolock-ui`) and install the core first, then the UI.
- On this machine, released `.lgx` files need the `linux-amd64-dev` variant workaround in
  `~/.claude/.../logos_basecamp_build.md`.
- Dev mode for the demo: copy the module folders into `~/.local/share/Logos/LogosBasecampDev/`.

## Tests
- **Program:** `cargo test` in `logos/registry-program`. 14 tests: the spec's 10 plus a fake party window, a
  non-admin station add, a forged revoke, and nullifier binding.
  - Signers and block windows are enforced by the chain before `apply` runs.
  - The tests cover the rules on stored state and the window arithmetic.
  - The end-to-end chain checks are in `evaluate_logos.py`.
- **Messaging:** `results_logos_messaging.json` (local socket). Genuine reply accepted; replayed reply, resting
  EEG, forged signature, unregistered person, wrong nonce and late reply all rejected.

## Remaining gaps
1. Nothing has been built or deployed yet (toolchain approval needed). The program has not been compiled, so
   expect small API fixes on the first `cargo test`.
2. Station and user share one wallet process in the demo. In production each device has its own wallet; the
   interface already separates them (`device_unlock`).
3. `logosctl`'s exact byte encoding for `send` / `messageReceived` is inferred from source (`json:[bytes]`). The
   transport decodes lists, base64 and raw text, but it is untested against a live node.
4. Step 5 (Storage backup, agent-to-agent messages) is not started.
5. Testnet resets wipe the program: rerun `deploy_registry.py deploy`.
