"""Deploy and administer the Neurolock registry on the LEZ testnet.

  python logos/deploy_registry.py deploy  [--party-blocks 2000]   # new program + config + station
  python logos/deploy_registry.py window  --start N --end M       # admin: move the party window
  python logos/deploy_registry.py status                          # print config read from chain

Keys (testnet only) go to `data_dir` (default ~/.neurolock_logos, outside the repo):
  fee_account        wallet account that pays fees (faucet-funded)
  admin.key          secp256k1 key of the registry admin
  station.key        secp256k1 key of this enrolment station
Writes logos_config.toml (git-ignored) with the program id.
"""
import argparse
import os
import secrets
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurolock.pop import lez                                   # noqa: E402
from neurolock.pop import logos_ledger as LL                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = os.path.join(ROOT, "logos_config.toml")


def key_file(path):
    """Read a hex key, creating a random one the first time (mode 600)."""
    if not os.path.exists(path):
        with open(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600), "w") as f:
            f.write(secrets.token_hex(32))
    return open(path).read().strip()


def write_config(d):
    lines = ["# Neurolock <-> LEZ testnet (git-ignored; testnet keys only)"]
    for k, v in d.items():
        lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
    open(CFG, "w").write("\n".join(lines) + "\n")


def admin_tx(w, program, fee, admin, op, ins):
    acc = [(lez.pda(program, LL.config_seed()), False), (admin, True)]
    t = time.time()
    tx, ok = w.send(program, acc, ins, fee)
    inc = ok and w.wait(tx)
    print(f"  {op}: tx {tx}  included={inc}  ({time.time() - t:.0f} s)")
    return tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["deploy", "window", "status"])
    ap.add_argument("--lib", default=os.environ.get("WALLET_FFI_LIB", os.path.join(ROOT, "logos_build/wallet-ffi/lib/libwallet_ffi.so")))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.neurolock_logos"))
    ap.add_argument("--bin", default=os.path.join(HERE, "registry-program/neurolock_registry.bin"))
    ap.add_argument("--party-blocks", type=int, default=2000)
    ap.add_argument("--start", type=int); ap.add_argument("--end", type=int)
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True)
    w = lez.Wallet(a.lib, a.data_dir)
    fee_path = f"{a.data_dir}/fee_account"
    if not os.path.exists(fee_path):
        open(fee_path, "w").write(w.new_account().hex())
    fee = bytes.fromhex(open(fee_path).read().strip())
    admin = w.import_key(key_file(f"{a.data_dir}/admin.key"))
    station = w.import_key(key_file(f"{a.data_dir}/station.key"))
    print("fee account", lez.b58e(fee), "| admin", lez.b58e(admin), "| station", lez.b58e(station))

    if a.cmd == "deploy":
        print("funding from the testnet faucet ...")
        w.ensure_gas(fee, at_least=2 * lez.LGO, top_up=5 * lez.LGO)
        elf = open(a.bin, "rb").read()
        t = time.time()
        program = w.deploy(elf, fee)
        print(f"deployed program {lez.b58e(program)} ({len(elf)} bytes, {time.time() - t:.0f} s)")
        time.sleep(5)
        now = w.block_height()
        start, end = now, now + a.party_blocks
        admin_tx(w, program, fee, admin, "init", lez.u8(LL.INIT) + lez.u64(start) + lez.u64(end))
        admin_tx(w, program, fee, admin, "add_station", lez.u8(LL.ADD_STATION) + station)
        write_config({"sequencer": lez.SEQUENCER, "program_id": lez.b58e(program),
                      "wallet_ffi_lib": a.lib, "data_dir": a.data_dir,
                      "station_key_file": f"{a.data_dir}/station.key", "block_seconds": 30,
                      "confirm_timeout_s": 240})
        print(f"party window blocks [{start}, {end}); wrote {CFG}")
    else:
        cfg = LL.load_config(CFG)
        program = lez.b58d(cfg["program_id"])
        if a.cmd == "window":
            admin_tx(w, program, fee, admin, "set_window", lez.u8(LL.SET_WINDOW) + lez.u64(a.start) + lez.u64(a.end))
        b = lez.read_shard(lez.b58e(lez.pda(program, LL.config_seed())), cfg["program_id"])
        n = int.from_bytes(b[32:36], "little"); o = 36 + 32 * n
        print({"program": cfg["program_id"], "admin": lez.b58e(b[:32]),
               "stations": [lez.b58e(b[36 + 32 * i:68 + 32 * i]) for i in range(n)],
               "party_window": [int.from_bytes(b[o:o + 8], "little"), int.from_bytes(b[o + 8:o + 16], "little")],
               "current_block": w.block_height()})


if __name__ == "__main__":
    main()
