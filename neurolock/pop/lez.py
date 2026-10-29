"""Minimal Python access to the Logos Execution Zone (LEZ) v0.3 testnet.

Three pieces, each small enough to read on stage:

1. READ   plain JSON-RPC over HTTPS to the sequencer: getAccount -> the bytes a
          program stored in an account ("shard"). No library needed.
2. DERIVE the same hashes the chain and our program use:
            public account id = sha256("/LEE/v0.3/AccountId/Public/"+pad ‖ x-only pubkey)
            program PDA       = sha256("/LEE/v0.2/AccountId/PDA/"+pad ‖ program ‖ seed)
          LEZ account keys are secp256k1 (BIP340 Schnorr, x-only public key).
3. WRITE  ctypes bindings to libwallet_ffi (the LEZ wallet's C library, built
          with `nix build github:logos-blockchain/logos-execution-zone?rev=411adc8…#wallet`).
          The wallet holds keys, signs, pays fees and submits transactions.

Testnet only. Keys live in a git-ignored data dir.
"""
import ctypes as C
import hashlib
import json
import time
import urllib.request

from cryptography.hazmat.primitives.asymmetric import ec

SEQUENCER = "https://testnet.lez.logos.co"
# Public testnet faucet used by the Logos tutorials (lez-guestbook). LGO has no value.
FAUCET_KEY_HEX = "717940b1cc55e5d6b2066dbf1d9a3f26f212f4db08d02388177fcfedd8a9be1b"
FAUCET_B58 = "7wHg9sbJwc6h3NP1S9bekfAzB8CHifEcxKswCKUt3YQo"
LGO = 1_000_000_000
SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# ============================================================ 1. read
def rpc(method, params, url=SEQUENCER, timeout=20):
    req = urllib.request.Request(url, json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                                                  "params": params}).encode(),
                                 {"content-type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    if "error" in out:
        raise RuntimeError(f"{method}: {out['error']}")
    return out["result"]


def read_shard(account_b58, program_b58, url=SEQUENCER):
    """Bytes `program` stored in `account` (empty if nothing)."""
    return bytes(rpc("getAccount", [account_b58], url)["data"]["shards"].get(program_b58, []))


# ============================================================ 2. derive
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58e(b: bytes) -> str:
    n = int.from_bytes(b, "big"); s = ""
    while n:
        n, r = divmod(n, 58); s = B58[r] + s
    return "1" * (len(b) - len(b.lstrip(b"\0"))) + s


def b58d(s: str) -> bytes:
    n = 0
    for c in s:
        n = n * 58 + B58.index(c)
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return (b"\0" * (len(s) - len(s.lstrip("1"))) + raw).rjust(32, b"\0")


def sha(*parts: bytes) -> bytes:
    return hashlib.sha256(b"".join(parts)).digest()


PUBLIC_PREFIX = b"/LEE/v0.3/AccountId/Public/" + b"\0" * 5
PDA_PREFIX = b"/LEE/v0.2/AccountId/PDA/" + b"\0" * 8


def pda(program: bytes, seed: bytes) -> bytes:
    assert len(seed) == 32
    return sha(PDA_PREFIX, program, seed)


def secp_key_from_secret(secret: bytes, domain: bytes) -> str:
    """Deterministic secp256k1 private key (hex) from a secret, e.g. the identity secret S."""
    k = int.from_bytes(sha(domain, secret), "big") % (SECP_N - 1) + 1
    return k.to_bytes(32, "big").hex()


def account_id_for_key(priv_hex: str) -> bytes:
    """Public account id of a secp256k1 private key (x-only public key, as LEE does)."""
    pk = ec.derive_private_key(int(priv_hex, 16), ec.SECP256K1()).public_key()
    x = pk.public_numbers().x.to_bytes(32, "big")
    return sha(PUBLIC_PREFIX, x)


# ============================================================ 3. write (libwallet_ffi)
class B32(C.Structure):
    _fields_ = [("data", C.c_uint8 * 32)]

    @classmethod
    def of(cls, b: bytes):
        x = cls(); C.memmove(x.data, b, 32); return x

    def bytes(self):
        return bytes(self.data)


class Identity(C.Structure):          # FfiAccountIdentity
    _fields_ = [("kind", C.c_int), ("account_id", B32), ("key_path", C.c_char_p),
                ("authority", B32), ("seed", B32), ("ask", B32), ("nsk", B32), ("npk", B32),
                ("vpk", C.POINTER(C.c_uint8)), ("vpk_len", C.c_size_t), ("identifier", B32)]


class Mention(C.Structure):           # FfiAccountMention
    _fields_ = [("identity", Identity), ("program_account_id", B32)]


class TxResult(C.Structure):          # FfiTransactionResult
    _fields_ = [("tx_hash", C.c_char_p), ("success", C.c_bool),
                ("secrets", C.POINTER(B32)), ("secrets_size", C.c_size_t)]


class TransferResult(C.Structure):    # FfiTransferResult
    _fields_ = [("tx_hash", C.c_char_p), ("success", C.c_bool)]


class CreateOut(C.Structure):         # FfiCreateWalletOutput
    _fields_ = [("wallet", C.c_void_p), ("mnemonic", C.c_char_p)]


class Wallet:
    """One LEZ wallet (keys + fee account) in `data_dir`. Nothing of value lives in it."""

    def __init__(self, lib_path, data_dir, sequencer=SEQUENCER):
        import os, secrets
        self.L = L = C.CDLL(lib_path)
        L.wallet_ffi_open.restype = C.c_void_p
        L.wallet_ffi_create_new.restype = CreateOut
        L.wallet_ffi_send_generic_public_transaction.argtypes = [
            C.c_void_p, C.POINTER(Mention), C.c_size_t, C.POINTER(C.c_uint8), C.c_size_t,
            B32, C.POINTER(B32), C.POINTER(TxResult)]
        L.wallet_ffi_resolve_public_account.argtypes = [B32, C.c_bool, C.POINTER(Identity)]
        L.wallet_ffi_poll_transaction_status.argtypes = [C.c_void_p, B32, C.POINTER(C.c_bool)]
        L.wallet_ffi_program_loader_deploy.argtypes = [
            C.c_void_p, C.POINTER(B32), C.POINTER(B32), C.c_size_t, C.POINTER(C.c_uint8), C.c_size_t,
            C.c_bool, C.POINTER(B32), C.POINTER(TxResult)]
        self.sequencer = sequencer
        os.makedirs(f"{data_dir}/wallet", exist_ok=True)
        cfg, sto, sta = (f"{data_dir}/wallet/{n}" for n in ("config.json", "storage.json", "statistics.json"))
        if not os.path.exists(cfg):
            json.dump({"sequencers": [{"sequencer_addr": sequencer}], "seq_poll_timeout": "30s",
                       "seq_tx_poll_max_blocks": 8, "seq_poll_max_retries": 8,
                       "seq_block_poll_max_amount": 100}, open(cfg, "w"), indent=2)
        if os.path.exists(sto):
            self.h = L.wallet_ffi_open(cfg.encode(), sto.encode(), sta.encode())
        else:
            out = L.wallet_ffi_create_new(cfg.encode(), sto.encode(), sta.encode(), secrets.token_hex(24).encode())
            self.h = out.wallet
        if not self.h:
            raise RuntimeError(f"could not open/create a LEZ wallet in {data_dir}")
        self.save()

    def check(self, err, what):
        if err != 0:
            raise RuntimeError(f"wallet_ffi_{what} failed with error code {err}")

    def save(self):
        self.L.wallet_ffi_save(C.c_void_p(self.h))

    def import_key(self, priv_hex: str) -> bytes:
        """Import a secp256k1 key (harmless if already there); returns its account id."""
        self.L.wallet_ffi_import_public_account(C.c_void_p(self.h), priv_hex.encode())
        self.save()
        return account_id_for_key(priv_hex)

    def new_account(self) -> bytes:
        out = B32()
        self.check(self.L.wallet_ffi_create_account_public(C.c_void_p(self.h), C.byref(out)), "create_account_public")
        self.save()
        return out.bytes()

    def block_height(self) -> int:
        h = C.c_uint64()
        self.check(self.L.wallet_ffi_get_current_block_height(C.c_void_p(self.h), C.byref(h)), "get_current_block_height")
        return h.value

    def balance(self, acct: bytes) -> int:
        out = (C.c_uint8 * 16)()
        if self.L.wallet_ffi_get_balance(C.c_void_p(self.h), C.byref(B32.of(acct)), C.c_bool(True), C.byref(out)) != 0:
            return 0
        return int.from_bytes(bytes(out), "little")

    def wait(self, tx_hex: str, timeout_s=240) -> bool:
        """Poll until the transaction is in a block (True) or the timeout passes (False)."""
        h = B32.of(bytes.fromhex(tx_hex)); inc = C.c_bool(False); t = time.time()
        while time.time() - t < timeout_s:
            if self.L.wallet_ffi_poll_transaction_status(C.c_void_p(self.h), h, C.byref(inc)) == 0 and inc.value:
                return True
            time.sleep(3)
        return False

    def ensure_gas(self, acct: bytes, at_least=LGO // 100, top_up=LGO):
        """Top up a fee account from the public testnet faucet if it is low."""
        if self.balance(acct) >= at_least:
            return None
        faucet = self.import_key(FAUCET_KEY_HEX)
        amount = (C.c_uint8 * 16)(*top_up.to_bytes(16, "little"))
        for attempt in range(4):
            res = TransferResult()
            self.L.wallet_ffi_transfer_public(C.c_void_p(self.h), C.byref(B32.of(faucet)), C.byref(B32.of(acct)),
                                              C.byref(amount), C.byref(res))
            tx = res.tx_hash.decode() if res.tx_hash else ""
            if tx:
                self.wait(tx)
            if self.balance(acct) >= at_least:
                return tx
            time.sleep(15 + 10 * attempt)     # shared faucet: nonce collisions, retry
        raise RuntimeError("could not get LGO from the testnet faucet")

    def send(self, program: bytes, accounts, instruction: bytes, payer: bytes):
        """accounts: list of (account_id bytes, needs_sign bool). Returns (tx_hash, sequencer_accepted)."""
        n = len(accounts)
        m = (Mention * n)()
        for i, (acct, sign) in enumerate(accounts):
            self.check(self.L.wallet_ffi_resolve_public_account(B32.of(acct), C.c_bool(sign), C.byref(m[i].identity)),
                       "resolve_public_account")
            m[i].program_account_id = B32.of(program)
        data = (C.c_uint8 * len(instruction)).from_buffer_copy(instruction)
        res = TxResult()
        err = self.L.wallet_ffi_send_generic_public_transaction(
            C.c_void_p(self.h), m, n, data, len(instruction), B32.of(program), C.byref(B32.of(payer)), C.byref(res))
        for i in range(n):
            self.L.wallet_ffi_free_account_identity(C.byref(m[i].identity))
        self.check(err, "send_generic_public_transaction")
        return (res.tx_hash.decode() if res.tx_hash else ""), bool(res.success)

    def deploy(self, elf: bytes, payer: bytes) -> bytes:
        """program_loader: write-once 96 KiB segments + a header; the header id is the program address."""
        seg = 96 * 1024
        header = self.new_account()
        segs = [self.new_account() for _ in range((len(elf) + seg - 1) // seg)]
        arr = (B32 * len(segs))(*[B32.of(s) for s in segs])
        data = (C.c_uint8 * len(elf)).from_buffer_copy(elf)
        res = TxResult()
        self.check(self.L.wallet_ffi_program_loader_deploy(C.c_void_p(self.h), C.byref(B32.of(header)), arr, len(segs),
                                                           data, len(elf), C.c_bool(True), C.byref(B32.of(payer)),
                                                           C.byref(res)), "program_loader_deploy")
        if not res.success:
            raise RuntimeError("deployment rejected")
        return header


# ============================================================ borsh (just what we need)
def u8(v): return bytes([v])
def u64(v): return int(v).to_bytes(8, "little")
def b32(v: bytes): assert len(v) == 32; return v
def bstr(s: str): b = s.encode(); return len(b).to_bytes(4, "little") + b
