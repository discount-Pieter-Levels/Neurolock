"""LogosLedger: the LedgerInterface, backed by the Neurolock registry program on LEZ.

Drop-in for LocalLedger (same four methods + the read-only views the pipeline
uses: `ids`, `delegations`, `verify_chain`). What changes is WHO enforces the rules:

  LocalLedger  this Python process checks signatures, windows and nullifiers
  LogosLedger  the LEZ program does, and the chain checks every signer

KEYS (all testnet, in the git-ignored data dir):
  fee account   pays transaction fees (topped up from the public testnet faucet)
  station key   secp256k1 key of this enrolment station (added on chain by the admin)
  user key      secp256k1 key DERIVED FROM THE IDENTITY SECRET S, imported into the
                wallet only after the person's veins have unlocked S (device_unlock).
                In a real deployment the station and each user run separate wallets;
                in the demo they share one process.

ON CHAIN (nothing biometric): commitment C = H(S), the user's LEZ account, the
user's Ed25519 id_pub, delegations {agent key, app, scope, expiry, active}.

A transaction can be INCLUDED and still REJECTED by the program (it pays the fee,
changes nothing). So every write is confirmed by reading the state back.
"""
import math
import time

from . import lez
from .ledger import LedgerError, LedgerInterface

# instruction tags = order of the Rust enum `Instruction`
INIT, ADD_STATION, SET_WINDOW, REGISTER, DELEGATE, REVOKE = range(6)


def load_config(path="logos_config.toml"):
    """Flat TOML (`key = "text"` or `key = number`, # comments): all logos_config.toml uses."""
    cfg = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = (x.strip() for x in line.split("=", 1))
            cfg[k] = v[1:-1] if v[:1] == '"' else (float(v) if "." in v else int(v))
    import os
    return {k: os.path.expanduser(v) if isinstance(v, str) else v for k, v in cfg.items()}


def user_key_hex(S: bytes) -> str:
    """The user's LEZ signing key, derived from the identity secret S."""
    return lez.secp_key_from_secret(S, b"neurolock/lez-user-key/v1")


def user_account(S: bytes) -> bytes:
    return lez.account_id_for_key(user_key_hex(S))


# PDA seeds — must match registry-program/src/main.rs
def _tag(t: bytes): return t.ljust(32, b"\0")
def config_seed(): return _tag(b"neurolock/config")
def commit_seed(c): return lez.sha(b"nl/commit", c)
def user_seed(u): return lez.sha(b"nl/user", u)
def deleg_seed(n): return lez.sha(b"nl/deleg", n)
def agent_seed(a, app): return lez.sha(b"nl/agent", a, app.encode())
def nullifier(u, app): return lez.sha(b"neurolock/nullifier/v1", u, app.encode())


def decode_delegation(b: bytes):
    """Borsh Delegation { nullifier, user, agent, app_id, scope, expiry_block, active }."""
    if not b:
        return None
    i = 96
    out = {"nullifier": b[:32].hex(), "user": b[32:64].hex(), "agent_pub": b[64:96].hex()}
    for k in ("app_id", "scope"):
        n = int.from_bytes(b[i:i + 4], "little"); out[k] = b[i + 4:i + 4 + n].decode(); i += 4 + n
    out["expiry_block"] = int.from_bytes(b[i:i + 8], "little"); out["active"] = bool(b[i + 8])
    return out


class _Ids:
    """`ledger.ids.get(id_pub)` -> {"commitment": hex} if that registration is on chain."""

    def __init__(self, led): self.led = led

    def get(self, id_pub, default=None):
        c = self.led._regs.get(id_pub)
        if c is None:
            return default
        rec = self.led._read(self.led.pda(commit_seed(bytes.fromhex(c))))
        if len(rec) == 64 and rec[32:].hex() == id_pub:
            return {"commitment": c, "id_pub": id_pub, "lez_user": rec[:32].hex()}
        return default

    def __contains__(self, id_pub): return self.get(id_pub) is not None


class _Delegations:
    """`ledger.delegations.get(delegation_id)` -> {"body": {...}} read from chain."""

    def __init__(self, led): self.led = led

    def get(self, did, default=None):
        d = decode_delegation(self.led._read(self.led.pda(deleg_seed(bytes.fromhex(did)))))
        return {"body": d} if d else default


class LogosLedger(LedgerInterface):
    def __init__(self, trusted_stations=(), config="logos_config.toml", log=None):
        """`trusted_stations` (Ed25519 keys) is ignored: on Logos the approved
        stations are the on-chain list the admin maintains."""
        cfg = load_config(config)
        self.cfg = cfg
        self.w = lez.Wallet(cfg["wallet_ffi_lib"], cfg["data_dir"], cfg.get("sequencer", lez.SEQUENCER))
        self.program = lez.b58d(cfg["program_id"])
        self.fee = bytes.fromhex(open(f"{cfg['data_dir']}/fee_account").read().strip())
        self.station = self.w.import_key(open(cfg["station_key_file"]).read().strip())
        self.block_s = float(cfg.get("block_seconds", 30))
        self.txlog = []                      # [{op, tx, seconds, accepted}]
        self._regs = {}                      # id_pub -> commitment hex (registrations made here)
        self.ids, self.delegations = _Ids(self), _Delegations(self)
        self.log = log or (lambda *a: None)
        self.w.ensure_gas(self.fee)

    # ------------------------------------------------------------ helpers
    def pda(self, seed): return lez.pda(self.program, seed)

    def _read(self, acct: bytes) -> bytes:
        return lez.read_shard(lez.b58e(acct), lez.b58e(self.program), self.w.sequencer)

    def party_window(self):
        b = self._read(self.pda(config_seed()))
        n = int.from_bytes(b[32:36], "little")
        o = 36 + 32 * n
        return int.from_bytes(b[o:o + 8], "little"), int.from_bytes(b[o + 8:o + 16], "little")

    def _tx(self, op, accounts, instruction, confirm):
        """Send, wait for a block, then check the program really changed state."""
        t = time.time()
        self.w.ensure_gas(self.fee)
        tx, sent = self.w.send(self.program, accounts, instruction, self.fee)
        included = sent and bool(tx) and self.w.wait(tx, timeout_s=self.cfg.get("confirm_timeout_s", 240))
        ok = included and confirm()
        self.txlog.append({"op": op, "tx": tx, "seconds": round(time.time() - t, 1),
                           "sequencer_accepted": sent, "included": bool(included), "state_changed": bool(ok)})
        self.log(f"  [chain]  {op}: tx {tx[:16]}..  {'OK' if ok else 'REJECTED'} ({time.time() - t:.0f} s)")
        return tx, ok

    def device_unlock(self, S: bytes):
        """Called on the user's device after the veins have recovered S: the
        LEZ key derived from S becomes available for signing delegations."""
        return self.w.import_key(user_key_hex(S))

    # ------------------------------------------------------------ LedgerInterface
    def register_commitment(self, entry):
        """entry: {commitment, id_pub, lez_user, ...}. The station's LEZ account signs."""
        c, u, idp = bytes.fromhex(entry["commitment"]), bytes.fromhex(entry["lez_user"]), bytes.fromhex(entry["id_pub"])
        start, end = self.party_window()
        ins = lez.u8(REGISTER) + c + u + idp + lez.u64(start) + lez.u64(end)
        acc = [(self.pda(config_seed()), False), (self.pda(commit_seed(c)), False),
               (self.pda(user_seed(u)), False), (self.station, True)]

        def landed():
            rec = self._read(self.pda(commit_seed(c)))
            return len(rec) == 64 and rec[:32] == u and rec[32:] == idp
        if landed():
            raise LedgerError("commitment already registered")
        tx, ok = self._tx("register", acc, ins, landed)
        if not ok:
            raise LedgerError(f"registry program rejected the registration (tx {tx})")
        self._regs[entry["id_pub"]] = entry["commitment"]
        return tx

    def post_delegation(self, deleg, now=None):
        """deleg body: {agent_pub, app_id, scope, expiry (unix s), lez_user}. Signed on chain by the user's LEZ key."""
        b = deleg["body"]
        u = bytes.fromhex(b["lez_user"]); agent = bytes.fromhex(b["agent_pub"]); app = b["app_id"]
        now_block = self.w.block_height()
        ttl = max(1.0, b["expiry"] - (time.time() if now is None else now))
        expiry_block = now_block + max(2, math.ceil(ttl / self.block_s))
        scope = ",".join(b["scope"]) if isinstance(b["scope"], (list, tuple)) else str(b["scope"])
        ins = (lez.u8(DELEGATE) + lez.bstr(app) + agent + lez.bstr(scope)
               + lez.u64(expiry_block) + lez.u64(now_block))
        n = nullifier(u, app)
        acc = [(self.pda(user_seed(u)), False), (self.pda(deleg_seed(n)), False),
               (self.pda(agent_seed(agent, app)), False), (u, True)]

        def landed():
            d = decode_delegation(self._read(self.pda(deleg_seed(n))))
            return bool(d and d["agent_pub"] == agent.hex() and d["active"] and d["expiry_block"] == expiry_block)
        tx, ok = self._tx("delegate", acc, ins, landed)
        if not ok:
            raise LedgerError("registry program rejected the delegation (one active agent per human per app, "
                              "or signer not enrolled)")
        return n.hex()

    def revoke(self, rev):
        b = rev["body"]
        d = decode_delegation(self._read(self.pda(deleg_seed(bytes.fromhex(b["delegation_id"])))))
        if d is None:
            raise LedgerError("no such delegation")
        u = bytes.fromhex(b.get("lez_user", d["user"]))
        ins = lez.u8(REVOKE) + lez.bstr(d["app_id"])
        acc = [(self.pda(deleg_seed(bytes.fromhex(b["delegation_id"]))), False), (u, True)]

        def landed():
            x = decode_delegation(self._read(self.pda(deleg_seed(bytes.fromhex(b["delegation_id"])))))
            return bool(x and not x["active"])
        tx, ok = self._tx("revoke", acc, ins, landed)
        if not ok:
            raise LedgerError("only the delegating human can revoke")
        return tx

    def is_active(self, delegation_id, now=None):
        d = decode_delegation(self._read(self.pda(deleg_seed(bytes.fromhex(delegation_id)))))
        return bool(d and d["active"] and self.w.block_height() < d["expiry_block"])

    def is_active_agent(self, agent_pub_hex, app_id):
        """Spec Step 1.4: is_active(agent, app) -> (bool, scope, expiry_block)."""
        n = self._read(self.pda(agent_seed(bytes.fromhex(agent_pub_hex), app_id)))
        if len(n) != 32:
            return False, None, None
        d = decode_delegation(self._read(self.pda(deleg_seed(n))))
        ok = bool(d and d["agent_pub"] == agent_pub_hex and d["active"] and self.w.block_height() < d["expiry_block"])
        return ok, (d or {}).get("scope"), (d or {}).get("expiry_block")

    def verify_chain(self):
        """Every write this client made was included in a block."""
        return all(t["included"] for t in self.txlog if t["state_changed"])
