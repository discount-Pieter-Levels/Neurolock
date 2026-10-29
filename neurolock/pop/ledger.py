"""Ledger interface for proof-of-personhood + LocalLedger stub (no blockchain).

The ledger stores only public, non-biometric things:
  - identity commitments C = H(S) with the human's public keys, signed by
    the enrolment station;
  - agent delegations and revocations, signed by the human.

LedgerInterface is the whole contract. To move to Logos, re-implement these
four methods there; nothing else in the pipeline touches storage.
"""
import hashlib
import json
import time
from abc import ABC, abstractmethod
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def canon(d: dict) -> bytes:
    """Canonical bytes of a record body (what gets signed)."""
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def check_sig(pub_hex: str, sig_hex: str, body: dict) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex)).verify(bytes.fromhex(sig_hex), canon(body))
        return True
    except Exception:
        return False


class LedgerInterface(ABC):
    @abstractmethod
    def register_commitment(self, entry: dict) -> str:
        """entry = {commitment, id_pub, station_pub, ts, station_sig}.
        Must reject a bad station signature or an unknown station."""

    @abstractmethod
    def post_delegation(self, deleg: dict) -> str:
        """deleg = {body: {agent_pub, app_id, scope, expiry, nullifier, id_pub, ts}, sig}.
        Must reject: unregistered id_pub, bad signature, expired, or a second
        ACTIVE delegation with the same nullifier (one agent per human per app)."""

    @abstractmethod
    def revoke(self, rev: dict) -> str:
        """rev = {body: {delegation_id, id_pub, ts}, sig}, signed by the same human."""

    @abstractmethod
    def is_active(self, delegation_id: str, now: float = None) -> bool:
        """Delegation exists, not revoked, not expired."""


class LedgerError(Exception):
    pass


class LocalLedger(LedgerInterface):
    """In-memory, hash-chained append-only log (same pattern as consent.py)."""

    def __init__(self, trusted_stations=()):
        self.chain = []
        self.stations = set(trusted_stations)
        self.ids = {}            # id_pub -> registration record
        self.delegations = {}    # delegation id -> record
        self.revoked = set()

    def _append(self, kind, rec):
        prev = self.chain[-1]["id"] if self.chain else "0" * 64
        r = {"kind": kind, "rec": rec, "prev": prev}
        r["id"] = hashlib.sha256(canon(r)).hexdigest()
        self.chain.append(r)
        return r["id"]

    def register_commitment(self, entry):
        body = {k: v for k, v in entry.items() if k != "station_sig"}
        if entry["station_pub"] not in self.stations:
            raise LedgerError("unknown enrolment station")
        if not check_sig(entry["station_pub"], entry["station_sig"], body):
            raise LedgerError("bad station signature")
        if entry["id_pub"] in self.ids:
            raise LedgerError("identity key already registered")
        self.ids[entry["id_pub"]] = entry
        return self._append("register", entry)

    def post_delegation(self, deleg, now=None):
        b = deleg["body"]
        now = time.time() if now is None else now
        if b["id_pub"] not in self.ids:
            raise LedgerError("signer is not a registered human")
        if not check_sig(b["id_pub"], deleg["sig"], b):
            raise LedgerError("bad delegation signature")
        if b["expiry"] <= now:
            raise LedgerError("delegation already expired")
        for did, d in self.delegations.items():
            if d["body"]["nullifier"] == b["nullifier"] and self.is_active(did, now):
                raise LedgerError("nullifier already has an active delegation (one agent per human per app)")
        did = self._append("delegate", deleg)
        self.delegations[did] = deleg
        return did

    def revoke(self, rev):
        b = rev["body"]
        d = self.delegations.get(b["delegation_id"])
        if d is None:
            raise LedgerError("no such delegation")
        if b["id_pub"] != d["body"]["id_pub"] or not check_sig(b["id_pub"], rev["sig"], b):
            raise LedgerError("only the delegating human can revoke")
        self.revoked.add(b["delegation_id"])
        return self._append("revoke", rev)

    def is_active(self, delegation_id, now=None):
        now = time.time() if now is None else now
        d = self.delegations.get(delegation_id)
        return d is not None and delegation_id not in self.revoked and d["body"]["expiry"] > now

    def verify_chain(self):
        prev = "0" * 64
        for r in self.chain:
            body = {k: r[k] for k in ("kind", "rec", "prev")}
            if r["prev"] != prev or hashlib.sha256(canon(body)).hexdigest() != r["id"]:
                return False
            prev = r["id"]
        return True
