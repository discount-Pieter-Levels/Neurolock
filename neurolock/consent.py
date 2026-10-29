"""Consent ledger + trustees.

Records are signed by the data subject (Ed25519) and hash-chained, so they
can't be forged or silently reordered. LocalLedger is a stand-in with the
same interface you'll implement against the Logos blockchain module.
"""
import hashlib
import json
import time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization


def pub_hex(pub: Ed25519PublicKey):
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


class LocalLedger:
    def __init__(self):
        self.chain = []

    def append(self, record: dict):
        prev = self.chain[-1]["id"] if self.chain else "0" * 64
        record = {**record, "prev": prev}
        record["id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        self.chain.append(record)
        return record["id"]

    def records_for(self, data_id):
        return [r for r in self.chain if r["data_id"] == data_id]


def sign_consent(sk: Ed25519PrivateKey, data_id: str, action: str, purpose: str):
    body = {"data_id": data_id, "action": action, "purpose": purpose,
            "subject": pub_hex(sk.public_key()), "ts": time.time()}
    sig = sk.sign(json.dumps(body, sort_keys=True).encode()).hex()
    return {**body, "sig": sig}


def verify(record):
    body = {k: record[k] for k in ("data_id", "action", "purpose", "subject", "ts")}
    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(record["subject"]))
    try:
        pub.verify(bytes.fromhex(record["sig"]), json.dumps(body, sort_keys=True).encode())
        return True
    except Exception:
        return False


class Trustee:
    """Holds one key share; releases it only if the latest valid record from
    the registered subject is a GRANT for the requested purpose."""

    def __init__(self, name, share, ledger, data_id, subject_pub):
        self.name, self.share, self.ledger = name, share, ledger
        self.data_id, self.subject = data_id, subject_pub

    def release(self, purpose):
        recs = [r for r in self.ledger.records_for(self.data_id)
                if r["subject"] == self.subject and verify(r) and r["purpose"] == purpose]
        if recs and recs[-1]["action"] == "GRANT":
            return self.share
        return None
