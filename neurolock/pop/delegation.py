"""Signed agent delegation: 'I, a verified human, let this agent act for me in app X'.

  nullifier = H(S || app_id)
     The same human always gets the same nullifier for the same app, so the
     ledger can refuse a second active delegation for that app. Different apps
     give unrelated nullifiers.
     NOTE: in this stub the delegation is also signed by the human's id_pub,
     so apps can link a person across apps by id_pub. A real deployment should
     replace (id_pub signature + nullifier) by a zero-knowledge proof of
     membership in the registered set (Semaphore-style). Out of scope here.

  Revocation = a record naming the delegation id, signed by the same key.
"""
import time
from .crypto import H
from .ledger import canon


def nullifier(S: bytes, app_id: str) -> str:
    return H(b"nullifier", S, app_id.encode()).hex()


def make_delegation(id_sk, id_pub_hex, S, agent_pub_hex, app_id, scope, ttl_s=3600, now=None):
    now = time.time() if now is None else now
    from .logos_ledger import user_account
    body = {"agent_pub": agent_pub_hex, "app_id": app_id, "scope": scope,
            "expiry": now + ttl_s, "nullifier": nullifier(S, app_id),
            "id_pub": id_pub_hex, "lez_user": user_account(S).hex(), "ts": now}
    return {"body": body, "sig": id_sk.sign(canon(body)).hex()}


def make_revocation(id_sk, id_pub_hex, delegation_id, now=None, lez_user=None):
    body = {"delegation_id": delegation_id, "id_pub": id_pub_hex,
            "ts": time.time() if now is None else now}
    if lez_user:
        body["lez_user"] = lez_user
    return {"body": body, "sig": id_sk.sign(canon(body)).hex()}


def agent_action_allowed(ledger, delegation_id, agent_sk, action: dict, now=None):
    """An app checks: is the delegation active, and did the delegated agent sign this action?"""
    from .ledger import check_sig
    d = ledger.delegations.get(delegation_id)
    if d is None or not ledger.is_active(delegation_id, now):
        return False, "delegation not active (revoked, expired or unknown)"
    sig = agent_sk.sign(canon(action)).hex()
    if not check_sig(d["body"]["agent_pub"], sig, action):
        return False, "action not signed by the delegated agent"
    return True, "ok"
