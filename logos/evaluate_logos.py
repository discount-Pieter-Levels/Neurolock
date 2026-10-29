"""Phase 3 evaluation against the LEZ testnet: 10 virtual people + every chain-side attack.

  python logos/evaluate_logos.py [--people 10]

Needs logos_config.toml (from deploy_registry.py) and the Phase 2 caches.
Writes results_logos.json: tx ids, timings, and accept/reject for each case.
Each transaction takes a block or more on the testnet (~30 s), so this takes ~20-30 min.
"""
import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurolock.liveness import pulse as P_, ssvep as SS                   # noqa: E402
from neurolock.pop import delegation as DL, lez                            # noqa: E402
from neurolock.pop import logos_ledger as LL                               # noqa: E402
from neurolock.pop.biometrics import ATTEMPTS, VeinData                    # noqa: E402
from neurolock.pop.enrol import Capture, Rejected, Station, pub_hex        # noqa: E402
from neurolock.pop.ledger import LedgerError                               # noqa: E402
from neurolock.pop.verify import verify                                    # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--people", type=int, default=10); A = ap.parse_args()
T0 = time.time()
P = pickle.load(open("results_pop/pop_params.pkl", "rb"))
VD = VeinData()
occ = np.load(os.path.expanduser("~/.cache/neurolock_pop/occipital_physionet.npz"))
SF = float(occ["sf"]); EIX = {s: k for k, s in enumerate(occ["subjects"].tolist())}
F, K, SEG, AMP = P["fingers"], P["ssvep"]["K"], P["ssvep"]["seg_s"], P["ssvep"]["amp_sim"]
rng = np.random.default_rng(31)
R = {"cases": [], "people": []}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def case(name, expect_ok, fn):
    t = time.time(); n0 = len(st.ledger.txlog)
    try:
        out = fn(); ok, why = True, "accepted"
    except (LedgerError, Rejected, RuntimeError) as e:
        out, ok, why = None, False, str(e)
    txs = [x["tx"] for x in st.ledger.txlog[n0:]]
    R["cases"].append({"case": name, "expected": "accept" if expect_ok else "reject",
                       "result": "accept" if ok else "reject", "pass": ok == expect_ok,
                       "reason": why, "tx": txs, "seconds": round(time.time() - t, 1)})
    log(f"{'PASS' if ok == expect_ok else 'FAIL'}  {name}: {'accepted' if ok else 'rejected'} ({why[:80]})")
    return out


def capture(person, session):
    v, e = person
    c = Capture()
    c.vein_enrol = {f: VD.images(v, f, session, range(6)) for f in F}
    c.vein_probe = {f: [VD.images(v, f, session, a) for a in ATTEMPTS] for f in F}
    c.pulse = (P_.synthetic_finger_video(bpm=70, seed=int(rng.integers(1e9))), 30)
    rest = occ["enrol" if session == 1 else "later"][EIX[e]]; need = int(K * SEG * SF)

    def respond(ch):
        o = rng.integers(0, rest.shape[1] - need)
        tgt = SS.attended(ch, "confirm") if "confirm" in ch else ch
        return SS.simulate_response(rest[:, o:o + need], SF, tgt, AMP, rng, P["occ_weights"]), SF
    c.ssvep = respond
    return c


st = Station(P, lambda trusted_stations: LL.LogosLedger(log=log))
L = st.ledger
R["program_id"] = L.cfg["program_id"]
R["party_window"] = L.party_window()
log("program", R["program_id"], "party window", R["party_window"], "block", L.w.block_height())

# ---------------------------------------------------------------- 10 people end to end
people = [p for p in P["virtual_test"]][: A.people]
creds = {}
for k, person in enumerate(people):
    ch = SS.make_challenge(K, SEG); t = time.time()
    try:
        creds[k], _ = st.enrol(capture(person, 1), ch, log=lambda *a: None)
        res = "enrolled"
    except (Rejected, LedgerError) as e:
        res = f"rejected: {e}"
    R["people"].append({"person": k, "enrol": res, "enrol_s": round(time.time() - t, 1)})
    log(f"person {k}: {res}")
for k, cred in creds.items():
    ch = SS.make_challenge(K, SEG); t = time.time()
    try:
        verify(st, cred, capture(people[k], 2), ch, mode="matcher", log=lambda *a: None)
        res = "verified"
    except Rejected as e:
        res = f"rejected by {e.layer}"
    R["people"][k].update(verify=res, verify_s=round(time.time() - t, 1))

# ---------------------------------------------------------------- chain attack table
k0 = next(k for k, p in enumerate(R["people"]) if p.get("verify") == "verified")
_, S0, id_sk = verify(st, creds[k0], capture(people[k0], 2), SS.make_challenge(K, SEG), mode="matcher", log=lambda *a: None)
idp = creds[k0]["id_pub"]; u0 = bytes.fromhex(creds[k0]["lez_user"])
C0 = L._regs[idp]

# duplicate commitment, sent straight to the program (bypassing the client-side pre-check)
other_user = lez.account_id_for_key(lez.secp_key_from_secret(b"someone else", b"x"))
s, e = L.party_window()


def raw_register(c, user, signer, start=s, end=e):
    ins = lez.u8(LL.REGISTER) + c + user + bytes(32) + lez.u64(start) + lez.u64(end)
    acc = [(L.pda(LL.config_seed()), False), (L.pda(LL.commit_seed(c)), False),
           (L.pda(LL.user_seed(user)), False), (signer, True)]
    tx, ok = L._tx("raw-register", acc, ins, lambda: L._read(L.pda(LL.commit_seed(c)))[:32] == user)
    if not ok:
        raise LedgerError("program rejected")
    return tx


case("duplicate commitment C (different user)", False, lambda: raw_register(bytes.fromhex(C0), other_user, L.station))
rogue = L.w.import_key(lez.secp_key_from_secret(os.urandom(32), b"rogue-station"))
case("register signed by an unknown station", False, lambda: raw_register(os.urandom(32), other_user, rogue))
case("register claiming a fake party window", False, lambda: raw_register(os.urandom(32), other_user, L.station, 0, 2**63))
# true out-of-window: the admin closes the party (window moved into the past), a station
# tries to register, then the admin restores the window.
admin = L.w.import_key(open(f"{L.cfg['data_dir']}/admin.key").read().strip())


def set_window(a_, b_):
    acc = [(L.pda(LL.config_seed()), False), (admin, True)]
    L._tx("admin-set-window", acc, lez.u8(LL.SET_WINDOW) + lez.u64(a_) + lez.u64(b_),
          lambda: L.party_window() == (a_, b_))


nowb = L.w.block_height()
set_window(max(0, nowb - 100), max(1, nowb - 50))
case("register outside the party window", False,
     lambda: raw_register(os.urandom(32), other_user, L.station, *L.party_window()))
set_window(s, e)
agent1, agent2 = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
D = lambda agent, app, ttl=3600: DL.make_delegation(id_sk, idp, S0, pub_hex(agent), app, ["vote"], ttl_s=ttl)
did = case("first delegation (fieldstation-vote)", True, lambda: L.post_delegation(D(agent1, "fieldstation-vote")))
case("second active delegation, same app", False, lambda: L.post_delegation(D(agent2, "fieldstation-vote")))
case("delegation for a different app", True, lambda: L.post_delegation(D(agent2, "fieldstation-forum")))
R["is_active_agent_after_delegate"] = L.is_active_agent(pub_hex(agent1), "fieldstation-vote")
mallory = L.w.import_key(lez.secp_key_from_secret(os.urandom(32), b"mallory"))
rev_forged = {"body": {"delegation_id": did, "id_pub": idp, "lez_user": mallory.hex()}}
case("revoke by someone else (forged proof)", False, lambda: L.revoke(rev_forged))
fake = {"body": {**D(agent2, "fieldstation-vote")["body"], "lez_user": mallory.hex()}}
case("delegate from an unenrolled account (forged proof)", False, lambda: L.post_delegation(fake))
case("revoke own delegation", True, lambda: L.revoke(DL.make_revocation(id_sk, idp, did, lez_user=u0.hex())))
R["revoked_delegation_active"] = L.is_active(did)
ok, why = DL.agent_action_allowed(L, did, agent1, {"vote": "yes"})
R["cases"].append({"case": "agent uses the revoked delegation", "expected": "reject",
                   "result": "accept" if ok else "reject", "pass": not ok, "reason": why})
case("delegate again after revoking", True, lambda: L.post_delegation(D(agent2, "fieldstation-vote")))
short = case("short delegation (expires in ~2 blocks)", True, lambda: L.post_delegation(D(agent1, "fieldstation-chat", ttl=1)))
if short:
    end = time.time() + 300
    while L.is_active(short) and time.time() < end:
        time.sleep(10)
    R["expired_delegation_reported_inactive"] = not L.is_active(short)
    log("expired delegation inactive:", R["expired_delegation_reported_inactive"])

# ---------------------------------------------------------------- what is on chain
R["on_chain_payload_audit"] = {
    "register": "C(32) user_account(32) id_pub(32) window(16)  -> commitment slot {user, id_pub}, user slot {C}",
    "delegate": "app_id, agent_pub(32), scope, expiry_block, now_block -> {nullifier, user, agent, app, scope, expiry, active}",
    "biometric_fields": "none: no vein map/code, EEG or signal is ever an argument or a stored field"}
R["transactions"] = L.txlog
tx_s = [t["seconds"] for t in L.txlog if t["included"]]
R["timing_s"] = {"per_included_tx_mean": float(np.mean(tx_s)) if tx_s else None,
                 "per_included_tx_max": float(np.max(tx_s)) if tx_s else None, "total": round(time.time() - T0, 1)}
R["all_cases_pass"] = all(c["pass"] for c in R["cases"])
json.dump(R, open("results_logos.json", "w"), indent=1, default=str)
log("done -> results_logos.json; all cases as expected:", R["all_cases_pass"])
