"""Neurolock proof-of-personhood: narrated stage demo (static data).

  python run_pop_demo.py          (run evaluate_pop.py once first: it writes
                                   results_pop/pop_params.pkl)

Cast (both are TEST-half virtual people: one FV-USM person + one EEG subject):
  Asha  - enrols, delegates to her agent, verifies, revokes
  Bob   - an impostor who presents his own veins and brain
A few other test people are enrolled first so dedup has a registry to search.

SIMULATED parts: finger-pulse videos and SSVEP responses (synthetic
oscillation added to the person's real resting EEG). Each line says which
layer made the decision.
"""
import argparse
import functools
import json
import pickle
import time

import cv2
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neurolock.liveness import pulse as P_, ssvep as SS
from neurolock.pop import delegation as DL
from neurolock.pop.biometrics import ATTEMPTS, VeinData
from neurolock.pop.enrol import Capture, Rejected, Station, pub_hex
from neurolock.pop.ledger import LedgerError, LocalLedger
from neurolock.pop.verify import verify

ap = argparse.ArgumentParser()
ap.add_argument("--ledger", choices=["local", "logos"], default="local",
                help="local = in-memory LocalLedger; logos = registry program on the LEZ testnet")
ARGS = ap.parse_args()
T0 = time.time()
P = pickle.load(open("results_pop/pop_params.pkl", "rb"))
VD = VeinData()
occ = np.load(__import__("os").path.expanduser("~/.cache/neurolock_pop/occipital_physionet.npz"))
SF = float(occ["sf"]); EIX = {s: k for k, s in enumerate(occ["subjects"].tolist())}
F = P["fingers"]; K, SEG, AMP = P["ssvep"]["K"], P["ssvep"]["seg_s"], P["ssvep"]["amp_sim"]
rng = np.random.default_rng(30)


def say(s=""):
    print(s, flush=True)


def capture(person, session, live=True, replay_of=None, intent="confirm"):
    """What the person presents. session 1 = enrolment visit, 2 = a later visit.
    replay_of=challenge: the 'EEG' is a recording made for that old challenge.
    intent: which consent button the person looks at."""
    v, e = person
    c = Capture()
    c.vein_enrol = {f: VD.images(v, f, session, range(6)) for f in F}
    c.vein_probe = {f: [VD.images(v, f, session, a) for a in ATTEMPTS] for f in F}
    img = cv2.resize(cv2.imread(VD.samples[VD.ix[(v, F[0], 1, 1)]].path, 0), (80, 60))
    c.pulse = (P_.synthetic_finger_video(bpm=72, seed=int(rng.integers(1e9))) if live
               else P_.synthetic_finger_video(still_image=img, seed=int(rng.integers(1e9))), 30)
    which = "enrol" if session == 1 else "later"
    rest = occ[which][EIX[e]]; need = int(K * SEG * SF)

    def respond(ch):            # SIMULATED SSVEP on the person's real resting EEG
        o = rng.integers(0, rest.shape[1] - need)
        target = replay_of or (SS.attended(ch, intent) if "confirm" in ch else ch)
        return SS.simulate_response(rest[:, o:o + need], SF, target, AMP, rng, P["occ_weights"]), SF
    c.ssvep = respond
    c.intent = intent
    return c


def attempt(title, fn):
    say(f"\n> {title}")
    try:
        out = fn()
        say("  RESULT: ACCEPTED")
        return out
    except (Rejected, LedgerError) as ex:
        layer = getattr(ex, "layer", "ledger")
        say(f"  RESULT: REJECTED by [{layer}]: {getattr(ex, 'reason', ex)}")
        return None


VP = P["virtual_test"]
D = P["demo"]
asha, bob = VP[D["asha"]], VP[D["bob"]]
# each on-chain registration is a testnet transaction (~30 s), so fewer people on Logos
others = [p for k, p in enumerate(VP) if k not in (D["asha"], D["bob"])][:8 if ARGS.ledger == "local" else 2]
if ARGS.ledger == "logos":
    from neurolock.pop.logos_ledger import LogosLedger
    ledger_cls = functools.partial(LogosLedger, log=print)
else:
    ledger_cls = LocalLedger
st = Station(P, ledger_cls)
say("=" * 72)
say("NEUROLOCK  proof of personhood  (static data; pulse + SSVEP SIMULATED)")
say(f"({D['note']})")
say("=" * 72)
say(f"Setting up: enrolling {len(others)} other people so dedup has a registry ...")
for o in others:
    ch = SS.make_challenge(K, SEG)
    try:
        st.enrol(capture(o, 1), ch, log=lambda *a: None)
    except Rejected:
        pass
say(f"  registry: {len(st.registry)} people, ledger: {len(st.ledger.chain)} records")

# 1. Asha enrols
ch = SS.make_challenge(K, SEG)
say(f"\n[1] Asha enrols. Flicker challenge: {[round(f, 2) for f in ch['freqs']]} Hz")
res = attempt("Asha: enrolment", lambda: st.enrol(capture(asha, 1), ch, log=say))
if res is None:
    raise SystemExit("Asha could not enrol on this run (see the layer above)")
cred, _ = res

# 2. Asha enrols again (new visit, session-2 veins)
ch = SS.make_challenge(K, SEG)
say("\n[2] Asha comes back and tries to enrol a second identity")
attempt("Asha: second enrolment", lambda: st.enrol(capture(asha, 2), ch, log=say))

# 3. delegation
ch = SS.make_challenge(K, SEG)
say("\n[3] Asha verifies, then delegates an agent for app 'fieldstation-vote'")
v = attempt("Asha: verify (fresh challenge)", lambda: verify(st, cred, capture(asha, 2), ch, log=say))
if v is None:
    raise SystemExit("Asha's verification failed on this run (see the layer above)")
_, S, id_sk = v
agent, agent2 = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
did = attempt("ledger: delegation to agent #1", lambda: st.ledger.post_delegation(
    DL.make_delegation(id_sk, cred["id_pub"], S, pub_hex(agent), "fieldstation-vote", ["vote"])))
attempt("ledger: a SECOND agent for the same app", lambda: st.ledger.post_delegation(
    DL.make_delegation(id_sk, cred["id_pub"], S, pub_hex(agent2), "fieldstation-vote", ["vote"])))

# 4. verification + attacks
ch = SS.make_challenge(K, SEG)
say("\n[4] High-stakes action 'vote yes on proposal 7': screen shows CONFIRM / DENY,")
say("    each flickering at a new random frequency every 1.5 s. Asha looks at CONFIRM.")
attempt("Asha: verify + consent", lambda: verify(st, cred, capture(asha, 2), ch, high_stakes=True,
                                                 action="vote yes on proposal 7", log=say))
ch = SS.make_challenge(K, SEG)
say("  Someone pressures Asha into 'transfer all funds'; she looks at DENY")
attempt("Asha looks at DENY", lambda: verify(st, cred, capture(asha, 2, intent="deny"), ch, high_stakes=True,
                                             action="transfer all funds", log=say))
old = SS.attended(SS.make_consent_challenge("vote yes on proposal 7", K, SEG), "confirm")
say("  attacker replays Asha's earlier CONFIRM brain response for 'transfer all funds'")
capR = capture(asha, 2)
_live = capR.ssvep


def replaying(ch):
    """Live for the liveness check, but for consent it plays back the old CONFIRM recording."""
    if "confirm" not in ch:
        return _live(ch)
    rest = occ["later"][EIX[asha[1]]][:, : int(K * SEG * SF)]
    return SS.simulate_response(rest, SF, old, AMP, rng, P["occ_weights"]), SF


capR.ssvep = replaying
ch = SS.make_challenge(K, SEG)
attempt("replayed consent", lambda: verify(st, cred, capR, ch, high_stakes=True, action="transfer all funds", log=say))
chA, chB = SS.make_challenge(K, SEG), SS.make_challenge(K, SEG)
say(f"  attacker replays Asha's EEG recorded for {[round(f, 2) for f in chA['freqs']]} Hz "
    f"against a new challenge {[round(f, 2) for f in chB['freqs']]} Hz")
attempt("replayed EEG", lambda: verify(st, cred, capture(asha, 2, replay_of=chA), chB, log=say))
ch = SS.make_challenge(K, SEG)
attempt("printed / still vein image", lambda: verify(st, cred, capture(asha, 2, live=False), ch, log=say))

# 5. Bob
ch = SS.make_challenge(K, SEG)
say("\n[5] Bob claims to be Asha, using his own veins and brain")
attempt("Bob as Asha", lambda: verify(st, cred, capture(bob, 2), ch, log=say))

# 6. revoke
say("\n[6] Asha revokes the agent")
if did:
    st.ledger.revoke(DL.make_revocation(id_sk, cred["id_pub"], did, lez_user=cred.get("lez_user")))
    ok, why = DL.agent_action_allowed(st.ledger, did, agent, {"vote": "yes"})
    say(f"  agent tries to vote -> {'ALLOWED' if ok else 'REJECTED by [ledger]'}: {why}")
say(f"\nledger ({ARGS.ledger}) valid: {st.ledger.verify_chain()}   ({time.time() - T0:.0f} s)")
if ARGS.ledger == "logos":
    json.dump({"program_id": st.ledger.cfg["program_id"], "transactions": st.ledger.txlog,
               "total_s": round(time.time() - T0, 1)}, open("results_logos_demo.json", "w"), indent=1)
    say("transactions -> results_logos_demo.json")
say("LIMIT: an attacker who controls the capture software can synthesise a matching")
say("SSVEP and pulse. Stopping that needs trusted sensor hardware (out of scope).")
