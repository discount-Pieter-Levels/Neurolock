"""Local service behind the Basecamp "Neurolock PoP" tab.

  python logos/pop_service.py [--ledger local|logos] [--port 47200]

Basecamp's QML tabs are sandboxed (no network, no processes), so the tab calls
our small `neurolock_pop` core module, which forwards to this service on
127.0.0.1. The service runs the Phase 2 pipeline for one demo person ("Asha", a
TEST-half virtual person: FV-USM veins + PhysioNet EEG; pulse/SSVEP SIMULATED).

  GET  /state                 -> {enrolled, verified, delegations, jobs, log}
  POST /enrol | /verify | /consent | /delegate?app=X | /revoke?app=X | /impostor | /replay
       -> {"job": id}; the job runs in the background, every layer's decision goes to the log.

Only 127.0.0.1 is bound. Nothing biometric is ever returned: the log carries
layer decisions and scores, the state carries public keys and tx ids.
"""
import argparse
import json
import os
import pickle
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurolock.liveness import pulse as P_, ssvep as SS                    # noqa: E402
from neurolock.pop import delegation as DL                                  # noqa: E402
from neurolock.pop.biometrics import ATTEMPTS, VeinData                     # noqa: E402
from neurolock.pop.enrol import Capture, Rejected, Station, pub_hex        # noqa: E402
from neurolock.pop.ledger import LedgerError, LocalLedger                  # noqa: E402
from neurolock.pop.verify import verify                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ledger", choices=["local", "logos"], default="local")
ap.add_argument("--port", type=int, default=47200)
A = ap.parse_args()

P = pickle.load(open("results_pop/pop_params.pkl", "rb"))
VD = VeinData()
occ = np.load(os.path.expanduser("~/.cache/neurolock_pop/occipital_physionet.npz"))
SF = float(occ["sf"]); EIX = {s: k for k, s in enumerate(occ["subjects"].tolist())}
F, K, SEG, AMP = P["fingers"], P["ssvep"]["K"], P["ssvep"]["seg_s"], P["ssvep"]["amp_sim"]
rng = np.random.default_rng(7)
ASHA, BOB = P["virtual_test"][P["demo"]["asha"]], P["virtual_test"][P["demo"]["bob"]]

LOG, JOBS, LOCK = [], [], threading.Lock()
S = {"enrolled": False, "verified": False, "cred": None, "secret": None, "id_sk": None,
     "delegations": {}, "agents": {}}


def say(line):
    with LOCK:
        LOG.append({"t": time.strftime("%H:%M:%S"), "line": str(line)})
        del LOG[:-200]


def capture(person, session, live=True, intent="confirm", replay_of=None):
    """What a person presents (SIMULATED pulse + SSVEP on real resting EEG)."""
    v, e = person
    c = Capture()
    c.vein_enrol = {f: VD.images(v, f, session, range(6)) for f in F}
    c.vein_probe = {f: [VD.images(v, f, session, a) for a in ATTEMPTS] for f in F}
    if live:
        c.pulse = (P_.synthetic_finger_video(bpm=72, seed=int(rng.integers(1e9))), 30)
    else:
        import cv2
        img = cv2.resize(cv2.imread(VD.samples[VD.ix[(v, F[0], 1, 1)]].path, 0), (80, 60))
        c.pulse = (P_.synthetic_finger_video(still_image=img, seed=int(rng.integers(1e9))), 30)
    rest = occ["enrol" if session == 1 else "later"][EIX[e]]; need = int(K * SEG * SF)

    def respond(ch):
        o = rng.integers(0, rest.shape[1] - need)
        tgt = replay_of or (SS.attended(ch, intent) if "confirm" in ch else ch)
        return SS.simulate_response(rest[:, o:o + need], SF, tgt, AMP, rng, P["occ_weights"]), SF
    c.ssvep = respond
    return c


if A.ledger == "logos":
    from neurolock.pop.logos_ledger import LogosLedger
    station = Station(P, lambda trusted_stations: LogosLedger(log=say))
else:
    station = Station(P, LocalLedger)


# ---------------------------------------------------------------- actions
def do_enrol(_):
    ch = SS.make_challenge(K, SEG)
    say(f"Enrolment: flicker challenge {[round(f, 2) for f in ch['freqs']]} Hz")
    S["cred"], _ = station.enrol(capture(ASHA, 1), ch, log=say)
    S["enrolled"] = True


def do_verify(high_stakes=False, action=""):
    ch = SS.make_challenge(K, SEG)
    _, S["secret"], S["id_sk"] = verify(station, S["cred"], capture(ASHA, 2), ch, mode="commitment",
                                        high_stakes=high_stakes, action=action, log=say)
    S["verified"] = True


def do_delegate(app):
    if not S["verified"]:
        raise Rejected("app", "verify first (the identity key comes from the vein unlock)")
    agent = Ed25519PrivateKey.generate()
    d = DL.make_delegation(S["id_sk"], S["cred"]["id_pub"], S["secret"], pub_hex(agent), app, ["act"])
    did = station.ledger.post_delegation(d)
    S["delegations"][app] = did; S["agents"][app] = pub_hex(agent)
    say(f"  [ledger] delegation for '{app}' -> agent {pub_hex(agent)[:12]}..")


def do_revoke(app):
    did = S["delegations"].get(app)
    if not did:
        raise LedgerError(f"no delegation for {app}")
    station.ledger.revoke(DL.make_revocation(S["id_sk"], S["cred"]["id_pub"], did, lez_user=S["cred"].get("lez_user")))
    say(f"  [ledger] delegation for '{app}' revoked")


def do_impostor(_):
    say("Bob claims to be Asha with his own veins and brain")
    verify(station, S["cred"], capture(BOB, 2), SS.make_challenge(K, SEG), log=say)


def do_replay(_):
    old = SS.make_challenge(K, SEG)
    say("Attacker replays Asha's EEG recorded for an old challenge")
    verify(station, S["cred"], capture(ASHA, 2, replay_of=old), SS.make_challenge(K, SEG), log=say)


ACTIONS = {"enrol": do_enrol, "verify": lambda a: do_verify(),
           "consent": lambda a: do_verify(True, a or "vote yes on proposal 7"),
           "delegate": lambda a: do_delegate(a or "fieldstation-vote"),
           "revoke": lambda a: do_revoke(a or "fieldstation-vote"),
           "impostor": do_impostor, "replay": do_replay}


def run_job(job):
    try:
        ACTIONS[job["action"]](job["arg"])
        job["status"] = "accepted"
    except (Rejected, LedgerError) as e:
        job["status"] = "rejected"; job["layer"] = getattr(e, "layer", "ledger")
        say(f"  RESULT: REJECTED by [{job['layer']}]: {getattr(e, 'reason', e)}")
    except Exception as e:                       # noqa: BLE001
        job["status"] = "error"; say(f"  ERROR: {e}"); traceback.print_exc()
    else:
        say("  RESULT: ACCEPTED")
    job["finished"] = time.time()


def state():
    dl = {}
    for app, did in S["delegations"].items():
        try:
            dl[app] = {"id": did[:16] + "..", "agent": S["agents"][app][:12] + "..", "active": station.ledger.is_active(did)}
        except Exception as e:               # noqa: BLE001
            dl[app] = {"id": did[:16] + "..", "active": None, "error": str(e)}
    with LOCK:
        return {"ledger": A.ledger, "person": "Asha (test virtual person)", "enrolled": S["enrolled"],
                "verified": S["verified"], "id_pub": (S["cred"] or {}).get("id_pub"),
                "delegations": dl, "jobs": JOBS[-10:], "log": LOG[-60:],
                "program_id": getattr(station.ledger, "cfg", {}).get("program_id")}


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        b = json.dumps(obj, default=str).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(state() if urlparse(self.path).path == "/state" else {"error": "unknown"}, 200)

    def do_POST(self):
        u = urlparse(self.path); action = u.path.strip("/")
        if action not in ACTIONS:
            return self._send({"error": f"unknown action {action}"}, 404)
        if any(j["status"] == "running" for j in JOBS):
            return self._send({"error": "another action is still running"}, 409)
        job = {"id": len(JOBS) + 1, "action": action, "arg": parse_qs(u.query).get("app", [""])[0],
               "status": "running", "started": time.time()}
        JOBS.append(job)
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        self._send({"job": job["id"]})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    say(f"Neurolock PoP service ready (ledger: {A.ledger}). Pulse and SSVEP are SIMULATED.")
    ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
