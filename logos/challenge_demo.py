"""Step 3 check: liveness challenge-response over messaging, plus its attacks.

  python logos/challenge_demo.py                      # local socket transport (fallback)
  python logos/challenge_demo.py --logosctl A_DIR B_DIR   # two logosctl daemons with delivery_module

The device side runs the SSVEP check locally (SIMULATED response on a real
person's resting EEG, as in Phase 2) and only sends pass/fail + signature.
Writes results_logos_messaging.json.
"""
import argparse
import json
import os
import pickle
import sys
import threading
import time

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurolock.liveness import challenge_msg as CM, ssvep as SS          # noqa: E402
from neurolock.pop.enrol import pub_hex                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--logosctl", nargs=2, metavar=("VERIFIER_DIR", "DEVICE_DIR"))
a = ap.parse_args()

P = pickle.load(open("results_pop/pop_params.pkl", "rb"))
occ = np.load(os.path.expanduser("~/.cache/neurolock_pop/occipital_physionet.npz"))
SF = float(occ["sf"]); rest = occ["later"][0]; rng = np.random.default_rng(3)


def run_checks(ch, live=True):
    """On the device: show the flicker, record, decide locally. SIMULATED EEG."""
    need = int(len(ch["freqs"]) * ch["seg_s"] * SF); o = rng.integers(0, rest.shape[1] - need)
    seg = rest[:, o:o + need]
    x = SS.simulate_response(seg, SF, ch, P["ssvep"]["amp_sim"], rng, P["occ_weights"]) if live else seg
    for _ in range(P["ssvep"]["max_tries"]):
        if SS.verify_response(x, SF, ch, P["ssvep"]["rho_thr"])[0]:
            return True
        x = SS.simulate_response(rest[:, o:o + need], SF, ch, P["ssvep"]["amp_sim"], rng, P["occ_weights"]) if live else seg
    return False


if a.logosctl:
    tv, td = CM.LogosDelivery(a.logosctl[0]), CM.LogosDelivery(a.logosctl[1])
else:
    CM.LocalBus.start_broker()
    tv, td = CM.LocalBus(), CM.LocalBus()

asha_sk = Ed25519PrivateKey.generate(); asha = pub_hex(asha_sk)


class Ids:                      # the verifier's view of the ledger: who is registered
    def __init__(self, s): self.s = s
    def get(self, k, d=None): return {"id_pub": k} if k in self.s else d


class L:
    ids = Ids({asha})


V = CM.Verifier(tv, ledger=L())
rows = []


def exchange(label, device_checks=run_checks, sk=asha_sk, pub=asha, tamper=None, delay=0.0):
    dev = CM.Device(td, sk, pub, device_checks)
    out = {}
    th = threading.Thread(target=lambda: out.update(r=dev.answer(timeout=30)))
    th.start(); time.sleep(0.3)
    t = time.time(); ch = V.issue(); th.join()
    reply = V.wait_reply(timeout=30)
    if tamper:
        reply = tamper(reply)
    time.sleep(delay)
    ok, why = V.check(reply, now=time.time() + delay * 0)
    rows.append({"case": label, "accepted": ok, "reason": why, "round_trip_s": round(time.time() - t, 2),
                 "message_fields": sorted(reply.keys())})
    print(f"  {label:42s} -> {'ACCEPT' if ok else 'REJECT'}: {why}")
    return ch, reply


print(f"transport: {tv.name}")
ch, good = exchange("genuine: live person, fresh challenge")
ok, why = V.check(good)
rows.append({"case": "replay the same reply again", "accepted": ok, "reason": why})
print(f"  {'replay the same reply again':42s} -> {'ACCEPT' if ok else 'REJECT'}: {why}")
exchange("no response (resting EEG)", device_checks=lambda c: run_checks(c, live=False))
mallory = Ed25519PrivateKey.generate()
exchange("forged signature (other key, Asha's id_pub)", sk=mallory)
exchange("unregistered person", sk=mallory, pub=pub_hex(mallory))
exchange("wrong nonce", tamper=lambda r: {**r, "nonce": "00" * 16})
# late: verifier's clock jumps past the window
_, late = exchange("(setup for late reply)")
V.answered.discard(late["challenge_id"]); rows.pop()
ok, why = V.check(late, now=time.time() + CM.MAX_AGE_S + 5)
rows.append({"case": "late reply", "accepted": ok, "reason": why})
print(f"  {'late reply':42s} -> {'ACCEPT' if ok else 'REJECT'}: {why}")
# what is on the wire
wire = {"challenge_fields": sorted(ch.keys()), "reply_fields": sorted(good.keys()),
        "contains_signals": any(isinstance(v, list) and len(v) > 16 for v in list(ch.values()) + list(good.values()))}
json.dump({"transport": tv.name, "topic": CM.TOPIC, "cases": rows, "wire": wire},
          open("results_logos_messaging.json", "w"), indent=1)
print("wire:", wire)
