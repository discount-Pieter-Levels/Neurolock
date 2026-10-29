"""Verification: prove 'I am an enrolled, live human' for a fresh challenge.

  1. VEIN    commitment mode: unwrap S with a finger's fuzzy-commitment key
                              (try each probe image; either finger).
             matcher mode:    the station matches against the stored template
                              and, on a match, releases the escrowed S.
  2. ALIVE   pulse + SSVEP response to THIS challenge (fresh nonce).
  3. CONSENT (high-stakes only) SSVEP confirm-target challenge bound to the
             action text: the person looks at CONFIRM; their EEG must follow
             CONFIRM's (fresh, random) frequency sequence. Looking at DENY,
             looking away, or a replayed recording gives no consent.
  4. PROOF   sign the challenge nonce with id_sk (derived from S) and, for
             high-stakes, a signed consent record {action_hash, nonce, decision}.
"""
import time
from cryptography.exceptions import InvalidTag

from . import crypto
from .enrol import Rejected, device_keys, pub_hex
from .ledger import canon
from ..liveness import pulse as pulse_mod, ssvep


def recover_S(station, cred, cap, mode):
    fingers = station.matcher.fingers
    if mode == "commitment":
        for f in fingers:
            v = cred["vein"][f]
            # all probe images of all attempts, one at a time
            imgs = [m for att in cap.vein_probe[f] for m in att[station.commit.method]]
            k, tries = station.commit.unlock(v["helper"], imgs)
            if k is None:
                continue
            key = crypto.kdf(crypto.bits_to_bytes(k), bytes.fromhex(v["salt"]), cred["kdf"])
            try:
                return crypto.unwrap(key, v["wrapped_S"], aad=cred["id_pub"].encode()), f"finger {f}, try {tries}"
            except InvalidTag:
                continue
        raise Rejected("vein", "no finger's fuzzy commitment opened")
    # matcher mode
    entry = next(e for e in station.registry if e["id_pub"] == cred["id_pub"])
    s, _ = station.matcher.person_score(station.templates(entry), cap.vein_probe)
    if s < station.P["vein_matcher"]["thr_verify"]:
        raise Rejected("vein", f"vein score {s:.2f} below threshold {station.P['vein_matcher']['thr_verify']:.2f}")
    return crypto.unwrap(station.at_rest_key, entry["escrow"], aad=b"escrow" + cred["id_pub"].encode()), f"score {s:.2f}"


def consent(station, cap, action, id_sk, log=print):
    """SSVEP consent for `action`. Up to max_tries fresh challenges while the
    answer is 'none' (no clear response); an explicit DENY stops immediately."""
    sp = station.P["ssvep"]
    for k in range(sp["max_tries"]):
        ch = ssvep.make_consent_challenge(action, sp["K"], sp["seg_s"])
        eeg, sf = cap.ssvep(ch)
        decision, per_seg = ssvep.decide_consent(eeg, sf, ch, sp["rho_thr"])
        if decision == "deny":
            raise Rejected("consent", f"person looked at DENY for '{action}'")
        if decision == "confirm":
            rec = {"action_hash": ch["action_hash"], "nonce": ch["nonce"], "decision": "confirm", "ts": ch["ts"]}
            log(f"  [consent] EEG followed CONFIRM for '{action}' (try {k + 1})")
            return {**rec, "sig": id_sk.sign(canon(rec)).hex()}
    raise Rejected("consent", f"no clear CONFIRM response in {sp['max_tries']} tries (last {per_seg})")


def verify(station, cred, cap, challenge, mode="commitment", high_stakes=False, action="", log=print):
    t0 = time.perf_counter()
    S, how = recover_S(station, cred, cap, mode)
    id_sk, id_pub = device_keys(S)
    if id_pub != cred["id_pub"] or station.ledger.ids.get(id_pub, {}).get("commitment") != crypto.H(b"commit", S).hex():
        raise Rejected("ledger", "recovered secret does not match the registered commitment")
    log(f"  [vein]   identity secret recovered ({mode}: {how})")
    unlock = getattr(station.ledger, "device_unlock", None)   # LogosLedger: S-derived LEZ key now usable
    if unlock:
        unlock(S)
    _, tries = station.check_liveness(cap, challenge)
    log(f"  [pulse]  heartbeat ok   [ssvep] fresh challenge followed (try {tries})")
    proof = {"id_pub": id_pub, "nonce": challenge["nonce"], "ts": challenge["ts"]}
    proof["sig"] = id_sk.sign(canon(proof)).hex()
    if high_stakes:
        proof["consent"] = consent(station, cap, action, id_sk, log)
    log(f"  [proof]  signed ({time.perf_counter() - t0:.1f} s)")
    return proof, S, id_sk
