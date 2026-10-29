"""Enrolment: one human -> one credential.

A Capture bundles everything a person presents at the enrolment station:
  vein_enrol  {finger: {method: [6 vein maps]}}      (enrolment images)
  pulse       (frames, fps)                          finger video
  ssvep       responder(challenge) -> (occipital EEG, sfreq): what the person's
              EEG shows while watching that flicker challenge
  intent      'confirm' or 'deny': which button the person looks at during a
              consent challenge (SIMULATION input; a real person just looks)

Steps (Station.enrol):
  1. DEDUP   vein matcher vs every registered template (encrypted at rest,
             decrypted only inside the station). Score >= dedup threshold -> reject.
  2. ALIVE   pulse check, then SSVEP challenge-response.
  3. SECRET  random identity secret S (32 bytes), generated on the user's device.
  4. WRAP    vein fuzzy commitment per finger -> key k_f; S is wrapped (AES-GCM)
             under Argon2id(k_f, salt_f). Either finger can unwrap S.
  5. LEDGER  register C = H(S) and id_pub (from S); station signs.
EEG is used ONLY for liveness and consent (SSVEP), never as a key: the
resting-EEG 'brain key' failed across sessions (0 % after one month).
             No templates or raw data go to the ledger.
"""
import os
import time
from dataclasses import dataclass, field

import numpy as np
from cryptography.hazmat.primitives import serialization

from . import crypto
from .biometrics import METHODS, VeinCommitment, VeinMatcher
from ..liveness import pulse as pulse_mod, ssvep


def pub_hex(sk):
    return sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


@dataclass
class Capture:
    vein_enrol: dict = None      # {finger: {method: [maps]}}
    vein_probe: dict = None      # {finger: [attempt {method: [maps]}, ...]}
    pulse: tuple = None          # (frames, fps)
    ssvep: object = None         # callable(challenge) -> (eeg, sfreq)
    intent: str = "confirm"
    label: str = ""


class Rejected(Exception):
    def __init__(self, layer, reason):
        super().__init__(f"[{layer}] {reason}")
        self.layer, self.reason = layer, reason


def _pack_maps(imgs):
    return b"".join(np.packbits(np.asarray(m, np.uint8)).tobytes() for f in sorted(imgs)
                    for meth in METHODS for m in imgs[f][meth])


def _unpack_maps(blob, fingers, n_img, shape):
    per = int(np.ceil(shape[0] * shape[1] / 8))
    out, i = {}, 0
    for f in sorted(fingers):
        out[f] = {}
        for meth in METHODS:
            out[f][meth] = []
            for _ in range(n_img):
                bits = np.unpackbits(np.frombuffer(blob[i:i + per], np.uint8))[: shape[0] * shape[1]]
                out[f][meth].append(bits.reshape(shape)); i += per
    return out


def device_keys(S):
    """Keys the user's device derives from the identity secret S."""
    id_sk = crypto.ed25519_from_seed(crypto.H(b"id-key", S))
    return id_sk, pub_hex(id_sk)


class Station:
    """Enrolment station. Holds a signing key, an at-rest template key and the
    encrypted template registry (matcher mode). Deletable after the event."""

    def __init__(self, params, ledger_cls, kdf_params=crypto.ARGON2):
        self.P = params
        self.sk = crypto.ed25519_from_seed(os.urandom(32))
        self.pub = pub_hex(self.sk)
        self.at_rest_key = os.urandom(32)          # AES-256-GCM key for stored templates
        self.ledger = ledger_cls(trusted_stations=[self.pub])
        self.registry = []                         # [{id_pub, blob}]  encrypted templates
        self.matcher = VeinMatcher(params["vein_matcher"])
        self.commit = VeinCommitment(params["vein_commit"])
        self.kdf_params = kdf_params
        self.map_shape = None

    # ------------------------------------------------------------ helpers
    def templates(self, entry):
        blob = crypto.unwrap(self.at_rest_key, entry["blob"], aad=entry["id_pub"].encode())
        return _unpack_maps(blob, self.matcher.fingers, 6, self.map_shape)

    def dedup_scores(self, enrol_imgs):
        """Newcomer's enrolment images (as two 3-image attempts) vs every registered person."""
        probe = {f: [{m: [enrol_imgs[f][m][i] for i in a] for m in METHODS} for a in ((0, 1, 2), (3, 4, 5))]
                 for f in self.matcher.fingers}
        return [self.matcher.person_score(self.templates(e), probe)[0] for e in self.registry]

    def check_liveness(self, cap, challenge):
        """Pulse, then SSVEP. A live person may fail one flicker challenge by
        chance, so up to max_tries challenges are given, each with a NEW random
        sequence and nonce (a replayed recording fails every one of them)."""
        frames, fps = cap.pulse
        ok, bpm, snr, _ = pulse_mod.check_pulse(frames, fps, self.P["pulse"]["min_snr_db"])
        if not ok:
            raise Rejected("pulse", f"no heartbeat in finger video (SNR {snr:.1f} dB)")
        sp = self.P["ssvep"]
        for k in range(sp["max_tries"]):
            ch = challenge if k == 0 else ssvep.make_challenge(sp["K"], sp["seg_s"])
            eeg, sf = cap.ssvep(ch)
            ok, per_seg = ssvep.verify_response(eeg, sf, ch, sp["rho_thr"])
            if ok:
                return bpm, k + 1
        raise Rejected("ssvep", f"EEG did not follow {sp['max_tries']} flicker challenges (last: {per_seg})")

    # ------------------------------------------------------------ enrolment
    def enrol(self, cap: Capture, challenge, log=print):
        t0 = time.perf_counter()
        fingers = self.matcher.fingers
        self.map_shape = self.map_shape or cap.vein_enrol[fingers[0]][METHODS[0]][0].shape
        # 1. dedup
        if self.registry:
            best = max(self.dedup_scores(cap.vein_enrol))
            if best >= self.P["vein_matcher"]["thr_dedup"]:
                raise Rejected("dedup", f"veins match an enrolled person (score {best:.2f} >= "
                                        f"{self.P['vein_matcher']['thr_dedup']:.2f})")
            log(f"  [dedup]  no match among {len(self.registry)} enrolled people (best {best:.2f})")
        else:
            log("  [dedup]  registry empty")
        # 2. liveness
        bpm, tries = self.check_liveness(cap, challenge)
        log(f"  [pulse]  heartbeat {bpm:.0f} bpm   [ssvep] challenge followed (try {tries})")
        # 3. identity secret (on device)
        S = os.urandom(32)
        id_sk, id_pub = device_keys(S)
        # 4. wrap S under each finger's vein key
        cred = {"id_pub": id_pub, "vein": {}, "kdf": self.kdf_params}
        for f in fingers:
            k, rec = self.commit.enrol(cap.vein_enrol[f][self.commit.method])
            salt = crypto.new_salt()
            key = crypto.kdf(crypto.bits_to_bytes(k), salt, self.kdf_params)
            cred["vein"][f] = {"helper": rec, "salt": salt.hex(),
                               "wrapped_S": crypto.wrap(key, S, aad=id_pub.encode())}
        # matcher-mode escrow: S wrapped under the station key, released on a vein match
        escrow = crypto.wrap(self.at_rest_key, S, aad=b"escrow" + id_pub.encode())
        # 5. ledger
        from .logos_ledger import user_account          # public LEZ account derived from S (no key leaves the device)
        entry = {"commitment": crypto.H(b"commit", S).hex(), "id_pub": id_pub,
                 "lez_user": user_account(S).hex(),
                 "station_pub": self.pub, "ts": time.time()}
        from .ledger import canon
        entry["station_sig"] = self.sk.sign(canon(entry)).hex()
        self.ledger.register_commitment(entry)
        cred["lez_user"] = entry["lez_user"]
        self.registry.append({"id_pub": id_pub, "escrow": escrow,
                              "blob": crypto.wrap(self.at_rest_key, _pack_maps(cap.vein_enrol),
                                                  aad=id_pub.encode())})
        log(f"  [ledger] registered commitment {entry['commitment'][:16]}..  "
            f"({time.perf_counter() - t0:.1f} s)")
        return cred, S
