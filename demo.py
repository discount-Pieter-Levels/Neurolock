"""End-to-end demo: grant consent -> data decrypts; revoke -> it doesn't.

Data key = SHA256(trustee secret || biometric key). Unlocking needs BOTH
2-of-3 trustees (who check the consent ledger) AND the person's fresh EEG.
"""
import argparse
import hashlib
import os
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from neurolock import data, features, fuzzy, shamir, consent

ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=["synthetic", "physionet", "kaggle"], default="synthetic")
ap.add_argument("--subject", type=int, default=1)
ap.add_argument("--path", help="folder with SubjectXX_1/_2 files (kaggle)")
ap.add_argument("--mode", choices=["task", "rest"], default="task")
args = ap.parse_args()
data.KAGGLE_DIR, data.KAGGLE_MODE = args.path, args.mode
loader = getattr(data, args.data)

# Population reference from other people
ref = np.array([features.band_features(*loader(s)[0::2]) for s in range(2, 22)])
enroll_sig, query_sig, sf = loader(args.subject)
med, mad = features.fit_population(np.vstack([ref, features.band_features(enroll_sig, sf)]))

# 1. ENROLL: neural signature -> biometric key
w, z = features.to_bits(features.band_features(enroll_sig, sf), med, mad)
mask = features.reliable_mask(z, min(105, (len(z) // 7) * 7))
bio_key, helper = fuzzy.enroll(w[mask], fuzzy.RepetitionCode(7))

# 2. ENCRYPT the raw recording
trustee_secret = int.from_bytes(os.urandom(32), "big")
dk = hashlib.sha256(trustee_secret.to_bytes(66, "big") + bio_key.tobytes()).digest()
nonce = os.urandom(12)
ciphertext = AESGCM(dk).encrypt(nonce, enroll_sig.astype(np.float32).tobytes(), b"subject-1")
print(f"encrypted {len(ciphertext)/1e6:.1f} MB of EEG")

# 3. SPLIT the trustee secret 2-of-3, set up consent
person = Ed25519PrivateKey.generate()
ledger, DATA_ID, PURPOSE = consent.LocalLedger(), "eeg:subject-1", "research:sleep-study"
trustees = [consent.Trustee(f"T{i}", sh, ledger, DATA_ID, consent.pub_hex(person.public_key()))
            for i, sh in enumerate(shamir.split(trustee_secret, n=3, t=2))]


def try_unlock(eeg, label):
    shares = [s for s in (t.release(PURPOSE) for t in trustees) if s]
    if len(shares) < 2:
        return print(f"[{label}] DENIED: only {len(shares)} trustee shares (no valid consent)")
    wq, _ = features.to_bits(features.band_features(eeg, sf), med, mad)
    k, ok = fuzzy.recover(wq[mask], helper)
    if not ok:
        return print(f"[{label}] DENIED: neural signature did not match")
    key = hashlib.sha256(shamir.combine(shares[:2]).to_bytes(66, "big") + k.tobytes()).digest()
    pt = AESGCM(key).decrypt(nonce, ciphertext, b"subject-1")
    print(f"[{label}] UNLOCKED: recovered {len(pt)/1e6:.1f} MB, matches original: "
          f"{pt == enroll_sig.astype(np.float32).tobytes()}")


try_unlock(query_sig, "no consent yet")
ledger.append(consent.sign_consent(person, DATA_ID, "GRANT", PURPOSE))
try_unlock(query_sig, "consent granted, own EEG")
try_unlock(loader(30)[1], "consent granted, someone else's EEG")
forger = Ed25519PrivateKey.generate()
ledger.append(consent.sign_consent(forger, DATA_ID, "REVOKE", PURPOSE))
try_unlock(query_sig, "forged revoke by stranger (ignored)")
ledger.append(consent.sign_consent(person, DATA_ID, "REVOKE", PURPOSE))
try_unlock(query_sig, "consent revoked")
