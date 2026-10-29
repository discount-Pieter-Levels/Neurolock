"""Small crypto helpers used by the PoP pipeline.

  kdf(secret, salt)    Argon2id: a slow, memory-hard hash. A biometric key has
                       only ~100-300 bits that are hard to predict, and an
                       attacker who steals helper data can guess offline. Argon2id
                       makes every guess cost ARGON2 time + memory, and a per-user
                       random salt stops one precomputed table working for everyone.
  wrap / unwrap        AES-256-GCM: encrypt + authenticate. Unwrapping with a wrong
                       key fails loudly instead of returning garbage.
  ed25519_from_seed    deterministic signing key from 32 secret bytes.
"""
import hashlib
import os
import time
from argon2.low_level import Type, hash_secret_raw
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Argon2id parameters (OWASP 2024 minimum is m=19 MiB, t=2, p=1; we use more memory)
ARGON2 = {"time_cost": 3, "memory_cost": 64 * 1024, "parallelism": 1}   # memory in KiB = 64 MiB


def H(*parts: bytes) -> bytes:
    """SHA-256 of length-prefixed parts (so ('ab','c') != ('a','bc'))."""
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(4, "big") + p)
    return h.digest()


def new_salt() -> bytes:
    return os.urandom(16)


def kdf(secret: bytes, salt: bytes, params=ARGON2) -> bytes:
    return hash_secret_raw(secret, salt, hash_len=32, type=Type.ID, **params)


def wrap(key: bytes, plaintext: bytes, aad: bytes = b"") -> dict:
    nonce = os.urandom(12)
    return {"nonce": nonce.hex(), "ct": AESGCM(key).encrypt(nonce, plaintext, aad).hex()}


def unwrap(key: bytes, blob: dict, aad: bytes = b"") -> bytes:
    """Raises cryptography.exceptions.InvalidTag if the key is wrong."""
    return AESGCM(key).decrypt(bytes.fromhex(blob["nonce"]), bytes.fromhex(blob["ct"]), aad)


def ed25519_from_seed(seed32: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(seed32)


def bits_to_bytes(bits) -> bytes:
    import numpy as np
    return np.packbits(np.asarray(bits, np.uint8)).tobytes()


def time_kdf(n=5):
    t = time.perf_counter()
    for _ in range(n):
        kdf(b"x" * 32, new_salt())
    return (time.perf_counter() - t) / n
