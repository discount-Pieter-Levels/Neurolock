"""Fuzzy commitment (Juels & Wattenberg, 1999).

Problem: hash(EEG) never matches twice, because EEG is noisy.
Trick: pick a random key k, encode it with an error-correcting code -> c.
Store only  helper = c XOR w  (w = enrollment bits) and hash(k).
Later, with fresh bits w':  helper XOR w' = c XOR (w XOR w') = c + small errors.
Decode -> k. If hash matches, it's the same person. Nothing stored reveals k or w alone.

Two codes:
  RepetitionCode(r)  - each key bit repeated r times; fixes (r-1)//2 flips per group.
  BCHCode(n, t)      - binary BCH; fixes any t flips anywhere in an n-bit block.
                       Inputs longer than n are split into blocks; a short final
                       block uses a shortened code.
"""
import hashlib
import secrets
import numpy as np


class RepetitionCode:
    """Simplest ECC: repeat each key bit r times, decode by majority vote."""

    def __init__(self, r):
        assert r % 2 == 1
        self.r = r

    def usable(self, L):
        return (L // self.r) * self.r

    def key_bits(self, L):
        return L // self.r

    def encode(self, k, L=None):
        return np.repeat(k, self.r)

    def decode(self, c):
        return (c.reshape(-1, self.r).sum(axis=1) > self.r // 2).astype(np.uint8)

    def decode_checked(self, c):
        """Majority vote never 'fails'; a wrong key is caught by the hash."""
        return self.decode(c), True

    def spec(self):
        return {"code": "rep", "r": self.r}


class BCHCode:
    """Binary BCH code of length n (2^m - 1) correcting t errors per block."""

    _cache = {}

    def __init__(self, n, t):
        import galois
        key = (n, t)
        if key not in self._cache:
            self._cache[key] = galois.BCH(n, d=2 * t + 1)
        self.bch = self._cache[key]
        self.n, self.k, self.t = self.bch.n, self.bch.k, self.bch.t

    def _blocks(self, L):
        """Block lengths for L input bits. A final block shorter than n-k+1 is dropped."""
        sizes = [self.n] * (L // self.n)
        rest = L % self.n
        if rest > self.n - self.k:
            sizes.append(rest)
        return sizes

    def usable(self, L):
        return sum(self._blocks(L))

    def key_bits(self, L):
        return sum(s - (self.n - self.k) for s in self._blocks(L))

    def encode(self, key, L):
        out, i = [], 0
        for s in self._blocks(L):
            m = s - (self.n - self.k)
            out.append(np.asarray(self.bch.encode(key[i:i + m].astype(int)), np.uint8))
            i += m
        return np.concatenate(out)

    def decode(self, c):
        return self.decode_checked(c)[0]

    def decode_checked(self, c):
        """Decode every block. Returns (key bits, ok).

        ok is False if ANY block had more than t errors. galois signals this
        with an error count of -1; without the check it silently returns the
        noisy message bits unchanged, which can occasionally still be the right
        key (the old 'decoder quirk', VEIN_README limitation 6)."""
        out, i, ok = [], 0, True
        for s in self._blocks(len(c)):
            m, n_err = self.bch.decode(c[i:i + s].astype(int), errors=True)
            out.append(np.asarray(m, np.uint8))
            ok &= int(n_err) >= 0
            i += s
        return np.concatenate(out), bool(ok)

    def spec(self):
        return {"code": "bch", "n": self.n, "t": self.t}


def code_from_spec(spec):
    if spec.get("code", "rep") == "bch":
        return BCHCode(spec["n"], spec["t"])
    return RepetitionCode(spec["r"])


def enroll(w, code):
    L = code.usable(len(w))
    if L == 0:
        raise ValueError(f"{len(w)} bits is too few for {code.spec()}")
    k_len = code.key_bits(len(w))
    k = np.frombuffer(secrets.token_bytes(k_len), dtype=np.uint8) & 1
    helper = code.encode(k, L) ^ w[:L]
    return k, {"helper": helper.tolist(), "hash": hashlib.sha256(k.tobytes()).hexdigest(), **code.spec()}


def recover(w_new, record):
    code = code_from_spec(record)
    helper = np.array(record["helper"], dtype=np.uint8)
    k, decoded = code.decode_checked(helper ^ w_new[: len(helper)])
    ok = decoded and hashlib.sha256(k.tobytes()).hexdigest() == record["hash"]
    return (k if ok else None), ok
