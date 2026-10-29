"""Shamir secret sharing: split a secret into n shares, any t rebuild it,
t-1 reveal nothing. Idea: a secret is the y-intercept of a random degree
(t-1) polynomial; t points pin down the polynomial, fewer don't."""
import secrets

P = 2**521 - 1  # Mersenne prime, larger than a 256-bit key


def split(secret: int, n: int, t: int):
    coeffs = [secret] + [secrets.randbelow(P) for _ in range(t - 1)]
    return [(x, sum(c * pow(x, i, P) for i, c in enumerate(coeffs)) % P) for x in range(1, n + 1)]


def combine(shares):
    s = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i != j:
                num = num * (-xj) % P
                den = den * (xi - xj) % P
        s = (s + yi * num * pow(den, -1, P)) % P
    return s
