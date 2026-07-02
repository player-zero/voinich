"""Test 17 — throw the modern cryptanalysis toolbox at the glyph stream.

1. Index of coincidence + Friedman/Kasiski kappa scan (lags 1..60):
   a polyalphabetic key of period k leaves periodic spikes.
2. Shannon invariants: bijective substitution preserves H2; compare the
   entropy signature against what each cipher family must produce.
3. An actual automated substitution solver (simulated annealing over
   glyph->letter mappings, scored by an English trigram model) — the
   Zodiac-340 method. If the text were a mono/simple substitution of a
   European-like language, this cracks it in minutes.
"""

import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens, culpeper_tokens
from morphology import glyphs, norm_glyph


def glyph_stream():
    out = []
    for _, toks in voynich_page_tokens():
        for w in toks:
            out.extend(norm_glyph(g) for g in glyphs(w))
            out.append(".")
    return out


def part1_ic(stream):
    print("=== 1. index of coincidence / kappa scan ===")
    s = [g for g in stream if g != "."]
    n = len(s)
    freq = Counter(s)
    ic = sum(c * (c - 1) for c in freq.values()) / (n * (n - 1))
    print(f"glyphs: {n}, alphabet {len(freq)}, IC = {ic:.4f}")
    print(f"  (uniform/polyalphabetic -> {1/len(freq):.4f}; "
          f"English letters ~0.066)")
    base = sum((c / n) ** 2 for c in freq.values())
    print("kappa(lag): P(s[i]==s[i+k]) vs random", f"{base:.4f}")
    peaks = []
    for k in range(1, 61):
        kap = sum(a == b for a, b in zip(s, s[k:])) / (n - k)
        peaks.append((k, kap))
    for k, kap in peaks[:8]:
        print(f"  lag {k:>2}: {kap:.4f}")
    tail = [kap for k, kap in peaks if k > 10]
    mx = max(peaks[10:], key=lambda x: x[1])
    print(f"  lags 11-60: mean {sum(tail)/len(tail):.4f}, "
          f"max {mx[1]:.4f} at lag {mx[0]} "
          f"(periodic key would spike far above {base:.4f})")


def part2_invariants():
    print("\n=== 2. Shannon invariants ===")
    print("Voynich glyph stream: H1=3.87, H2=2.13 (measured earlier)")
    print("English letters:      H1=4.08, H2=3.24 (Culpeper control)")
    print("cipher family -> required ciphertext H2 vs observed 2.13:")
    print("  monoalphabetic subst.  H2 preserved -> need language with "
          "H2~2.1: none known (European 3.0-3.5)")
    print("  polyalphabetic (Vigenere etc.)  flattens -> H2 ~ log2(A) "
          "= 4.5+ : observed is the OPPOSITE")
    print("  homophonic             adds choice noise -> H2 > plaintext "
          "H2 >= 3.0 : observed 2.13")
    print("  transposition          destroys local order -> H2 -> H1 "
          "= 3.9 : observed 2.13")


def part3_solver(stream, iters=12000, restarts=3):
    print("\n=== 3. automated substitution solver (annealing + "
          "trigram LM) ===")
    # English trigram model incl. space
    eng = " ".join(culpeper_tokens(limit=200000))
    tri = Counter(eng[i:i + 3] for i in range(len(eng) - 2))
    tot = sum(tri.values())
    V = 27 ** 3
    logp = {t: math.log2((c + 0.5) / (tot + 0.5 * V))
            for t, c in tri.items()}
    floor = math.log2(0.5 / (tot + 0.5 * V))

    # glyph stream -> ids; '.' maps to space, rest optimized
    syms = [g for g, c in Counter(stream).most_common() if g != "."]
    text = stream[:45000]
    tri_g = Counter(tuple(text[i:i + 3]) for i in range(len(text) - 2))
    letters = "abcdefghijklmnopqrstuvwxyz"

    def score(mapping):
        s = 0.0
        for g3, c in tri_g.items():
            t = "".join(" " if x == "." else mapping[x] for x in g3)
            s += c * logp.get(t, floor)
        return s

    n3 = sum(tri_g.values())
    best_overall = None
    rng = random.Random(2024)
    for r in range(restarts):
        mapping = {g: rng.choice(letters) for g in syms}
        cur = score(mapping)
        T = 2000.0
        for it in range(iters):
            g = rng.choice(syms)
            old = mapping[g]
            mapping[g] = rng.choice(letters)
            new = score(mapping)
            if new >= cur or rng.random() < math.exp((new - cur) / T):
                cur = new
            else:
                mapping[g] = old
            T *= 0.9995
        if best_overall is None or cur > best_overall[0]:
            best_overall = (cur, dict(mapping))
        print(f"  restart {r}: {-cur/n3:.2f} bits/char")

    cur, mapping = best_overall
    # baseline: English scored by its own model
    held = " ".join(culpeper_tokens(limit=250000)[210000:215000])
    hb = sum(logp.get(held[i:i + 3], floor)
             for i in range(len(held) - 2)) / (len(held) - 2)
    print(f"best decode: {-cur/n3:.2f} bits/char   "
          f"(real English on this model: {-hb:.2f})")
    demo = stream[:120]
    out = "".join(" " if g == "." else mapping[g] for g in demo)
    print(f"  'decrypted' opening of f1r:\n    {out}")


def main():
    stream = glyph_stream()
    part1_ic(stream)
    part2_invariants()
    part3_solver(stream)


if __name__ == "__main__":
    main()
