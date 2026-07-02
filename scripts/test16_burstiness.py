"""Test 16 — fine frequency structure: Heaps' law and word burstiness.

Natural languages show:
  - Heaps: vocabulary grows ~ n^beta, beta 0.4-0.6;
  - two word populations: function words spread evenly (CV of
    inter-occurrence gaps ~1), content words come in bursts (CV >> 1).

Compare: Voynich vs English herbal vs M2v3 synthetic vs shuffled.
"""

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens, culpeper_tokens
from generators import skeleton_and_buckets, gen_m2v3
from battery import flatten_pages


def heaps_beta(tokens):
    """slope of log V(n) vs log n over the second half of the curve."""
    seen = set()
    pts = []
    for i, w in enumerate(tokens, 1):
        seen.add(w)
        if i & (i - 1) == 0 and i >= 256:   # powers of two
            pts.append((math.log(i), math.log(len(seen))))
    xs, ys = zip(*pts)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)


def burstiness(tokens, min_count=15):
    """per-word CV of inter-occurrence gaps; returns {word: (count, CV)}."""
    pos = defaultdict(list)
    for i, w in enumerate(tokens):
        pos[w].append(i)
    out = {}
    for w, ps in pos.items():
        if len(ps) < min_count:
            continue
        gaps = [b - a for a, b in zip(ps, ps[1:])]
        mu = sum(gaps) / len(gaps)
        sd = math.sqrt(sum((g - mu) ** 2 for g in gaps) / len(gaps))
        out[w] = (len(ps), sd / mu)
    return out


def summarize(name, tokens, rng):
    beta = heaps_beta(tokens)
    b = burstiness(tokens)
    cvs = sorted(cv for _, cv in b.values())
    med = cvs[len(cvs) // 2]
    bursty = sum(1 for cv in cvs if cv > 1.3) / len(cvs)
    sh = tokens[:]
    rng.shuffle(sh)
    b0 = burstiness(sh)
    cvs0 = sorted(cv for _, cv in b0.values())
    med0 = cvs0[len(cvs0) // 2]
    print(f"{name:<18} Heaps beta {beta:.2f}   median CV {med:.2f} "
          f"(shuffled {med0:.2f})   words with CV>1.3: {bursty:.0%}   "
          f"(n words {len(cvs)})")
    return b


def main():
    rng = random.Random(1)
    voy = [w for _, t in voynich_page_tokens() for w in t]
    eng = culpeper_tokens(limit=len(voy))
    struct, buckets = skeleton_and_buckets()
    synth = [w for _, t in flatten_pages(gen_m2v3(struct, buckets))
             for w in t]

    print("=== Heaps' law and burstiness ===")
    bv = summarize("VOYNICH", voy, rng)
    be = summarize("English herbal", eng, rng)
    summarize("M2v3 synthetic", synth, rng)

    def extremes(b, label):
        items = sorted(b.items(), key=lambda kv: kv[1][1])
        print(f"  {label} most even: "
              + ", ".join(f"{w}({cv:.2f})" for w, (_, cv) in items[:5]))
        print(f"  {label} most bursty: "
              + ", ".join(f"{w}({cv:.2f})" for w, (_, cv) in items[-5:]))

    print()
    extremes(bv, "VOY")
    extremes(be, "ENG")

    # is burstiness correlated with frequency the way languages do?
    # (in language, top function words are even, mid-freq content bursty)
    print("\nmedian CV by frequency band:")
    for name, b in (("VOYNICH", bv), ("English", be)):
        bands = {"15-30": [], "31-100": [], "100+": []}
        for w, (c, cv) in b.items():
            k = "15-30" if c <= 30 else "31-100" if c <= 100 else "100+"
            bands[k].append(cv)
        row = "  ".join(
            f"{k}: {sorted(v)[len(v)//2]:.2f} (n={len(v)})"
            for k, v in bands.items() if v)
        print(f"  {name:<10} {row}")


if __name__ == "__main__":
    main()
