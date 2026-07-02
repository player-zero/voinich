"""Test 2 — do adjacent words depend on each other (syntax-like glue)?

Mutual information between adjacent words, restricted to the top-K most
frequent words so the estimate is stable. Bias is handled by comparing
against the same computation on shuffled pages and against a natural
text processed identically. Also: immediate-repetition rate.
"""

import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens, culpeper_page_tokens

K = 150


def mi_adjacent(pages, topk):
    pairs = Counter()
    left = Counter()
    right = Counter()
    for _, toks in pages:
        for a, b in zip(toks, toks[1:]):
            if a in topk and b in topk:
                pairs[(a, b)] += 1
                left[a] += 1
                right[b] += 1
    n = sum(pairs.values())
    mi = 0.0
    for (a, b), c in pairs.items():
        p = c / n
        mi += p * math.log2(p / (left[a] / n * right[b] / n))
    return mi, n


def rep_rate(pages):
    rep = tot = 0
    for _, toks in pages:
        for a, b in zip(toks, toks[1:]):
            tot += 1
            rep += a == b
    return rep / tot


def analyse(name, pages, rounds=5, seed=7):
    vocab = Counter(w for _, t in pages for w in t)
    topk = {w for w, _ in vocab.most_common(K)}
    mi, n = mi_adjacent(pages, topk)
    rng = random.Random(seed)
    mis = []
    reps = []
    for _ in range(rounds):
        sh = [(p, rng.sample(t, len(t))) for p, t in pages]
        mis.append(mi_adjacent(sh, topk)[0])
        reps.append(rep_rate(sh))
    bias = sum(mis) / len(mis)
    rep = rep_rate(pages)
    rep0 = sum(reps) / len(reps)
    print(f"{name}  (pairs used: {n})")
    print(f"  MI adjacent words: real {mi:.3f} bits, shuffled {bias:.3f} "
          f"-> excess {mi-bias:+.3f} bits")
    print(f"  immediate repeat w_i==w_i+1: real {rep:.2%}, "
          f"shuffled {rep0:.2%}, ratio {rep/rep0:.2f}")
    return mi - bias


def top_pairs(pages, min_count=15):
    """Most attracted / repulsed adjacent pairs (PMI with count floor)."""
    vocab = Counter(w for _, t in pages for w in t)
    topk = {w for w, _ in vocab.most_common(K)}
    pairs = Counter()
    uni = Counter()
    n = 0
    for _, toks in pages:
        for a, b in zip(toks, toks[1:]):
            if a in topk and b in topk:
                pairs[(a, b)] += 1
                uni[a] += 1
                uni[b] += 1
                n += 1
    scored = []
    for (a, b), c in pairs.items():
        if c < min_count:
            continue
        pmi = math.log2(c * n / (uni[a] * uni[b]))
        scored.append((pmi, a, b, c))
    scored.sort(reverse=True)
    print("  most attracted pairs (PMI):")
    for pmi, a, b, c in scored[:8]:
        print(f"    {a:>8} {b:<10} n={c:<4} pmi={pmi:+.2f}")
    print("  most repulsed pairs:")
    for pmi, a, b, c in scored[-5:]:
        print(f"    {a:>8} {b:<10} n={c:<4} pmi={pmi:+.2f}")


def main():
    voy = voynich_page_tokens()
    eng = culpeper_page_tokens()
    ev = analyse("VOYNICH", voy)
    print()
    ee = analyse("CONTROL herbal English", eng)
    print(f"\nexcess MI, voynich vs control: {ev:.3f} vs {ee:.3f} bits")
    print()
    top_pairs(voy)


if __name__ == "__main__":
    main()
