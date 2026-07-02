"""Uniform measurement battery for the generator arbitrage.

A corpus is a page structure: [(page_id, [paragraph, ...])] where a
paragraph is [line, ...] and a line is [word, ...]. All synthetic
corpora share the REAL manuscript's skeleton (same pages, paragraphs,
lines, words-per-line), so layout metrics are directly comparable.
"""

import math
import random
from collections import Counter, defaultdict


def flatten_pages(struct):
    return [(p, [w for par in pars for line in par for w in line])
            for p, pars in struct]


def all_tokens(struct):
    return [w for _, t in flatten_pages(struct) for w in t]


def zipf_slope(vocab, top=1000):
    ranked = [c for _, c in vocab.most_common(top)]
    xs = [math.log(r + 1) for r in range(len(ranked))]
    ys = [math.log(c) for c in ranked]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)


def h2_char(tokens):
    text = ".".join(tokens)
    bi = Counter(zip(text, text[1:]))
    first = Counter()
    for (a, _), c in bi.items():
        first[a] += c
    n = sum(bi.values())
    return -sum(c / n * math.log2(c / first[a]) for (a, _), c in bi.items())


def mi_excess(pages, rng, rounds=3, topk=150):
    vocab = Counter(w for _, t in pages for w in t)
    top = {w for w, _ in vocab.most_common(topk)}

    def mi(pgs):
        pairs = Counter()
        for _, t in pgs:
            pairs.update((a, b) for a, b in zip(t, t[1:])
                         if a in top and b in top)
        n = sum(pairs.values())
        if not n:
            return 0.0
        l, r = Counter(), Counter()
        for (a, b), c in pairs.items():
            l[a] += c
            r[b] += c
        return sum(c / n * math.log2((c / n) / (l[a] / n * r[b] / n))
                   for (a, b), c in pairs.items())

    real = mi(pages)
    null = []
    for _ in range(rounds):
        null.append(mi([(p, rng.sample(t, len(t))) for p, t in pages]))
    return real - sum(null) / len(null)


def repeat_ratio(pages, rng, rounds=3):
    def rate(pgs):
        rep = tot = 0
        for _, t in pgs:
            for a, b in zip(t, t[1:]):
                tot += 1
                rep += a == b
        return rep / tot

    real = rate(pages)
    null = sum(rate([(p, rng.sample(t, len(t))) for p, t in pages])
               for _ in range(rounds)) / rounds
    return real / null if null else float("nan")


def selfsim_ratio(pages, rng, rounds=2, window=20):
    def rate(pgs):
        hit = tot = 0
        for _, t in pgs:
            for i in range(1, len(t)):
                tot += 1
                hit += t[i] in t[max(0, i - window):i]
        return hit / tot

    real = rate(pages)
    null = sum(rate([(p, rng.sample(t, len(t))) for p, t in pages])
               for _ in range(rounds)) / rounds
    return real / null if null else float("nan")


def _tv(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    keys = set(c1) | set(c2)
    return 0.5 * sum(abs(c1.get(k, 0) / n1 - c2.get(k, 0) / n2)
                     for k in keys)


def positional_z(struct, rng, rounds=10):
    """z-scores of total-variation distance between (a) first-char dist
    of line-first vs line-mid words, (b) last-char dist of line-last vs
    line-mid words. Shuffle words within paragraphs for the null."""

    def collect(st):
        first, mid, last, mid_end = Counter(), Counter(), Counter(), \
            Counter()
        for _, pars in st:
            for par in pars:
                for line in par:
                    for wi, w in enumerate(line):
                        if wi == 0:
                            first[w[0]] += 1
                        elif wi == len(line) - 1:
                            last[w[-1]] += 1
                        else:
                            mid[w[0]] += 1
                            mid_end[w[-1]] += 1
        return _tv(first, mid), _tv(last, mid_end)

    real_s, real_e = collect(struct)
    null_s, null_e = [], []
    for _ in range(rounds):
        sh = []
        for p, pars in struct:
            npars = []
            for par in pars:
                ws = [w for line in par for w in line]
                rng.shuffle(ws)
                it = iter(ws)
                npars.append([[next(it) for _ in line] for line in par])
            sh.append((p, npars))
        s, e = collect(sh)
        null_s.append(s)
        null_e.append(e)

    def z(real, null):
        mu = sum(null) / len(null)
        sd = (sum((x - mu) ** 2 for x in null) / len(null)) ** 0.5 or 1e-9
        return (real - mu) / sd

    return z(real_s, null_s), z(real_e, null_e)


def run_battery(struct, eva=True, seed=99):
    rng = random.Random(seed)
    pages = flatten_pages(struct)
    toks = all_tokens(struct)
    vocab = Counter(toks)
    m = {}
    m["types"] = len(vocab)
    m["hapax%"] = 100 * sum(1 for c in vocab.values() if c == 1) / len(vocab)
    m["top1%"] = 100 * vocab.most_common(1)[0][1] / len(toks)
    m["zipf"] = zipf_slope(vocab)
    m["MIexc"] = mi_excess(pages, rng)
    m["rep"] = repeat_ratio(pages, rng)
    m["selfsim"] = selfsim_ratio(pages, rng)
    zs, ze = positional_z(struct, rng)
    m["posZ-start"] = zs
    m["posZ-end"] = ze
    if eva:
        from slot_grammar import conforms, GRAMMAR2
        m["H2"] = h2_char(toks)
        conf = sum(c for w, c in vocab.items() if conforms(w, GRAMMAR2))
        m["gram%"] = 100 * conf / len(toks)
    else:
        m["H2"] = float("nan")
        m["gram%"] = float("nan")
    return m
