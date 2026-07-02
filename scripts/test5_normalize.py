"""Test 5 — apply the data-supported normalizations and recompute the
headline statistics.

Rules applied:
  R1 (strong):    m -> ir  (line-final flourish of the i-series;
                  cosine 0.93 at a 0.96 noise ceiling)
  R2 (tentative): strip line-initial y- when the remainder is attested
                  elsewhere (prepended ornament, 86% vs 77% baseline)
Refuted, NOT applied: p->t / f->k (p/f are a distinct word class).
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_pages
from morphology import glyphs
from test2_context import analyse

def norm_m(w):
    if "m" not in w:
        return w
    out = []
    for i, ch in enumerate(w):
        if ch == "m":
            out.append("r" if i and w[i - 1] == "i" else "ir")
        else:
            out.append(ch)
    return "".join(out)


def normalized_pages(strip_y=True):
    pages = voynich_pages()
    vocab = Counter(w for _, pars in pages for par in pars
                    for line in par for w in line)
    out = []
    for p, pars in pages:
        npars = []
        for par in pars:
            nlines = []
            for line in par:
                nline = []
                for wi, w in enumerate(line):
                    if (strip_y and wi == 0 and w.startswith("y")
                            and len(w) > 2 and w[1:] in vocab):
                        w = w[1:]
                    nline.append(norm_m(w))
                nlines.append(nline)
            npars.append(nlines)
        out.append((p, npars))
    return out


def flatten(pages):
    return [(p, [w for par in pars for line in par for w in line])
            for p, pars in pages]


def entropy_stats(tokens):
    text = ".".join(tokens)
    uni = Counter(text)
    bi = Counter(zip(text, text[1:]))
    tot = sum(uni.values())
    h1 = -sum(c / tot * math.log2(c / tot) for c in uni.values())
    first = Counter()
    for (a, _), c in bi.items():
        first[a] += c
    nb = sum(bi.values())
    h2 = -sum(c / nb * math.log2(c / first[a]) for (a, _), c in bi.items())
    return h1, h2


def summary(name, pages):
    toks = [w for _, t in flatten(pages) for w in t]
    vocab = Counter(toks)
    hapax = sum(1 for c in vocab.values() if c == 1)
    h1, h2 = entropy_stats(toks)
    print(f"{name}: tokens {len(toks)}, types {len(vocab)}, "
          f"hapax {hapax/len(vocab):.1%}, H1 {h1:.3f}, H2 {h2:.3f}")


def positional_check(pages):
    """z-scores for the key features after normalization."""
    import random
    from collections import defaultdict

    def feats(w):
        f = set()
        gs = glyphs(w)
        if w.endswith("m"):
            f.add("ends-m")
        if w.startswith("y"):
            f.add("starts-y")
        if gs and gs[0] in ("k", "t", "p", "f", "cth", "ckh", "cph", "cfh"):
            f.add("starts-gallows")
        if w.startswith("q"):
            f.add("starts-q")
        return f

    def collect(pgs):
        slots = defaultdict(Counter)
        for _, pars in pgs:
            for par in pars:
                for line in par:
                    for wi, w in enumerate(line):
                        slot = ("start" if wi == 0 else
                                "end" if wi == len(line) - 1 else "mid")
                        slots[slot]["_n"] += 1
                        for f in feats(w):
                            slots[slot][f] += 1
        return slots

    real = collect(pages)
    rng = random.Random(11)
    null = defaultdict(lambda: defaultdict(list))
    for _ in range(20):
        sh = []
        for p, pars in pages:
            npars = []
            for par in pars:
                ws = [w for line in par for w in line]
                rng.shuffle(ws)
                it = iter(ws)
                npars.append([[next(it) for _ in line] for line in par])
            sh.append((p, npars))
        for slot, cnt in collect(sh).items():
            n = cnt["_n"]
            for f in ("ends-m", "starts-y", "starts-gallows", "starts-q"):
                null[slot][f].append(cnt[f] / n)
    print("\nremaining positional effects (|z|>=4 shown):")
    for slot in ("start", "mid", "end"):
        n = real[slot]["_n"]
        for f in ("ends-m", "starts-y", "starts-gallows", "starts-q"):
            xs = null[slot][f]
            mu = sum(xs) / len(xs)
            sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5 or 1e-9
            z = (real[slot][f] / n - mu) / sd
            if abs(z) >= 4:
                print(f"  {f:<15} @line-{slot:<6} ratio "
                      f"{real[slot][f]/n/mu if mu else 0:5.2f}  z {z:6.1f}")


def main():
    raw = voynich_pages()
    norm = normalized_pages()

    summary("RAW       ", raw)
    summary("NORMALIZED", norm)

    raw_v = Counter(w for _, t in flatten(raw) for w in t)
    nrm_v = Counter(w for _, t in flatten(norm) for w in t)
    merged = {w for w in raw_v if w not in nrm_v}
    print(f"types eliminated by merging: {len(merged)}")

    print("\n--- adjacent-word MI, raw vs normalized ---")
    analyse("RAW", flatten(raw))
    analyse("NORMALIZED", flatten(norm))

    positional_check(norm)


if __name__ == "__main__":
    main()
