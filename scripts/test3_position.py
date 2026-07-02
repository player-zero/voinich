"""Test 3 — does word structure depend on layout position?

In a plain-language text, how a word is built cannot depend on where the
line breaks. We measure glyph-class features by: word position in line
(first/middle/last) and line position in paragraph (first/other).
Significance: shuffle words within each paragraph (layout kept, words
redealt), 20 rounds -> z-scores.
"""

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_pages
from morphology import glyphs, norm_glyph

GALLOWS = {"k", "t", "p", "f"}
BENCHED = {"cth", "ckh", "cph", "cfh"}
PF = {"p", "f", "cph", "cfh"}


def features(w):
    gs = glyphs(w)
    f = set()
    g0 = gs[0]
    if g0 in GALLOWS | BENCHED:
        f.add("starts-gallows")
    if any(g in PF for g in gs):
        f.add("has-p/f")
    if g0 == "y":
        f.add("starts-y")
    if g0 in ("s", "sh"):
        f.add("starts-s/sh")
    if g0 == "d":
        f.add("starts-d")
    if g0 == "o":
        f.add("starts-o")
    if g0 == "q":
        f.add("starts-q")
    gl = norm_glyph(gs[-1])
    if gl in ("Im", "m"):
        f.add("ends-m")
    if gl == "In":
        f.add("ends-in")
    if gl == "y":
        f.add("ends-y")
    return f


ALL_FEATS = ["starts-q", "starts-o", "starts-d", "starts-y", "starts-s/sh",
             "starts-gallows", "has-p/f", "ends-m", "ends-in", "ends-y"]


def collect(pages):
    """slots: dict position-name -> Counter of features (+ '_n' total)."""
    slots = defaultdict(Counter)

    def add(slot, w):
        slots[slot]["_n"] += 1
        for f in features(w):
            slots[slot][f] += 1

    for _, pars in pages:
        for par in pars:
            for li, line in enumerate(par):
                lrole = "par-first-line" if li == 0 else "line-other"
                for wi, w in enumerate(line):
                    add(lrole, w)
                    if wi == 0:
                        add("word-line-first", w)
                    elif wi == len(line) - 1:
                        add("word-line-last", w)
                    else:
                        add("word-line-mid", w)
    return slots


def shuffle_pages(pages, rng):
    out = []
    for p, pars in pages:
        new_pars = []
        for par in pars:
            ws = [w for line in par for w in line]
            rng.shuffle(ws)
            it = iter(ws)
            new_pars.append([[next(it) for _ in line] for line in par])
        out.append((p, new_pars))
    return out


def main():
    pages = voynich_pages()
    real = collect(pages)

    rng = random.Random(3)
    rounds = 20
    null = defaultdict(lambda: defaultdict(list))
    for _ in range(rounds):
        sh = collect(shuffle_pages(pages, rng))
        for slot, cnt in sh.items():
            n = cnt["_n"]
            for f in ALL_FEATS:
                null[slot][f].append(cnt[f] / n)

    print(f"{'feature':<16}{'slot':<18}{'real':>7}{'expected':>10}"
          f"{'ratio':>7}{'z':>7}")
    for slot in ["word-line-first", "word-line-mid", "word-line-last",
                 "par-first-line", "line-other"]:
        n = real[slot]["_n"]
        for f in ALL_FEATS:
            obs = real[slot][f] / n
            xs = null[slot][f]
            mu = sum(xs) / len(xs)
            sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
            if sd == 0:
                continue
            z = (obs - mu) / sd
            if abs(z) >= 4:
                print(f"{f:<16}{slot:<18}{obs:>7.2%}{mu:>10.2%}"
                      f"{obs/mu:>7.2f}{z:>7.1f}")
    print(f"\n(shown: |z| >= 4; {rounds} shuffle rounds; "
          f"n per slot: " +
          ", ".join(f"{s}={real[s]['_n']}" for s in real))

    # word length by position
    print("\nmean word length (glyphs):")
    for slot in ["word-line-first", "word-line-mid", "word-line-last"]:
        pass
    lens = defaultdict(list)
    for _, pars in pages:
        for par in pars:
            for line in par:
                for wi, w in enumerate(line):
                    slot = ("first" if wi == 0 else
                            "last" if wi == len(line) - 1 else "mid")
                    lens[slot].append(len(glyphs(w)))
    for slot in ("first", "mid", "last"):
        v = lens[slot]
        print(f"  {slot:<6} {sum(v)/len(v):.2f}")


if __name__ == "__main__":
    main()
