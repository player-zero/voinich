"""Test 7 — is the line a structural unit ("record with fields")?

(a) word-class share as a function of position across the line
    (5 bins, lines of 6+ words) — gradients beyond the first/last word?
(b) MI between adjacent word CLASSES, real vs shuffled-within-line —
    do classes have preferred ordering inside the line?
"""

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_pages
from morphology import glyphs, norm_glyph

GALLOWS = {"k", "t", "cth", "ckh"}
PF = {"p", "f", "cph", "cfh"}


def wclass(w):
    gs = [norm_glyph(g) for g in glyphs(w)]
    if any(g in PF for g in gs):
        return "pf"
    g0 = gs[0]
    if g0 == "q":
        return "q-"
    if g0 in GALLOWS:
        return "gallows-"
    if g0 in ("ch", "sh"):
        return "ch/sh-"
    if g0 in ("o", "a"):
        return "o/a-"
    if g0 == "y":
        return "y-"
    if g0 in ("d", "s", "l", "r"):
        return "dslr-"
    return "other"


CLASSES = ["q-", "gallows-", "ch/sh-", "o/a-", "y-", "dslr-", "pf", "other"]


def lines_all():
    out = []
    for _, pars in voynich_pages():
        for par in pars:
            out.extend(par)
    return out


def part_a(lines):
    print("=== (a) class share by position bin (lines of 6+ words) ===")
    long_lines = [l for l in lines if len(l) >= 6]
    print(f"lines used: {len(long_lines)}")
    bins = defaultdict(Counter)
    for l in long_lines:
        n = len(l)
        for i, w in enumerate(l):
            b = min(4, int(i / n * 5))
            bins[b][wclass(w)] += 1
            bins[b]["_n"] += 1
    header = "class     " + "".join(f"{'bin'+str(b):>8}" for b in range(5))
    print(header + "   (start ... end)")
    for c in CLASSES:
        row = "".join(f"{bins[b][c]/bins[b]['_n']:>8.1%}" for b in range(5))
        print(f"{c:<10}{row}")


def part_b(lines, rounds=10, seed=17):
    print("\n=== (b) MI of adjacent word classes within lines ===")

    def mi(ls):
        pairs = Counter()
        for l in ls:
            cs = [wclass(w) for w in l]
            pairs.update(zip(cs, cs[1:]))
        n = sum(pairs.values())
        left, right = Counter(), Counter()
        for (a, b), c in pairs.items():
            left[a] += c
            right[b] += c
        return sum(c / n * math.log2((c / n) / (left[a] / n * right[b] / n))
                   for (a, b), c in pairs.items())

    real = mi(lines)
    rng = random.Random(seed)
    null = []
    for _ in range(rounds):
        sh = []
        for l in lines:
            l2 = l[:]
            rng.shuffle(l2)
            sh.append(l2)
        null.append(mi(sh))
    mu = sum(null) / len(null)
    sd = (sum((x - mu) ** 2 for x in null) / len(null)) ** 0.5 or 1e-9
    print(f"real {real:.4f} bits, within-line shuffle {mu:.4f} "
          f"(z={ (real-mu)/sd:.1f})")

    # which class transitions drive it
    pairs = Counter()
    for l in lines:
        cs = [wclass(w) for w in l]
        pairs.update(zip(cs, cs[1:]))
    n = sum(pairs.values())
    left, right = Counter(), Counter()
    for (a, b), c in pairs.items():
        left[a] += c
        right[b] += c
    scored = sorted(((c / n * math.log2((c / n) / (left[a] / n
                     * right[b] / n)), a, b, c)
                    for (a, b), c in pairs.items() if c >= 50),
                    key=lambda x: -abs(x[0]))
    print("strongest class transitions (contribution to MI):")
    for contrib, a, b, c in scored[:10]:
        exp = left[a] * right[b] / n
        print(f"  {a:>9} -> {b:<9} n={c:<5} expected {exp:6.0f}  "
              f"ratio {c/exp:4.2f}")


def main():
    lines = lines_all()
    part_a(lines)
    part_b(lines)


if __name__ == "__main__":
    main()
