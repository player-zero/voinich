"""Test 13 — three pre-registered attempts to REFUTE the generator
verdict.

A. qo-rule across line breaks. Motor-flow predicts the -y -> qo- pull
   weakens when the pen is interrupted by a line break; a system rule
   (or encoding) predicts it persists.
B. Page vocabulary: smooth drift (habit) vs discrete per-page topics
   (content). Cosine similarity of page word-distributions as a
   function of distance in manuscript order, within herbal/Currier-A.
C. Long repeated phrases at long range. Recipes repeat formulas across
   distant pages; a 20-word-window copyist cannot. Compare real vs the
   M2v3 synthetic corpus vs the English herbal.
"""

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_pages, culpeper_page_tokens
from ivtff import parse, words, SECTIONS
from generators import skeleton_and_buckets, gen_m2v3
from battery import flatten_pages


def part_a():
    print("=== A. qo-rule: within lines vs across line breaks ===")
    within, across = [], []
    for _, pars in voynich_pages():
        for par in pars:
            for li, line in enumerate(par):
                for a, b in zip(line, line[1:]):
                    within.append((a, b))
                if li + 1 < len(par) and line and par[li + 1]:
                    across.append((line[-1], par[li + 1][0]))

    def rule(pairs):
        hy = ty = hc = tc = 0
        for a, b in pairs:
            if a.endswith("y"):
                ty += 1
                hy += b.startswith("qo")
            elif a[-1:] in "nlrsm":
                tc += 1
                hc += b.startswith("qo")
        return hy / ty, hc / tc, ty, tc

    for name, pairs in (("within-line", within), ("across-break", across)):
        py, pc, ny, nc = rule(pairs)
        print(f"  {name:<13} P(qo|-y)={py:6.1%} (n={ny})   "
              f"P(qo|cons)={pc:5.1%} (n={nc})   pull ratio {py/pc:4.1f}x")


def part_b():
    print("\n=== B. page vocabulary: drift vs discrete topics ===")
    # herbal, Currier A only, in manuscript order
    pages = []
    for l in parse("data/ZL3b-n.txt"):
        if l.ltype[1] != "P":
            continue
        if SECTIONS.get(l.page_vars.get("I")) != "herbal":
            continue
        if l.page_vars.get("L") != "A":
            continue
        if not pages or pages[-1][0] != l.page:
            pages.append((l.page, []))
        pages[-1][1].extend(words(l.text))
    pages = [(p, ws) for p, ws in pages if len(ws) >= 60]
    print(f"  herbal-A pages used: {len(pages)}")

    def cos(c1, c2):
        n1 = math.sqrt(sum(v * v for v in c1.values()))
        n2 = math.sqrt(sum(v * v for v in c2.values()))
        return sum(v * c2.get(k, 0) for k, v in c1.items()) / (n1 * n2)

    dists = defaultdict(list)
    cs = [(i, Counter(ws)) for i, (_, ws) in enumerate(pages)]
    for i, ci in cs:
        for j, cj in cs:
            if j <= i:
                continue
            d = j - i
            bin_ = (1 if d == 1 else 2 if d <= 3 else 4 if d <= 7
                    else 8 if d <= 15 else 16 if d <= 31 else 32)
            dists[bin_].append(cos(ci, cj))
    # split-half internal coherence baseline
    rng = random.Random(3)
    internal = []
    for _, ws in pages:
        w2 = ws[:]
        rng.shuffle(w2)
        h = len(w2) // 2
        internal.append(cos(Counter(w2[:h]), Counter(w2[h:])))
    print(f"  same-page split-half coherence: "
          f"{sum(internal)/len(internal):.3f}")
    for b in sorted(dists):
        v = dists[b]
        label = {1: "d=1", 2: "d=2-3", 4: "d=4-7", 8: "d=8-15",
                 16: "d=16-31", 32: "d=32+"}[b]
        print(f"  {label:<9} mean cos {sum(v)/len(v):.3f}  (n={len(v)})")


def ngram_repeats(pages_tokens, n, min_gap_pages=1):
    """n-gram types repeated with occurrences on different pages."""
    seen = {}
    far = []
    for pi, (_, toks) in enumerate(pages_tokens):
        for i in range(len(toks) - n + 1):
            g = tuple(toks[i:i + n])
            if g in seen and pi - seen[g] >= min_gap_pages:
                far.append((g, seen[g], pi))
            if g not in seen:
                seen[g] = pi
    return far


def part_c():
    print("\n=== C. repeated phrases across distant pages ===")
    struct, buckets = skeleton_and_buckets()
    real = flatten_pages(struct)
    synth = flatten_pages(gen_m2v3(struct, buckets))
    eng = culpeper_page_tokens()
    for name, pgs in (("REAL", real), ("M2v3 synthetic", synth),
                      ("English herbal", eng)):
        row = []
        for n in (3, 4, 5):
            far = ngram_repeats(pgs, n)
            row.append(f"{n}-grams:{len(far):>4}")
        print(f"  {name:<16} cross-page repeats  " + "  ".join(row))
    # show the real long ones
    far4 = ngram_repeats(real, 4)
    far5 = ngram_repeats(real, 5)
    print("  real repeated 4+-word phrases (first page -> later page):")
    shown = 0
    for g, p1, p2 in (far5 + far4):
        print(f"    [{' '.join(g)}]  pages #{p1}->#{p2} "
              f"(gap {p2-p1})")
        shown += 1
        if shown >= 8:
            break


def main():
    part_a()
    part_b()
    part_c()


if __name__ == "__main__":
    main()
