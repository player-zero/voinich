"""Fit an explicit slot grammar to Voynich words and measure coverage.

The grammar is a regex over glyph-class strings:

  word = [q] [o/y] [d/s/l/r] [bench+gallows] [e-run] [bench+gallows] [e-run]
         [d/s/l/r] [o/a] [i-group | l/r/m/n/d/s] [y]

Controls: within-word glyph shuffle (does the grammar just accept
anything?) and per-type coverage.
"""

import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words
from morphology import glyphs, norm_glyph

CLASS = {"q": "Q", "o": "O", "y": "Y", "a": "A",
         "ch": "C", "sh": "C",
         "cth": "G", "ckh": "G", "cph": "G", "cfh": "G",
         "k": "K", "t": "K", "p": "K", "f": "K",
         "E": "E", "d": "D", "l": "L", "r": "R", "s": "S",
         "n": "N", "m": "M",
         "In": "J", "Ir": "J", "Il": "J", "Im": "J", "I-": "J"}

GRAMMAR = re.compile(
    r"^"
    r"Q?"                       # 1  q-
    r"[OY]?"                    # 2  o/y prefix
    r"[DSLR]?"                  # 3  initial consonant (d-, s-, l-, r-)
    r"(?:[CG]K?|K[CG]?)?"       # 4  bench and/or gallows
    r"E?"                       # 5  e-run
    r"(?:[CG]K?|K[CG]?)?"       # 6  second bench/gallows
    r"E?"                       # 7  second e-run
    r"[DSLR]?"                  # 8  final consonant
    r"[OA]?"                    # 9  o/a
    r"(?:J|[LRMNDS])?"          # 10 i-group or terminal consonant
    r"Y?"                       # 11 final -y
    r"$")


# v2: the tail (slots 8-11) may repeat, absorbing stacked suffixes
# (aral, daldy) and o-infixes (chodaiin = cho+daiin).
GRAMMAR2 = re.compile(
    r"^Q?[OY]?[DSLR]?"
    r"(?:[CG]K?|K[CG]?)?E?"
    r"(?:[CG]K?|K[CG]?)?E?"
    r"(?:[DSLR]?[OA]?(?:J|[LRMNDS])?Y?){0,2}$")


def classify(word):
    try:
        return "".join(CLASS[norm_glyph(g)] for g in glyphs(word))
    except KeyError:
        return None


def coverage(wordlist, grammar=GRAMMAR):
    ok = tot = 0
    misses = Counter()
    for w in wordlist:
        cs = classify(w)
        tot += 1
        if cs is not None and grammar.match(cs):
            ok += 1
        else:
            misses[w] += 1
    return ok / tot, misses


def conforms(word, grammar=GRAMMAR):
    cs = classify(word)
    return cs is not None and bool(grammar.match(cs))


def main():
    random.seed(42)
    loci = parse("data/ZL3b-n.txt")
    voy = [w for l in loci for w in words(l.text)]

    cov, misses = coverage(voy)
    types = set(voy)
    cov_t = sum(1 for w in types
                if (c := classify(w)) and GRAMMAR.match(c)) / len(types)
    print(f"tokens covered by slot grammar: {cov:.1%}")
    print(f"types  covered by slot grammar: {cov_t:.1%}")

    print("\ntop 20 words the grammar rejects:")
    for w, c in misses.most_common(20):
        print(f"  {w:<14} {c:>4}   {classify(w)}")

    # v2 grammar: repeatable tail
    cov2, misses2 = coverage(voy, GRAMMAR2)
    cov2_t = sum(1 for w in types if conforms(w, GRAMMAR2)) / len(types)
    print(f"\nv2 (repeatable tail): tokens {cov2:.1%}  types {cov2_t:.1%}")
    print("top 10 words v2 still rejects:")
    for w, c in misses2.most_common(10):
        print(f"  {w:<14} {c:>4}   {classify(w)}")

    # compound test: can v2-rejected words be split into two v2 words?
    rej_tokens = sum(misses2.values())
    split_ok = 0
    examples = []
    for w, c in misses2.items():
        for i in range(2, len(w) - 1):
            a, b = w[:i], w[i:]
            if conforms(a, GRAMMAR2) and conforms(b, GRAMMAR2):
                split_ok += c
                if len(examples) < 12:
                    examples.append(f"{a}+{b}")
                break
    print(f"\nof rejected tokens, splittable into two v2-words: "
          f"{split_ok/rej_tokens:.1%}")
    print("  e.g.", ", ".join(examples))

    # control 1: shuffle glyphs within each word
    shuf = []
    for w in voy:
        gs = glyphs(w)
        random.shuffle(gs)
        shuf.append("".join(gs))
    cov_s, _ = coverage(shuf)
    cov_s2, _ = coverage(shuf, GRAMMAR2)
    print(f"\ncontrol — glyphs shuffled within words: "
          f"v1 {cov_s:.1%}, v2 {cov_s2:.1%} covered")

    # control 2: how big is the grammar's language vs. observed vocabulary?
    # count how many of the grammar's possible short class-strings occur
    all_cs = Counter(classify(w) for w in voy if classify(w))
    gram_cs = [c for c in all_cs if GRAMMAR.match(c)]
    print(f"\ndistinct class-templates observed: {len(all_cs)}; "
          f"of them grammar-conform: {len(gram_cs)}")

    # slot-fill profile: which slots are used how often (rough, via groups)
    # re-parse with capturing groups for a profile
    cap = re.compile(
        r"^(Q?)([OY]?)([DSLR]?)((?:[CG]K?|K[CG]?)?)(E?)"
        r"((?:[CG]K?|K[CG]?)?)(E?)([DSLR]?)([OA]?)((?:J|[LRMNDS])?)(Y?)$")
    names = ["q", "o/y", "d/s/l/r", "bench/gallows", "e-run",
             "bench/gallows-2", "e-run-2", "d/s/l/r fin", "o/a",
             "i-grp/cons", "-y"]
    fill = Counter()
    n_ok = 0
    for w in voy:
        cs = classify(w)
        m = cap.match(cs) if cs else None
        if not m:
            continue
        n_ok += 1
        for i, g in enumerate(m.groups()):
            if g:
                fill[i] += 1
    print("\nslot usage among conforming tokens:")
    for i, name in enumerate(names):
        share = fill[i] / n_ok
        print(f"  {i+1:>2} {name:<16} {share:>6.1%} {'#' * int(share*50)}")


if __name__ == "__main__":
    main()
