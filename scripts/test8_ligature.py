"""Test 8 — is p (f) a ligature of t (k) plus an absorbed element?

If p = t+e written as one sign, then:
  (i)  the distribution of what FOLLOWS p should match what follows t+e;
  (ii) the distribution of what PRECEDES p should match what precedes t;
  (iii) rewriting p -> te inside words should produce attested words
        with correlated frequencies, better than p -> t or controls.
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens
from morphology import glyphs, norm_glyph
from test4_variants import cosine


def context_dists(tokens):
    """after-X and before-X glyph distributions for interesting X."""
    after = {k: Counter() for k in
             ("p", "f", "t", "k", "tE", "kE", "pE", "ch")}
    before = {k: Counter() for k in ("p", "f", "t", "k")}
    for w in tokens:
        gs = [norm_glyph(g) for g in glyphs(w)] + ["<end>"]
        for i, g in enumerate(gs[:-1]):
            base = {"p": "p", "cph": "p", "f": "f", "cfh": "f",
                    "t": "t", "cth": "t", "k": "k", "ckh": "k",
                    "ch": "ch"}.get(g)
            if base is None:
                continue
            if base in before:
                before[base][gs[i - 1] if i else "<start>"] += 1
            after[base][gs[i + 1]] += 1
            # what follows the e-run that follows a gallows
            if base in ("t", "k", "p") and gs[i + 1] == "E":
                after[base + "E"][gs[i + 2]] += 1
    return after, before


def show(counter, label, n=6):
    tot = sum(counter.values())
    top = "  ".join(f"{g}:{v/tot:.0%}" for g, v in counter.most_common(n))
    print(f"  {label:<6} (n={tot:>5}): {top}")


def rewrites(tokens):
    vocab = Counter(tokens)
    p_types = [(w, c) for w, c in vocab.items()
               if "p" in w and "cph" not in w]
    n_tok = sum(c for _, c in p_types)
    print(f"\np-word types: {len(p_types)}, tokens: {n_tok}")
    print(f"{'rewrite':<12}{'attested (tok)':>15}{'freq corr':>11}")
    for name, fn in [
            ("p -> t", lambda w: w.replace("p", "t")),
            ("p -> te", lambda w: w.replace("p", "te")),
            ("p -> tch", lambda w: w.replace("p", "tch")),
            ("p -> tk (ctl)", lambda w: w.replace("p", "tk")),
            ("p -> d (ctl)", lambda w: w.replace("p", "d"))]:
        ok = sum(c for w, c in p_types if fn(w) in vocab)
        # rank correlation between freq(p-word) and freq(image), attested only
        pairs = [(c, vocab[fn(w)]) for w, c in p_types if fn(w) in vocab]
        corr = float("nan")
        if len(pairs) > 10:
            xs = [math.log(a) for a, _ in pairs]
            ys = [math.log(b) for _, b in pairs]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
            sy = math.sqrt(sum((y - my) ** 2 for y in ys))
            corr = sum((x - mx) * (y - my)
                       for x, y in zip(xs, ys)) / (sx * sy)
        print(f"{name:<12}{ok/n_tok:>14.1%}{corr:>11.2f}")


def main():
    tokens = [w for _, t in voynich_page_tokens() for w in t]
    after, before = context_dists(tokens)

    print("=== what FOLLOWS ===")
    for k in ("p", "tE", "t", "ch", "f", "kE", "k"):
        show(after[k], k)
    print("cosine similarities (after-distributions):")
    for a, b in [("p", "tE"), ("p", "t"), ("p", "ch"), ("f", "kE"),
                 ("f", "k"), ("f", "tE"), ("p", "kE"), ("tE", "kE")]:
        print(f"  after-{a:<3} vs after-{b:<3}: "
              f"{cosine(after[a], after[b]):.3f}")

    print("\n=== what PRECEDES ===")
    for k in ("p", "t", "f", "k"):
        show(before[k], k)
    print("cosine similarities (before-distributions):")
    for a, b in [("p", "t"), ("p", "k"), ("f", "k"), ("f", "t"),
                 ("t", "k")]:
        print(f"  before-{a:<3} vs before-{b:<3}: "
              f"{cosine(before[a], before[b]):.3f}")

    rewrites(tokens)


if __name__ == "__main__":
    main()
