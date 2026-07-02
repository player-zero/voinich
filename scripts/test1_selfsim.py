"""Test 1 — local self-similarity: is each word a small variation of a
word seen shortly before it on the same page?

Metric: min edit distance (over glyphs for Voynich, chars for the
control) from each token to the previous W tokens on the page.
Controls: same pages with word order shuffled (kills locality, keeps
vocabulary), and a natural-language herbal of the same size.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens, culpeper_page_tokens
from morphology import glyphs

W = 20


def edit(a, b, cap=3):
    """Banded Levenshtein with early cutoff at cap."""
    la, lb = len(a), len(b)
    if abs(la - lb) >= cap:
        return cap
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = cur[0]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
            best = min(best, cur[j])
        if best >= cap:
            return cap
        prev = cur
    return min(prev[lb], cap)


def stats(pages, tokenizer):
    n = exact = close = 0
    for _, toks in pages:
        seqs = [tokenizer(w) for w in toks]
        for i in range(1, len(seqs)):
            lo = max(0, i - W)
            d = min(edit(seqs[i], s) for s in seqs[lo:i])
            n += 1
            exact += d == 0
            close += d <= 1
    return exact / n, close / n


def shuffled(pages, rounds=3, seed=1):
    rng = random.Random(seed)
    out = []
    for r in range(rounds):
        out.append([(p, rng.sample(toks, len(toks))) for p, toks in pages])
    return out


def lag_curve(pages, tokenizer, maxlag=40):
    """P(exact repeat at lag k)."""
    from collections import Counter
    hit, tot = Counter(), Counter()
    for _, toks in pages:
        for i, w in enumerate(toks):
            for k in range(1, min(maxlag, i) + 1):
                tot[k] += 1
                hit[k] += toks[i - k] == w
    return {k: hit[k] / tot[k] for k in tot}


def report(name, pages, tokenizer):
    ex, cl = stats(pages, tokenizer)
    exs = cls = 0
    rounds = shuffled(pages)
    for sp in rounds:
        e, c = stats(sp, tokenizer)
        exs += e / len(rounds)
        cls += c / len(rounds)
    print(f"{name}")
    print(f"  exact copy in prev {W} words:  real {ex:6.1%}   "
          f"shuffled {exs:6.1%}   ratio {ex/exs:4.2f}")
    print(f"  within 1 edit of prev {W}:     real {cl:6.1%}   "
          f"shuffled {cls:6.1%}   ratio {cl/cls:4.2f}")
    return ex, cl


def main():
    voy = voynich_page_tokens()
    eng = culpeper_page_tokens()
    print(f"tokens: voynich {sum(len(t) for _, t in voy)}, "
          f"control {sum(len(t) for _, t in eng)}\n")
    report("VOYNICH (glyph edit distance)", voy, glyphs)
    print()
    report("CONTROL herbal English (char edit distance)", eng, list)

    print("\nP(exact repeat) by distance k (voynich, real vs expectation):")
    curve = lag_curve(voy, glyphs)
    base = sum(curve[k] for k in range(30, 41)) / 11
    for k in (1, 2, 3, 5, 10, 20, 40):
        print(f"  k={k:<3} {curve[k]:6.2%}   (far baseline ~{base:.2%})")


if __name__ == "__main__":
    main()
