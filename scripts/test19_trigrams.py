"""Test 19 — autopsy of the 166 cross-page 3-grams.

We waved them off as 'stock micro-sequences'. Check properly:
  - proper null: words shuffled within pages (chance level of cross-page
    trigram repeats), 5 rounds;
  - composition: are they chains of globally frequent words, or do they
    contain rare words (content-like)?
  - concentration: by section, and how often the same trigram repeats.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words, SECTIONS
from test13_falsify import ngram_repeats


def pages_with_sections():
    pages = []
    secs = {}
    for l in parse("data/ZL3b-n.txt"):
        if l.ltype[1] != "P":
            continue
        ws = words(l.text)
        if not ws:
            continue
        if not pages or pages[-1][0] != l.page:
            pages.append((l.page, []))
            secs[l.page] = SECTIONS.get(l.page_vars.get("I"), "?")
        pages[-1][1].extend(ws)
    return pages, secs


def main():
    pages, secs = pages_with_sections()
    vocab = Counter(w for _, t in pages for w in t)
    top20 = {w for w, _ in vocab.most_common(20)}
    top100 = {w for w, _ in vocab.most_common(100)}

    far = ngram_repeats(pages, 3)
    print(f"cross-page 3-gram repeat events: {len(far)}")

    # proper null: shuffle within pages
    rng = random.Random(9)
    null = []
    for _ in range(5):
        sh = [(p, rng.sample(t, len(t))) for p, t in pages]
        null.append(len(ngram_repeats(sh, 3)))
    mu = sum(null) / len(null)
    print(f"null (within-page shuffle, 5 rounds): mean {mu:.0f} "
          f"-> excess over chance: {len(far)-mu:.0f} "
          f"({len(far)/mu:.1f}x)")

    # composition
    toks = [w for g, _, _ in far for w in g]
    in20 = sum(w in top20 for w in toks) / len(toks)
    in100 = sum(w in top100 for w in toks) / len(toks)
    rare = sum(vocab[w] < 20 for w in toks) / len(toks)
    print(f"composition of repeated trigrams: {in20:.0%} top-20 words, "
          f"{in100:.0%} top-100, {rare:.0%} rare (freq<20)")

    # multiplicity and section spread
    types = Counter(g for g, _, _ in far)
    print(f"distinct trigram types: {len(types)}; "
          f"repeated 3+ times: {sum(1 for c in types.values() if c >= 2)}")
    by_sec = Counter()
    page_ids = [p for p, _ in pages]
    for g, p1, p2 in far:
        by_sec[secs[page_ids[p2]]] += 1
    tot_by_sec = Counter()
    for p, t in pages:
        tot_by_sec[secs[p]] += len(t)
    print("rate per 1000 tokens by section (of later occurrence):")
    for s, c in by_sec.most_common():
        print(f"  {s:<15} {1000*c/tot_by_sec[s]:6.2f}   (n={c})")

    print("\ntop repeated trigrams:")
    for g, c in types.most_common(10):
        fr = [vocab[w] for w in g]
        print(f"  {' '.join(g):<28} x{c+1}   word freqs {fr}")


if __name__ == "__main__":
    main()
