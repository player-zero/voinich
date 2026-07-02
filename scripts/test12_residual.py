"""Test 12 — dissect the 0.31-bit adjacent-word MI residual.

Ladder of increasingly constrained shuffles (each preserves one more
known factor). The excess that survives every rung is the truly
unexplained word-pair structure.

  N0: shuffle words within page                     -> E0 (=0.31)
  N1: + preserve layout bucket (line role x pos)    -> E1
  N2: + preserve coarse word class                  -> E2
  pre-step: merge known split-words (s+aiin...)     -> repeat ladder
  side check: contribution of exact-repeat pairs (diagonal)
"""

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_pages
from test7_line import wclass


def token_table():
    """per page: list of (word, bucket, class)."""
    pages = []
    for p, pars in voynich_pages():
        rows = []
        for li_par, par in enumerate(pars):
            for li, line in enumerate(par):
                role = "pf" if li == 0 else "ot"
                for wi, w in enumerate(line):
                    pos = ("s" if wi == 0 else
                           "e" if wi == len(line) - 1 else "m")
                    rows.append((w, role + pos, wclass(w)))
        pages.append((p, rows))
    return pages


def merge_splits(pages):
    """Merge adjacent pairs with PMI>1, n>=5, concat attested."""
    toks = [r[0] for _, rows in pages for r in rows]
    vocab = Counter(toks)
    pairs = Counter()
    for _, rows in pages:
        ws = [r[0] for r in rows]
        pairs.update(zip(ws, ws[1:]))
    n = sum(pairs.values())
    nt = len(toks)
    merge = set()
    for (a, b), c in pairs.items():
        if c < 5:
            continue
        pmi = math.log2((c / n) / (vocab[a] / nt * vocab[b] / nt))
        if pmi > 1 and a + b in vocab:
            merge.add((a, b))
    out = []
    n_merged = 0
    for p, rows in pages:
        nrows = []
        i = 0
        while i < len(rows):
            if (i + 1 < len(rows)
                    and (rows[i][0], rows[i + 1][0]) in merge):
                w = rows[i][0] + rows[i + 1][0]
                nrows.append((w, rows[i][1], wclass(w)))
                n_merged += 1
                i += 2
            else:
                nrows.append(rows[i])
                i += 1
        out.append((p, nrows))
    return out, n_merged


def mi_top(pages_words, topk=150, drop_diag=False):
    vocab = Counter(w for ws in pages_words for w in ws)
    top = {w for w, _ in vocab.most_common(topk)}
    pairs = Counter()
    for ws in pages_words:
        pairs.update((a, b) for a, b in zip(ws, ws[1:])
                     if a in top and b in top)
    if drop_diag:
        pairs = Counter({k: c for k, c in pairs.items() if k[0] != k[1]})
    n = sum(pairs.values())
    l, r = Counter(), Counter()
    for (a, b), c in pairs.items():
        l[a] += c
        r[b] += c
    return sum(c / n * math.log2((c / n) / (l[a] / n * r[b] / n))
               for (a, b), c in pairs.items())


def excess(pages, keyfn, rng, rounds=5, drop_diag=False):
    """MI(real) - MI(shuffle constrained by keyfn(row))."""
    real = mi_top([[r[0] for r in rows] for _, rows in pages],
                  drop_diag=drop_diag)
    null = []
    for _ in range(rounds):
        shuffled = []
        for p, rows in pages:
            groups = defaultdict(list)
            for i, row in enumerate(rows):
                groups[keyfn(row)].append(row[0])
            for g in groups.values():
                rng.shuffle(g)
            it = {k: iter(v) for k, v in groups.items()}
            shuffled.append([next(it[keyfn(row)]) for row in rows])
        null.append(mi_top(shuffled, drop_diag=drop_diag))
    return real - sum(null) / len(null)


def ladder(pages, rng, label):
    e0 = excess(pages, lambda r: 0, rng)
    e1 = excess(pages, lambda r: r[1], rng)
    e2 = excess(pages, lambda r: (r[1], r[2]), rng)
    print(f"{label}")
    print(f"  E0 (vs free shuffle):            {e0:+.3f} bits")
    print(f"  E1 (+layout preserved):          {e1:+.3f}  "
          f"-> layout explains {e0-e1:+.3f}")
    print(f"  E2 (+word class preserved):      {e2:+.3f}  "
          f"-> classes explain {e1-e2:+.3f}")
    return e0, e2


def main():
    rng = random.Random(42)
    pages = token_table()

    e0, _ = ladder(pages, rng, "RAW corpus")

    e0d = excess(pages, lambda r: 0, rng, drop_diag=True)
    print(f"  E0 without exact-repeat pairs:   {e0d:+.3f}  "
          f"-> repeats explain {e0-e0d:+.3f}")

    merged, k = merge_splits(pages)
    print(f"\nafter merging split-words ({k} merges):")
    ladder(merged, rng, "MERGED corpus")

    # the surviving pairs: what does the hard residual look like?
    print("\nstrongest surviving pairs (merged corpus, class+layout "
          "unexplainable, PMI, n>=12):")
    toks = [r[0] for _, rows in merged for r in rows]
    vocab = Counter(toks)
    top = {w for w, _ in vocab.most_common(150)}
    pairs = Counter()
    for _, rows in merged:
        ws = [r[0] for r in rows]
        pairs.update((a, b) for a, b in zip(ws, ws[1:])
                     if a in top and b in top)
    n = sum(pairs.values())
    l, r = Counter(), Counter()
    for (a, b), c in pairs.items():
        l[a] += c
        r[b] += c
    scored = [(math.log2((c / n) / (l[a] / n * r[b] / n)), a, b, c)
              for (a, b), c in pairs.items() if c >= 12]
    scored.sort(reverse=True)
    for pmi, a, b, c in scored[:12]:
        print(f"  {a:>8} {b:<10} n={c:<4} pmi={pmi:+.2f}")


if __name__ == "__main__":
    main()
