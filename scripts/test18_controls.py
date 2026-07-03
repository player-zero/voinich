"""Test 18 — patch the language yardsticks.

Our H2 and word-MI baselines rested on ONE control (printed English,
rigid word order). Add: classical Latin (free word order) and Maori
(tiny alphabet, strict CV phonotactics — the 'Hawaiian-type' scenario
that could in principle produce low character entropy).
"""

import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens, culpeper_tokens
from test2_context import analyse

MAORI_LEGAL = set("aeiouhkmnprtwg")


def gutenberg_body(path):
    t = Path(path).read_text(encoding="utf-8", errors="ignore")
    a = t.find("*** START")
    b = t.find("*** END")
    return t[a:b] if a >= 0 and b > a else t


def latin_tokens():
    txt = gutenberg_body("data/latin1.txt") + \
        gutenberg_body("data/latin2.txt")
    # drop bracketed editorial notes and numbers
    txt = re.sub(r"\[[^\]]*\]", " ", txt)
    toks = [w.lower() for w in re.findall(r"[A-Za-z]+", txt)]
    return toks


def maori_tokens():
    """Keep lines that are overwhelmingly Maori-legal words."""
    txt = gutenberg_body("data/maori_raw.txt")
    txt = txt.replace("ā", "a").replace("ē", "e") \
             .replace("ī", "i").replace("ō", "o") \
             .replace("ū", "u")
    out = []
    for line in txt.splitlines():
        ws = [w.lower() for w in re.findall(r"[A-Za-z]+", line)]
        if len(ws) < 4:
            continue
        legal = [w for w in ws if set(w) <= MAORI_LEGAL]
        if len(legal) >= 0.9 * len(ws):
            out.extend(legal)
    return out


def h1_h2(tokens):
    text = ".".join(tokens)
    uni = Counter(text)
    bi = Counter(zip(text, text[1:]))
    n = sum(uni.values())
    h1 = -sum(c / n * math.log2(c / n) for c in uni.values())
    first = Counter()
    for (a, _), c in bi.items():
        first[a] += c
    nb = sum(bi.values())
    h2 = -sum(c / nb * math.log2(c / first[a]) for (a, _), c in bi.items())
    return h1, h2, len([g for g in uni if g != "."])


def mi_excess(tokens, seed=5, page=170, rounds=3, topk=150):
    pages = [(i, tokens[i:i + page])
             for i in range(0, len(tokens), page)]
    rng = random.Random(seed)

    def mi(pgs):
        vocab = Counter(w for _, t in pgs for w in t)
        top = {w for w, _ in vocab.most_common(topk)}
        pairs = Counter()
        for _, t in pgs:
            pairs.update((a, b) for a, b in zip(t, t[1:])
                         if a in top and b in top)
        n = sum(pairs.values())
        l, r = Counter(), Counter()
        for (a, b), c in pairs.items():
            l[a] += c
            r[b] += c
        return sum(c / n * math.log2((c / n) / (l[a] / n * r[b] / n))
                   for (a, b), c in pairs.items())

    real = mi(pages)
    null = sum(mi([(p, rng.sample(t, len(t))) for p, t in pages])
               for _ in range(rounds)) / rounds
    return real - null


def main():
    voy = [w for _, t in voynich_page_tokens() for w in t]
    corpora = [
        ("Voynich (glyph text)", voy),
        ("English (Culpeper)", culpeper_tokens(limit=len(voy))),
        ("Latin (Caesar)", latin_tokens()[:len(voy)]),
        ("Maori (Hinemoa)", maori_tokens()),
    ]
    print(f"{'corpus':<22}{'tokens':>8}{'alphabet':>9}"
          f"{'H1':>7}{'H2':>7}{'word-MI exc':>13}")
    for name, toks in corpora:
        h1, h2, alpha = h1_h2(toks)
        mie = mi_excess(toks) if len(toks) > 3000 else float("nan")
        print(f"{name:<22}{len(toks):>8}{alpha:>9}"
              f"{h1:>7.2f}{h2:>7.2f}{mie:>13.2f}")
    print("\nsample Maori kept:", " ".join(maori_tokens()[:12]))


if __name__ == "__main__":
    main()
