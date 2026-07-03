"""Test 21 — the small-alphabet showdown: run Maori through the SAME
battery as the Voynich text. If 'Voynich is a Maori-type language in an
unknown script', Maori must match it not just on H2 but on word MI,
burstiness anchors, formulas, and the slot-machine signature.

Corpora: Voynich (glyphs), English herbal, Latin (Caesar), Maori
(Grey's Nga Mahinga 1854 + Hinemoa, charset-filtered OCR).
"""

import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens, culpeper_tokens
from morphology import glyphs, norm_glyph
from battery import zipf_slope, h2_char, mi_excess, repeat_ratio, \
    selfsim_ratio
from test16_burstiness import heaps_beta, burstiness
from test13_falsify import ngram_repeats
from test18_controls import latin_tokens, maori_tokens, gutenberg_body

MAORI_LEGAL = set("aeiouhkmnprtwg")


def maori_grey_tokens():
    txt = Path("data/grey_maori.txt").read_text(encoding="utf-8",
                                                errors="ignore")
    out = []
    for line in txt.splitlines():
        ws = [w.lower() for w in re.findall(r"[A-Za-z]+", line)]
        if len(ws) < 4:
            continue
        legal = [w for w in ws if set(w) <= MAORI_LEGAL]
        if len(legal) >= 0.8 * len(ws):
            out.extend(legal)
    return out


def paginate(tokens, size=170):
    return [(i, tokens[i:i + size]) for i in range(0, len(tokens), size)]


def monotone_share(wordlist, tok):
    pos = defaultdict(list)
    for w in wordlist:
        gs = tok(w)
        if len(gs) < 2:
            continue
        for i, g in enumerate(gs):
            pos[g].append(i / (len(gs) - 1))
    cnt = {g: len(v) for g, v in pos.items()}
    mean = {g: sum(v) / len(v) for g, v in pos.items()}
    rank = {g: r for r, g in enumerate(
        sorted((g for g in mean if cnt[g] >= 30), key=lambda g: mean[g]))}
    ok = tot = 0
    for w in wordlist:
        gs = tok(w)
        tot += 1
        rs = [rank.get(g) for g in gs]
        if None in rs:
            continue
        ok += all(a <= b for a, b in zip(rs, rs[1:]))
    return ok / tot


def cross_mi(pages, rng, rounds=3):
    def cm(pgs):
        pairs = Counter()
        for _, t in pgs:
            for a, b in zip(t, t[1:]):
                pairs[(a[-1], b[0])] += 1
        n = sum(pairs.values())
        l, r = Counter(), Counter()
        for (a, b), c in pairs.items():
            l[a] += c
            r[b] += c
        return sum(c / n * math.log2((c / n) / (l[a] / n * r[b] / n))
                   for (a, b), c in pairs.items())
    real = cm(pages)
    null = sum(cm([(p, rng.sample(t, len(t))) for p, t in pages])
               for _ in range(rounds)) / rounds
    return real - null


def profile(name, pages, tok):
    rng = random.Random(11)
    toks = [w for _, t in pages for w in t]
    vocab = Counter(toks)
    b = burstiness(toks)
    cvs = sorted(cv for _, cv in b.values())
    top_band = [cv for w, (c, cv) in b.items() if c > 100]
    m = {
        "tokens": len(toks),
        "types": len(vocab),
        "hapax%": 100 * sum(1 for c in vocab.values() if c == 1)
        / len(vocab),
        "zipf": zipf_slope(vocab),
        "heaps": heaps_beta(toks),
        "H2": h2_char(toks),
        "MIexc": mi_excess(pages, rng),
        "rep": repeat_ratio(pages, rng),
        "selfsim": selfsim_ratio(pages, rng),
        "medCV": cvs[len(cvs) // 2] if cvs else float("nan"),
        "topCV": (sorted(top_band)[len(top_band) // 2]
                  if top_band else float("nan")),
        "mono%": 100 * monotone_share(toks, tok),
        "xMI": cross_mi(pages, rng),
        "3grams": len(ngram_repeats(pages, 3)),
        "4grams": len(ngram_repeats(pages, 4)),
    }
    return m


def main():
    voy_pages = voynich_page_tokens()
    n_voy = sum(len(t) for _, t in voy_pages)
    maori = (maori_grey_tokens() + maori_tokens())[:n_voy]
    corpora = [
        ("Voynich", voy_pages,
         lambda w: [norm_glyph(g) for g in glyphs(w)]),
        ("Maori", paginate(maori), list),
        ("Latin", paginate(latin_tokens()[:n_voy]), list),
        ("English", paginate(culpeper_tokens(limit=n_voy)), list),
    ]
    res = {}
    for name, pages, tok in corpora:
        res[name] = profile(name, pages, tok)
        print(f"done {name}: {res[name]['tokens']} tokens",
              file=sys.stderr)

    metrics = ["tokens", "types", "hapax%", "zipf", "heaps", "H2",
               "MIexc", "rep", "selfsim", "medCV", "topCV", "mono%",
               "xMI", "3grams", "4grams"]
    fmt = {"tokens": "{:.0f}", "types": "{:.0f}", "hapax%": "{:.1f}",
           "zipf": "{:.2f}", "heaps": "{:.2f}", "H2": "{:.2f}",
           "MIexc": "{:.2f}", "rep": "{:.2f}", "selfsim": "{:.2f}",
           "medCV": "{:.2f}", "topCV": "{:.2f}", "mono%": "{:.0f}",
           "xMI": "{:.3f}", "3grams": "{:.0f}", "4grams": "{:.0f}"}
    print(f"{'metric':<9}" + "".join(f"{n:>11}" for n, _, _ in corpora))
    for m in metrics:
        print(f"{m:<9}" + "".join(
            f"{fmt[m].format(res[n][m]):>11}" for n, _, _ in corpora))
    print("\nsample of Maori corpus:", " ".join(maori[100:115]))


if __name__ == "__main__":
    main()
