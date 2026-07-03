"""Test 23 — the conlang door: Neo-Quenya through the same battery.

Quenya is an INVENTED language (invented phonology, grammar, lexicon) —
exactly what the 'Voynichese is an artificial language' hypotheses
propose. But our corpus (Fauskanger's Neo-Quenya NT translation)
CARRIES MEANING. Prediction: constructedness does not erase the
information signature — only meaninglessness does.

Corpus: 4 gospels + Acts, diacritics folded, tokenized like the rest.
"""

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens, culpeper_tokens
from morphology import glyphs, norm_glyph
from test18_controls import latin_tokens
from test21_maori import (maori_grey_tokens, maori_tokens, paginate,
                          profile)
from test22_code import cpp_tokens

ENG_STOP = {"the", "of", "and", "to", "is", "that", "with", "for",
            "revision", "translation", "notes"}


def quenya_tokens():
    out = []
    for p in sorted(Path("data").glob("q_*.txt")):
        for line in p.read_text(encoding="utf-8",
                                errors="ignore").splitlines():
            ws = [w.lower() for w in re.findall(r"[A-Za-zÀ-ſ]+",
                                                line)]
            if not ws:
                continue
            if sum(w in ENG_STOP for w in ws) >= 2:
                continue
            for w in ws:
                w = unicodedata.normalize("NFD", w)
                w = "".join(c for c in w if not unicodedata.combining(c))
                if w.isalpha():
                    out.append(w)
    return out


def main():
    voy_pages = voynich_page_tokens()
    n = sum(len(t) for _, t in voy_pages)
    corpora = [
        ("Voynich", voy_pages,
         lambda w: [norm_glyph(g) for g in glyphs(w)]),
        ("Quenya", paginate(quenya_tokens()[:n]), list),
        ("C++", paginate(cpp_tokens()[:n]), list),
        ("Maori", paginate((maori_grey_tokens()
                            + maori_tokens())[:n]), list),
        ("Latin", paginate(latin_tokens()[:n]), list),
        ("English", paginate(culpeper_tokens(limit=n)), list),
    ]
    res = {}
    for name, pages, tok in corpora:
        res[name] = profile(name, pages, tok)
        print(f"done {name}", file=sys.stderr)

    metrics = ["types", "hapax%", "zipf", "heaps", "H2", "MIexc",
               "rep", "selfsim", "medCV", "topCV", "mono%", "xMI",
               "3grams", "4grams"]
    fmt = {"types": "{:.0f}", "hapax%": "{:.1f}", "zipf": "{:.2f}",
           "heaps": "{:.2f}", "H2": "{:.2f}", "MIexc": "{:.2f}",
           "rep": "{:.2f}", "selfsim": "{:.2f}", "medCV": "{:.2f}",
           "topCV": "{:.2f}", "mono%": "{:.0f}", "xMI": "{:.3f}",
           "3grams": "{:.0f}", "4grams": "{:.0f}"}
    print(f"{'metric':<9}" + "".join(f"{nm:>10}" for nm, _, _ in corpora))
    for m in metrics:
        print(f"{m:<9}" + "".join(
            f"{fmt[m].format(res[nm][m]):>10}" for nm, _, _ in corpora))
    q = quenya_tokens()
    print("\nsample:", " ".join(q[200:214]))


if __name__ == "__main__":
    main()
