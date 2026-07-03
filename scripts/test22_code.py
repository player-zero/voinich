"""Test 22 — the third pole: source code through the same battery.

Code is MEANINGFUL but not a natural language. If the manuscript were a
formal notation carrying content, it should behave like code: rigid
syntax (word MI even higher than language), massive boilerplate
formulas, and keyword anchors spread evenly.

Corpus: verbose C++ (nlohmann/json.hpp + Catch2 single headers),
tokenized the same way as the other corpora ([A-Za-z]+ runs,
lowercased).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_page_tokens, culpeper_tokens
from morphology import glyphs, norm_glyph
from test18_controls import latin_tokens
from test21_maori import (maori_grey_tokens, maori_tokens, paginate,
                          profile)


def cpp_tokens():
    txt = Path("data/cpp1.txt").read_text(encoding="utf-8",
                                          errors="ignore") + \
        Path("data/cpp2.txt").read_text(encoding="utf-8",
                                        errors="ignore")
    return [w.lower() for w in re.findall(r"[A-Za-z]+", txt)]


def main():
    voy_pages = voynich_page_tokens()
    n = sum(len(t) for _, t in voy_pages)
    corpora = [
        ("Voynich", voy_pages,
         lambda w: [norm_glyph(g) for g in glyphs(w)]),
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


if __name__ == "__main__":
    main()
