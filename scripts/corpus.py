"""Shared corpus loaders for the three blind tests.

Voynich: running text only (generic locus type P), grouped by page and
paragraph, labels/circles/radials excluded. Control: Culpeper's herbal
(1653), same token count as the Voynich corpus.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words


def voynich_pages():
    """[(page, [line1_words, line2_words, ...] per paragraph)] ->
    returns list of pages; each page is a list of paragraphs; each
    paragraph a list of lines; each line a list of words."""
    loci = parse("data/ZL3b-n.txt")
    pages = {}
    order = []
    for l in loci:
        if l.ltype[1] != "P":
            continue
        if l.page not in pages:
            pages[l.page] = []
            order.append(l.page)
        page = pages[l.page]
        new_par = "<%>" in l.text or not page
        ws = words(l.text)
        if not ws:
            continue
        if new_par:
            page.append([ws])
        else:
            page[-1].append(ws)
    return [(p, pages[p]) for p in order if pages[p]]


def voynich_page_tokens():
    """[(page, [w, w, ...])] flattened per page."""
    out = []
    for p, pars in voynich_pages():
        toks = [w for par in pars for line in par for w in line]
        out.append((p, toks))
    return out


def culpeper_tokens(limit=38000):
    txt = Path("data/culpeper.txt").read_text(encoding="utf-8",
                                              errors="ignore")
    start = txt.find("*** START")
    end = txt.find("*** END")
    txt = txt[start:end]
    toks = [w.lower() for w in re.findall(r"[A-Za-z]+", txt)]
    return toks[:limit]


def culpeper_page_tokens(limit=38000, page_size=170):
    """Chunk the control text into pseudo-pages comparable in size to
    Voynich pages (~170 running-text tokens per page on average)."""
    toks = culpeper_tokens(limit)
    return [(f"c{i}", toks[i:i + page_size])
            for i in range(0, len(toks), page_size)]
