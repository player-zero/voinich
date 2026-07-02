"""Parser for IVTFF 2.0 transliteration files (voynich.nu).

Yields loci as (page, page_vars, locus_id, locus_type, text) and provides
cleaning/tokenization helpers for basic Eva text.
"""

import re
from dataclasses import dataclass, field


PAGE_HEADER = re.compile(r"^<(f\S+?)>\s*(?:<!\s*(.*?)>)?\s*$")
LOCUS = re.compile(r"^<(f\S+?)\.(\S+?),(\S+?)>\s*(.*)$")
VAR = re.compile(r"\$(\w)=(\S+)")


@dataclass
class Locus:
    page: str
    num: str
    ltype: str  # e.g. @P0, +P0, =Pt, *P0, @L0 ...
    text: str
    page_vars: dict = field(default_factory=dict)


def parse(path):
    """Yield Locus objects from an IVTFF file. Continuation lines (ending
    with '/') are joined."""
    loci = []
    cur_vars = {}
    cur_page = None
    pending = None  # (page, num, ltype, text) awaiting continuation

    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                continue
            m = PAGE_HEADER.match(line)
            if m and "." not in m.group(1):
                cur_page = m.group(1)
                cur_vars = dict(VAR.findall(m.group(2) or ""))
                continue
            m = LOCUS.match(line)
            if m:
                if pending:
                    loci.append(Locus(*pending[:4], page_vars=pending[4]))
                page, num, ltype, text = m.groups()
                if text.endswith("/"):
                    pending = [page, num, ltype, text[:-1], cur_vars]
                else:
                    loci.append(Locus(page, num, ltype, text, cur_vars))
                    pending = None
            elif pending is not None and line:
                # continuation line (starts after a trailing '/')
                cont = line.lstrip()
                if cont.startswith("/"):
                    cont = cont[1:]
                if cont.endswith("/"):
                    pending[3] += cont[:-1]
                else:
                    pending[3] += cont
                    loci.append(Locus(*pending[:4], page_vars=pending[4]))
                    pending = None
    if pending:
        loci.append(Locus(*pending[:4], page_vars=pending[4]))
    return loci


INLINE_COMMENT = re.compile(r"<![^>]*>")
MARKUP = re.compile(r"<[^>]*>")  # <%>, <$>, <->, <~> etc.
HIGH_ASCII = re.compile(r"@\d+;")
ALTERNATE = re.compile(r"\[([^:\]]*):[^\]]*\]")  # [a:b] -> a (first reading)


def clean(text):
    """Strip IVTFF markup down to plain Eva with '.' word separators.

    - inline comments/markup removed
    - alternate readings: first option kept
    - ligature braces {} dropped, ' (half-space) dropped
    - high-ascii codes and illegible '?' become '?'
    - ',' (uncertain space) normalized to '.'
    """
    t = INLINE_COMMENT.sub("", text)
    t = ALTERNATE.sub(r"\1", t)
    t = MARKUP.sub("", t)
    t = HIGH_ASCII.sub("?", t)
    t = t.replace("{", "").replace("}", "").replace("'", "")
    t = t.replace(",", ".")
    t = re.sub(r"[!%=\-~\s]", "", t)
    t = re.sub(r"\.+", ".", t).strip(".")
    return t


def words(text, drop_uncertain=True):
    ws = [w for w in clean(text).split(".") if w]
    if drop_uncertain:
        ws = [w for w in ws if "?" not in w]
    return ws


SECTIONS = {
    "H": "herbal", "A": "astro", "Z": "zodiac", "B": "biological",
    "C": "cosmological", "P": "pharmaceutical", "S": "stars/recipes",
    "T": "text-only",
}
