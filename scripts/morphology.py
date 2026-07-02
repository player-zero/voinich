"""Word-structure analysis: tokenize Eva words into glyphs, measure how
strictly glyphs are ordered within words, and compare against an English
control text."""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words

# Glyph tokenizer: benched gallows and ch/sh are single glyphs; runs of e
# and a+i-run+final are treated as units the way Voynich researchers
# conventionally group them.
GLYPH = re.compile(r"cth|ckh|cph|cfh|ch|sh|e+|i+[nrlm]?|.")


def glyphs(word):
    return GLYPH.findall(word)


def norm_glyph(g):
    """Collapse run-length variants: ee/eee -> e+, iin/iiin -> in-group."""
    if set(g) <= {"e"}:
        return "E"          # e-run
    if g.startswith("i"):
        return "I" + (g.lstrip("i") or "-")   # In, Ir, Il, Im, I-
    return g


def mean_positions(wordlist):
    """Mean relative position (0=start, 1=end) of each glyph type."""
    pos = defaultdict(list)
    for w in wordlist:
        gs = [norm_glyph(g) for g in glyphs(w)]
        if len(gs) < 2:
            continue
        for i, g in enumerate(gs):
            pos[g].append(i / (len(gs) - 1))
    return {g: sum(v) / len(v) for g, v in pos.items()}, \
           {g: len(v) for g, v in pos.items()}


def monotone_share(wordlist, rank):
    """Share of tokens whose glyph sequence never decreases in rank.
    Glyphs missing from `rank` make a word non-conforming."""
    ok = tot = 0
    bad = Counter()
    for w in wordlist:
        gs = [norm_glyph(g) for g in glyphs(w)]
        tot += 1
        rs = [rank.get(g) for g in gs]
        if None in rs:
            bad[w] += 1
            continue
        if all(a <= b for a, b in zip(rs, rs[1:])):
            ok += 1
        else:
            bad[w] += 1
    return ok / tot, bad


def main():
    loci = parse("data/ZL3b-n.txt")
    voy = [w for l in loci for w in words(l.text)]

    # --- English control: words from the README prose ---
    eng_text = Path("data/000_README.txt").read_text(encoding="latin-1")
    eng = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", eng_text)]

    print("=== glyph inventory (Voynich) ===")
    mpos, cnt = mean_positions(voy)
    total = sum(cnt.values())
    print(f"{'glyph':<6}{'count':>7}  {'freq':>6}  mean-pos")
    for g in sorted(mpos, key=lambda g: mpos[g]):
        if cnt[g] < 30:
            continue
        bar = " " * int(mpos[g] * 40) + "*"
        print(f"{g:<6}{cnt[g]:>7}  {cnt[g]/total:>6.2%}  {mpos[g]:.2f} |{bar}")
    rare = {g for g in mpos if cnt[g] < 30}
    print(f"(hidden rare glyphs, <30 occurrences: {len(rare)}: "
          f"{' '.join(sorted(rare))})")

    # --- strict-order test ---
    # rank = mean position, ties broken arbitrarily; rare glyphs excluded
    common = [g for g in mpos if cnt[g] >= 30]
    order = sorted(common, key=lambda g: mpos[g])
    rank = {g: i for i, g in enumerate(order)}
    share, bad = monotone_share(voy, rank)
    print(f"\nVoynich: {share:.1%} of tokens follow the strict glyph order")
    print("most common violating words:")
    for w, c in bad.most_common(10):
        print(f"  {w:<12} {c}")

    # same procedure, blind, on English
    mpos_e, cnt_e = mean_positions(eng)
    common_e = [g for g in mpos_e if cnt_e[g] >= 10]
    rank_e = {g: i for i, g in
              enumerate(sorted(common_e, key=lambda g: mpos_e[g]))}
    share_e, _ = monotone_share(eng, rank_e)
    print(f"English control ({len(eng)} tokens): "
          f"{share_e:.1%} follow their best strict order")

    # --- slot templates: map words to class strings, count coverage ---
    CLASS = {}
    for g, c in [("q", "Q"), ("o", "O"), ("y", "Y"), ("a", "A"),
                 ("ch", "C"), ("sh", "C"),
                 ("cth", "G"), ("ckh", "G"), ("cph", "G"), ("cfh", "G"),
                 ("k", "K"), ("t", "K"), ("p", "K"), ("f", "K"),
                 ("E", "E"), ("d", "D"), ("l", "L"), ("r", "R"),
                 ("s", "S"), ("n", "N"), ("m", "M"), ("x", "X"),
                 ("In", "J"), ("Ir", "J"), ("Il", "J"), ("Im", "J"),
                 ("I-", "J")]:
        CLASS[g] = c
    tmpl = Counter()
    for w in voy:
        gs = [norm_glyph(g) for g in glyphs(w)]
        try:
            tmpl["".join(CLASS[g] for g in gs)] += 1
        except KeyError:
            tmpl["<rare>"] += 1
    n = len(voy)
    print(f"\n=== top 25 word templates (classes: Q=q O=o Y=y A=a "
          f"C=ch/sh G=benched-gallows K=k/t/p/f E=e-run D=d L=l R=r S=s "
          f"J=i-group) ===")
    cum = 0
    for t, c in tmpl.most_common(25):
        cum += c
        ex = [w for w in list(dict.fromkeys(voy))
              if "".join(CLASS.get(norm_glyph(g), "?")
                         for g in glyphs(w)) == t][:3]
        print(f"  {t:<10} {c:>5} {c/n:>6.2%}  cum {cum/n:>6.1%}   "
              f"e.g. {', '.join(ex)}")
    print(f"distinct templates: {len(tmpl)}")


if __name__ == "__main__":
    main()
