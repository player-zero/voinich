"""Test 4 — identify what the positional glyphs are variants OF.

Method: distributional matching. If m is a line-final flourish of some
ending Z, then the stems of line-final -m words should look like the
stems of mid-line -Z words. Same logic for p/f (paragraph-first lines)
vs candidates elsewhere, and for line-initial y-.

All comparisons are ranked against several candidates including
implausible controls, so the data can refuse the hypothesis.
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_pages
from morphology import glyphs, norm_glyph


def tokens_with_position():
    """(word, line_role, word_pos) for every running-text token."""
    out = []
    for _, pars in voynich_pages():
        for par in pars:
            for li, line in enumerate(par):
                role = "first" if li == 0 else "other"
                for wi, w in enumerate(line):
                    pos = ("start" if wi == 0 else
                           "end" if wi == len(line) - 1 else "mid")
                    out.append((w, role, pos))
    return out


def cosine(c1, c2):
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    dot = sum(v * c2.get(k, 0) for k, v in c1.items())
    return dot / (n1 * n2) if n1 and n2 else 0.0


def part1_final_m(toks):
    print("=== part 1: what ending does line-final -m replace? ===")
    stems = {}
    for w, role, pos in toks:
        gs = [norm_glyph(g) for g in glyphs(w)]
        if len(gs) < 2:
            continue
        stem, fin = "".join(gs[:-1]), gs[-1]
        if fin in ("m", "Im") and pos == "end":
            stems.setdefault("m@end", Counter())[stem] += 1
        elif pos == "mid":
            stems.setdefault(fin, Counter())[stem] += 1
    target = stems["m@end"]
    print(f"line-final -m tokens: {sum(target.values())}, "
          f"distinct stems: {len(target)}")
    print(f"{'candidate':<10}{'cosine':>8}{'stem overlap':>14}")
    for cand in ["In", "Ir", "r", "l", "n", "y", "s", "d", "o"]:
        if cand not in stems:
            continue
        cov = sum(c for s, c in target.items() if s in stems[cand]) \
            / sum(target.values())
        print(f"  {cand:<8}{cosine(target, stems[cand]):>8.3f}{cov:>13.1%}")


def part2_pf(toks):
    print("\n=== part 2: what do p/f (paragraph-first lines) replace? ===")
    frames = {}
    for w, role, pos in toks:
        gs = [norm_glyph(g) for g in glyphs(w)]
        for target_set, key, want_role in (
                ({"p", "cph"}, "p@first", "first"),
                ({"f", "cfh"}, "f@first", "first"),
                ({"t", "cth"}, "t", "other"),
                ({"k", "ckh"}, "k", "other"),
                ({"d"}, "d", "other"),
                ({"ch"}, "ch", "other")):
            if role != want_role:
                continue
            hits = [i for i, g in enumerate(gs) if g in target_set]
            if len(hits) != 1:
                continue
            i = hits[0]
            benched = gs[i].startswith("c")
            frame = "".join(gs[:i]) + ("_B_" if benched else "_") \
                + "".join(gs[i + 1:])
            frames.setdefault(key, Counter())[frame] += 1
    for tgt in ("p@first", "f@first"):
        print(f"{tgt}: {sum(frames[tgt].values())} tokens")
        for cand in ("t", "k", "d", "ch"):
            cov = sum(c for fr, c in frames[tgt].items()
                      if fr in frames[cand]) / sum(frames[tgt].values())
            print(f"  {cand:<4} cosine {cosine(frames[tgt], frames[cand]):.3f}"
                  f"   frame overlap {cov:.1%}")


def part3_y(toks):
    print("\n=== part 3: line-initial y- — prefix or substitute? ===")
    vocab_mid = Counter(w for w, r, p in toks if p == "mid")
    y_start = Counter(w for w, r, p in toks
                      if p == "start" and w.startswith("y") and len(w) > 2)
    n = sum(y_start.values())
    # model A: y is prepended to a normal word
    strip_ok = sum(c for w, c in y_start.items() if w[1:] in vocab_mid)
    # model B: y substitutes another initial glyph
    print(f"line-initial y- tokens: {n}")
    print(f"  model A (y+word, rest attested mid-line): {strip_ok/n:.1%}")
    for repl in ("o", "d", "s", "q", "ch"):
        sub_ok = sum(c for w, c in y_start.items()
                     if repl + w[1:] in vocab_mid)
        print(f"  model B (y replaces {repl:<2}): image attested {sub_ok/n:.1%}")
    # baseline: random mid-line words with first glyph stripped
    mid_multi = [(w, c) for w, c in vocab_mid.items()
                 if len(glyphs(w)) > 2]
    base = sum(c for w, c in mid_multi
               if "".join(w[len(glyphs(w)[0]):]) in vocab_mid) \
        / sum(c for _, c in mid_multi)
    print(f"  baseline (any mid word, first glyph stripped): {base:.1%}")


def main():
    toks = tokens_with_position()
    part1_final_m(toks)
    part2_pf(toks)
    part3_y(toks)


if __name__ == "__main__":
    main()
