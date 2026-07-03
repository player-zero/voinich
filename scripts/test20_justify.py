"""Test 20 — are the line-edge effects scribal JUSTIFICATION (fitting
words to the physical margin) rather than positional ritual?

If the scribe adapts the last word to remaining space, then across
lines with the same word count, the last word's length should
anti-correlate with the length of everything before it (fuller line ->
shorter last word). Same logic for the -m flourish: justification
predicts it depends on line fullness; ritual predicts it doesn't.

Caveat: we have no physical widths — glyph count is the proxy.
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_pages
from morphology import glyphs


def lines_all():
    out = []
    for _, pars in voynich_pages():
        for par in pars:
            out.extend(par)
    return out


def pearson(xy):
    n = len(xy)
    mx = sum(x for x, _ in xy) / n
    my = sum(y for _, y in xy) / n
    sx = (sum((x - mx) ** 2 for x, _ in xy)) ** 0.5
    sy = (sum((y - my) ** 2 for _, y in xy)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in xy) / (sx * sy)


def corr_last_vs_rest(lines, rng=None, target="last"):
    """within same word-count groups; optional within-group shuffle of
    the target values as null."""
    groups = defaultdict(list)
    for line in lines:
        if len(line) < 4:
            continue
        gl = [len(glyphs(w)) for w in line]
        if target == "last":
            rest, tgt = sum(gl[:-1]), gl[-1]
        else:  # control: a middle word
            mid = len(gl) // 2
            rest, tgt = sum(gl) - gl[mid], gl[mid]
        groups[len(line)].append((rest, tgt))
    tot = sum(len(v) for v in groups.values())
    r = 0.0
    for n, xy in groups.items():
        if len(xy) < 30:
            continue
        if rng is not None:
            ys = [y for _, y in xy]
            rng.shuffle(ys)
            xy = [(x, y) for (x, _), y in zip(xy, ys)]
        r += len(xy) / tot * pearson(xy)
    return r


def m_vs_fullness(lines):
    """P(last word ends in m) by tercile of line fullness (total glyphs
    of the line before the last word, within same word count)."""
    groups = defaultdict(list)
    for line in lines:
        if len(line) < 4:
            continue
        gl = [len(glyphs(w)) for w in line]
        groups[len(line)].append((sum(gl[:-1]), line[-1].endswith("m")))
    terc = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for n, rows in groups.items():
        if len(rows) < 30:
            continue
        rows.sort(key=lambda r: r[0])
        k = len(rows) // 3
        for i, (_, is_m) in enumerate(rows):
            t = min(2, i // k) if k else 1
            terc[t][0] += is_m
            terc[t][1] += 1
    return {t: (h / n if n else 0, n) for t, (h, n) in terc.items()}


def main():
    lines = lines_all()
    rng = random.Random(4)

    r_last = corr_last_vs_rest(lines)
    nulls = [corr_last_vs_rest(lines, rng=random.Random(100 + i))
             for i in range(10)]
    mu = sum(nulls) / len(nulls)
    sd = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** 0.5 or 1e-9
    r_mid = corr_last_vs_rest(lines, target="mid")
    print("corr(length of last word, total length of the rest of line),")
    print("within same word-count groups:")
    print(f"  last word:  r = {r_last:+.3f}   "
          f"(shuffle null {mu:+.3f}±{sd:.3f}, z={(r_last-mu)/sd:+.1f})")
    print(f"  mid word (control): r = {r_mid:+.3f}")

    print("\nP(last word ends in -m) by line fullness tercile:")
    for t, (p, n) in m_vs_fullness(lines).items():
        lab = ["short lines", "medium", "full lines"][t]
        print(f"  {lab:<12} {p:6.2%}  (n={n})")


if __name__ == "__main__":
    main()
