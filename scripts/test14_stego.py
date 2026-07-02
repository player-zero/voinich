"""Test 14 — probe the steganographic thin-channel hypothesis.

Candidate 'free choice' channels: ch/sh, k/t, p-vs-t in paragraph-first
lines, i-run length (in/iin). For each, in manuscript order:

  capacity  — n * H(choice): the naive payload ceiling
  VIF       — per-page rate overdispersion (chi2/df); habit drifts,
              encrypted payload is homogeneous
  bursts    — P(choice repeats previous choice) vs independence
  context   — how much H(choice) drops given (prev glyph, prev choice,
              hand): habit is predictable, payload is not

Verdict per channel: 'habit-like' (structured) or 'payload-compatible'
(indistinguishable from fair noise), with residual capacity.
"""

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words
from morphology import glyphs


def h_bin(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def extract_channels():
    """Return {name: [(page, hand, prev_glyph, bit), ...]} in order."""
    ch = defaultdict(list)
    for l in parse("data/ZL3b-n.txt"):
        if l.ltype[1] != "P":
            continue
        page = l.page
        hand = l.page_vars.get("H", "?")
        par_first = "<%>" in l.text
        for w in words(l.text):
            gs = glyphs(w)
            for i, g in enumerate(gs):
                prev = gs[i - 1] if i else "<s>"
                if g == "ch":
                    ch["ch/sh"].append((page, hand, prev, 0))
                elif g == "sh":
                    ch["ch/sh"].append((page, hand, prev, 1))
                elif g == "k":
                    ch["k/t"].append((page, hand, prev, 0))
                elif g == "t":
                    ch["k/t"].append((page, hand, prev, 1))
                if par_first and g in ("t", "k", "p", "f"):
                    ch["pf-vs-tk@par1"].append(
                        (page, hand, prev, 1 if g in ("p", "f") else 0))
                if g.startswith("i") and g.rstrip("nrlm") != g:
                    ilen = len(g) - 1
                    ch["i-run(1vs2+)"].append(
                        (page, hand, prev, 0 if ilen == 1 else 1))
    return ch


def analyse(name, rows):
    n = len(rows)
    bits = [b for _, _, _, b in rows]
    p = sum(bits) / n
    H = h_bin(p)
    cap = n * H
    print(f"\n--- {name}: n={n}, P(1)={p:.3f}, "
          f"H={H:.3f} bit/use, raw capacity {cap/8:.0f} bytes ---")

    # per-page overdispersion
    by_page = defaultdict(list)
    for pg, _, _, b in rows:
        by_page[pg].append(b)
    chi = df = 0
    for pg, bs in by_page.items():
        m = len(bs)
        if m < 10:
            continue
        e = m * p
        v = m * p * (1 - p)
        chi += (sum(bs) - e) ** 2 / v
        df += 1
    vif = chi / df
    z_vif = (chi - df) / math.sqrt(2 * df)
    print(f"  page drift: VIF={vif:.2f} (z={z_vif:+.1f}) "
          f"{'<- HABIT DRIFT' if z_vif > 4 else '(homogeneous)'}")

    # burstiness within page
    same = tot = 0
    for pg, bs in by_page.items():
        for a, b in zip(bs, bs[1:]):
            tot += 1
            same += a == b
    exp = p * p + (1 - p) * (1 - p)
    obs = same / tot
    z_b = (same - tot * exp) / math.sqrt(tot * exp * (1 - exp))
    print(f"  burstiness: P(same as prev)={obs:.3f} vs {exp:.3f} "
          f"(z={z_b:+.1f}) {'<- STICKY' if z_b > 4 else ''}")

    # conditional entropy given context
    def cond_H(keyfn):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            prev_bit = rows[i - 1][3] if i else -1
            groups[keyfn(row, prev_bit)].append(row[3])
        h = 0.0
        for g in groups.values():
            if len(g) < 5:
                h += len(g) / n * H
                continue
            h += len(g) / n * h_bin(sum(g) / len(g))
        return h

    h_hand = cond_H(lambda r, pb: r[1])
    h_prev = cond_H(lambda r, pb: r[2])
    h_all = cond_H(lambda r, pb: (r[1], r[2], pb))
    print(f"  H given hand: {h_hand:.3f}  given prev glyph: {h_prev:.3f}  "
          f"given all+prev bit: {h_all:.3f}")
    resid = n * h_all
    print(f"  residual capacity after habit structure: "
          f"{resid/8:.0f} bytes ({100*h_all/H:.0f}% of raw)")
    return cap, resid


def main():
    channels = extract_channels()
    total_raw = total_res = 0.0
    for name in ("ch/sh", "k/t", "pf-vs-tk@par1", "i-run(1vs2+)"):
        cap, res = analyse(name, channels[name])
        total_raw += cap
        total_res += res
    print(f"\n=== TOTAL: raw {total_raw/8:.0f} bytes, "
          f"residual after structure {total_res/8:.0f} bytes ===")


if __name__ == "__main__":
    main()
