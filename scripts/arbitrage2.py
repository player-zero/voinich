"""Final head-to-head: REAL vs the augmented copyist (M2v3), including
the cross-boundary glyph MI as an extra metric."""

import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from battery import run_battery, flatten_pages
from generators import skeleton_and_buckets, gen_m2v3


def cross_mi_excess(struct, rounds=5, seed=7):
    pages = flatten_pages(struct)
    rng = random.Random(seed)

    def cm(pgs):
        pairs = Counter()
        for _, t in pgs:
            for a, b in zip(t, t[1:]):
                pairs[(a[-1], "qo" if b.startswith("qo") else b[0])] += 1
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


def qo_after_y(struct):
    pages = flatten_pages(struct)
    hit = tot = hit_c = tot_c = 0
    for _, t in pages:
        for a, b in zip(t, t[1:]):
            if a.endswith("y"):
                tot += 1
                hit += b.startswith("qo")
            elif a[-1:] in "nlrsm":
                tot_c += 1
                hit_c += b.startswith("qo")
    return hit / tot, hit_c / tot_c


def main():
    struct, buckets = skeleton_and_buckets()
    m2v3 = gen_m2v3(struct, buckets)

    rows = [("REAL", struct), ("M2v3 copyist+flow", m2v3)]
    res = {}
    for name, st in rows:
        r = run_battery(st, eva=True)
        r["crossMI"] = cross_mi_excess(st)
        r["qo|y"], r["qo|cons"] = qo_after_y(st)
        res[name] = r

    metrics = ["types", "hapax%", "top1%", "zipf", "H2", "gram%",
               "MIexc", "crossMI", "qo|y", "qo|cons", "rep", "selfsim",
               "posZ-start", "posZ-end"]
    fmt = {"types": "{:.0f}", "hapax%": "{:.1f}", "top1%": "{:.2f}",
           "zipf": "{:.2f}", "H2": "{:.2f}", "gram%": "{:.1f}",
           "MIexc": "{:.2f}", "crossMI": "{:.3f}", "qo|y": "{:.1%}",
           "qo|cons": "{:.1%}", "rep": "{:.2f}", "selfsim": "{:.2f}",
           "posZ-start": "{:.1f}", "posZ-end": "{:.1f}"}
    print(f"{'metric':<12}" + "".join(f"{n:>20}" for n, _ in rows))
    for m in metrics:
        print(f"{m:<12}" + "".join(
            f"{fmt[m].format(res[n][m]):>20}" for n, _ in rows))


if __name__ == "__main__":
    main()
