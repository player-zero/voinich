"""The last duel: REAL vs copyist v4 (multi-scale memory) on the full
battery PLUS the fine-structure metrics (burstiness, Heaps, qo-flow)."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from battery import run_battery, flatten_pages
from generators import skeleton_and_buckets, gen_m2v4
from arbitrage2 import cross_mi_excess, qo_after_y
from test16_burstiness import heaps_beta, burstiness


def fine(struct):
    toks = [w for _, t in flatten_pages(struct) for w in t]
    b = burstiness(toks)
    cvs = sorted(cv for _, cv in b.values())
    return {"heaps": heaps_beta(toks),
            "medCV": cvs[len(cvs) // 2],
            "CV>1.3": 100 * sum(1 for cv in cvs if cv > 1.3) / len(cvs)}


def main():
    struct, buckets = skeleton_and_buckets()
    v4 = gen_m2v4(struct, buckets)
    rows = [("REAL", struct), ("M2v4", v4)]
    res = {}
    for name, st in rows:
        r = run_battery(st, eva=True)
        r["crossMI"] = cross_mi_excess(st)
        r["qo|y"], r["qo|cons"] = qo_after_y(st)
        r.update(fine(st))
        res[name] = r
    metrics = ["types", "hapax%", "top1%", "zipf", "H2", "gram%",
               "MIexc", "crossMI", "qo|y", "qo|cons", "rep", "selfsim",
               "posZ-start", "posZ-end", "heaps", "medCV", "CV>1.3"]
    fmt = {"types": "{:.0f}", "hapax%": "{:.1f}", "top1%": "{:.2f}",
           "zipf": "{:.2f}", "H2": "{:.2f}", "gram%": "{:.1f}",
           "MIexc": "{:.2f}", "crossMI": "{:.3f}", "qo|y": "{:.1%}",
           "qo|cons": "{:.1%}", "rep": "{:.2f}", "selfsim": "{:.2f}",
           "posZ-start": "{:.1f}", "posZ-end": "{:.1f}",
           "heaps": "{:.2f}", "medCV": "{:.2f}", "CV>1.3": "{:.0f}%"}
    print(f"{'metric':<12}" + "".join(f"{n:>14}" for n, _ in rows))
    for m in metrics:
        print(f"{m:<12}" + "".join(
            f"{fmt[m].format(res[n][m]):>14}" for n, _ in rows))


if __name__ == "__main__":
    main()
