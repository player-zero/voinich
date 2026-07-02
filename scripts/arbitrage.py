"""The showdown: run the uniform battery on the real manuscript and on
all candidate generators filling the same page skeleton."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from battery import run_battery
from generators import (skeleton_and_buckets, gen_m1, gen_m2, gen_m3,
                        gen_m4, gen_m5)

METRICS = ["types", "hapax%", "top1%", "zipf", "H2", "gram%",
           "MIexc", "rep", "selfsim", "posZ-start", "posZ-end"]


def main():
    struct, buckets = skeleton_and_buckets()

    corpora = [
        ("REAL", struct, True),
        ("M1 iid+modes", gen_m1(struct, buckets), True),
        ("M2 copyist", gen_m2(struct, buckets), True),
        ("M3 table", gen_m3(struct, buckets), True),
        ("M4 ciph-prose", gen_m4(struct, buckets, mode="prose"), True),
        ("M4 ciph-lists", gen_m4(struct, buckets, mode="lists", seed=44),
         True),
        ("M5 anagram", gen_m5(struct), False),
    ]

    results = {}
    for name, st, eva in corpora:
        results[name] = run_battery(st, eva=eva)
        print(f"done: {name}", file=sys.stderr)

    header = f"{'metric':<11}" + "".join(f"{n:>14}" for n, _, _ in corpora)
    print(header)
    print("-" * len(header))
    fmt = {"types": "{:.0f}", "hapax%": "{:.1f}", "top1%": "{:.2f}",
           "zipf": "{:.2f}", "H2": "{:.2f}", "gram%": "{:.1f}",
           "MIexc": "{:.2f}", "rep": "{:.2f}", "selfsim": "{:.2f}",
           "posZ-start": "{:.1f}", "posZ-end": "{:.1f}"}
    for m in METRICS:
        row = f"{m:<11}"
        for name, _, _ in corpora:
            row += f"{fmt[m].format(results[name][m]):>14}"
        print(row)


if __name__ == "__main__":
    main()
