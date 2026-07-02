"""Test 15 — do the manuscript's sections replicate the global profile?

Run the arbitrage battery + the qo-rule independently on every section
with >=2000 running-text tokens. Generator verdict predicts: same
qualitative profile everywhere (one machine, drifting parameters).
A section behaving like language would refute it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words, SECTIONS
from battery import run_battery, flatten_pages


def section_structs():
    by_sec = {}
    for l in parse("data/ZL3b-n.txt"):
        if l.ltype[1] != "P":
            continue
        ws = words(l.text)
        if not ws:
            continue
        sec = SECTIONS.get(l.page_vars.get("I"), "?")
        pages = by_sec.setdefault(sec, {})
        if l.page not in pages:
            pages[l.page] = []
        page = pages[l.page]
        if "<%>" in l.text or not page:
            page.append([ws])
        else:
            page[-1].append(ws)
    return {sec: [(p, pars) for p, pars in pages.items()]
            for sec, pages in by_sec.items()}


def qo_rule(struct):
    hy = ty = hc = tc = 0
    for _, t in flatten_pages(struct):
        for a, b in zip(t, t[1:]):
            if a.endswith("y"):
                ty += 1
                hy += b.startswith("qo")
            elif a[-1:] in "nlrsm":
                tc += 1
                hc += b.startswith("qo")
    return (hy / ty) / (hc / tc) if ty and tc and hc else float("nan")


def main():
    structs = section_structs()
    rows = []
    for sec, st in sorted(structs.items(),
                          key=lambda kv: -sum(len(t) for _, t in
                                              flatten_pages(kv[1]))):
        n = sum(len(t) for _, t in flatten_pages(st))
        if n < 2000:
            continue
        m = run_battery(st, eva=True)
        m["qo-pull"] = qo_rule(st)
        m["tokens"] = n
        rows.append((sec, m))

    metrics = ["tokens", "hapax%", "zipf", "H2", "gram%", "MIexc",
               "rep", "selfsim", "posZ-start", "posZ-end", "qo-pull"]
    fmt = {"tokens": "{:.0f}", "hapax%": "{:.1f}", "zipf": "{:.2f}",
           "H2": "{:.2f}", "gram%": "{:.1f}", "MIexc": "{:.2f}",
           "rep": "{:.2f}", "selfsim": "{:.2f}", "posZ-start": "{:.1f}",
           "posZ-end": "{:.1f}", "qo-pull": "{:.1f}"}
    print(f"{'metric':<11}" + "".join(f"{s[:13]:>14}" for s, _ in rows))
    for m in metrics:
        print(f"{m:<11}" + "".join(
            f"{fmt[m].format(r[m]):>14}" for _, r in rows))


if __name__ == "__main__":
    main()
