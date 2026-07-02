"""Test 6 — what are the p/f words?

(a) profile: where p/f sits in the word, hapax rate, twins under p->t;
(b) where they live: body vs paragraph-first line vs physical titles
    (Pt loci: short right-justified lines) vs section;
(c) are paragraph-first lines title-like: vocabulary sharing among
    first lines vs among body lines, cohesion with own paragraph.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words, SECTIONS
from morphology import glyphs, norm_glyph

PF = {"p", "f", "cph", "cfh"}
TK = {"t", "k", "cth", "ckh"}


def has_pf(w):
    return any(g in PF for g in glyphs(w))


def load():
    """loci -> list of dicts with page, section, kind, line words.
    kind: 'title' (Pt), 'par-first', 'body'."""
    out = []
    for l in parse("data/ZL3b-n.txt"):
        if l.ltype[1] != "P":
            continue
        ws = words(l.text)
        if not ws:
            continue
        kind = ("title" if l.ltype[2] == "t"
                else "par-first" if "<%>" in l.text
                else "body")
        out.append({"page": l.page, "sec": SECTIONS.get(
            l.page_vars.get("I"), "?"), "kind": kind, "ws": ws})
    return out


def part_a(lines):
    print("=== (a) profile of p/f words ===")
    all_w = [w for l in lines for w in l["ws"]]
    vocab = Counter(all_w)
    pf_types = {w for w in vocab if has_pf(w)}
    pf_tok = sum(vocab[w] for w in pf_types)
    print(f"p/f tokens: {pf_tok} ({pf_tok/len(all_w):.1%}), "
          f"types: {len(pf_types)}")
    hap = lambda ts: sum(1 for w in ts if vocab[w] == 1) / len(ts)
    tk_types = {w for w in vocab if any(g in TK for g in glyphs(w))
                and not has_pf(w)}
    print(f"hapax rate: p/f words {hap(pf_types):.1%}, "
          f"t/k words {hap(tk_types):.1%}, "
          f"all {sum(1 for w in vocab if vocab[w]==1)/len(vocab):.1%}")
    # position of the p/f glyph inside the word
    pos = Counter()
    for w in pf_types:
        gs = glyphs(w)
        i = next(i for i, g in enumerate(gs) if g in PF)
        pos["initial" if i == 0 else
            "after 1 glyph" if i == 1 else "deeper"] += vocab[w]
    tot = sum(pos.values())
    for k, v in pos.most_common():
        print(f"  p/f sits {k}: {v/tot:.1%}")
    print("top p-words with their p->t twins (freq):")
    top_pf = [w for w, _ in vocab.most_common() if has_pf(w)][:10]
    for w in top_pf:
        twin = w.replace("p", "t").replace("f", "k")
        print(f"  {w:<12} {vocab[w]:>4}   {twin:<12} {vocab.get(twin,0):>4}")


def part_b(lines):
    print("\n=== (b) where p/f words live ===")
    for kind in ("body", "par-first", "title"):
        ls = [l for l in lines if l["kind"] == kind]
        toks = [w for l in ls for w in l["ws"]]
        share = sum(has_pf(w) for w in toks) / len(toks)
        print(f"  {kind:<10} lines {len(ls):>5}  tokens {len(toks):>6}  "
              f"p/f-token share {share:6.2%}")
    print("  p/f share by section (running text):")
    for sec in sorted({l['sec'] for l in lines}):
        toks = [w for l in lines if l["sec"] == sec for w in l["ws"]]
        if len(toks) < 500:
            continue
        share = sum(has_pf(w) for w in toks) / len(toks)
        print(f"    {sec:<15} {share:6.2%}  (n={len(toks)})")


def part_c(lines, seed=13):
    print("\n=== (c) are paragraph-first lines title-like? ===")
    rng = random.Random(seed)
    first = [l["ws"] for l in lines if l["kind"] == "par-first"]
    body = [l["ws"] for l in lines if l["kind"] == "body"]

    def cross_line_sharing(group, rounds=30):
        """P(token's type occurs in another line of the same group);
        groups downsampled to equal size."""
        m = min(len(first), len(body))
        vals = []
        for _ in range(rounds):
            g = rng.sample(group, m)
            vocab_by_line = [set(ws) for ws in g]
            share = tot = 0
            for i, ws in enumerate(g):
                others = set().union(*(vocab_by_line[:i]
                                       + vocab_by_line[i + 1:]))
                for w in ws:
                    tot += 1
                    share += w in others
            vals.append(share / tot)
        return sum(vals) / len(vals)

    print(f"  token repeated in another line of same kind: "
          f"first-lines {cross_line_sharing(first):.1%}, "
          f"body-lines {cross_line_sharing(body):.1%}")

    # cohesion: does a paragraph's first line share vocab with ITS body
    # more than with a random other paragraph's body?
    pars = []
    cur = None
    for l in lines:
        if l["kind"] == "par-first":
            cur = {"first": l["ws"], "body": []}
            pars.append(cur)
        elif l["kind"] == "body" and cur is not None:
            cur["body"].extend(l["ws"])
    pars = [p for p in pars if len(p["body"]) >= 10]
    own = rnd = tot = 0
    for i, p in enumerate(pars):
        other = pars[rng.randrange(len(pars) - 1)]["body"]
        bset, oset = set(p["body"]), set(other)
        for w in p["first"]:
            tot += 1
            own += w in bset
            rnd += w in oset
    print(f"  first-line word found in own paragraph body: {own/tot:.1%}, "
          f"in random other paragraph: {rnd/tot:.1%}  "
          f"(paragraphs: {len(pars)})")


def main():
    lines = load()
    part_a(lines)
    part_b(lines)
    part_c(lines)


if __name__ == "__main__":
    main()
