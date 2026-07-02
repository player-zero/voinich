"""Test 9 — how reliable are the spaces?

(a) ZL marks uncertain spaces with ',' (vs certain '.'). Compare corpus
    statistics with uncertain spaces treated as spaces vs as no-space.
    Do comma-joined tokens conform to the slot grammar?
(b) On certain spaces: adjacent pairs with high PMI whose concatenation
    is an attested word = split-word candidates. How much text is that?
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ivtff
from ivtff import parse, INLINE_COMMENT, ALTERNATE, MARKUP, HIGH_ASCII
from slot_grammar import conforms, GRAMMAR2
from test2_context import analyse


def clean2(text, comma="."):
    t = INLINE_COMMENT.sub("", text)
    t = ALTERNATE.sub(r"\1", t)
    t = MARKUP.sub("", t)
    t = HIGH_ASCII.sub("?", t)
    t = t.replace("{", "").replace("}", "").replace("'", "")
    t = t.replace(",", comma)
    t = re.sub(r"[!%=\-~\s]", "", t)
    t = re.sub(r"\.+", ".", t).strip(".")
    return t


def corpus(comma):
    pages = {}
    order = []
    n_comma_pairs = 0
    for l in parse("data/ZL3b-n.txt"):
        if l.ltype[1] != "P":
            continue
        ws = [w for w in clean2(l.text, comma).split(".")
              if w and "?" not in w]
        if not ws:
            continue
        if l.page not in pages:
            pages[l.page] = []
            order.append(l.page)
        pages[l.page].extend(ws)
    return [(p, pages[p]) for p in order]


def entropy2(tokens):
    text = ".".join(tokens)
    bi = Counter(zip(text, text[1:]))
    first = Counter()
    for (a, _), c in bi.items():
        first[a] += c
    n = sum(bi.values())
    return -sum(c / n * math.log2(c / first[a]) for (a, _), c in bi.items())


def summary(name, pages):
    toks = [w for _, t in pages for w in t]
    vocab = Counter(toks)
    conf = sum(c for w, c in vocab.items() if conforms(w, GRAMMAR2))
    print(f"{name}: tokens {len(toks)}, types {len(vocab)}, "
          f"hapax {sum(1 for c in vocab.values() if c==1)/len(vocab):.1%}, "
          f"H2 {entropy2(toks):.3f}, grammar-conform {conf/len(toks):.1%}")
    return vocab


def comma_tokens_check():
    """Tokens that contain an uncertain space: do the joined forms
    conform / exist as attested words more than chance?"""
    vocab_dot = Counter()
    joined = Counter()
    for l in parse("data/ZL3b-n.txt"):
        if l.ltype[1] != "P":
            continue
        for w in clean2(l.text, ".").split("."):
            if w and "?" not in w:
                vocab_dot[w] += 1
        for w in clean2(l.text, "").split("."):
            if w and "?" not in w:
                joined[w] += 1
    # tokens that exist only under joining
    only = {w: c for w, c in joined.items() if w not in vocab_dot}
    n = sum(only.values())
    conf = sum(c for w, c in only.items() if conforms(w, GRAMMAR2))
    print(f"\ncomma-joined new forms: {len(only)} types / {n} tokens; "
          f"grammar-conform {conf/n:.1%}")


def pmi_merge(pages):
    toks = [w for _, t in pages for w in t]
    vocab = Counter(toks)
    pairs = Counter()
    for _, t in pages:
        pairs.update(zip(t, t[1:]))
    n = sum(pairs.values())
    uni = Counter(toks)
    nt = len(toks)
    cands = []
    for (a, b), c in pairs.items():
        if c < 5:
            continue
        pmi = math.log2((c / n) / (uni[a] / nt * uni[b] / nt))
        cat = a + b
        if pmi > 1 and cat in vocab:
            cands.append((pmi, a, b, c, vocab[cat]))
    cands.sort(reverse=True)
    tot = sum(c for _, _, _, c, _ in cands)
    print(f"\nsplit-word candidates (PMI>1, pair>=5, concat attested): "
          f"{len(cands)} pair types, {tot} tokens "
          f"({tot/nt:.2%} of corpus)")
    for pmi, a, b, c, vc in cands[:15]:
        print(f"  {a:>6}+{b:<8} n={c:<4} pmi={pmi:4.1f}  "
              f"concat attested {vc}x")
    return {(a, b) for _, a, b, _, _ in cands}


def apply_merge(pages, merge_set):
    out = []
    for p, t in pages:
        nt = []
        i = 0
        while i < len(t):
            if i + 1 < len(t) and (t[i], t[i + 1]) in merge_set:
                nt.append(t[i] + t[i + 1])
                i += 2
            else:
                nt.append(t[i])
                i += 1
        out.append((p, nt))
    return out


def main():
    dot = corpus(".")
    nospace = corpus("")

    print("=== (a) uncertain spaces: space vs joined ===")
    summary("comma=space ", dot)
    summary("comma=joined", nospace)
    comma_tokens_check()

    print("\n=== (b) PMI split-word candidates on the certain corpus ===")
    merge = pmi_merge(nospace)

    print("\n--- MI excess before/after merging split words ---")
    analyse("comma=joined", nospace)
    analyse("merged      ", apply_merge(nospace, merge))


if __name__ == "__main__":
    main()
