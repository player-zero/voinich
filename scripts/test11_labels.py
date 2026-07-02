"""Test 11 — the labels: nomenclature or noise?

Labels are single words physically attached to drawn objects (stars,
plant parts, containers, nymphs). If the text is nomenclature/reference:
  (i)   labels should be name-like: high uniqueness, low q- share;
  (ii)  labels should recur in the running text of THEIR OWN page more
        than on other pages of the same section (text talks about the
        labelled things);
  (iii) label morphology should be a subset of the same slot grammar.
"""

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words, SECTIONS
from slot_grammar import conforms, GRAMMAR2
from morphology import glyphs


def load():
    labels = []   # (page, sec, subtype, word)
    text_by_page = defaultdict(list)
    sec_of_page = {}
    for l in parse("data/ZL3b-n.txt"):
        sec = SECTIONS.get(l.page_vars.get("I"), "?")
        sec_of_page[l.page] = sec
        ws = words(l.text)
        if not ws:
            continue
        if l.ltype[1] == "P":
            text_by_page[l.page].extend(ws)
        elif l.ltype[1] == "L":
            for w in ws:
                labels.append((l.page, sec, l.ltype[2], w))
    return labels, text_by_page, sec_of_page


def main():
    labels, text_by_page, sec_of_page = load()
    lab_words = [w for _, _, _, w in labels]
    lab_vocab = Counter(lab_words)
    text_vocab = Counter(w for ws in text_by_page.values() for w in ws)

    print(f"labels: {len(lab_words)} tokens, {len(lab_vocab)} types, "
          f"pages with labels: {len({p for p,_,_,_ in labels})}")
    print("by subtype:", dict(Counter(s for _, _, s, _ in labels)))

    # (i) name-likeness
    print("\n=== (i) are labels name-like? ===")
    hap_lab = sum(1 for c in lab_vocab.values() if c == 1) / len(lab_vocab)
    print(f"hapax among label types: {hap_lab:.1%} "
          f"(running text: 70.2%)")
    rep = sum(c for c in lab_vocab.values() if c > 1) / len(lab_words)
    print(f"label tokens that repeat as labels elsewhere: {rep:.1%}")
    for feat, fn in [("q-", lambda w: w.startswith("q")),
                     ("o-", lambda w: w.startswith("o")),
                     ("d-", lambda w: w.startswith("d")),
                     ("y-", lambda w: w.startswith("y")),
                     ("ch/sh-", lambda w: w.startswith(("ch", "sh")))]:
        a = sum(fn(w) for w in lab_words) / len(lab_words)
        b = sum(c for w, c in text_vocab.items() if fn(w)) \
            / sum(text_vocab.values())
        print(f"  starts {feat:<7} labels {a:6.1%}  text {b:6.1%}")
    mean_lab = sum(len(glyphs(w)) for w in lab_words) / len(lab_words)
    print(f"mean length: labels {mean_lab:.2f} glyphs (text 4.09)")

    # (iii) grammar
    conf = sum(conforms(w, GRAMMAR2) for w in lab_words) / len(lab_words)
    print(f"slot-grammar conformity: labels {conf:.1%} (text ~90%)")
    novel = sum(1 for w in lab_vocab if w not in text_vocab)
    print(f"label types never seen in running text: {novel}/{len(lab_vocab)}"
          f" ({novel/len(lab_vocab):.1%})")

    # (ii) do labels appear in their own page's text?
    print("\n=== (ii) labels vs running text of their own page ===")
    rng = random.Random(31)
    pages_by_sec = defaultdict(list)
    for p, ws in text_by_page.items():
        if ws:
            pages_by_sec[sec_of_page[p]].append(p)
    own = ctrl = tot = 0
    for p, sec, sub, w in labels:
        own_text = text_by_page.get(p)
        others = [q for q in pages_by_sec[sec] if q != p]
        if not own_text or not others:
            continue
        q = rng.choice(others)
        tot += 1
        own += w in set(own_text)
        ctrl += w in set(text_by_page[q])
    print(f"label found in own page text: {own/tot:.1%}, "
          f"in random page of same section: {ctrl/tot:.1%}  (n={tot})")

    # star labels recurring across zodiac pages
    print("\n=== star/zodiac label reuse across pages ===")
    for sub in ("s", "z"):
        ls = [(p, w) for p, sec, s, w in labels if s == sub]
        if not ls:
            continue
        byw = defaultdict(set)
        for p, w in ls:
            byw[w].add(p)
        multi = {w: ps for w, ps in byw.items() if len(ps) > 1}
        print(f"subtype '{sub}': {len(ls)} tokens, {len(byw)} types; "
              f"types on 2+ pages: {len(multi)}")
        ex = sorted(multi.items(), key=lambda kv: -len(kv[1]))[:5]
        for w, ps in ex:
            print(f"    {w:<10} on {len(ps)} pages")


if __name__ == "__main__":
    main()
