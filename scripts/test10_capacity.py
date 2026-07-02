"""Test 10 — how much information does one Voynich word carry?

- H(word): unigram word entropy (plug-in + Miller-Madow correction),
  also per section, and for the English control processed identically.
- Slot decomposition: entropy of each of the 11 slots, their sum vs the
  joint -> how strongly slots are coupled.
- Yardsticks: entropy of one Latin/English letter, one digit.
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words, SECTIONS
from corpus import culpeper_tokens
from slot_grammar import classify
from test2_context import analyse


CAP = re.compile(
    r"^(Q?)([OY]?)([DSLR]?)((?:[CG]K?|K[CG]?)?)(E?)"
    r"((?:[CG]K?|K[CG]?)?)(E?)((?:[DSLR]?[OA]?(?:J|[LRMNDS])?Y?){0,2})$")
SLOT_NAMES = ["q", "o/y", "d/s/l/r", "bench+gallows", "e-run",
              "bench/gallows 2", "e-run 2", "tail"]


def H(counter, correct=True):
    n = sum(counter.values())
    h = -sum(c / n * math.log2(c / n) for c in counter.values())
    if correct:
        h += (len(counter) - 1) / (2 * n * math.log(2))
    return h


def main():
    loci = [l for l in parse("data/ZL3b-n.txt") if l.ltype[1] == "P"]
    toks = [w for l in loci for w in words(l.text)]
    vocab = Counter(toks)

    h_word = H(vocab)
    print(f"H(word) = {h_word:.2f} bits  "
          f"(effective vocabulary 2^H = {2**h_word:.0f} words)")

    # per section (topic removed)
    by_sec = {}
    for l in loci:
        sec = SECTIONS.get(l.page_vars.get("I"), "?")
        by_sec.setdefault(sec, []).extend(words(l.text))
    hs = 0.0
    for sec, ws in by_sec.items():
        hs += len(ws) / len(toks) * H(Counter(ws))
    print(f"H(word | section) = {hs:.2f} bits")

    # English control, identical pipeline
    eng = culpeper_tokens()
    h_eng = H(Counter(eng))
    print(f"control: H(English word) = {h_eng:.2f} bits")

    # yardsticks
    letters = Counter(c for w in eng for c in w)
    print(f"yardsticks: H(English letter) = {H(letters):.2f} bits, "
          f"H(digit 0-9) = {math.log2(10):.2f} bits, "
          f"H(Latin letter ~23) = {math.log2(23):.2f} bits max")

    # slot decomposition over grammar-conforming tokens
    slots = [Counter() for _ in SLOT_NAMES]
    joint = Counter()
    n_conf = 0
    for w, c in vocab.items():
        cs = classify(w)
        m = CAP.match(cs) if cs else None
        if not m:
            continue
        n_conf += c
        joint[w] += c
        # slot content: use the actual glyphs? class string groups suffice
        for i, g in enumerate(m.groups()):
            slots[i][g or "-"] += c
    print(f"\nslot entropies (over {n_conf} conforming tokens, "
          f"class level):")
    total = 0.0
    for name, cnt in zip(SLOT_NAMES, slots):
        h = H(cnt, correct=False)
        total += h
        used = 1 - cnt.get("-", 0) / n_conf
        print(f"  {name:<16} H={h:5.2f} bits  (filled {used:.0%})")
    h_joint_class = H(Counter(
        {cs: c for cs, c in
         ((classify(w), c) for w, c in vocab.items())
         if cs and CAP.match(cs)}), correct=False)
    print(f"  sum of slots      {total:5.2f} bits")
    print(f"  joint (class str) {h_joint_class:5.2f} bits "
          f"-> inter-slot coupling {total - h_joint_class:.2f} bits")

    # information actually used per word, sequence-aware:
    # entropy rate upper bound via H(word) and the adjacent-word excess
    print(f"\nper-word rate estimate: H(word|section) {hs:.2f} "
          f"- adjacent excess ~0.3 = ~{hs-0.3:.1f} bits/word")
    print(f"words needed per English letter: "
          f"{H(letters)/ (hs-0.3):.2f}")
    print(f"English letters encodable per Voynich word: "
          f"{(hs-0.3)/H(letters):.1f}")


if __name__ == "__main__":
    main()
