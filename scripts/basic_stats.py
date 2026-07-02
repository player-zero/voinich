"""First-pass statistics over the ZL transliteration: corpus size, word
frequencies, Zipf fit, character entropy, word-length distribution, and
Currier A/B split."""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ivtff import parse, words, SECTIONS


def entropy(counter):
    total = sum(counter.values())
    return -sum(c / total * math.log2(c / total) for c in counter.values())


def cond_entropy(bigrams, unigrams):
    """H(X2|X1) from bigram and unigram counts."""
    total_bi = sum(bigrams.values())
    h = 0.0
    for (a, b), c in bigrams.items():
        p_ab = c / total_bi
        p_b_given_a = c / unigrams[a]
        h -= p_ab * math.log2(p_b_given_a)
    return h


def char_stats(wordlist, label):
    text = ".".join(wordlist)
    uni = Counter(text)
    bi = Counter(zip(text, text[1:]))
    # unigrams restricted to chars that start a bigram, for conditioning
    uni_first = Counter(a for a, _ in bi.elements()) if False else Counter()
    for (a, _), c in bi.items():
        uni_first[a] += c
    h1 = entropy(uni)
    h2 = cond_entropy(bi, uni_first)
    print(f"  [{label}] alphabet={len(uni)}  H1={h1:.3f} bits  "
          f"H2(cond)={h2:.3f} bits")


def main():
    loci = parse("data/ZL3b-n.txt")
    print(f"loci: {len(loci)}  pages: {len({l.page for l in loci})}")

    all_words = []
    by_lang = {"A": [], "B": []}
    by_section = {}
    for l in loci:
        ws = words(l.text)
        all_words.extend(ws)
        lang = l.page_vars.get("L")
        if lang in by_lang:
            by_lang[lang].extend(ws)
        sec = SECTIONS.get(l.page_vars.get("I"), l.page_vars.get("I"))
        by_section.setdefault(sec, []).extend(ws)

    vocab = Counter(all_words)
    n_tok, n_typ = len(all_words), len(vocab)
    print(f"tokens: {n_tok}  types: {n_typ}  TTR: {n_typ/n_tok:.3f}")
    hapax = sum(1 for c in vocab.values() if c == 1)
    print(f"hapax legomena: {hapax} ({hapax/n_typ:.1%} of types)")

    print("\ntop 20 words:")
    for w, c in vocab.most_common(20):
        print(f"  {w:<10} {c:>5}  {c/n_tok:.2%}")

    # Zipf: slope of log(freq) vs log(rank) over top 1000
    ranked = [c for _, c in vocab.most_common(1000)]
    xs = [math.log(r + 1) for r in range(len(ranked))]
    ys = [math.log(c) for c in ranked]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)
    print(f"\nZipf slope (top 1000): {slope:.3f}")

    lens = Counter(len(w) for w in all_words)
    mean_len = sum(l * c for l, c in lens.items()) / n_tok
    print(f"\nword length: mean={mean_len:.2f}")
    for L in sorted(lens):
        bar = "#" * round(lens[L] / n_tok * 200)
        print(f"  {L:>2} {lens[L]:>6} {bar}")

    print("\ncharacter entropy (word text incl. '.' separator):")
    char_stats(all_words, "all")
    for lang, ws in by_lang.items():
        char_stats(ws, f"Currier {lang}")

    print("\ntokens per section:")
    for sec, ws in sorted(by_section.items(), key=lambda kv: -len(kv[1])):
        print(f"  {str(sec):<15} {len(ws):>6}")

    print("\ntop 10 words, Currier A vs B:")
    ca, cb = Counter(by_lang["A"]), Counter(by_lang["B"])
    ta, tb = sum(ca.values()), sum(cb.values())
    fmt = "  {:<10}{:>6}  {:<7}|  {:<10}{:>6}  {:<7}"
    for (wa, na), (wb, nb) in zip(ca.most_common(10), cb.most_common(10)):
        print(fmt.format(wa, na, f"{na/ta:.2%}", wb, nb, f"{nb/tb:.2%}"))


if __name__ == "__main__":
    main()
