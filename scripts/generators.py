"""Five candidate generators. Each fills the REAL manuscript's page
skeleton (same paragraphs/lines/word counts) with its own words.

M1 iid+modes   — words drawn independently from position-conditioned
                 empirical distributions (layout matters, no memory)
M2 copyist     — like M1, but with prob. p_copy the word is a mutated
                 copy of one of the last 20 words (motor habits)
M3 table       — Rugg-style: prefix/core/tail component lists sampled
                 by slowly drifting indices (mechanical combinatorics)
M4 cipher      — verbose homophonic cipher: each PLAINTEXT LETTER of an
                 English herbal becomes one (real) Voynich word;
                 'prose' encodes running text, 'lists' shuffled text
M5 anagram     — English herbal words with letters canonically sorted
                 (the alphagram hypothesis)
"""

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import voynich_pages, culpeper_tokens


def skeleton_and_buckets():
    """Real structure + bucket-conditional word distributions."""
    struct = voynich_pages()
    buckets = defaultdict(list)
    for _, pars in struct:
        for pi, par in enumerate(pars):
            for li, line in enumerate(par):
                role = "pf" if li == 0 else "ot"
                for wi, w in enumerate(line):
                    pos = ("s" if wi == 0 else
                           "e" if wi == len(line) - 1 else "m")
                    buckets[role + pos].append(w)
    return struct, buckets


def fill(struct, wordgen):
    """Rebuild the structure, asking wordgen(role, pos, emitted) for
    each slot; emitted = list of words generated so far (page-local)."""
    out = []
    for p, pars in struct:
        emitted = []
        npars = []
        for pi, par in enumerate(pars):
            nlines = []
            for li, line in enumerate(par):
                role = "pf" if li == 0 else "ot"
                nline = []
                for wi in range(len(line)):
                    pos = ("s" if wi == 0 else
                           "e" if wi == len(line) - 1 else "m")
                    w = wordgen(role, pos, emitted)
                    nline.append(w)
                    emitted.append(w)
                nlines.append(nline)
            npars.append(nlines)
        out.append((p, npars))
    return out


def gen_m1(struct, buckets, seed=1):
    rng = random.Random(seed)

    def g(role, pos, emitted):
        return rng.choice(buckets[role + pos])
    return fill(struct, g)


ENDINGS = ["y", "dy", "in", "iin", "ar", "al", "or", "ol", "ain",
           "aiin", "ey", "eey", "am", "ir"]


def mutate(w, rng):
    ops = []
    if w.startswith("qo"):
        ops.append(lambda w: w[1:])
    elif w.startswith("o"):
        ops.append(lambda w: "q" + w)
    if "ch" in w:
        ops.append(lambda w: w.replace("ch", "sh", 1))
    elif "sh" in w:
        ops.append(lambda w: w.replace("sh", "ch", 1))
    if "k" in w:
        ops.append(lambda w: w.replace("k", "t", 1))
    elif "t" in w:
        ops.append(lambda w: w.replace("t", "k", 1))
    if "ee" in w:
        ops.append(lambda w: w.replace("ee", "e", 1))
    elif "e" in w:
        ops.append(lambda w: w.replace("e", "ee", 1))
    for e in ENDINGS:
        if w.endswith(e) and len(w) > len(e):
            stem = w[:-len(e)]
            ops.append(lambda w, s=stem: s + rng.choice(ENDINGS))
            break
    if not ops:
        return w
    for op in rng.sample(ops, min(len(ops), rng.choice([1, 1, 2]))):
        w = op(w)
    return w


def gen_m2(struct, buckets, seed=2, p_copy=0.18, p_exact=0.01,
           p_sloppy=0.40):
    """Copyist: mostly writes 'fresh' words from positional habit, but
    recall is sloppy (a fresh word is often a slight distortion of a
    remembered one), and sometimes copies from the last two lines."""
    rng = random.Random(seed)

    def g(role, pos, emitted):
        r = rng.random()
        if emitted and r < p_exact:
            return rng.choice(emitted[-20:])
        if emitted and r < p_copy:
            return mutate(rng.choice(emitted[-20:]), rng)
        w = rng.choice(buckets[role + pos])
        if rng.random() < p_sloppy:
            w = mutate(w, rng)
        return w
    return fill(struct, g)


def gen_m2v3(struct, buckets, seed=6, p_copy=0.10, p_exact=0.006,
             p_sloppy=0.50, p_qo_boost=0.14, p_qo_avoid=0.55,
             p_split=0.013):
    """Copyist v3: adds (1) cross-boundary flow — after a -y word the
    next word is pulled toward qo-, after a consonant it avoids qo-;
    (2) occasional word-splitting; (3) perseveration (exact repeats
    come from the immediately preceding words)."""
    rng = random.Random(seed)
    vocab = set(w for ws in buckets.values() for w in ws)
    queue = []

    def base(role, pos, emitted, fresh_only=False):
        r = rng.random()
        if not fresh_only and emitted and r < p_exact:
            return rng.choice(emitted[-3:])
        if not fresh_only and emitted and r < p_copy:
            return mutate(rng.choice(emitted[-20:]), rng)
        w = rng.choice(buckets[role + pos])
        if rng.random() < p_sloppy:
            w = mutate(w, rng)
        return w

    def g(role, pos, emitted):
        if queue:
            return queue.pop(0)
        w = base(role, pos, emitted)
        prev = emitted[-1] if emitted else ""
        if prev.endswith("y") and rng.random() < p_qo_boost:
            for _ in range(30):
                if w.startswith("qo"):
                    break
                w = base(role, pos, emitted, fresh_only=True)
        elif prev[-1:] in "nlrsmd" and w.startswith("qo") \
                and rng.random() < p_qo_avoid:
            for _ in range(5):
                w = base(role, pos, emitted, fresh_only=True)
                if not w.startswith("qo"):
                    break
        if len(w) >= 5 and rng.random() < p_split:
            for i in range(2, len(w) - 1):
                if w[:i] in vocab and w[i:] in vocab:
                    queue.append(w[i:])
                    return w[:i]
        return w
    return fill(struct, g)


def gen_m2v4(struct, buckets, seed=8, p_copy=0.10, p_exact=0.006,
             p_sloppy=0.50, p_qo_boost=0.14, p_qo_avoid=0.55,
             p_split=0.013, p_page=0.25, p_page_mut=0.2):
    """Copyist v4 = v3 + multi-scale memory: besides the 20-word window,
    the scribe reuses words already written on the CURRENT PAGE (page-
    scale bursts -> burstiness, vocabulary drift)."""
    rng = random.Random(seed)
    vocab = set(w for ws in buckets.values() for w in ws)
    queue = []

    def base(role, pos, emitted, fresh_only=False):
        r = rng.random()
        if not fresh_only and emitted:
            if r < p_exact:
                return rng.choice(emitted[-3:])
            if r < p_copy:
                return mutate(rng.choice(emitted[-20:]), rng)
            if r < p_copy + p_page:
                w = rng.choice(emitted)
                if rng.random() < p_page_mut:
                    w = mutate(w, rng)
                return w
        w = rng.choice(buckets[role + pos])
        if rng.random() < p_sloppy:
            w = mutate(w, rng)
        return w

    def g(role, pos, emitted):
        if queue:
            return queue.pop(0)
        w = base(role, pos, emitted)
        prev = emitted[-1] if emitted else ""
        if prev.endswith("y") and rng.random() < p_qo_boost:
            for _ in range(30):
                if w.startswith("qo"):
                    break
                w = base(role, pos, emitted, fresh_only=True)
        elif prev[-1:] in "nlrsmd" and w.startswith("qo") \
                and rng.random() < p_qo_avoid:
            for _ in range(5):
                w = base(role, pos, emitted, fresh_only=True)
                if not w.startswith("qo"):
                    break
        if len(w) >= 5 and rng.random() < p_split:
            for i in range(2, len(w) - 1):
                if w[:i] in vocab and w[i:] in vocab:
                    queue.append(w[i:])
                    return w[:i]
        return w
    return fill(struct, g)


SPLIT = re.compile(
    r"^(q?[oy]?[dslr]?)"
    r"((?:c[tkpf]h|ch|sh)?[ktpf]?e*(?:c[tkpf]h|ch|sh)?[ktpf]?e*)"
    r"((?:[dslr]?[oa]?(?:i+[nrlm]?|[lrmnds])?y?){0,2})$")


def gen_m3(struct, buckets, seed=3):
    """Component table with drifting indices (no layout awareness)."""
    rng = random.Random(seed)
    all_words = [w for ws in buckets.values() for w in ws]
    pre, core, tail = Counter(), Counter(), Counter()
    for w in all_words:
        m = SPLIT.match(w)
        if not m or m.group(1) + m.group(2) + m.group(3) != w:
            continue
        pre[m.group(1)] += 1
        core[m.group(2)] += 1
        tail[m.group(3)] += 1
    cols = []
    for cnt in (pre, core, tail):
        ranked = [x for x, c in cnt.most_common() if c >= 3]
        cols.append(ranked)
    idx = [0, 0, 0]

    def geom(n):
        # frequency-skewed cell choice: top rows of the table are used
        # much more often (as a real table user would)
        i = 0
        while rng.random() > 0.12 and i < n - 1:
            i += 1
        return i

    def g(role, pos, emitted):
        w = ""
        for k in range(3):
            if rng.random() < 0.35:
                idx[k] = geom(len(cols[k]))
            else:
                idx[k] = min(len(cols[k]) - 1,
                             max(0, idx[k] + rng.choice([-1, 0, 0, 1])))
            w += cols[k][idx[k]]
        return w or "ol"
    return fill(struct, g)


def gen_m4(struct, buckets, seed=4, mode="prose", homophones=1500):
    """Verbose homophonic cipher of the English herbal."""
    rng = random.Random(seed)
    eng = culpeper_tokens(limit=60000)
    if mode == "lists":
        rng.shuffle(eng)
    letters = [c for w in eng for c in w]
    freq = Counter(letters)
    n = sum(freq.values())
    # homophone table: real Voynich word types, common letters get more
    voy_types = [w for w, _ in Counter(
        w for ws in buckets.values() for w in ws).most_common()]
    table = {}
    i = 0
    for ch, c in freq.most_common():
        k = max(1, round(c / n * homophones))
        table[ch] = voy_types[i:i + k] or [voy_types[0]]
        i += k
    stream = iter(letters)

    def g(role, pos, emitted):
        ch = next(stream)
        return rng.choice(table[ch])
    return fill(struct, g)


def gen_m5(struct, seed=5):
    """Alphagram English: letters sorted within each word."""
    eng = culpeper_tokens(limit=40000)
    stream = iter(eng)

    def g(role, pos, emitted):
        return "".join(sorted(next(stream)))
    return fill(struct, g)
