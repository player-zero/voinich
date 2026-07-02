# Data

The corpora are not stored in the repository. Fetch them from the original
sources (run from the repo root):

```sh
sh scripts/fetch_data.sh
```

This downloads:

- `ZL3b-n.txt` — Zandbergen–Landini transliteration of the Voynich MS
  (EVA alphabet, IVTFF 2.0 format). Source and documentation:
  http://www.voynich.nu/transcr.html
- `IT2a-n.txt` — Takahashi transliteration (optional cross-check).
- `culpeper.txt` — Nicholas Culpeper, *The Complete Herbal* (1653),
  Project Gutenberg #49513, public domain. Used as the same-size,
  same-genre natural-language control.
