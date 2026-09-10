# Data

This directory stores the data used by the paper experiments. It was copied from the research workspace without modifying the source files.

The API pipeline uses the following core layout:

```text
data/
  en_bea/
    train/
      bea-train.src
      bea-train.ref0
    dev/
      bea-dev.src
      bea-dev.ref0
      bea-dev.m2
    test/
      bea-test.src
      bea-test.m2
  conll-2014/
    conll14.test.src
    conll14.test.ref0
    conll14.test.ref1
    conll14.test.m2
```

Additional bundled material includes detokenized variants, document-level BEA development data, curated few-shot pools, official CoNLL-2014 M2 variants, and the original conversion helper. The derived train sample and train/test-matched subsets used during development are not included.

The eight few-shot examples are sampled deterministically from the bundled BEA train files using the seed recorded in each configuration. BEA test is scored on CodaBench; its source file is used as a line-aligned placeholder reference during local prediction generation, and the M2/CodaBench evaluator supplies the actual scoring references.

## Licensing and release

These are third-party datasets and are not covered by any future project-level license. The available Write & Improve and LOCNESS notices are retained under `en_bea/dev/doc_level/`. Before making this repository public, confirm that redistribution through the intended venue is allowed by the original BEA-2019 and CoNLL-2014 terms. If it is not, keep this directory local and restore a download-based setup for the public release.
