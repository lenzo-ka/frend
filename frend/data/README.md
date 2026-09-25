# frend/data

## `exceptions/en.json`

Curated English word- and sentence-break suppressions imported from the foreign
`break_exceptions/en.txt` seed by `tools/import_break_exceptions.py`. The importer
keeps reviewed title, rank, organization, credential, name, road, and country
abbreviations only when followed by a capital letter. It drops all regex records,
day/month names that can naturally end sentences, and ambiguous domain-dependent
entries (`Pt.`, `v./vs.`, `Md.`, `Nr.`, and `Num.`). Every build prints the exact
kept and dropped records with source line numbers and reasons; `--check` verifies
the vendored inventory byte-for-byte and also runs icukit's transactional witness
gate. Each retained rule claims both levels: at the word level its abbreviation is
one atomic token including the final period, and at the sentence level that period
does not end the sentence.

## `type_priors.json`

Vendored corpus prior table for the Layer-2 base-rate tiebreak: raw
`shape -> class -> count` integer counts, plus provenance. Runtime
(`frend.type_priors`) derives `P(class | shape)` and the sample size `n(shape)`
from these counts on demand; no smoothing and no ratios are stored.

### Attribution

This attribution applies to `type_priors.json` and `spoken_priors.json`: their counts are derived from the CC BY-SA 4.0 Google/Sproat "en_with_types" English text-normalization corpus (Sproat & Jaitly, *RNN Approaches to Text Normalization: A Challenge*, 2016; arXiv:1611.00068; https://github.com/rwsproat/text-normalization-data), are aggregate counts over the sample declared by each artifact, and retain provenance in the JSON; whether ShareAlike attaches to an aggregate is not asserted here either way.

**This data was filtered and aggregated**, not copied: surfaces containing a Unicode decimal digit (category `Nd`) use a fixed corpus-class → frend class mapping, while single uppercase-letter surfaces admit the corpus's own classes so their alphabetic and numeric interpretations share a denominator. Other surfaces and the PLAIN/PUNCT/LETTERS/VERBATIM/ELECTRONIC classes are dropped. Each retained surface is reduced to its `frend.shape` signature, and only the resulting integer `shape -> class` counts are retained. The `single_uppercase_letters` section preserves the corresponding per-letter aggregates for inspection. No corpus text is reproduced. The raw corpus itself is never vendored here.

The table is rebuilt (or verified) by `tools/build_type_priors.py`, which streams
the corpus shards line by line. `--check` re-derives the table and diffs it
against this file, so it requires the corpus present at the corpus path
(`--corpus-dir`, or the `FREND_TN_CORPUS_DIR` environment variable);
`tools/fetch_corpora.py` fetches and verifies it. There is no
corpus-free build. `--corpus` also accepts a path to any Google-TN-format shard directory,
such as a small fixture, and records that path in the provenance.

## `spoken_priors.json`

This table measures which sources among the alternatives returned by `frend.verbalize` match spoken forms in the Google/Sproat English text-normalization corpus. It records per-kind matched rows, unmatched rows split among unrecognized, unverbalized, and no-alternative-matched outcomes, source match counts, and frequent unmatched spoken forms with the same reason breakdown. End-to-end recall is `matched / total`. Verbalizer recall is `matched / (matched + no_alternative_matched)`; the loader and runtime do not compute or apply it.

The builder reads every tenth sorted shard and keeps the first configured limit of eligible rows for each mapped corpus class in each selected shard. The exact shard names, limit, and resulting row counts live in the JSON provenance.

frend has no canonical detector registry, so the builder declares its own recognition profile rather than claiming to reuse one. Cardinal and decimal recognition includes ICU-derived long and short compact-number patterns. Money recognition includes symbol and ICU-derived display-name detectors for each recorded currency code. Measure recognition includes percent and a measure detector for each recorded unit, whose written forms icukit derives from ICU, and the recorded mixed measures (foot and inch, pound and ounce). Time recognition reads hours, minutes, a written day period ("5pm" is spoken "five p m") and a written time zone, spoken as its letters ("10 PM ET", "ten p m e t"); numeric durations ("1:47.22") are read by icukit's duration detector and filed under TIME, as the corpus files race times; ordinal recognition includes Roman numerals, which read as cardinals or ordinals; date recognition includes plural numerals ("1990s", "nineteen nineties", filed as DATE by the corpus). A letter-digit token read as its runs ("3D", "three d") has no corpus class of its own, so its detector joins the date, time and ordinal profiles, where such tokens occur. The JSON records the locale, detector classes by kind, date skeletons, currency codes, and full-span rule. Each full-span detection is resolved and verbalized separately through the normal frend lattice; no detection payload is constructed by the builder.

The verbalizer deduplicates equal alternative text in ICU rule-set order. Each matched row is therefore credited to the first retained source that produces the text, and source counts sum to matched rows. The measurement does not recover sources discarded by that deduplication. The table stores raw counts. At runtime a kind-level share is the source count divided by matched rows; at a sub-key (a fraction's denominator, or a measure's ICU unit identifier), every source's share blends the sub-key's own evidence toward the kind-level share, `(sub-key count + A × kind share) / (sub-key matched + A)` with `A = frend.spoken_priors.SUB_KEY_PRIOR_STRENGTH` (5), so a denominator seen once barely leaves the kind and one seen hundreds of times dominates it. `SourceMeasurement` carries the sub-key's own matched count beside the share. Source ranking is disabled while building the table because measuring through the ranking the table drives would be circular.

Before recognition, the builder removes trailing spaces and commas from corpus written tokens. It does not remove any other suffix. Matching lowercases spoken forms and alternatives, replaces Unicode punctuation with spaces, and collapses whitespace. `<self>` and `sil` spoken sentinels are skipped.

The shared attribution above applies to this table. The corpus is not shipped here.

`tools/build_spoken_priors.py --check` repeats the declared sample and compares the canonical JSON byte-for-byte. A corpus path argument selects a fixture or another Google-TN-format shard directory.

## `tlds-alpha-by-domain.txt`

IANA's list of top-level domains, vendored unchanged from
https://data.iana.org/TLD/tlds-alpha-by-domain.txt; its first line carries IANA's version
(`tld_version()`). frend's electronic detector reads a bare domain ("boston.com") only
when its last label is in this list, so "e.g." and "report.pdf" are not domains. Refresh
it by fetching the same URL; the electronic priors record the version they were built
with.

## `electronic_priors.json`

How the runs of a URL, email address or domain are said, measured from the corpus's
ELECTRONIC class by `tools/build_electronic_priors.py` (`--check` repeats the sample and
compares byte for byte). Each row of the sampled shards is split into runs as frend's
detector splits it and aligned with its spoken form, run by run; rows that do not align
are counted and left out. The table holds integer counts only: letter runs said as a
word or spelled, keyed by shape (case, length, whether a vowel letter occurs) and, for a
top-level domain, by the domain; digit runs read as ICU's cardinal or year or digit by
digit, keyed by length; and each separator character's spoken words. No corpus text is
stored. The shared attribution above applies.

In `spoken_priors.json`, ELECTRONIC is measured like the other kinds, with the corpus's
per-letter spoken notation ("b_letter o_letter") joined into words first. It records no
`top_unmatched` examples, which would copy URLs from the corpus.
