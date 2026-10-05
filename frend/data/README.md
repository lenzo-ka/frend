# frend/data

## Layout

Each table sits under the locale it is for: `<locale>/<table>` (`en/type_priors.json`).
`frend.locale_data` looks a locale up along its chain, found by truncating subtags
(`en_US` → `en` → `root`). A measured table (counted from one language's corpus) is
looked up along the chain but never in `root`, so a locale with no measured table gets
none, never another language's counts. The cross-locale tables (the ICU shape backfill,
IANA's top-level domains) are in `root/`. Each measured table's provenance names its
`locale` and the `corpus` it was counted from (`google-tn:en_with_types`), beside the
shards it counted.

## The corpus and its split

Most measured tables under `en/` are counted from the Google TN English corpus,
`en_with_types` (store id `google/tn-en_with_types`: the Kaggle dataset's
`en_with_types.tgz`, 100 shards `output-000NN-of-00100`). Its README splits the shards
training 00-89, runtime eval 90-94 and test 95-99, and frend keeps that split
(`tools/google_tn_rows.py`: `TRAINING_SHARDS`, `RUNTIME_EVAL_SHARDS`, `TEST_SHARDS`).
A table counts training shards only: `type_priors.json` and
`spellout_dictionary.json`, `spelled_token_priors.json` and `number_priors.json` all 90
(00-89), the sampled tables every tenth
(00, 10, ..., 80), and `range_priors.json` both (its emit counts all 90, its reading
counts every tenth). The runtime-eval and test shards are held out
(`HELD_OUT_SHARDS`) and no builder opens one. Shard 99 is the published test shard
the evaluator reports; shard 95 is the held-out running-text shard changes are
accepted on (`--held-out-shard`); decision runs add `--skip-report-shard` so they do
not verify or score shard 99. Shards 90-94 are a second held-out pool.

## `en/exceptions.json`

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

## `en/type_priors.json`

Vendored corpus prior table for the Layer-2 base-rate tiebreak: raw
`shape -> class -> count` integer counts, plus provenance. Runtime
(`frend.type_priors`) derives `P(class | shape)` and the sample size `n(shape)`
from these counts on demand; no smoothing and no ratios are stored.

### Attribution

This attribution applies to `type_priors.json`, `spoken_priors.json`,
and `number_priors.json`: their counts are derived
from the CC BY-SA 4.0 Google/Sproat "en_with_types" English text-normalization corpus
(Sproat & Jaitly, *RNN Approaches to Text Normalization: A Challenge*, 2016;
arXiv:1611.00068; https://github.com/rwsproat/text-normalization-data), are aggregates
over the sample declared by each artifact, and retain provenance in the JSON; whether
ShareAlike attaches to an aggregate is not asserted here either way.

**This data was filtered and aggregated**, not copied: surfaces containing a Unicode decimal digit (category `Nd`) use a fixed corpus-class → frend class mapping, while single uppercase-letter surfaces admit the corpus's own classes so their alphabetic and numeric interpretations share a denominator. Other surfaces and the PLAIN/PUNCT/LETTERS/VERBATIM/ELECTRONIC classes are dropped. Each retained surface is reduced to its `frend.shape` signature, and only the resulting integer `shape -> class` counts are retained. The `single_uppercase_letters` section preserves the corresponding per-letter aggregates for inspection. No corpus text is reproduced. The raw corpus itself is never vendored here.

The table is rebuilt (or verified) by `tools/build_type_priors.py`, which streams
the corpus shards line by line. It counts the 90 training shards (00-89) and none of
the held-out 90-99 (see the split above); `provenance.shards` lists the 90 it counted.
`--check` re-derives the table and diffs it
against this file, so it requires the corpus present at the corpus path
(`--corpus-dir`, or the `FREND_TN_CORPUS_DIR` environment variable);
`tools/fetch_corpora.py` fetches and verifies it. There is no
corpus-free build. `--corpus` also accepts a path to any Google-TN-format shard directory,
such as a small fixture, and records that path in the provenance.

## `en/spoken_priors.json`

This table measures which sources among the alternatives returned by `frend.verbalize` match spoken forms in the Google/Sproat English text-normalization corpus. It records per-kind matched rows, unmatched rows split among unrecognized, unverbalized, and no-alternative-matched outcomes, source match counts, and frequent unmatched spoken forms with the same reason breakdown. End-to-end recall is `matched / total`. Verbalizer recall is `matched / (matched + no_alternative_matched)`; the loader and runtime do not compute or apply it.

The builder reads every tenth sorted shard, less the held-out shards (00, 10, ..., 80 of the full corpus), and keeps the first configured limit of eligible rows for each mapped corpus class in each selected shard. The exact shard names, limit, and resulting row counts live in the JSON provenance.

frend has no canonical detector registry, so the builder declares its own recognition profile rather than claiming to reuse one. Cardinal and decimal recognition includes ICU-derived long and short compact-number patterns. Money recognition includes symbol and ICU-derived display-name detectors for each recorded currency code. Digit recognition (the corpus's DIGIT class, digit strings read one digit at a time) is frend's number reader and its written-forms reader. Measure recognition includes percent and a measure detector for each recorded unit, whose written forms icukit derives from ICU, and the recorded mixed measures (foot and inch, pound and ounce). Time recognition reads hours, minutes, a written day period ("5pm" is spoken "five p m") and a written time zone, spoken as its letters ("10 PM ET", "ten p m e t"); numeric durations ("1:47.22") are read by icukit's duration detector and filed under TIME, as the corpus files race times; ordinal recognition includes Roman numerals, which read as cardinals or ordinals; date recognition includes plural numerals ("1990s", "nineteen nineties", filed as DATE by the corpus). A letter-digit token read as its runs ("3D", "three d") has no corpus class of its own, so its detector joins the date, time and ordinal profiles, where such tokens occur. The JSON records the locale, detector classes by kind, date skeletons, currency codes, and full-span rule. Each full-span detection is resolved and verbalized separately through the normal frend lattice; no detection payload is constructed by the builder.

The verbalizer deduplicates equal alternative text in ICU rule-set order. Each matched row is therefore credited to the first retained source that produces the text, and source counts sum to matched rows. The measurement does not recover sources discarded by that deduplication. The table stores raw counts. At runtime a kind-level share is the source count divided by matched rows; at a sub-key (a fraction's denominator, or a measure's ICU unit identifier), every source's share blends the sub-key's own evidence toward the kind-level share, `(sub-key count + A × kind share) / (sub-key matched + A)` with `A = frend.spoken_priors.SUB_KEY_PRIOR_STRENGTH` (5), so a denominator seen once barely leaves the kind and one seen hundreds of times dominates it. `SourceMeasurement` carries the sub-key's own matched count beside the share. Source ranking is disabled while building the table because measuring through the ranking the table drives would be circular.

Before recognition, the builder removes trailing spaces and commas from corpus written tokens. It does not remove any other suffix. Matching lowercases spoken forms and alternatives, replaces Unicode punctuation with spaces, and collapses whitespace. `<self>` and `sil` spoken sentinels are skipped.

The shared attribution above applies to this table. The corpus is not shipped here.

## `en/telephone_priors.json`

Grouped readings for conservative North American telephone shapes, counted from all
Google-TN training shards (00--89) by `tools/build_telephone_priors.py`. The detector
requires an observed exact punctuation shape, a strict TELEPHONE majority over every
corpus class with that shape, and at least three TELEPHONE rows. It accepts only the
trained ASCII digit/separator alphabet, requires the NANP `NXX-NXX-XXXX` structure (with
N11 excluded), and rejects a candidate directly after an alphanumeric or symbol. The
table also counts the immediately preceding corpus token by its own gold class and ships
only its surface-free character-class shape (case/run kind, length bucket, punctuation
or symbol class) for typed strict-majority non-TELEPHONE contexts with at least three
observations; ordinary PLAIN words and punctuation do not veto. Reading counts are
stratified by whether each digit group contains zero or repeats one digit; each group
is classified against ICU's cardinal and digit spellout, with the locale lexical
table supplying the corpus's `o` realization of zero. No corpus surface string is stored.

The Google-TN target contains the literal marker `sil` between telephone groups, and the
headline scorer retains embedded markers. Telephone alternatives therefore retain that
marker too. Unsupported ISBN, ZIP-code, identifier, score, and year/range shapes stay
outside the detector even when Google-TN labels them TELEPHONE; a learned typed label
can veto a supported shape. The shared attribution above applies to this table.

`tools/build_telephone_priors.py --check` rescans the declared training shards and
compares the canonical JSON byte-for-byte. A corpus path argument selects a fixture or
another Google-TN-format shard directory.

## `en/grouped_id_priors.json`

Exact grouped-digit shapes counted from all Google-TN training shards (00--89) by
`tools/build_grouped_id_priors.py`. A shape is retained only with at least three
targets that say one digit per written digit and a strict grouped-reading majority over
every corpus row with that shape. Each row contains only the shape, gold-class counts,
and grouped/other reading counts; no written or spoken corpus surface is stored.

At runtime, non-ISBN identifiers additionally require the conservative
development-selected structure of exactly three groups and nine ASCII digits. ISBNs
instead pass the ISBN-10 modulus-11 or ISBN-13 modulus-10 checksum algorithm, never an
identifier list. The International ISBN Agency describes ISBN elements and separators,
the ISBN-13 alternating 1/3 weights, and the check digit in its
[ISBN overview](https://www.isbn-international.org/content/what-isbn/10); its
[Users' Manual](https://www.isbn-international.org/index.php/content/isbn-users-manual/29)
is the definitive system guide. ISBN-10 remains supported for pre-2007 identifiers,
including X as the value-ten check digit.

ICU supplies digit and letter names; only the locale lexical table supplies the ``o``
form of zero. Embedded ``sil`` retains written group boundaries, as the headline
Google-TN scorer does. A same-span #68 telephone detection wins before this detector is
added, so the measured telephone behavior is unchanged. The shared Google-TN
attribution above applies.

`tools/build_grouped_id_priors.py --check` verifies the canonical aggregate by
rescanning the declared training inputs.

## `root/tlds-alpha-by-domain.txt`

IANA's list of top-level domains, vendored unchanged from
https://data.iana.org/TLD/tlds-alpha-by-domain.txt; its first line carries IANA's version
(`tld_version()`). frend's electronic detector reads a bare domain ("boston.com") only
when its last label is in this list, so "e.g." and "report.pdf" are not domains. Refresh
it by fetching the same URL; the electronic priors record the version they were built
with.

## `en/electronic_priors.json`

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

## `en/spellout_dictionary.json`

The first decision for a spell-or-say token. `tools/build_spellout_dictionary.py`
counts case-preserved Google TN `LETTERS` and `PLAIN` surfaces on training shards
00--89. The shipped operating point is support 5 and 99% purity, selected on S0
(shards 90--94). A row counts only when its spoken form literally says the token
or names its letters. It stores exact surfaces whose selected decision differs from
the casefolded AEIOU fallback, except spell decisions with unambiguous exact-headword
Wiktionary lexical evidence. Translation words, abbreviation-like or symbolic senses,
inflection-only forms, and non-titlecase mixed-case technical forms do not qualify.

The compact JSON rows have the header-declared shape `[surface, decision, say_count,
spell_count]`, are sorted by surface, and share one Google TN source/license record in
the provenance header. They contain no sentence, expansion, gloss, IPA, neighbor, or
other context. Exact-case lookup is the default. Callers may opt in with
`case_variant_lookup=True`; unanimous observed families reuse the same filtered rows,
so no second token table is required and ordinary-word exclusions apply in both modes.

`--check` reopens verified shards 00--89 and compares the complete compact file. A
missing bounded-token row uses the AEIOU rule, whose vowel test casefolds. An unattested
capital run retains the established acronym shape/CV fallback. The old bounded-token
detector gate also still applies. Thus the dictionary is the single source only for
stored spell/say exceptions; it is not the sole decision mechanism. `exceptions.json`
remains unrelated ICU word/sentence-break suppression.

## `en/acronym_priors.json`

Whether an acronym is spelled or said as a word, measured from the corpus by
`tools/build_acronym_priors.py` (`--check` repeats the sample and compares byte for
byte): an all-capitals token of two or more letters counts as spelled when the corpus
files it as LETTERS ("FBI" "f b i") and as a word when it files it as PLAIN ("NASA").
The packaged table remains sampled every tenth training shard: counts are kept by the
token's shape (`letter_key`: case, length, whether a vowel letter occurs), by its
consonant-vowel pattern up to seven letters (`cv:cvc`), by the surface for an acronym
icukit's lexicon lists (`surface:NASA`) and for a Roman numeral icukit reads (`roman:II`,
which also counts the corpus's CARDINAL and ORDINAL readings as `numeral`), with `*`
pooling all. It stores no corpus surface inventory.

The off-by-default `google-tn` conformance profile uses a separate exact-surface table
built from the caller's own licensed corpus. `tools/build_acronym_priors.py` writes it
outside the package to `--profile-out`; the default is
`FREND_GOOGLE_TN_PROFILE_PATH`, then the user's cache under
`frend/profiles/google-tn/acronym_surfaces.json`. It counts training shards 00--89,
preserves case, and strips only the detector's suffix into `bare`, `plural`, or
`possessive`. An exact row is used only at the development-selected support floor;
below it, the established shape/CV ranking is unchanged. The file records the floor,
smoothing strength, and the 00--79 training / 80--89 development split. Because it
contains corpus surfaces verbatim it is never included in frend's package; the shared
Google TN attribution and CC BY-SA 4.0 license above apply to that derived file.

The same profile's PLAIN-token spelling table is built by
`tools/build_britishisms.py`. Its default external path is the acronym table's directory
under `britishisms.json`, overridable with `FREND_GOOGLE_TN_BRITISHISMS_PATH`. Written
forms retain separate lower, Title, and UPPER evidence and record converted and left-as-is
counts over shards 00--89. Every differing pair is classified as respelling, diacritic
removal, expansion/abbreviation, or other. Shards 00--79 train the exact pairs and a small
fixed inventory of edit types (`our` to `or`, `ise` to `ize`, `re` to `er`, `ogue` to
`og`, `mme` to `m`, doubled consonants, and related measured suffixes); shards 80--89
choose which pair classes to admit, their support floors, the common edit-rate threshold,
and whether lowercase evidence may back off another case shape. The loader requires
digests for all 90 training shards and the profile refuses to score unless they exactly
match the acronym table's verified shard digests. The default profile never reads either
external table, so its output is unchanged.

## `en/spelled_token_priors.json`

Spell-or-say exceptions for non-uppercase tokens of two through six letters in the
locale's script. `tools/build_spelled_token_priors.py` counts all 90 training shards
(00--89). A row is relevant only
when `LETTERS` literally equals the locale's authoritative letter names or `PLAIN`
literally equals the written token after spoken normalization. The default rule says a
token containing a/e/i/o/u and spells one without; y is separate. Only a measured
majority that disagrees is stored (60,982 exceptions from 898,326 measured tokens).
Each exception holds spell/say counts and unsmoothed shares. Both readings are offered
for every structurally eligible token. `--check` recounts verified training shards byte
for byte. Its only inputs are those rows, locale-authoritative letter names, and spoken
normalization.

## `en/abbreviation_priors.json`

How the corpus says each abbreviation of icukit's lexicon as running text writes it,
measured by `tools/build_abbreviation_priors.py` (`--check` as above). A key is the
written token's ICU lower case with one trailing period removed (`st` for "st", "St."
and "ST"), for each lexicon abbreviation ending in "." with a lowercase letter, no inner
period and more than one letter, and each dotted chain of single letters with an
expansion (`e.g`). Under the key, rows are kept by the token's written case (`lower`,
`title`, `upper`) and counted by what the corpus says: `spelled`, `as-written`, the
lexicon expansion ("saint"), or `other`. The corpus writes none of these keys with a
period, and some (st, dr, mr, ...) never in title case; a case with no row reads the
key. Only counts are stored; the shared attribution applies.

## Provenance: declared sources only, nothing from LDC

Every file here names what it was made from by a source id, and that id must be one
of `frend.data_sources.SHIPPABLE_SOURCES`, each listed with its license and license
class: `google/tn-en_with_types` (the store id; shippable-share-alike),
`icu/<version>/<family>`, `iana/tlds-alpha-by-domain`, `lenzo/break_exceptions`,
`frend/curated` (the hand-written forms in `lexical.json`) and `festvox/festival`
(Festival's curated word lists, `en/context/festival_classes.json`, which carries
Festival's notice as its license asks). A JSON table names it in `corpus` or `source`,
at its top level or in its `provenance`; a binary context tree (`.cart`) names it in its
embedded metadata; the older labels the
measured tables carry (`google-tn:en_with_types`, `icu-reflective-generation`) are
mapped to their ids there (`SOURCE_LABELS`), so the tables keep their bytes. A file that
is a source vendored unchanged (IANA's list) is named by its source's `vendored` entry.
Nothing is derived from an LDC corpus: every `ldc/*` store id is internal-only for
frend (kal, 2026-09-28), usable for evaluation and development but never in a shipped
table, count or fixture. `tests/test_locale_data.py` refuses a file naming no source,
an undeclared one, an `ldc/` one, or whose provenance mentions an LDC corpus anywhere
("LDC93S6A", "ldc:wsj0").

## `en/zero_priors.json`

How the corpus says a zero digit, "o", "oh" or "zero", per reading kind, measured by
`tools/build_zero_priors.py` (`--check` as above): the zero words in the spoken form of
each token written with a 0, after "point" for DECIMAL, MEASURE and MONEY and anywhere
for DATE, TIME, DIGIT and ELECTRONIC. Readings that differ only in a zero's word share
their weight by these counts. Only counts are stored; the shared attribution applies.

SYMBOL (`spoken_priors.json`) measures the corpus's VERBATIM and PUNCT rows with frend's
symbol reader; unlike the other kinds, a spoken `sil` is a target (silence), not a
skipped row, and the written token is not trimmed. Its sub-key is a symbol's code point
(`U+0026`) or a letter's script (`Grek`).

## `<locale>/lexical.json`

The spoken forms frend writes by hand because ICU and CLDR do not give them, one
table per locale, loaded through `frend.locale_data.lexical_forms`. Each entry under
`forms` holds its `value` and `why`: the reason it is hand-written, stating only what
was checked. Forms emitted into a reading carry the source `lexical:en_US`, the key the
measured tables count them under. A locale with no forms (`ru/lexical.json`) has each
lexical feature off. `range.connector`'s "to" pattern and `range.separator`'s ranges
are read by the range connector (a separator between two numbers is also offered "to";
see `en/context/`); `range.connector`'s `range` patterns ("to" and silent) join the two
ends of a range ICU writes (`frend.ranges`: icukit's number, measure and date-interval
range readers); `range.separator`'s classes (range "-", ratio ":", dimension "x" and
"×") and each class's `range.connector` patterns are frend's own range reader's
(`frend.ranges.RangeDetector`, "5-10", "16:79", "3x4"; see `en/range_priors.json`).

## `en/range_priors.json`

The range table, measured by `tools/build_range_priors.py` (`--check` as above), read by
`frend.ranges` for a range written in running text ("5-10", "5 - 10", "16:79", "3x4",
"5-10 kg"). Rules decide the candidate span (R1-R4, R6, R7 in `frend.ranges`); the table
holds only what is trained:

- `readings` (E, the emit decision), over all 90 training shards: per emit key (the
  separator class, `dash` pooling "-" and the en dash, then the two ends' ASCII digit
  lengths, `ratio:clock` where icukit's `time:flexible` reads the span, else `other`)
  and per class, the corpus triples the rules can emit on (`range`) against the single
  corpus tokens of that written shape by class (`single_token`: a TELEPHONE "555-1212",
  a TIME "10:30"). A span is emitted where `range` is at least `EMIT_RATIO` times the
  singles; a key never observed reads its class.
- `kinds` (J, the joint reading prior), over every tenth training shard, in the spoken
  priors' schema per class and sub-key (`4+2:0` for a right end written with a leading
  zero): the joint source (`range:<left kind>+<connector>+<right kind>`) of the first
  builder-mode reading that says the corpus's reading of the triple.
- `provenance.relevance`: what each part counts, and what it sets apart: triples no rule
  can emit on, and R6's punctuation dash. A silent dash is valid when the actual range
  reading says both ends correctly; only silent rows whose ends are misread are set
  apart and counted nowhere.

Only counts are stored; the shared attribution applies.

## `en/context/`

The context trees (`frend.context`): which of a span's readings the running text
around it favors. `tools/build_context_trees.py` trains one cartlet decision tree per
reading-choice problem (a problem is the sorted labels of a span's readings) from P7's
stored example set on kalman (`frend/google/tn-en_with_types/p7-examples/d1fcf656b9eb98dc`:
training shards 05, 15, ..., 85, disjoint from the shards the other tables count; every
token where frend offers two or more readings, labeled by the reading the corpus says,
at most 30,000 per problem, seed 20260928, and every lone "-" with its spoken form).
Seeded and reproducible: `--check` rebuilds from the stored set (each file checked
against its receipt) and compares byte for byte, so it needs the kalman mount.

- `index.json`: provenance (the corpus, the set's fingerprint, receipt and file
  hashes, the training parameters), the 200 frequent words (the words most often
  within two of a training span that no other class claims: Flite's frequent-word
  class, learned here rather than copied), the connectors (the lone separator's
  problem, read also for a separator written inside a range, "5-10"; the en dash,
  which the set never writes between numbers, reads the hyphen's), and every tree's
  file, hash, size and label counts.
- Only the `range:range` problem from the ranges plan's P6 is trained, using its own stored set
  (`frend/google/tn-en_with_types/p6-range-examples/36373cb1123ca6d5`, derived by
  `--derive-range-examples` from the same shards 05, ..., 85: every range triple the
  rules can emit on, less the punctuation dashes, with its sentence either side), with
  the range family of features added; `index.json`'s `range_examples` names that set,
  and the trained range tree names its fingerprint. `range:ratio` and `range:dimension`
  are excluded because neither made an override on the seeded samples of shards 90--94
  or shard 95. The main set's trees are built exactly as without the range set
  (`--check --no-range-examples` compares them alone).
- `index.json` also states exact problem keys omitted after held-out measurement. The
  retained `range:range` tree is trained, indexed, and consulted at runtime;
  `range:ratio`, `range:dimension`, and 54 P7 problem keys are excluded because they
  made no override on seeded samples of both shards 90--94 and shard 95.
- `trees/<id>.cart`: the trees, cartlet model format 2, each naming its source and the
  set's fingerprint in its metadata; read with cartlet's dependency-free runner.
- `festival_classes.json`: Festival's hand-curated lists (`lib/tokenpos.scm`: regnal
  names, king-like titles, section words), unchanged, with Festival's notice. Curated,
  not built.

The trees' features are Flite's number-tree features (the span's length, a day-of-
month number, the class of the words at -2..+2 with ICU's month and weekday names),
ICU's Word_Break, General_Category and Script of three characters either side, the
Festival classes, and frend's own first choice and its weight. A tree's top reading
replaces frend's first choice only at probability 0.7 or more. The trees hold only
split values and label counts (and the frequent words); no corpus text. The shared
attribution above applies.

## `en/number_priors.json`

Bare four-digit tokens alone receive DATE, CARDINAL and DIGIT alternatives. Structured
icukit dates (month, day, era, decade or interval) retain their date structure and do
not use this choice. `tools/build_number_priors.py` streams all training shards and
stores raw class counts by the token's first two digits, plus an all-bucket fallback;
there is no hand-set year or value window.

The builder's `--check` mode verifies corpus identity and reproduces the shipped
artifact byte for byte. The artifact retains no corpus text.
