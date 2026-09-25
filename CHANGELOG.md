# Changelog

## Unreleased
- Spoke icukit #99's numeric durations ("1:47.22", "2:30") as TIME, where the corpus
  files them. Each field reads as a cardinal in ICU's wide unit form, joined by ICU's
  list patterns for units ("one minute nineteen seconds"). A written fraction of the
  seconds reads as ICU's decimal and, as the corpus reads a race time, as its digits in
  milliseconds after "and" ("... eighteen seconds and eighty five milliseconds",
  lexical). TIME matched rises from 1700 to 1856 of 2000 sampled rows; rows left
  unverbalized, mostly times with seconds, fall from 94 to 10. The tests and the table
  builder need an icukit release carrying #99 (`FlexibleNumericDurationDetector`).
- Said a written era as it is written, following icukit #103's capture form. A wide
  name ("300 Before Christ", "5 Common Era") is said as its words rather than spelled;
  an abbreviation is still spelled ("b c e"), and its wide alternative is now the name
  of its own CLDR family, read from ICU's data ("2000 CE" also "two thousand Common
  Era", not "Anno Domini").
- Remeasured the spoken-priors table on icukit `3421be9` (#102 to #106, unreleased):
  DATE matched rises from 1993 to 1996 of 2000 (weekday-first day-first dates, other
  locales' month names). MEASURE is unchanged at 1933; 11 rates written with no amount
  ("/km²", icukit's new `UnitValue`) move from unrecognized to unverbalized.
- Recognized and spoke URLs, email addresses and bare domains, which ICU and icukit leave
  to frend (`frend.electronic.ElectronicDetector`). A span is a scheme URL, a "www."
  address, an email address, or a domain whose last label is in IANA's top-level-domain
  list (vendored); every host must pass ICU's IDNA (UTS #46) and only maximal spans are
  read, so icukit's readings inside a URL lose to it in the 1-best. Each run is said as
  the corpus is measured to say it (`data/electronic_priors.json`, from 49,001 aligned
  rows): a letter run as a word or spelled by its shape, a top-level domain by its own
  evidence ("com" a word, "edu" spelled), a digit run as ICU's cardinal or year or digit
  by digit, a separator by its measured name ("dot", "slash", "dash"). Only "@" is
  lexical ("at"): the corpus holds no email address. ELECTRONIC is a new measured kind
  in the spoken priors: 1605 of 2000 sampled rows match.
- Stated in the README that frend reads plain text: Markdown, HTML, XML (including
  SSML) and rich text must be handled before the string reaches frend and icukit, and
  frend has no markup modes yet.
- Spoke and measured measures. A measure (icukit #97) reads its amount as frend reads
  any number and its unit as ICU names it: icukit formats the value in the unit's wide
  form and frend cuts ICU's own number out, leaving the unit in the plural and order
  CLDR gives ("60 km" "sixty kilometers", "60 km/h" "sixty kilometers per hour"). A
  rate also reads with the unit's plural after "per", as the corpus does ("578.3/km2"
  "... per square kilometers"). A mixed measure speaks each component and joins them
  with ICU's list pattern for units ("5'10\"" "five feet, ten inches"). A percent now
  reads its written fraction digits ("79.20%" "seventy nine point two o percent").
  MEASURE is a new measured kind: 1924 of 2000 sampled rows match. Its shares are
  conditioned on the reading's ICU unit, so a rate learns the corpus's plural after
  "per" (0.902) without every unit taking it; percent is its own sub-key.
- Spoke icukit #97's era years and year-less dates, from ICU where locale data says
  it. An era year reads as a cardinal and the letters of its written era, as the
  corpus reads it ("500 BC" "five hundred b c"), with ICU's wide era name ("five
  hundred Before Christ") as an alternative. A day-month date also reads day first
  ("18 September" "the eighteenth of september"), as a full date already did. A year
  whose ICU reading says "oh" also reads "o" ("1908" "nineteen o eight"). DATE matched
  rises from 1949 to 1993 of 2000 sampled rows; rows with no matching alternative
  fall from 45 to 1. Percent by name ("12 percent", "5 per cent") was already spoken
  through ICU's percent name; the corpus files it under MEASURE, measured next.
- Spoke icukit #95's time zones and Roman ordinals. A written time zone follows the
  time as its letters ("10 PM ET" "ten p m e t", "18:00 UTC" "eighteen hundred u t c"),
  as the corpus reads it, so it is no longer left unspoken. A Roman ordinal also reads
  with "the" ("V." "the fifth", "XIVth" "the fourteenth"); a written arabic ordinal
  does not. A dot-separated time ("3.14" as a time) loses to the decimal on captures
  and on the corpus prior for its shape, and a test pins both. TIME matched rises from
  1547 to 1700 of 2000 sampled rows and ORDINAL from 1976 to 1994.
- Raised the icukit floor to `icukit>=0.5`, the first release carrying the readings
  frend now speaks (#90 through #97: letter-digit runs, day-period and zoned times,
  decades, Roman ordinals). Against 0.4.0 frend failed to import.
- Remeasured the spoken-priors table against icukit #97, which reads dates without a
  year ("1 July"), dotted months ("Oct. 2006") and era years ("500 BC"), measures by
  wide names and ASCII marks, mixed measures, and "percent". DATE matched rises from
  1946 to 1949 of 2000; unrecognized DATE rows fall from 30 to 6 and those with no
  matching alternative rise from 24 to 45, dates frend does not yet say. No other kind
  changed: frend measures no MEASURE kind yet, and the mixed-measure detector is not in
  its profile.
- Remeasured the spoken-priors table against icukit #95, which reads time zones after a
  time, "." between hour and minute, and Roman ordinals ("V.", "XIVth"). TIME matched
  rises from 1485 to 1547 of 2000 sampled rows and ORDINAL from 1971 to 1976; no other
  kind changed. Rows #95 now recognizes and frend does not yet speak are counted as no
  alternative matched: TIME 14 to 160 (mostly a time zone, which is not yet said) and
  ORDINAL 2 to 20 (Roman ordinals). Unrecognized TIME rows fall from 417 to 199.
- Renamed the project from irn to frend, the front end. The package, import and
  distribution name is now `frend` (`import frend`; entries below name the old
  `irn.*` modules, which are now `frend.*`). The corpus directory variable is
  `FREND_TN_CORPUS_DIR` (was `IRN_TN_CORPUS_DIR`), the reading-profile class is
  `FrendReadingProfile`, and the graph namespaces are `https://ogion.org/frend/...`
  with the `frend` prefix. Nothing else changed.
- Spoke and measured the reading types icukit #90 added. A time with a written day
  period speaks its written hour and the period's letters ("5pm" "five p m", "12am"
  "twelve a m"), a zero-led minute reads "oh five" or "o five", and an on-the-hour
  24-hour time also reads "twenty hundred"; seconds stay unverbalized. A plural numeral
  reads year-style and cardinal-style ("1990s" "nineteen nineties", "'90s" "nineties"),
  and the type priors file it under DATE, as the corpus does. A letter-digit token reads
  as its runs ("3D" "three d", "1080p" "ten eighty p"). A Roman numeral also reads as an
  ordinal ("II" "the second"), and a written possessive keeps its "'s". The
  spoken-priors table now measures TIME (1485 of 2000 sampled rows matched) and
  ORDINAL (1971 of 2000); DATE rises from 1903 to 1946 with decades; no kind fell.
- Marked spelled-out abbreviation readings with their own source. With icukit's
  spell-out expansion type, "MD" gives "M D" (`icukit-spell-out:title/follows-name`)
  beside "Maryland" (`icukit-abbreviation:region/address`) and the Roman reading
  "one thousand five hundred", all three competing in the lattice.
- Blended sparse sub-key shares toward the kind-level share. A fraction denominator's
  source share is now `(sub-key count + 5 × kind share) / (sub-key matched + 5)`
  (`irn.spoken_priors.SUB_KEY_PRIOR_STRENGTH`), so the 186 of 317 denominators seen
  once no longer decide their ranking outright, while well-attested ones barely move
  ("1/2" → "one half" at 0.986 from 283 rows). A source the sub-key never saw now
  keeps a kind-derived share instead of going unmeasured. The top-ranked source
  changes for 4 denominators. The shipped table is unchanged.
- Spoke a written plus sign: "+5" gives "plus five", with "five" kept as the fallback,
  so no capture on a verbalized reading is left unspoken across the recognition
  profile. The spoken-priors recognition profile now builds its date group from
  `date_detectors` instead of `all_detectors`, which also returned two number
  detectors and made every number a duplicate reading; the regenerated table differs
  only in its recorded detector classes.
- Verbalized two reading families that previously fell back to their surface. A written
  ordinal (`ordinal:flexible`, "29th") is spoken through ICU's ordinal spellout
  ("twenty-ninth"), and outcovers its bare digits in the 1-best. An abbreviation
  (icukit's `abbreviation` reading) yields every lexicon expansion as an alternative,
  keeping ambiguity ("Dr." gives Doctor and Drive), with the sense and cue in each
  provenance; a surface the lexicon does not expand ("Ms.") stays unverbalized. A
  written leading zero may go unsaid: "0.75" also gives "point seven five".
- Stopped dropping written material that a reading captures but no spoken form says.
  Each verbalizer now declares the captures it speaks, and every other capture on a
  verbalized reading is recorded in the new `VerbalizedUnit.unspoken`, so it is known at
  verbalization time instead of lost. A written weekday is now spoken: "Monday, March 16,
  1908" leads every date form with "Monday, " (ICU's `EEEE` name, from either of
  icukit's weekday encodings), after ranking, so curated keys and measured shares are
  unchanged. A written plus sign is recorded as unspoken, not yet said. The weekday
  moves three sampled corpus dates from unmatched to matched, so `spoken_priors.json`
  was regenerated: date matched 1900 to 1903 of 2000.
- Added `tools/fetch_corpora.py` and its manifest `tools/corpora.json`, which fetch and
  verify the public corpora the data tables are built from: the Google/Sproat English
  text-normalization corpus from Kaggle, pinned by archive and by each of its 100 shards,
  and NeMo-text-processing's English test tables at a pinned commit. Nothing is used
  unless its sha256 matches. The data README no longer advertises the corpus-free
  `--corpus nemo` build, which was removed earlier.
- Added attribution conditioned on one decoder token sequence, with per-reading
  posteriors, output-equivalent classes, exact rational measured-prior label splits
  built directly from Decimal measurements, explicit no-evidence members,
  foreign-output refusals, and zero-mass results. Raised the uncapped tiergraph floor
  to `tiergraph>=0.2.3`, where `OutputPlan` first appears.
- Added deterministic finite-state-grammar and AT&T acceptor text exports for
  alignment DAGs, with polynomial max-product epsilon removal, Decimal factors,
  float32 representability checks, explicit work budgets, symbol tables, and
  provenance manifests. FSG decimal probabilities preserve the exact language,
  and their best products agree with exact graph products within a computed
  manifest tolerance derived from precision and longest weighted path; every
  manifest transition also carries its exact rational factor. Products recovered
  from AT&T negated natural-log weights agree within their computed manifest
  tolerance; finite decimal logs are not claimed to round-trip exactly. Neither
  format claims path sums or reader score arithmetic.
- Added a word-level alignment DAG over all carrier tilings and distinct token forms,
  with measured span-relative reading factors, explicit evidence markers, resource
  refusals, and cached counting and log plans. The exact 1-best resolver is unchanged.
- Raised the tiergraph floor to `tiergraph>=0.2.2`, where `PathPlan` first appears,
  without an upper cap.
- Defined spoken normalization by splitting at Unicode punctuation and whitespace
  before lowering each token. The builder and alignment graph share this contract;
  Greek final sigma is now local to each token.
- Stopped cutting date spoken forms to a prefix. A date's month-first and day-first
  alternatives were sliced to a fixed count in generation order, which is not a ranking
  because the alternatives are unweighted, so a longer set lost forms silently. Every
  generated form is now kept; the composed graph's own bound refuses an oversized set
  instead. A test widens the year forms and fails when either slice is restored.
- Declared icukit as a runtime dependency, `icukit>=0.4`, now that it has a PyPI release.
  The floor is the release irn is verified against, installed from the wheel.
- Extended sampled spoken-form coverage with trailing-token normalization, ICU-derived compact and currency-name recognition, and alternatives for compact numbers, leading-point decimals, decimal `o`, day-first dates, signed fractions, and region-expanded US dollar names. Each generated table records the full recognition profile, and every added alternative remains unweighted.
- Added exact decimal, conventional fraction, and bare currency spoken alternatives from ICU locale data and English lexical forms attested by the sampled corpus. Currency minor units and zero-minor-unit collapse are included without changing alternative ranking.
- Added a sampled corpus table measuring which verbalizer sources reproduce attested spoken forms, with unmatched-form evidence and a read-only lookup API. The measurement does not affect alternative ranking or verbalization output.
- Dependency requirements are floors only, `tiergraph>=0.2` with no upper cap. A
  sibling release that breaks irn should break irn's tests where it can be seen,
  rather than be held back silently by a cap. Verified against the PyPI wheel.
- Enumerate the covers once instead of discovering completeness by doubling. The
  resolver used to ask the fold for a small number of witnesses, judge from the
  ranking whether it had them all, double the request and run the whole fold
  again. It now asks once for a bound far above any practical input and relies on
  the fold's own truncation flag, which it already reported. If that bound is ever
  reached the resolver **refuses** rather than returning what came back: a prefix
  handed to canonical selection looks exactly like a complete answer, and that is
  the failure the loop existed to avoid. A test lowers the bound and asserts the
  refusal, and fails behaviorally without it.
- Recorded what `_geometry_rank` is for, having established it the hard way. Its
  docstring said the fold "encodes exactly this", which is false: the fold ranks by
  a weight varying per position and candidate, so two covers can share the geometry
  rank and carry different fold values. Grouping by the fold's value instead --
  which looked like removing a duplicated rule -- drops structural ambiguity and
  changes a resolved winner, measured against four tests. The rank is a
  **coarsening** of the fold's order, and the coarsening is what makes two span
  signatures at one geometry a structural ambiguity rather than a ranking.
- Moved to tiergraph 0.2.0, which is now the floor rather than a preference. A
  fold with `ranked_output` **required** a `tie_policy` at 0.1.0 and **refuses**
  one from 0.2.0, so no source satisfies both versions. `resolve_cover`'s
  geometry fold declared `TiePolicy.CHOOSE_FIRST` alongside ranked output and
  now declares neither. Nothing changed in what it returns: ranked selection
  orders equal-valued witnesses by their canonical witness path, which is total
  wherever item labels are distinct, so the policy was never consulted. The
  frozen verbalization test is the evidence -- it passes unchanged, and it would
  not if the ordering had moved. Measured: 103 tests pass on 0.1.0 before the
  change and 103 pass on 0.2.0 after it, the same tests.
- Project scaffold: package skeleton, house-style tooling (ruff, pytest), and the
  composition design (tiergraph substrate + icukit recognition + ipakit phonetics).
