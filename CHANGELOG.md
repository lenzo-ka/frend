# Changelog

## Unreleased
- Read abbreviations as running text writes them. icukit's lexicon lists "Mr.", "St.",
  "vol."; the corpus writes "mr", "st", "Vol". `frend.abbreviation_variants` derives each
  lexicon abbreviation's period-less, ICU lower-case and ICU title-case forms (from the
  135 surfaces ending in "." with a lowercase letter, no inner period and more than one
  letter, so "IN", "OK" and "p." make none; a form that is itself a lexicon surface is
  left to it) and `AbbreviationVariantDetector` reads them ("abbreviation:variant"), in
  the evaluator's profile (`reading_detectors`). A variant is spoken as its lexicon
  expansions and as written, and every abbreviation reading, a variant's or an exact
  lexicon surface's ("St."), is ranked by a new measured table,
  `data/en/abbreviation_priors.json` (`tools/build_abbreviation_priors.py`, every tenth
  training shard): how the corpus says each key (the ICU lower case less one trailing
  period) by its written case (lower, title, upper), blended toward the key at the spoken
  priors' strength. The corpus writes no key with a period and 23 keys (st, dr, mr, ...)
  never in title case, so "Mr" ranks by `mr` (mister 8,621 of 8,756). So "Mr McVeigh"
  reads "mister mcveigh", "st Paul" "saint paul", "Vol 3" "volume three", "ltd"
  "limited", while "no", "sat", "Mrs" and "miss" stay as written first (the corpus says
  them so) with the expansion kept. An all-capitals form is still a letter run, spelled
  first ("MR" m r), and is now also offered its expansions after the letters ("mister");
  which should lead is left to context. A dotted chain is spelled in any case ("e.g." e
  g, "j.r.r." j r r: the corpus spells every chain it writes, "e.g." 2,432 of 2,437
  lower rows), and a chain in another case borrows the lexicon's expansions ("E.G." for
  example, after the letters). Published first choice (shard 99) 99.05% becomes 99.16%
  (+99 tokens: PLAIN +95, LETTERS +4), any reading 99.27% becomes 99.38%, sentences
  90.23% become 91.30%; held-out shard 95 first choice 99.07% becomes 99.17% (+94:
  PLAIN +89, LETTERS +5), any reading 99.28% becomes 99.39%, sentences 90.79% become
  91.79%; no token loses its first choice or its any-reading match on either shard, and
  running text is unchanged. ARCTIC's own profile reads the same graphs until
  `build_graphs-P4.patch` adds the variant reader; with it, the seven "Mr"/"Mrs" prompts
  gain their spoken reading (909 of 909 known readings), 77 graphs gain a branch, total
  paths 4,248 become 4,586 and the largest graph stays at 216 paths.
- Ship nothing derived from an LDC corpus: a test checks that no file under
  `frend/data/` names an `ldc/` corpus in its provenance (kal: every `ldc/*` store id is
  internal-only for frend).
- Train on the corpus's own split. The en_with_types README splits its 100 shards
  training 00-89, runtime eval 90-94 and test 95-99; `tools/google_tn_rows.py` now
  names the three pools (`TRAINING_SHARDS`, `RUNTIME_EVAL_SHARDS`, `TEST_SHARDS`), and
  `HELD_OUT_SHARDS` is 90-99, not 95 and 99 alone. Every builder reads training shards
  only: `type_priors.json` all 90, and the spoken, zero, acronym and electronic tables
  every tenth (00, 10, ..., 80), where shard 90 was the tenth. Shard 95 stays the
  held-out running-text shard and 99 the published test; 90-94 are a second held-out
  pool. Every table is rebuilt:
  - `type_priors.json`, 98 shards to 90: 1,848 shapes become 1,800 (48 seen only in
    90-94 drop out, none is new) and 56,237,940 surfaces become 51,649,056 (`N:N` time
    61,570 becomes 56,603).
  - `spoken_priors.json`, 10 shards to 9: 2,000 rows per class become 1,800 (4,000 to
    3,600 for SYMBOL); no kind-level source share moves by more than 0.6 of a point
    (SYMBOL's silence, 81.9% to 82.4%), and the fraction denominator 2 is matched 252
    times, not 283.
  - `zero_priors.json`, 10 shards to 9 (DIGIT "o" 33,028 becomes 29,579).
  - `acronym_priors.json`, 10 shards to 9: spelled 853,152 becomes 767,364 and word
    321,069 becomes 288,728; `roman:CIV`, `roman:CML`, `roman:DVI` and `roman:LXV`
    fall under the 20-sighting floor (452 keys become 448).
  - `electronic_priors.json`, 10 shards to 9: 53,257 rows (49,001 aligned) become
    47,981 (44,151); five top-level domains fall out (`army`, `bd`, `gm`, `jm`,
    `play`).

  The one-best golden is regenerated on the rebuilt type table (9 of its 13 rows move
  their hashes; no first choice moves). Re-baselined, the published report (shard 99)
  moves by two tokens: first choice 99.05% and any reading 99.27% as before, sentences
  90.24% become 90.23%; "DVI" is no longer spelled (LETTERS first choice 86.1% becomes
  86.0%) and one long URL loses its any-reading match (ELECTRONIC any reading 77.6%
  becomes 75.5%). Running text (1 of 110 first) is unchanged, and shard 95 is
  identical per token (99.07%, 99.28%, 90.79%) and in running text (13,167 triples).
  ARCTIC keeps every path count and alternative (902 of 902 known readings, max 216
  paths); 192 graphs change only in their prior values.
- Read a run of capitals in any script. The letters reader matches capitals by Unicode
  general category `Lu` (ICU's set), in one ICU script, not `[A-Z]`: "ÉCU" and "СССР"
  are letter runs, and capitals of two scripts ("AΒC", Latin and Greek) are not one run.
  Initials follow the same rule. The acronym builder counts a token by the same
  predicate (`frend.letters.is_letter_run`), so it no longer counts capitals the reader
  cannot match; on the ten sampled shards no token moves, and `acronym_priors.json` is
  unchanged. The vowels behind the acronym keys (`letters.cv_pattern`,
  `electronic.letter_key`) now come from the locale's `lexical.json` (`letter.vowels`);
  a locale without them forms no vowel key. The evaluator reads a shared profile,
  `tools/reading_profile.py`: `reading_detectors(locale)`, the spoken-priors profile
  plus the abbreviation and letters readers, every reader built for the locale asked
  for; `build_spoken_priors._detectors(locale="en_US")` takes the locale too, and
  Russian builds no English reader. ARCTIC keeps its own reader set and does not read
  this profile. No reading moves: every corpus builder's `--check` passes, the
  published and held-out reports are identical, and the ARCTIC graphs are
  byte-identical.
- Hold every hand-written spoken form in a per-locale table,
  `frend/data/<locale>/lexical.json`, each form with the reason ICU and CLDR do not
  give it (`why`). `frend/data/en/lexical.json` holds every form frend labels
  `lexical:en_US`: the spoken zero ("o", "oh" before a minute, "o" for ICU's year
  "oh", and the zero words weight is shared across), "o'clock" and "hundred" for a
  time, the possessive "'s", "at" for "@", the bare currency names and "united states
  dollar", "half" and "quarter" and "a half", "the" before an ordinal, "plus", the
  plural of a number word, "and" before milliseconds, and the plural after "per". The
  range connectors and separators and the acronym vowels are in the table too, unread
  until the readers that use them land. `frend/data/ru/lexical.json` is empty, so for
  Russian each of those features is off (a zero-led minute reads as ICU's cardinal).
  The source label stays `lexical:en_US` byte for byte (the spoken table keys on it),
  and no reading moves: every corpus builder's `--check` passes, the published and
  held-out reports are identical, and the ARCTIC graphs are byte-identical. The range
  connector's `why` states only what was checked: no resource walked gives English a
  spoken range connector; CLDR's calendar data does hold "at" and "of" in English, but
  outside its interval patterns (a test walks `calendar/gregorian` in root, en and
  en_US, and fails on Chinese, whose interval patterns write 至).
- Keep the data by locale: each table moves under the locale it is for,
  `frend/data/<locale>/` (`en/type_priors.json`, `en/spoken_priors.json`,
  `en/zero_priors.json`, `en/acronym_priors.json`, `en/electronic_priors.json`, and
  `en/exceptions.json`, formerly `exceptions/en.json`), and the cross-locale tables go
  to `frend/data/root/` (the ICU shape backfill, IANA's top-level domains). The new
  `frend.locale_data` looks a locale up along its chain, `en_US` → `en` → `root`, but
  never finds a measured table in `root`. A locale with none gets `None`, never English
  counts. The measured loaders take a keyword-only `locale` (default `en_US`):
  `source_prior`, `load_spoken_prior_table`, `load_prior_table`,
  `load_electronic_priors`, and the zero and acronym tables. Each measured table's
  provenance now names its `locale` (`en`) and `corpus` (`google-tn:en_with_types`)
  beside the shards it counted. No count moves: `type_priors.json` keeps its 98
  shards, its `generated` date and its counts (the one-best golden's counts pin is
  unchanged and only its path is re-pointed), every builder's `--check` passes, and
  no reading moves (the published and held-out reports are identical). The
  package-data globs are now `data/*/*.json` and `data/*/*.txt`.
- Hold two shards out of every corpus table (`tools/google_tn_rows.py`):
  `output-00099-of-00100`, the published test shard, and `output-00095-of-00100`, a
  held-out shard to accept and select changes on. Every builder now reads its shards
  through `training_shards`, and each table lists the shards it counted
  (`provenance.shards`, or `sample_rule.shards` in `spoken_priors.json`).
  - **Counting overlap, fixed.** `data/type_priors.json` counted all 100 shards,
    including the test shard it is scored on. It is rebuilt from the other 98 (1,864
    shapes become 1,848; 57,386,447 surfaces become 56,237,940; for example `N:N` time
    62,839 becomes 61,570). The rebuild changes no reading: the published report is
    byte-identical (first choice 99.05%, any reading 99.27%, sentences 90.24%, running
    text 1 of 110 first), and so is shard 95's. The spoken, zero, acronym and electronic
    tables never read shard 95 or 99 and are unchanged. The one-best golden is
    regenerated on the rebuilt table (9 of its 13 rows move their hashes; no first
    choice moves) and now pins the table's counts.
  - **Selection overlap, disclosed.** Earlier entries here chose and justified changes
    by their figures "on the published test set", shard 99. Those figures were
    selected on the test shard and are not held-out results. From now on changes are
    accepted and selected on shard 95, and shard 99 is reported once per change.
  - `tools/evaluate_google_tn.py --held-out-shard NAME` adds a held-out section: per
    token over NAME's first 100,000 lines, cut as the test shard is, and running text
    over the whole shard. Shard 95's baseline: first choice 99.07%, any reading 99.28%,
    sentences 90.79% (92,425 tokens); 13,167 running-text triples.
- Spell a chain of initials ("J.R.R. Tolkien" "j r r", "C.S. Lewis" "c s"): a
  capitals abbreviation icukit's lexicon lists with no expansion (a sentence-break
  entry) is now spelled rather than left as written, and a chain the lexicon does not
  list reads one initial at a time.
- Say a date written year first ("2008-09-30") day first, as the corpus does ("the
  thirtieth of september two thousand eight"): the date measurement's sub-key gains
  `year-first` beside `month-first` and `day-first`, so a year-first date no longer
  ranks like "March 5, 2024". On the published test set DATE rises from 97.1% to 99.8%
  first choice and all tokens from 98.97% to 99.05%.
- Require icukit 0.8.0, the first release with CLDR symbol names, so the symbol
  reading below reaches every install; the guards for older icukit are gone.
- Say a zero as the corpus does (kal, reconsidering the earlier "oh"): readings that
  differ only in a zero's word ("three point zero five" / "three point o five";
  ICU's "nineteen oh-five" / "nineteen o five") share their weight by a measured
  per-zero table, `data/zero_priors.json` (`tools/build_zero_priors.py`). The corpus
  never says "oh": a date's zero is "o", a digit run's always "o", a decimal's
  fractional zero "o" 74% of the time, a measure's 54%, a time's 68%; an integer part
  stays "zero". On the published test set DATE rises from 95.7% to 97.1% first choice,
  DECIMAL from 89.1% to 93.5%, MEASURE from 90.8% to 93.7%, and all tokens from 98.92%
  to 98.97%.
- Spell out a run of capitals or an initial (`frend.letters`): "ATM" "a t m", "UFOs"
  "u f o's", "S." "s" (over icukit's "South"), each also said as a word, ranked by the
  acronym measure. `data/acronym_priors.json` gains a consonant-vowel pattern key
  ("GUS" is mostly said, "GWR" spelled) and per-surface counts for runs icukit reads as
  Roman numerals, so "II" stays "two" and "CD" is spelled. On the published test set
  LETTERS rises from 41.1% to 86.1% first choice and all tokens from 98.31% to 98.92%.
- Read a standalone symbol or letter of another script by name (`frend.symbols`), from
  icukit's CLDR symbol names (icukit #136, `icu_abbreviations(kinds=["symbol"])`) and
  ICU's formal character names, with silence always offered: the corpus picks "and" for
  "&", "number" for "#", a Greek letter's name ("alpha"), and nothing for "." "," "-"
  and a character of a script it does not read ("風"). A character inside a word ("R&D")
  is left alone. SYMBOL is a new measured kind over the corpus's VERBATIM and PUNCT rows
  (silence counts as a target there), ranked per code point for a symbol and per script
  for a letter: 3286 of 4000 match. On the published test set VERBATIM rises from 9.2%
  to 89.6% first choice, PUNCT stays at 100%, and all tokens from 97.44% to 98.31%.
- Added `tools/evaluate_google_tn.py`, which scores frend's first choice on the published
  Google text-normalization test set (the first 100,000 lines of `output-00099-of-00100`,
  as Sproat & Jaitly 2016 and Bakhturina et al. 2022 use), per class, overall and per
  sentence, beside whether any reading frend offers matches. The corpus's own token
  boundaries are used, so tokenization is not scored; frend reads each token alone,
  where the published models see its sentence. First result: 97.44% of tokens first
  choice (97.64% any reading), 82.40% of sentences.
- Read a number written as plain digits digit by digit as well, as spelled-out letters
  are (kal: "it's like spell out"): "2013" also "two zero one three" and the corpus's
  "two o one three", "068" also "o six eight" (the zero its value drops). Only the
  number reading gets it: a year read as a date is never spelled digit by digit (kal:
  "I wouldn't read a year like that if I knew it was a year"). A single digit, a grouped
  ("1,000"), signed or fractional number does not. The corpus's DIGIT class is now a
  measured kind: 1779 of 2000 sampled rows match.
- Fixed what an independent review (Codex, at kal's request) found in #19 to #24:
  - A case citation is read only where a page follows ("339 U.S. 629"), so "The 339
    U.S. troops" is no longer a citation that drops "U.S.".
  - An era written "BC." or "AD." is left to icukit: the period ends the sentence.
    Dotted letters ("500 B.C.") are still read.
  - An acronym ranks by its own corpus evidence where icukit's lexicon lists it
    ("NASA" is said as a word 2118 times to 4, "FBI" always spelled), blended toward
    its shape; a dotted acronym ("U.S.") gets no word reading ("us").
  - A date ranks by its written order, measured: the spoken-priors table conditions
    DATE on it (month-first, day-first, other), so "March 5, 2024" leads with "March
    fifth" and "5 March 2024" with "the fifth of March" (349 to 22 and 524 to 21 in the
    corpus sample), with or without an era or a time.
  - The written "at" in a date and time is labeled as written words, not lexical.
- Raised the icukit floor to `icukit>=0.7`, the release carrying what frend now uses from
  icukit #111 to #132: dates with times, eras and quarters, relative dates, the curated
  acronyms, and zone captures by IANA ID. Verified from the PyPI wheel: 465 passed.
- Offered only a zone's own long names where icukit captures the zone's IANA ID (icukit
  #132, 0.7): "9 AM IST", read once as Asia/Kolkata and once as Europe/Dublin, now gives
  "India Standard Time" on the first reading and "Irish Standard Time" on the second,
  each from ICU's display names for that zone. Where the capture holds no IANA ID
  (icukit 0.6), every listed name is still offered.
- Remeasured the spoken-priors table on icukit `59172f9` (#130: two- and three-digit
  trailing years read only under their own opt-in type). DATE matched is unchanged at
  1997 of 2000; one row moves from "no alternative matched" to "unrecognized".
- Read an acronym spelled and as a word, weighted as the corpus measures its shape, before
  icukit's long forms (kal's ruling, split by source): "FBI" gives "f b i", "fbi" and
  "Federal Bureau of Investigation"; "BBC", which has no vowel, is spelled first. The
  weights come from a new table, `data/acronym_priors.json`
  (`tools/build_acronym_priors.py`), which counts how often the corpus spells an
  all-capitals token (LETTERS) or leaves it as a word (PLAIN) by length and vowel. A
  spelled form icukit already gives ("M D") takes the weight rather than being repeated.
- Spoke relative dates (icukit's `date:relative`) as ICU's wide style says them. A named
  phrase reads as ICU's wide phrase for its direction and unit ("last Fri." "last
  Friday", "last mo." "last month", "yesterday"); a numeric one as ICU's wide numeric
  form with frend's number in place of ICU's ("1 hr. ago" "one hour ago", "in 2h" "in
  two hours"). The formatter is built as icukit builds it.
- Spoke a date that carries a time, an era or a quarter (icukit #115, #118, #121, #123),
  part by part in written order through frend's own date and time speech: "March 5,
  2024 at 2:07 PM" "March fifth, twenty twenty-four at two oh seven p m" ("at" said
  where it is written, as in ICU's long date-time pattern; nothing for a comma), "Tue
  2:07 PM" "Tuesday, two oh seven p m", "Mar 5, 2024 AD" "... a d" or "... Anno
  Domini", and a quarter as ICU names it ("Q1 2024" "first quarter twenty twenty-four",
  also "q one ..."). The corpus holds none of these as one token, so the table is
  unchanged.
- Read and spoke written forms ICU writes nowhere, which icukit leaves to frend (kal's
  ruling; `frend.written_forms.WrittenFormsDetector`): a time with a space after the
  colon ("7: 31" "seven thirty one"), a case citation ("339 U.S.", the corpus's
  "three hundred thirty nine", also with "u s"), spaced digits ("6 3" "six three"), and
  an era written with periods, attached, or first ("500 B.C.", "4AD", "A.D. 1066"),
  spoken through frend's existing time, number and era speech. TIME matched rises from
  1856 to 1971 of 2000. The corpus's DIGIT class turned out to be plain digit strings
  read digit by digit ("2013" "two o one three"), not spaced digits; it stays outside
  the measured kinds.
- Read a time-zone abbreviation both spelled and as its long name (kal: "EST should have
  two reads"): "5:30 pm EST" gives "... e s t" and "... Eastern Standard Time". The names
  come from icukit's list of ICU's abbreviations (`icu_abbreviations`, icukit 0.6.0); an
  ambiguous abbreviation keeps every name ("IST": India and Irish Standard Time), and
  one ICU does not name ("Z") stays spelled. The ranking is the corpus's.
- Said a time zone or day period written in words as those words. icukit #116 reads long
  zone names ("10 PM Eastern Standard Time", "7 PM New York Time") and worded day
  periods ("2 in the afternoon"), which frend had spelled letter by letter ("e a s t e r
  n ..."); an abbreviation ("ET", "PM", "Z") is still spelled, as the corpus reads it.
- Remeasured the spoken-priors table on icukit `a365345` (#111 to #114, after 0.6.0,
  unreleased): three-digit years and dotted weekdays raise DATE matched from 1996 to
  1997 of 2000, and "Rs" read as rupees raises MONEY from 1911 to 1916. No other kind
  changed. frend does not yet speak relative dates, so #114's relative weekdays stay as
  written.
- Raised the icukit floor to `icukit>=0.6`, the first release carrying what frend now
  uses from icukit #99 to #109: numeric durations, units written with no amount
  (`UnitValue`), CLDR's era variants and wide eras, and curated unit spellings. Against
  0.5.0 the tests failed to import `FlexibleNumericDurationDetector`.
- Remeasured the spoken-priors table on icukit `e91dd58` (#107 to #109, unreleased):
  composed units and the curated unit spellings ICU does not write ("lbs", "sq km",
  "per km²") raise MEASURE matched from 1944 to 1954 of 2000. No other kind changed.
- Spoke icukit #99's numeric durations ("1:47.22", "2:30") as TIME, where the corpus
  files them. Each field reads as a cardinal in ICU's wide unit form, joined by ICU's
  list patterns for units ("one minute nineteen seconds"). A written fraction of the
  seconds reads as ICU's decimal and, as the corpus reads a race time, as its digits in
  milliseconds after "and" ("... eighteen seconds and eighty five milliseconds",
  lexical). TIME matched rises from 1700 to 1856 of 2000 sampled rows; rows left
  unverbalized, mostly times with seconds, fall from 94 to 10. The tests and the table
  builder need an icukit release carrying #99 (`FlexibleNumericDurationDetector`).
- Spoke a unit written with no amount ("/km²", "per second"), which icukit #102 reads as
  a `UnitValue`: ICU's wide form of the unit for one, with ICU's number cut out ("per
  square kilometer"), the corpus's own form. MEASURE matched rises from 1933 to 1944 of
  2000; no MEASURE row is left unverbalized. frend matches the value by name, so it still
  runs on icukit 0.5.0.
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
