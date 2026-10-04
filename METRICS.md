# Metrics and evaluation

This file defines frend's English text-normalization measurements. Percentages are
accuracies; `pp` means percentage points. Results are not comparable unless the data
cut, scorer, profile, and commit all match. Paths beginning with `lanes/` are relative
to `~/dev/lenzo/agents/untracked/frend/lanes/`.

## Data and split

The Google TN `en_with_types` split is fixed as follows:

- Training: shards 00--89. Only these shards may build shipped priors or local profile
  tables.
- Tuning: shards 90--94. S0 is a uniform reservoir sample of 20,000 complete sentences
  from each shard with seed `20260930`: 100,000 sentences and 1,200,533 tokens. Its
  fingerprint is `de8ecfe19becd1dd7a141b3ae8b481016c7f40600e5a875264c7bd16be04c1b6`.
- Acceptance: shard 95. The standard acceptance run scores its first 100,000 lines,
  which contain 7,576 sentences and 92,425 tokens.
- Report: shard 99 is report-only and may be scored at milestones, never for tuning or
  acceptance. The current `5db817b` checkpoint has not been reported on that cut; it
  was neither verified nor scored. An earlier commit, `276ef32`, has a same-cut report
  in `sota/report.md`. Decision runs pass `--held-out-shard output-00095-of-00100
  --skip-report-shard`, so shard 99 is not even verified or opened. Shards 96--98 are
  unused.

Sources: `checkpoint-5db817b/results.md`, its `shard95-*-receipt.json` files, and
`tools/google_tn_rows.py`; `sota/report.md` and `LOG.md` (2026-10-01) cover the earlier
report and milestone ruling.

The prepared NeMo comparison cut derives from Google TN shard 99. It was used for
measurement and keep/drop decisions in #55 and #56 before that provenance was noticed.
Those decisions stand, but since the 2026-10-01 ruling the cut is report-only and must
not select changes. Source: `LOG.md` (2026-10-01) and `nemo-scorer/results.md`.

## Scoring

- **First-choice tokens**: the fraction of corpus rows for which frend's best reading
  equals the target.
- **Any-reading tokens**: the fraction for which any enumerated reading equals the
  target. Enumeration is bounded to the first 64 alternative combinations per path.
- **Sentence accuracy**: the fraction of corpus sentences for which every token's first
  choice is correct.

Both prediction and target pass `normalize_spoken`: Unicode punctuation becomes a
space, whitespace is split, tokens are lowercased, and the result is joined with single
spaces. A target of `<self>` means the normalized written token; a target of `sil`
means empty output. ELECTRONIC letter notation is decoded before comparison. Embedded
`sil` words are retained by the headline scorer. `triage_misses.py
--strip-embedded-sil` is an opt-in diagnostic and is never a headline result.
The evaluator counts recognition, resolution, and verbalization exceptions as misses
rather than dropping them. `triage_misses.py` currently has a known bug: after such an
exception it substitutes the normalized written surface and can count the failed token
as correct when that surface matches the target; a fix is queued. Sources:
`tools/evaluate_google_tn.py`, `tools/google_tn_rows.py`, `tools/triage_misses.py`, and
`span-errors/results.md`.

At the checkpoint, the sentence miss rate is about ten times the token miss rate under
both profiles. This is expected: one bad token makes the whole sentence wrong. Source:
`checkpoint-5db817b/results.md`.

### Known blind spots

The row scorer reads one Google TN row at a time, with the rest of its sentence supplied
only as context. It therefore cannot score a reading that joins multiple corpus rows
into one span. This is why the useful `range:range` tree has no row-score effect.

Google TN's pre-normalization is undocumented. Typographic folding, UK-to-US
respelling, diacritic removal, and case rewriting are observations of the released
data and Kestrel targets, not stated corpus guarantees. The S0 sample contains none of
the 13 code points handled by frend's typographic fold, so that fold is score-inert on
S0. Sources: `tree-prune/probe.md`, `typographic-fold/results.md`, `ukus/results.md`,
and `LOG.md` (2026-10-02).

## Evaluation cuts

**Token strata.** SEEN means that the exact, case-preserved written form occurs in the
vocabulary extracted from all training shards 00--89; otherwise it is UNSEEN. The
extractor ID is `google-tn-tab-column-2-non-eos-v1`. The vocabulary is private external
evaluation state and is never shipped. Source: `strata/results.md`.

**Sentence strata.** `ALL_SEEN_NONTRIVIAL` ignores rows whose targets are `<self>` or
`sil`; `HAS_UNSEEN_NONTRIVIAL` is its complement. `ALL_SEEN` and `HAS_UNSEEN` are the
strict partition over every token. Headline strata below use the strict partition.
Source: `strata/results.md` and `tools/seen_strata.py`.

**Miss triage.** These classes partition first-choice misses; `capped` is a separate,
overlapping diagnostic.

| Class | Meaning |
|---|---|
| D | No detection span was produced. |
| S | Detections exist, but none covers the complete written token. |
| R | The correct normalized form is offered, but is not ranked first. |
| V | A complete span exists, but the correct form is absent from bounded readings. |
| P | A PLAIN miss whose normalized surface and target have different spellings but the same nonzero number of alphabetic words; most observed cases are respellings, diacritic changes, or case changes. |
| O | Another PLAIN or surface mismatch, such as expansion, spelling, or silence. |
| E | The verbalizer raised an exception. |

Sources: `s0-triage/results.md`, `ukus/results.md`, and `tools/triage_misses.py`.

## Intervals and change gates

Intervals are percentile bootstraps clustered by sentence. Absolute intervals sample
sentences; paired intervals draw the same sentence multiplicities for both arms and
report B minus A. The standard is 1,000 replicates, seed `20261002`, and a 95% interval.
A ratio draw with a zero denominator is undefined and omitted. An interval is reported
only when at least 95% of draws are defined; otherwise it is unreliable. Sources:
`intervals/results.md`, `tools/bootstrap_intervals.py`, and `tools/compare_runs.py`.

Score-changing candidates ship only when the S0 tuning interval excludes zero in the
favorable direction and the shard-95 acceptance result does not lose. Score-inert
robustness changes, such as #70's typographic fold, instead ship when S0 and shard 95 do
not lose and their change-specific property gates pass; #70's S0 interval included zero
while its identity, exact-example/equivalence, and latency gates passed. The score-change
rule rejected the R lane's connector recalibration after it gained on S0 but lost two
shard-95 tokens. Sources: `ranking-r/results.md` and `typographic-fold/results.md`.

## Profiles

The default profile is general-purpose frend. `google-tn` is an off-by-default
conformance profile. It loads locally built, externally stored tables for per-surface
acronym spell/say choices and Kestrel-style PLAIN conversions, including observed
UK-to-US spellings, diacritics, expansions, and other rewrites. The tables are built
from shards 00--89, are never package resources, and must share the same verified shard
identities. Sources: `ukus/results.md`, `kbest-eval/results.md`, and
`frend/profiles.py`.

The profile's aggregate gain is primarily memorization. On S0 it adds 2,213
first-choice tokens: 2,212 SEEN and one UNSEEN. At checkpoint 5db817b its UNSEEN delta
is +0.0203 pp with a 95% interval of [-0.0411, +0.0842], while its SEEN delta is
+0.1850 pp [+0.1769, +0.1934]. Source: `checkpoint-5db817b/results.md`.

## Frozen checkpoint: 5db817b

All intervals in this section use the standard 1,000-replicate sentence bootstrap.
Values are percent, except triage shares, which are percentage points of all tokens.
Source for all three tables: `checkpoint-5db817b/results.md` and the JSON receipts in
that lane.

| Set | Profile | First-choice tokens (95% CI) | Any-reading tokens (95% CI) | Sentences (95% CI) |
|---|---|---:|---:|---:|
| S0 | default | 99.5635 [99.5499, 99.5759] | 99.7648 [99.7551, 99.7740] | 95.4330 [95.3050, 95.5581] |
| S0 | google-tn | 99.7479 [99.7372, 99.7579] | 99.8439 [99.8354, 99.8519] | 97.4120 [97.3150, 97.5040] |
| shard 95 | default | 99.5477 [99.4948, 99.5999] | 99.7263 [99.6860, 99.7664] | 95.3933 [94.9050, 95.8421] |
| shard 95 | google-tn | 99.7263 [99.6841, 99.7691] | 99.8117 [99.7779, 99.8467] | 97.2941 [96.9377, 97.6640] |

| Set | Profile | Stratum (tokens) | First choice (95% CI) | Any reading (95% CI) |
|---|---|---|---:|---:|
| S0 | default | SEEN (1,195,618) | 99.5955 [99.5820, 99.6073] | 99.7856 [99.7763, 99.7943] |
| S0 | default | UNSEEN (4,915) | 91.7803 [90.9570, 92.6197] | 94.6897 [94.0386, 95.3354] |
| S0 | google-tn | SEEN (1,195,618) | 99.7805 [99.7704, 99.7900] | 99.8650 [99.8574, 99.8723] |
| S0 | google-tn | UNSEEN (4,915) | 91.8006 [91.0032, 92.6399] | 94.7101 [94.0873, 95.3533] |
| shard 95 | default | SEEN (92,052) | 99.5665 [99.5148, 99.6165] | 99.7382 [99.6985, 99.7781] |
| shard 95 | default | UNSEEN (373) | 94.9062 [92.3507, 97.0331] | 96.7828 [94.8424, 98.4931] |
| shard 95 | google-tn | SEEN (92,052) | 99.7458 [99.7041, 99.7866] | 99.8240 [99.7906, 99.8583] |
| shard 95 | google-tn | UNSEEN (373) | 94.9062 [92.3507, 97.0331] | 96.7828 [94.8424, 98.4931] |

| Set | Profile | D | S | R | V | P | O | E |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | default | .0233 [.0196,.0272] | .0330 [.0294,.0364] | .2012 [.1922,.2105] | .0466 [.0417,.0514] | .1293 [.1229,.1359] | .0031 [.0021,.0042] | 0 [0,0] |
| S0 | google-tn | .0245 [.0207,.0284] | .0330 [.0294,.0364] | .0960 [.0899,.1033] | .0466 [.0417,.0514] | .0490 [.0451,.0532] | .0031 [.0021,.0042] | 0 [0,0] |
| shard 95 | default | .0454 [.0262,.0667] | .0303 [.0185,.0422] | .1785 [.1472,.2141] | .0541 [.0368,.0731] | .1417 [.1158,.1675] | .0022 [0,.0055] | 0 [0,0] |
| shard 95 | google-tn | .0454 [.0262,.0667] | .0303 [.0185,.0422] | .0855 [.0635,.1104] | .0541 [.0368,.0731] | .0563 [.0405,.0727] | .0022 [0,.0055] | 0 [0,0] |

These tables remain frozen. The #70 typographic fold has no Google TN fold-score effect
on S0 or shard 95. After that checkpoint, #71 adds 10 first-choice tokens, 10 any-reading
tokens, and 10 correct sentences on S0 under each profile; shard 95 is byte-identical.
Sources: `typographic-fold/results.md`, `ranges-ids/results.md`, and
`ranges-ids/s0-*-paired.json`.

## Latency and robustness

#63 was measured on `kale.local` (arm64, Python 3.13.15, ICU 78.3). This table gives
pipeline p50 in milliseconds for warm calls and parent-observed process-wall p50 for a
fresh process per utterance. Source: `latency-ab/results.md`.

| Words | Warm default / google-tn | Cold default / google-tn |
|---|---:|---:|
| <=5 | 2.06 / 2.11 | 651.32 / 988.57 |
| 6--10 | 2.22 / 2.75 | 673.13 / 1006.24 |
| 11--20 | 3.49 / 4.61 | 684.57 / 1012.75 |
| 21--40 | 5.13 / 6.61 | 671.86 / 1060.92 |

Cold profile-on verbalization spends about 240--255 ms at the median loading and
validating its two external tables. In the 457-case fold benchmark, keep-all completes
457/457 cases, public k=1 completes 455/457, and direct ranked PATH k=64 completes
351/457 within its two-second per-mode bound; 99/100 long English PATH cases hit the
bound. Source: `latency-ab/results.md`.

#65 retains a 1,024-code-point sentence-break recommendation. It covers 49,367/49,469
measured literary sentences (99.794%) and Alice's 919-code-point maximum; the isolated
513--1,024 bucket has a 40.34 ms pipeline p99. The hard limits are 8,192 code points per
resolution unit, 8,192 per URL/domain candidate, 254 per email candidate, and 4,194,304
per document. These are refusal bounds, not truncation points. Source:
`length-bounds/results.md` and `length-bounds/timing-isolated.json`.

#64 finds that 99.002% of default tokens and 99.023% of google-tn tokens stabilize with
zero whole-token lookahead; both profiles have token-lookahead p50/p90/p99 of 0 and a
maximum of 113 tokens, or 226 code points. The proposed static bound is short for 78
tokens per profile, so these observations do not justify a shipping stream API. Source:
`streaming-eval/results.md`.

## Rulings and measured costs

- **2026-09-28, refined 2026-09-30:** filter demonstrable corpus non-readings from
  training, but retain valid silent connectors and genuine negatives. The refined
  misread-end filter drops 236 emission rows and 35 sampled joined rows, versus
  121,241 and 11,962 under the overbroad version. Source: `LOG.md` and
  `numbers/results.md`.
- **2026-10-01:** keep `range:range`. It is correct on 32/33 offline fires, while the
  row scorer observes none of those joined-span changes; pruning scores therefore stay
  unchanged. Source: `LOG.md`, `tree-prune/probe.md`, and `tree-prune/results.md`.
- **2026-10-01:** the NeMo cut is report-only because it derives from shard 99. It had
  already informed #55 and #56; those decisions stand. Source: `LOG.md`.
- **2026-10-02:** drop triage class A. Its corrected count is zero because Ruling A is a
  training filter; the other triage counts are unchanged. Source: `LOG.md` and
  `s0-triage/results.md`.
- **2026-10-02:** corpus-fitted acronym and orthographic behavior belongs in the
  off-by-default `google-tn` profile. Enabling the whole profile adds 2,213 S0 tokens,
  almost entirely SEEN; default output remains general-purpose. Source: `LOG.md` and
  `ukus/results.md`.

## Reproduction and receipts

Use the repository `.venv`, the catalog-verified corpus, bounded commands, and external
outputs. The important harness-to-figure mapping is:

| Figure | Command | Durable output |
|---|---|---|
| S0 score, strata, triage, absolute CIs | `tools/triage_misses.py --strata VOCAB --intervals 1000 --interval-seed 20261002 --per-sentence-out PATH` | `/Volumes/k02/processed/frend/google/tn-en_with_types/{strata,intervals}/` |
| Shard-95 score and CIs | `tools/evaluate_google_tn.py --held-out-shard output-00095-of-00100 --skip-report-shard --strata VOCAB --intervals 1000 --per-sentence-out PATH` | evaluator JSON/receipt paths; checkpoint copies are in `checkpoint-5db817b/` |
| Paired deltas and gates | `tools/compare_runs.py --a ARM_A --b ARM_B --intervals 1000 --interval-seed 20261002 --json OUT` | `.../intervals/s0-comparison.json` and lane `*-paired.json` files |
| Bootstrap implementation | imported `tools/bootstrap_intervals.py` | method, seed, defined draws, and floor are embedded in each report |
| Warm/cold profile latency | `tools/benchmark_latency.py --profile google-tn --corpus-dir CORPUS --output OUT` | `/Volumes/k02/processed/frend/latency-profile-ab-fugu-fixes-a50ee50.json` |
| Fold scaling | `tools/benchmark_fold.py --data-dir /Volumes/k02/processed/frend/fold-bench --output OUT --mode-timeout 2` | `/Volumes/k02/processed/frend/fold-bench/results/frend-lt-87a5e6a.json` |
| Streaming lookahead | `tools/measure_lookahead.py --corpus-dir CORPUS --output OUT` | `/Volumes/k02/processed/frend/streaming/s0-lookahead*.json` and `s0-static-*-fugu.json` |
| Length evidence | `lanes/length-bounds/measure-isolated-buckets.py` | `/Volumes/k02/processed/frend/length-bounds/` |

For `google-tn`, set `FREND_GOOGLE_TN_PROFILE_PATH` to the external acronym table and
`FREND_GOOGLE_TN_BRITISHISMS_PATH` to the external spelling table. The current tables
and their build/check receipts live under
`/Volumes/k02/processed/frend/google/tn-en_with_types/{ranker,ukus}/`. Exact invocations
and artifact names are recorded in the named lane `results.md` files.

## Published comparison target

Zhang, Sproat et al. report 99.84% all-token accuracy and 98.24% exact-sentence accuracy
on the standard English shard-99 cut. Those values are the comparison target, not a
claim about the S0 or shard-95 tables above. The current development split uses 90--94
for tuning and 95 for acceptance, keeps 99 report-only, scores corpus rows separately
with sentence context, and offers both default and corpus-conformance profiles.
Comparison to Zhang--Sproat remains indicative for the current `5db817b` checkpoint,
which has not been reported on the same shard-99 cut. An earlier commit, `276ef32`, has
a same-cut report, and milestone reports are permitted. Scorer differences must be
stated. Sources: `sota/report.md`,
`checkpoint-5db817b/results.md`, and `LOG.md` (2026-10-01).
