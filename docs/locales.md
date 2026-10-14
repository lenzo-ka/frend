# Bringing up and extending locales

This guide is for frend contributors and for teams maintaining a fork with their own
locale data. It describes the code and data shipped in frend 0.1.0; it does not describe
a future extension interface.

Run repository commands from the checkout root, with the development environment active
and the checkout importable:

```console
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

Do not use `uv run`. Write reports and temporary probe output outside the repository.

## What a locale receives

frend has no accepted-locale allowlist. A well-formed ICU locale identifier first gets
the common ICU/CLDR-backed recognition and verbalization surface through icukit. The
locale chain is CLDR-aware, and package data is looked up from the most specific locale
to its parents. Therefore a locale can be useful before frend has any locale-specific
files for it.

On top of that baseline, `frend/generated_rule_locales.py` declares which locales have
compiled composition bundles for six families: date, time, fraction, money, electronic,
and telephone. The source declarations and receipts live under
`frend/data/normalization`; builders compile them into the family directories under
`frend/data`, which are the runtime artifacts. Date is the exception: its builder derives
its bundle directly from the installed ICU/CLDR data.

Finally, measured or trained tables are loaded along the locale chain but never from
another language. In 0.1.0 these are principally the English tables under
`frend/data/en`. A locale without such a table gets no measured prior; it does not borrow
English counts.

Reading order is additive. Measured evidence ranks first. Among unmeasured alternatives,
lexical or curated sourced forms rank ahead of generated forms. Generated composition
preserves existing sourced readings and alternatives; raw `surface:passthrough` and
`surface:unsupported` completeness fallbacks rank after spoken readings. Do not remove an
alternative merely to make another one preferred.

The release gate checks these eleven service profiles: `en_US`, `es_MX`, `es_ES`,
`fr_FR`, `de_DE`, `pt_BR`, `it_IT`, `zh_CN`, `ko_KR`, `ja_JP`, and `pt_PT`. Other valid
ICU locale identifiers use ICU fallback on a best-effort basis. They are accepted by the
API, but they are not release-qualified.

## Bring up a new locale

### 1. Probe the baseline

Choose the canonical ICU spelling and inspect what ICU actually supplies. This `nl_NL`
example records a useful date-only baseline:
records ICU, PyICU, and icukit versions, RBNF rules and fallback source, plural categories,
script, and CLDR formatter availability:

```console
export LOCALE=nl_NL
export ARTIFACT_DIR="$(mktemp -d)"
python -B tools/locale_recipe.py probe --locale "$LOCALE" --out "$ARTIFACT_DIR/probe.json"
python -c 'import os, frend; print(repr(frend.normalize("123", locale=os.environ["LOCALE"])))'
```

`tools/locale_recipe.py` only has the `probe` command in 0.1.0. It does not register or
build a locale.

If the ICU baseline is enough, no package registration is needed. Add a locale directory
such as `frend/data/<locale>` only when the locale needs package data. A `lexical.json`
entry must have `schema_version`, the matching `locale`, `source`, and nonempty `forms`;
every form has exactly a `value` and a specific `why`.

Making a locale a checked service profile is a separate, exact population change. Add
the locale to every item below in one change:

- the appropriate language entry in `LOCALES` in `tools/locale_gate.py`, which derives
  `GATE_LOCALES`;
- `tools/date_latency_inputs.json`, `tools/fraction_latency_inputs.json`, and
  `tools/time_latency_inputs.json`; the runtime gate requires each manifest to name
  exactly `GATE_LOCALES`, even when the locale adds only date rules, because the other
  probes verify the unchanged ICU baseline;
- `LOCALES` in `tests/test_service_locales.py` and the service-profile matrix in
  `README.md`, plus the checked-profile population sentence in this guide;
  `tests/test_project_metadata.py` checks that the code and README populations agree;
- either the language's NeMo fixture workload or a checked TSV workload described under
  [Gate setup](#gate-setup).

The date composition benchmark also reads `tools/date_latency_inputs.json`; use a probe
with a full-span year, month, and day detection. The fraction manifest needs `fraction`
and `percent` probes, and the time manifest needs one full-span clock probe. Run the
focused tests for those three manifests before the resource gates.

### 2. Add only the rule families you can support

Add the locale and family to `GENERATED_RULE_CAPABILITIES` in
`frend/generated_rule_locales.py`. That declaration controls builders and runtime
loaders, so the builder cannot create a new locale until the declaration exists. Prepare
the source document and tests, then declare the capability, build the compiled bundle,
and run its `--check` in the same change. Do not submit a declaration without its source,
compiled bundle, and tests.

| Family | Source declaration | Runtime bundle | Builder |
| --- | --- | --- | --- |
| date | installed ICU/CLDR date patterns | `frend/data/date_rules/<locale>.json` | `tools/build_date_rules.py` |
| time | `frend/data/normalization/time_<locale>.json` | `frend/data/time_rules/<locale>.json` | `tools/build_time_rules.py` |
| fraction | `frend/data/normalization/<locale>.json` | `frend/data/fraction_rules/<locale>.json` | `tools/build_fraction_rules.py` |
| money | `frend/data/normalization/money_<locale>.json` | `frend/data/money_rules/<locale>.json` | `tools/build_money_rules.py` |
| electronic | `frend/data/normalization/electronic_<locale>.json` | `frend/data/electronic_rules/<locale>.json` | `tools/build_electronic_rules.py` |
| telephone | `frend/data/normalization/telephone_<locale>.json` | `frend/data/telephone_rules/<locale>.json` | `tools/build_telephone_rules.py` |

The time, fraction, money, electronic, and telephone builders validate the
`frend-normalization/2` schema before compiling. Follow a neighboring locale of the same
family; the schemas are family-specific and the builders reject missing fields, unknown
selector values, incomplete selectors, invalid directions, ancestry gaps, and excess
derivations.

### 3. Record sources before rules

A source must state the spoken form or construction being encoded. A spelling list,
translation, convention inferred from examples, or another normalizer's output is not a
source claim. NeMo fixtures are compatibility evidence only and must never choose or rank
a reading.

For each source:

1. Register a stable source ID in `SHIPPABLE_SOURCES` in `frend/data_sources.py`. Record
   its license identifier, notice, allowed use, shipping class, and any unchanged vendored
   files. Namespace sources such as versioned ICU and CLDR use the existing namespace
   entries.
2. Add an acquisition receipt under `frend/data/normalization`. A receipt contains the
   registered `source`, a stable `locator` or `repository`, `revision`, timezone-bearing
   `fetched_at`, lowercase SHA-256 values for the acquired material and its license, and a
   reproducible `transform_command`. Add the permitted `use` where the source policy
   requires it.
3. Add the source to the normalization document's `sources`. Give every selector,
   resource, rule, weight, or curated row a nonempty `derived_from`. Each claim key must
   close through `ancestry` to the declared receipt.
4. Put the spoken form in the relevant resource or rule and cite the exact claim that
   states it. Electronic additions also need the reviewed claim TSV required by
   `tools/build_electronic_rules.py`.

If the source's terms are unknown or incompatible, stop. Do not mark it shippable by
analogy. Update `THIRD_PARTY_NOTICES.md` and include the required license text or notice
for every distributed source or derivative.

### 4. Build and verify bundles

After preparing the source declaration, add the capability declaration, run the builder
without `--check`, and immediately run its `--check` mode, which recompiles in memory and
compares the checked-in runtime artifact. Declaration, generated file, and successful
check travel together before submission. Run all affected builders; the commands are
intentionally explicit:

```console
python -B tools/build_date_rules.py
python -B tools/build_date_rules.py --check
python -B tools/build_time_rules.py
python -B tools/build_time_rules.py --check
python -B tools/build_fraction_rules.py
python -B tools/build_fraction_rules.py --check
python -B tools/build_money_rules.py
python -B tools/build_money_rules.py --check
python -B tools/build_electronic_rules.py --claims "$ELECTRONIC_CLAIMS"
python -B tools/build_electronic_rules.py --claims "$ELECTRONIC_CLAIMS" --check
python -B tools/build_telephone_rules.py
python -B tools/build_telephone_rules.py --check
```

`ELECTRONIC_CLAIMS` is the reviewed TSV for the electronic change, with the schema
enforced by the builder. A change that does not touch electronic rules does not run that
builder. Never edit a compiled family bundle as the source of truth.

### 5. Add behavior tests and regression pins

Add focused tests beside the family tests (`tests/test_date_rules.py`,
`tests/test_time_rules.py`, `tests/test_fraction_rules.py`, `tests/test_money_rules.py`,
`tests/test_electronic_rules.py`, or `tests/test_telephone_rules.py`). Cover:

- exact source-to-record and record-to-runtime provenance;
- a full-span detection and the newly sourced spoken alternative;
- the reading being **offered** in the keep-all alternatives;
- relevant near misses, caps, lazy loading, and unchanged locales;
- a first-choice assertion only when independent source or measured ranking justifies it.

NeMo targets may show a coverage opportunity, but a NeMo fixture must never establish
rank. Pin observed behavior with an independently sourced fixture. When a legacy sourced
reading remains ahead of the new one, assert that the new reading is offered rather than
forcing it to rank first.

## Extend or modify an existing locale

Start from the source declaration, not the generated bundle.

- To add a reading or alternative, add a cited resource or rule with its own stable ID and
  `derived_from`, then rebuild. Keep existing valid alternatives.
- To change the preferred reading, add or correct the source and adjust the declared
  priority or measured rank. Do not delete a valid alternative to manufacture a winner.
- To add a family the locale lacks, create that family's source declaration and receipts,
  add the family to `GENERATED_RULE_CAPABILITIES`, build its runtime bundle, and add the
  complete family tests and gates described above.
- To change a hand-written lexical form, edit the locale's `lexical.json`, preserve a
  concrete `why`, and run `tests/test_lexical_forms.py`.

After a coverage-changing fraction or time change, regenerate the exact change and
recovery manifests from paired base/head coverage reports, then verify the regenerated
files before committing:

```console
python -B tools/build_fraction_manifests.py --base "$BASE_COVERAGE" --head "$HEAD_COVERAGE"
python -B tools/build_fraction_manifests.py --base "$BASE_COVERAGE" --head "$HEAD_COVERAGE" --check
python -B tools/build_time_manifests.py --base "$BASE_COVERAGE" --head "$HEAD_COVERAGE"
python -B tools/build_time_manifests.py --base "$BASE_COVERAGE" --head "$HEAD_COVERAGE" --check
```

The other families pin their recovery witnesses in their focused test and witness files;
update the affected witness only from a qualified base/head run.

Any change that can affect alignment must regenerate the mutation receipt before a push.
This harness temporarily edits production files, so start it in a clean worktree, run it
alone, do not interrupt it, and inspect its only expected output change:

```console
python -B tests/run_align_mutations.py
git diff -- tests/data/align_mutations.json
```

## Pre-PR gates

Use a clean base worktree and the candidate worktree, the same environment, the same
external fixture checkout, and output directories outside both repositories. The gate
tools write reports; inspect their `failures`, `breached`, and negative-flip rows rather
than treating process success alone as evidence.

### Gate setup

Use the public NVIDIA NeMo text-processing fixture repository. Clone it outside frend,
record the revision used by the qualification report, and point `NEMO_ROOT` at the
language directories under its test tree:

```console
git clone https://github.com/NVIDIA/NeMo-text-processing.git /path/to/NeMo-text-processing
export NEMO_ROOT=/path/to/NeMo-text-processing/tests/nemo_text_processing
git -C /path/to/NeMo-text-processing rev-parse HEAD
```

`HEAD_REPO` is the candidate checkout. `BASE_REPO` is a separate clean worktree at the
commit the candidate is compared against; it must not be the candidate directory at an
older in-memory state. `BASE_GATE` and `HEAD_GATE` are separate output directories, and
`ARTIFACT_DIR` holds comparisons and benchmarks. Set and verify them before running a
gate:

```console
export HEAD_REPO="$PWD"
export BASE_REPO=/path/to/clean/base-worktree
export ARTIFACT_DIR="$(mktemp -d)"
export BASE_GATE="$ARTIFACT_DIR/base-gate"
export HEAD_GATE="$ARTIFACT_DIR/head-gate"
test -f "$HEAD_REPO/tools/locale_gate.py"
test -f "$BASE_REPO/pyproject.toml"
test -d "$NEMO_ROOT"
```

The candidate gate program must drive both coverage arms. That gives base and head the
same new locale population and probe manifests while `PYTHONPATH` selects the frend code
under measurement. `runtime-pair` performs the same alternating base/head isolation
internally; do not replace it with two unrelated `runtime` runs.

NeMo has no fixture directory for every language. If
`$NEMO_ROOT/<language>/data_text_normalization` is absent or empty, add a repository TSV
such as `tests/data/locales/<locale>_checked.tsv` with exactly these columns:

```text
written	spoken	class	source
02-03-2003	twee maart tweeduizenddrie	date	Exact public source locator and claim
```

Every row needs a nonempty, reviewable source locator. Repeat a written form on separate
rows when multiple spoken targets are admissible. Set
`CHECKED_WORKLOAD="$HEAD_REPO/tests/data/locales/${LOCALE}_checked.tsv"` and pass the file
to coverage and runtime with `--checked "$LOCALE=$CHECKED_WORKLOAD"`. The built-in
`pt_PT` checked workload remains included. A trial-only row must say `TRIAL` in its
source field and must not be committed. Confirm every new service locale has cases in
the resulting coverage report; a zero-row locale is not qualified.

### Unit, lint, and path gates

```console
ruff check .
ruff format --check .
python -m pytest tests/test_no_local_paths.py
python -m pytest
```

### Bundle checks and crash sweeps

Run every affected builder with `--check`, as above. Then run the sweep for every affected
family. The full time sweep intentionally omits `--quick`:

```console
python -B tools/sweep_fraction_normalization.py
python -B tools/sweep_time_normalization.py
python -B tools/sweep_money_measure.py
python -B tools/sweep_electronic_normalization.py --out "$ARTIFACT_DIR/electronic-sweep.json"
python -B tools/sweep_telephone.py --output "$ARTIFACT_DIR/telephone-sweep.json"
```

Date coverage is exercised by the common locale gate and its composition benchmark.

### Coverage, witness, and negative flips

Run coverage once from the base and once from the candidate against the same external
NeMo fixture checkout. NeMo remains report-only compatibility evidence:

```console
PYTHONPATH="$BASE_REPO" python -B "$HEAD_REPO/tools/locale_gate.py" coverage --nemo-root "$NEMO_ROOT" --out-dir "$BASE_GATE" --checked "$LOCALE=$CHECKED_WORKLOAD"
PYTHONPATH="$HEAD_REPO" python -B "$HEAD_REPO/tools/locale_gate.py" coverage --nemo-root "$NEMO_ROOT" --out-dir "$HEAD_GATE" --checked "$LOCALE=$CHECKED_WORKLOAD"
PYTHONPATH="$HEAD_REPO" python -B "$HEAD_REPO/tools/locale_gate.py" compare --base "$BASE_GATE/coverage.json" --head "$HEAD_GATE/coverage.json" --require-no-negative-flips --out "$ARTIFACT_DIR/coverage-comparison.json"
```

The comparison must show the intended reading is offered, no unexplained negative flip,
and no loss in unchanged locales. Measure the affected family's frozen recovery witness
when it has one:

```console
python -B tools/fraction_recovery_witness.py measure --repo "$HEAD_REPO" --nemo-root "$NEMO_ROOT" --witness tools/fraction_recovery_witness.json --out "$ARTIFACT_DIR/fraction-witness.json"
python -B tools/money_recovery_witness.py measure --repo "$HEAD_REPO" --nemo-root "$NEMO_ROOT" --witness tools/money_recovery_witness.json --out "$ARTIFACT_DIR/money-witness.json"
python -B tools/electronic_recovery_witness.py --repo "$HEAD_REPO" --witness tools/electronic_recovery_witness.json --manifest "$ELECTRONIC_MANIFEST" --out "$ARTIFACT_DIR/electronic-witness.json"
python -B tools/telephone_recovery_witness.py --subject-root "$BASE_REPO" --output "$ARTIFACT_DIR/telephone-base.json"
python -B tools/telephone_recovery_witness.py --subject-root "$HEAD_REPO" --output "$ARTIFACT_DIR/telephone-head.json"
python -B tools/telephone_recovery_witness.py --output "$ARTIFACT_DIR/telephone-comparison.json" --compare "$ARTIFACT_DIR/telephone-base.json" "$ARTIFACT_DIR/telephone-head.json" pr10/expected-recovery.json
```

`ELECTRONIC_MANIFEST` is the source-claim manifest bound by the frozen electronic
witness. Run only the affected witness tools. Freeze a new witness only after review,
never merely to make a failing comparison pass.

### Runtime and RSS ceilings

Collect alternating fresh-process base/head measurements and make the locale gate apply
its embedded latency and RSS ceilings:

```console
PYTHONPATH="$HEAD_REPO" python -B "$HEAD_REPO/tools/locale_gate.py" runtime-pair --base-repo "$BASE_REPO" --head-repo "$HEAD_REPO" --nemo-root "$NEMO_ROOT" --out-dir "$ARTIFACT_DIR/runtime-pair" --checked "$LOCALE=$CHECKED_WORKLOAD"
PYTHONPATH="$HEAD_REPO" python -B "$HEAD_REPO/tools/locale_gate.py" soak-pair --base-repo "$BASE_REPO" --head-repo "$HEAD_REPO" --out-dir "$ARTIFACT_DIR/soak-pair" --unseen
PYTHONPATH="$HEAD_REPO" python -B "$HEAD_REPO/tools/locale_gate.py" compare --base "$BASE_GATE/coverage.json" --head "$HEAD_GATE/coverage.json" --base-runtime "$ARTIFACT_DIR/runtime-pair/base/runtime.json" --head-runtime "$ARTIFACT_DIR/runtime-pair/head/runtime.json" --base-soak "$ARTIFACT_DIR/soak-pair/base/soak.json" --head-soak "$ARTIFACT_DIR/soak-pair/head/soak.json" --require-no-negative-flips --out "$ARTIFACT_DIR/resource-comparison.json"
```

Also run the absolute composition benchmark for every affected family:

```console
python -B tools/bench_date_compose.py --out "$ARTIFACT_DIR/date-benchmark.json"
python -B tools/bench_time_compose.py --out "$ARTIFACT_DIR/time-benchmark.json"
python -B tools/bench_fraction_compose.py --out "$ARTIFACT_DIR/fraction-benchmark.json"
python -B tools/bench_money_compose.py --out "$ARTIFACT_DIR/money-benchmark.json"
python -B tools/benchmark_electronic_rules.py --repo "$PWD" --out "$ARTIFACT_DIR/electronic-benchmark.json"
python -B tools/bench_telephone_compose.py --output "$ARTIFACT_DIR/telephone-benchmark.json"
```

### English S0 must not lose

If a change can affect English, run the same fixed S0 tuning sample on base and head and
write text-free per-sentence records outside the repository. Do not use evaluation
holdout data to choose rules or ranks.

```console
PYTHONPATH="$BASE_REPO" python -B "$BASE_REPO/tools/triage_misses.py" --corpus-dir "$CORPUS_DIR" --output-dir "$BASE_S0" --accuracy-only --per-sentence-out "$BASE_S0/per-sentence.json"
PYTHONPATH="$HEAD_REPO" python -B "$HEAD_REPO/tools/triage_misses.py" --corpus-dir "$CORPUS_DIR" --output-dir "$HEAD_S0" --accuracy-only --per-sentence-out "$HEAD_S0/per-sentence.json"
python -B tools/compare_runs.py --a "$BASE_S0/per-sentence.json" --b "$HEAD_S0/per-sentence.json" --intervals 1000 --json "$ARTIFACT_DIR/s0-comparison.json"
```

The report is `b_minus_a`. A score-changing change needs a favorable interval; a
score-inert robustness change must be non-negative and satisfy its property gates. Keep
all evaluation holdout data out of training, source selection, rule selection, and rank
tuning.

## Rules of the road

- Use ICU and CLDR first. Hand-roll detection or composition only when they supply no
  implementation; a frend rule still requires a source that states the spoken form.
- Do not add unsourced word lists. Hand-write a lexical form only when no source data
  interface supplies it, and record the specific reason in `lexical.json`.
- Train decisions that reproduce another system or corpus; do not hardcode that system's
  observed outputs as general linguistic rules.
- Never train or tune on evaluation holdout data. Keep training, tuning, acceptance, and
  report roles separate.
- Never commit local paths, private mount names, or client filesystem layout. Repository
  paths are relative; external artifact locations are caller-supplied.
- Treat corpus-derived tables according to their effective license. Record inputs,
  transformations, and license ancestry, and update `THIRD_PARTY_NOTICES.md`. Exact
  surface tables may need to remain outside the package even when aggregates can ship.

## Using your own data and locales

Today frend loads locale material only from the installed package's `frend/data`
directory. There is no runtime API for out-of-tree locale material in 0.1.0. A runtime
data API is planned, but this release does not contain one.

The supported route is to fork or vendor frend, add the locale and data using this guide,
build and check the bundles, run the gates, and install that fork. Keep the fork easy to
update:

- keep client data in its own locale, normalization, and receipt files;
- avoid unrelated edits to frend's builders and runtime;
- merge upstream regularly, then regenerate every affected compiled bundle;
- rerun bundle checks, focused tests, coverage, resource gates, English S0 when relevant,
  and the full suite before deploying the merged fork.

Client-supplied data has the same provenance obligations as upstream data. Use your own
source IDs and acquisition receipts, register their actual licenses in
`frend.data_sources`, preserve required notices, and ship only material whose license
permits the intended distribution. Do not reuse frend's source ID or license receipt for
different client material.
