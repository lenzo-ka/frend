# frend

**frend** -- the front end. It irons out running text:
recognize the formatted values in it, resolve overlapping readings into a best
non-overlapping cover, and verbalize the result toward a spoken form.

frend builds its text-normalization pipeline on two sibling projects:

- **tiergraph** -- the substrate: a layered hypergraph with a general, law-checked
  semiring fold (Boolean / counting / min-plus / max-plus, n-best, provenance). Resolution
  is a fold over the candidate hypergraph; the parse trace is the provenance dial of the
  same fold.
- **icukit** -- recognition, recognition-only: strict ICU-inverting detectors plus flexible
  CLDR-generated recognizers that deposit candidate readings (including the non-canonical
  spellings real text carries).
- **ipakit** (caller-owned) -- an optional downstream phonetics/IPA step after frend
  returns words.

frend's pipeline is `text -> recognize (icukit) -> resolve (tiergraph.fold) -> verbalize
(words)`. A caller can then connect those words to its own `phonetize (ipakit)` step.

**Input is plain text.** frend and icukit read a text string: the characters a reader
would see, with nothing else in them. Markup of any kind -- Markdown, HTML or XML
(including SSML), rich-text formats -- must be handled before the string reaches frend:
removed, or turned into the text it stands for. Otherwise its syntax is read as text
("**5**" is not the number 5 to a reader of plain text, and a tag's attribute values
are recognized like any other characters). frend has no markup modes today; reading
through a markup layer, keeping its structure and offsets, may come later.

URLs, email addresses and bare domains are recognized by frend itself (ICU has no link
recognition, and icukit leaves them to frend), and spoken as the corpus is measured to
say them.

## Install

Once frend is published on PyPI, install it with `python -m pip install frend`. After
`v0.1.0` is tagged and until PyPI publication, install the tagged release directly with
`python -m pip install "frend @ git+https://github.com/lenzo-ka/frend.git@v0.1.0"`.
frend requires Python 3.12 or newer and icukit 0.10.1 or newer.

## Quick start

```python
import frend

print(repr(frend.normalize("Meet me at 5:30. It costs $12.50.")))
```

Expected output (the spaces are reader output and are significant):

```text
' meet   me   at   five thirty .  it   costs   twelve dollars and fifty cents .'
```

## API usage

```python
import frend

spoken = frend.normalize("Meet me at 5:30. It costs $12.")
aligned = frend.normalize("Meet me at 5:30.", offsets=True)

stream = frend.NormalizationStream(offsets=True)
first = stream.feed("Meet me at 5:30. ")
last = stream.close()
behavior = frend.resolve_behavior(["google-tn"])
print(behavior.kwargs)
print(aligned.text)
for unit in aligned.units:
    print(unit.output_span, unit.source_span, unit.reader, unit.provenance)
```

`normalize` sentence-breaks documents with icukit and returns the first-choice spoken
text. Leading and trailing source whitespace is discarded in both return forms, while
inter-sentence source whitespace contributes one space. Reader emissions can add spaces of
their own, including at the edges of the returned text and next to inter-sentence spacing.
Pass `offsets=True` when a screen reader or aligner also needs the source mapping; offset
units cover discarded boundary-whitespace spans explicitly.

Production sentence work is bounded at 1,024 code points by default. Set
`max_sentence_chars=None` only for an already bounded complete evaluation unit, select an
opt-in published model with `sentence_base=`, and use `reflow="lines"`, `"paragraphs"`, or
`"auto"` when hard-wrapped source should be interpreted through icukit's offset-preserving
view. `max_unit_chars` remains the independent lattice-work bound. Icukit
`RegexLimitError` exceptions propagate unchanged.

`NormalizationStream` accepts the same policy options. Each batch contains only settled
speech, optional absolute source/output alignments, and a monotone `pending_from`
watermark. It coordinates icukit's linear sentence and detection streams and keeps
retained source bounded. Output normally settles at a paragraph, a configured cap, an
explicit flush, or close; a cap or flush is an explicit context cut, so constructs that
straddle it need not match a single whole-document `normalize` call. Chunk boundaries
inside the same settlement windows do not change the answer.

Named behavior schemas compose in caller order, with later presets winning key by key.
Resolve shipped names or JSON files with `resolve_behavior`, then spread its immutable
`kwargs` into `normalize`; the result also records member digests, setters, and validated
icukit sections. Icukit sentence options are applied through the same runtime keywords;
schemas that set icukit detection, stream, or regex groups are rejected because frend
does not expose those receiving APIs.
Explicit call keywords remain Python's normal duplicate-key error.

Resolving a behavior schema only composes its configuration; it does not load external
profile data. Applying the `google-tn` behavior to `normalize` does load profile data and
requires both corpus-derived tables, built from the same verified Google TN inventory.
Configure both local tables explicitly before using that profile:

Set `FREND_GOOGLE_TN_PROFILE_PATH` to the built `acronym_surfaces.json` and
`FREND_GOOGLE_TN_BRITISHISMS_PATH` to the matching `britishisms.json`. The exact build
and verification commands are in [the data guide](frend/data/README.md#enacronym_priorsjson).

### Eleven-locale service profile

All eleven checked locales receive the common ICU-backed recognition surface. The
matrix lists frend's additional measured data or generated linguistic bundles. These
are checked service profiles, not an accepted-locale allowlist.

| Locale | Checked frend coverage beyond common ICU-backed recognition |
| --- | --- |
| `en_US` | English measured/default tables; measured telephone route |
| `es_MX` | generated date, time, fraction, and money bundles |
| `es_ES` | generated date, time, and fraction bundles |
| `fr_FR` | generated date, time, fraction, money, and telephone bundles |
| `de_DE` | generated date, time, fraction, and electronic bundles |
| `pt_BR` | generated date, time, fraction, money, and telephone bundles |
| `it_IT` | generated date, time, fraction, and money bundles |
| `zh_CN` | generated date, time, and fraction bundles |
| `ko_KR` | generated date, time, fraction, and money bundles |
| `ja_JP` | generated date, time, and fraction bundles |
| `pt_PT` | generated date, time, fraction, and telephone bundles |

Other ICU locale identifiers are accepted through ICU's fallback behavior on a
best-effort basis. They are not part of the checked service profile, so their available
readers and output can differ and are not release-qualified.

### icukit disk cache

On first use, icukit may compute deterministic Python detector tables and store their
marshal snapshots in the platform user cache directory (normally
`$XDG_CACHE_HOME/icukit` on Linux, or the platform equivalent). ICU objects, detector
instances, and application text remain in memory. Set `ICUKIT_CACHE_DIR` to choose
another cache root, or `ICUKIT_CACHE=0` to disable both disk reads and writes. Because
the cache uses Python marshal files, use only a directory trusted against modification
by other users.

### Input limits

The recognizer's document-scale guard accepts at most 4,194,304 Unicode code points by
default (`max_input_chars`). Lattice/choice resolution and verbalization have a
separate `max_unit_chars` work bound of 8,192 code points because they construct
per-position graph state. They refuse rather than truncate over either bound, and the
unit-bound error tells callers to sentence-break first.

Plain-text validation also refuses byte strings, NUL characters, every lone surrogate,
and text in which more than 1% of all code points have category `Cn` (unassigned) or
category `Cc` (control) other than the `Cc` members of Unicode White_Space: tab, line
feed, vertical tab, form feed, carriage return, and U+0085 NEXT LINE. All other Unicode
White_Space characters have separator categories and are accepted. Decode bytes as
strict UTF-8 and remove markup before calling frend.

`normalize` sentence-breaks documents before resolution. Callers of the lower-level
resolution APIs should sentence-break documents themselves and preserve sentence
offsets and order. At an untrusted ingress, call `validate_input` before passing text
directly to icukit's detectors; the resolution and verbalization entry points repeat
the check. Passing `max_input_chars=None` turns off only the document check, and
`max_unit_chars=None` turns off only the separate work bound; the plain-text checks
remain active.

By default, public recognition applies the declared `fold="typographic"` input fold:
curly single and double quotes become their ASCII forms; NBSP, figure space, thin space
and narrow NBSP become a space; and Unicode hyphen, non-breaking hyphen and minus become
hyphen-minus. Fullwidth digits U+FF10–U+FF19 become ASCII digits `0`–`9`; fullwidth
decimal/grouping punctuation between digits and signs beside digits become their ASCII
forms, as do recognized fullwidth Latin measure symbols directly after a number. Every
replacement is one code point, so aligned source spans still slice the raw input directly.
U+2013 EN DASH is not folded: the installed icukit accepts ICU's en dash in generated date
intervals but not an ASCII hyphen in the same patterns.
`NormalizedText.fold` and `ReadingLattice.fold` declare the applied fold; their lattice
also retains `raw_source_text`. Pass `fold=None` to disable it. Plain-string `normalize`
output has no metadata, so request `offsets=True` when fold provenance is required. The
convenience call `resolve_lattice(source_text=text)` performs recognition after folding.
A lattice built from caller-supplied detections reports `fold=None`, because the resolver
cannot prove which text those detections saw.

`ElectronicDetector` validates its input before recognition. It does not emit a prefix
of an oversized address: a contiguous URL/domain candidate over 8,192 code points or an
email candidate over 254 code points is omitted. Both recognition caps are configurable.
They are engineering work limits informed by, but not dictated by, the protocols: RFC
9110 recommends that HTTP senders and recipients support URIs of at least 8,000 octets
(a minimum, not a maximum), while RFC 5321 limits an SMTP forward or reverse path to 256
octets. frend's caps count Unicode code points, not protocol octets.

Status: recognize, resolve and verbalize are built, including a keep-all mode that
carries every reading and its spoken forms instead of one best cover. A word-level
alignment graph built from that output exports as finite-state grammar and acceptor text
for a decoder, and attributes aligned output back to reading classes. Phonetization is
outside frend; a caller may pass the returned words to ipakit.

Evaluation definitions, frozen results, and reproduction commands are in [METRICS.md](METRICS.md).

## API reference

[The public API reference](docs/api.md) lists every name exported by `frend`, with one
line describing its contract. The primary convenience API is `normalize`; request
`offsets=True` when source/output alignment and fold provenance are needed.

## Stability

frend 0.x is alpha software. Public APIs and normalization output may change between
minor releases. The exact outputs covered by the test suite are the release contract;
callers that depend on other output details should pin a version and qualify upgrades.

## License

frend is distributed under `BSD-2-Clause AND CC-BY-SA-4.0`: the original code and
project-authored data use the [BSD 2-Clause License](LICENSE), while Google-TN-derived
data files use CC BY-SA 4.0. See [Third-party notices](THIRD_PARTY_NOTICES.md) for the
file-level scope and all required license and attribution notices.

## Data provenance

[Third-party notices](THIRD_PARTY_NOTICES.md) summarizes every shipped third-party
source, license, attribution, and modification. [The data guide](frend/data/README.md)
records the per-table inputs, held-out splits, builders, and provenance policy.

## Development

Install an editable development checkout with `python -m pip install -e ".[dev]"`.
Run the suite with `python -m pytest`, lint with `ruff check .`, and check formatting
with `ruff format --check .`.

The sibling repos (tiergraph, icukit, ipakit) may be installed editable during development.
CI tests compatible released dependencies and, in a separate floor job, explicitly tests
icukit 0.10.1.

### Reproducing the data tables

The tables under `frend/data` are measured from public sources, and their builders'
`--check` modes rederive them byte for byte. The complete corpus setup, verification,
and builder commands live in [the data guide](frend/data/README.md); the large external
corpora are not part of this repository or package.

## The name

frend is the front end. It began as irn, the Intermediate Representation Normalizer,
which was a backronym: the name always came from the front end, Fe, then iron, then irn,
celebrating the immortal Bond of the Burghs, Edinburgh and Pittsburgh.
