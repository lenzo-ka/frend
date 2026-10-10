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

### Usage

```python
import frend

spoken = frend.normalize("Meet me at 5:30. It costs $12.")
aligned = frend.normalize("Meet me at 5:30.", offsets=True)
behavior = frend.resolve_behavior(["google-tn"])
print(behavior.kwargs)
print(aligned.text)
for unit in aligned.units:
    print(unit.output_span, unit.source_span, unit.reader, unit.provenance)
```

`normalize` sentence-breaks documents with icukit and returns the first-choice spoken
text. Pass `offsets=True` when a screen reader or aligner also needs the source mapping;
inter-sentence whitespace is one space in the returned text, and leading and trailing
whitespace is trimmed. Offset units still cover those trimmed boundary spans explicitly.

Named behavior schemas compose in caller order, with later presets winning key by key.
Resolve shipped names or JSON files with `resolve_behavior`, then spread its immutable
`kwargs` into `normalize`; the result also records member digests, setters, and any opaque
icukit sections. Explicit call keywords remain Python's normal duplicate-key error.

Resolving a behavior schema only composes its configuration; it does not load external
profile data. Applying the `google-tn` behavior to `normalize` does load profile data and
requires both corpus-derived tables, built from the same verified Google TN inventory.
Configure both local tables explicitly before using that profile:

```sh
FREND_GOOGLE_TN_PROFILE_PATH=acronym_surfaces.json \
FREND_GOOGLE_TN_BRITISHISMS_PATH=britishisms.json \
python - <<'PY'
import frend

behavior = frend.resolve_behavior(["google-tn"])
print(frend.normalize("****", **behavior.kwargs))
PY
```

### Eleven-locale service profile

The checked service profile covers these eleven locales, in this order:

- `en_US`
- `es_MX`
- `es_ES`
- `fr_FR`
- `de_DE`
- `pt_BR`
- `it_IT`
- `zh_CN`
- `ko_KR`
- `ja_JP`
- `pt_PT`

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

## Development

```
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

The sibling repos (tiergraph, icukit, ipakit) are installed editable during development. CI tests
against the released icukit and, in a second job, against icukit's `main`, so frend can
use icukit changes before they are released; a frend release needs the first job green.

### Reproducing the data tables

The tables under `frend/data` are measured from public corpora, and each builder's
`--check` re-derives its table from them byte for byte. To get the corpora:

```
python tools/fetch_corpora.py            # both sources, into ../tn-corpus
python tools/fetch_corpora.py --verify   # check a copy you already have
python tools/build_type_priors.py --check
python tools/build_spoken_priors.py --check
.venv/bin/python -B tools/build_grouped_id_priors.py \
  --locale en_US --source-id google/tn-en_with_types --pool training \
  --receipt /tmp/frend-grouped-id-priors-rebuild.json
.venv/bin/python -B tools/build_grouped_id_priors.py --check \
  --locale en_US --source-id google/tn-en_with_types --pool training \
  --receipt /tmp/frend-grouped-id-priors-check.json
```

`tools/corpora.json` pins every source by sha256: the Google/Sproat English
text-normalization corpus from Kaggle (CC BY-SA 4.0; about 3.9 GB, 20 GB unpacked; no
account needed), and the English text-normalization test tables of NVIDIA
NeMo-text-processing at a fixed commit (Apache 2.0). Pass `--root` to put them elsewhere,
and point the builders at the shards with `FREND_TN_CORPUS_DIR`.

## The name

frend is the front end. It began as irn, the Intermediate Representation Normalizer,
which was a backronym: the name always came from the front end, Fe, then iron, then irn,
celebrating the immortal Bond of the Burghs, Edinburgh and Pittsburgh.
