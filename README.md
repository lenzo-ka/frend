# frend

**frend** -- the front end. It irons out running text:
recognize the formatted values in it, resolve overlapping readings into a best
non-overlapping cover, and verbalize the result toward a spoken form.

frend is a *composition* over three sibling projects:

- **tiergraph** -- the substrate: a layered hypergraph with a general, law-checked
  semiring fold (Boolean / counting / min-plus / max-plus, n-best, provenance). Resolution
  is a fold over the candidate hypergraph; the parse trace is the provenance dial of the
  same fold.
- **icukit** -- recognition, recognition-only: strict ICU-inverting detectors plus flexible
  CLDR-generated recognizers that deposit candidate readings (including the non-canonical
  spellings real text carries).
- **ipakit** (later) -- phonetics/IPA, for the pronunciation half of verbalization.

Pipeline shape: `text -> recognize (icukit) -> resolve (tiergraph.fold) -> verbalize -> phonetize (ipakit)`.

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

Status: recognize, resolve and verbalize are built, including a keep-all mode that
carries every reading and its spoken forms instead of one best cover. A word-level
alignment graph built from that output exports as finite-state grammar and acceptor text
for a decoder, and attributes aligned output back to reading classes. Phonetization
(ipakit) is not built yet.

## Development

```
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

The sibling repos (tiergraph, icukit, ipakit) are installed editable during development.

### Reproducing the data tables

The tables under `frend/data` are measured from public corpora, and each builder's
`--check` re-derives its table from them byte for byte. To get the corpora:

```
python tools/fetch_corpora.py            # both sources, into ../tn-corpus
python tools/fetch_corpora.py --verify   # check a copy you already have
python tools/build_type_priors.py --check
python tools/build_spoken_priors.py --check
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
