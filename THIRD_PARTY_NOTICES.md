# Third-party notices

frend's original software and project-authored data are licensed under the BSD
2-Clause License in `LICENSE`. The distribution also contains the derived data and
third-party material described below. A license named for one set of files does not
change the license of unrelated files.

## Google text-normalization corpus

- **Source and attribution:** the English `en_with_types` data from Richard Sproat and
  Navdeep Jaitly, *RNN Approaches to Text Normalization: A Challenge* (2016),
  <https://github.com/rwsproat/text-normalization-data>.
- **Source ID:** `google/tn-en_with_types`.
- **License:** Creative Commons Attribution-ShareAlike 4.0 International
  (`CC-BY-SA-4.0`). The full license is in `LICENSES/CC-BY-SA-4.0.txt`.
- **Covered files:** `frend/data/en/abbreviation_priors.json`,
  `acronym_priors.json`, `electronic_priors.json`, `electronic_span_priors.json`,
  `grouped_id_priors.json`, `inch_context_priors.json`, `number_priors.json`,
  `range_priors.json`, `spelled_token_priors.json`, `spellout_dictionary.json`,
  `spoken_priors.json`, `telephone_priors.json`, `type_priors.json`, and
  `zero_priors.json`, plus `frend/data/en/context/index.json` and every `.cart` file in
  `frend/data/en/context/trees/`.
- **Modifications:** frend filtered the corpus's training split, normalized and
  classified records, counted or aggregated observations, selected measured
  spell-or-say surfaces, and trained and serialized context decision trees. The
  original corpus shards and sentences are not distributed. The two exact-surface
  tables are `spellout_dictionary.json` and `spelled_token_priors.json`.

The covered files, and adaptations made from them, are offered under CC BY-SA 4.0.
Attribution, an indication of the modifications, and ShareAlike apply to those files.
Per-file provenance records give the exact builder, inputs, split, and further
transformation details.

## Unicode CLDR and ICU

- **Source and attribution:** Unicode CLDR and ICU data and generated behavior from
  Unicode, Inc., including CLDR 48 and ICU 78.3.
- **Source IDs:** `cldr/48`, `icu/78.3/cldr-date-patterns`,
  `icu/78.3/fraction-percent`, `icu/78.3/reflective-generation`, and
  `icu/78.3/time`.
- **License:** Unicode License V3 (`Unicode-3.0`). The full copyright and permission
  notice is in `LICENSES/Unicode-3.0.txt` and must accompany copies.
- **Covered files:** `frend/data/date_rules/*.json`,
  `frend/data/root/icu_shape_backfill.json`, and the citation/build receipts
  `frend/data/normalization/cldr-48-electronic.json`, `icu-78.3.json`, and
  `icu-78.3-time.json`.
- **Modifications:** frend generated locale-specific date patterns, recognition-shape
  estimates, and compact normalization-rule receipts from selected CLDR/ICU data. The
  JSON files record their exact source version and build command.

## libphonenumber

- **Source and attribution:** Google's libphonenumber project, release v9.0.41,
  `PhoneNumberMetadata.xml`,
  <https://github.com/google/libphonenumber/tree/v9.0.41>.
- **Source ID:** `libphonenumber`.
- **License:** Apache License 2.0 (`Apache-2.0`). The full license is in
  `LICENSES/Apache-2.0.txt`.
- **Covered file:** `frend/data/normalization/libphonenumber-9.0.41.json`.
- **Modifications:** selected descriptor fields were normalized as canonical JSON and
  hashed. The source XML is not distributed.

The separately installed `phonenumbers` runtime dependency carries its own license and
distribution notices.

## IANA top-level-domain list

- **Source and attribution:** Internet Assigned Numbers Authority (IANA),
  `tlds-alpha-by-domain.txt`, <https://data.iana.org/TLD/tlds-alpha-by-domain.txt>.
- **Source ID:** `iana/tlds-alpha-by-domain`.
- **License:** `LicenseRef-IANA-public`; IANA publishes the file as root-zone registry
  data for importing into software.
- **Covered file:** `frend/data/root/tlds-alpha-by-domain.txt`.
- **Modifications:** none. The file is vendored unchanged, including its version line.

## Festival

- **Source and attribution:** The Festival Speech Synthesis System, Centre for Speech
  Technology Research, University of Edinburgh, `lib/tokenpos.scm`, commit
  `5db6ee949f4e4e9da25c87bd03b90e69f9011393`.
- **Source ID:** `festvox/festival`.
- **License:** `LicenseRef-Festival` under the notice reproduced below.
- **Covered file:** `frend/data/en/context/festival_classes.json`.
- **Modifications:** the word lists `regnal_names`, `titles`, and `sections` are retained
  unchanged, including their original spellings, but serialized as JSON and used only
  as context feature classes. Festival's rules that consume those lists are not copied.

> The Festival Speech Synthesis System. Centre for Speech Technology Research,
> University of Edinburgh, UK. Copyright (c) 1996,1997. All Rights Reserved.
> Permission is hereby granted, free of charge, to use and distribute this software and
> its documentation without restriction, including without limitation the rights to
> use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this
> work, and to permit persons to whom this work is furnished to do so, subject to the
> following conditions: 1. The code must retain the above copyright notice, this list
> of conditions and the following disclaimer. 2. Any modifications must be clearly
> marked as such. 3. Original authors' names are not deleted. 4. The authors' names are
> not used to endorse or promote products derived from this software without specific
> prior written permission. THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS
> WORK DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE UNIVERSITY OF
> EDINBURGH NOR THE CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
> DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
> IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
> CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

## Public linguistic citation receipts

The files listed below are frend-authored normalization receipts. They cite public
linguistic examples or rules but do not redistribute source manuals, pages, articles,
or books. They are recorded in `frend.data_sources` as
`LicenseRef-Citation-Only`; each JSON receipt supplies its source URL, revision,
retrieval method, and the manual or scripted transformation. The resulting rule records
are licensed as frend project data under BSD-2-Clause.

| Source attribution | Covered receipts |
| --- | --- |
| german-grammar.de (`german-grammar/fractions`); Serlo Education e.V. (`serlo/fractions`); Rat fuer deutsche Rechtschreibung (`rechtschreibrat/amtliches-regelwerk-2018`); Duden online (`duden/online`); Goethe-Institut (`goethe/a1-deutsch-online`) | `german-fractions.json`, `serlo-fractions.json`, `rechtschreibrat-fractions.json`, `duden-*.json`, `goethe-electronic.json` |
| Boletin Oficial del Estado (`boe/gaceta`); Real Academia Espanola and ASALE (`rae/brae`, `rae/dpd-2`, `rae/gramatica-basica`); Secretaria de Educacion Publica (`sep/nueva-escuela-mexicana`); Universidad de Alicante (`ua/matesfacil`) | `boe-clock-seconds.json`, `rae-*.json`, `sep-*.json`, `ua-mixed-fractions.json` |
| Office quebecois de la langue francaise (`oqlf/bdl`); Academie francaise (`academie-francaise/dictionnaire`, `academie-francaise/questions-de-langue`); Ministere de l'Education nationale, Eduscol (`eduscol/fractions-cycle-3`) | `oqlf-fractions.json`, `academie-*.json`, `eduscol-fractions.json` |
| Accademia della Crusca (`accademia-crusca/consulenza`); Istituto della Enciclopedia Italiana, Treccani (`treccani/enciclopedia-italiano`, `treccani/vocabolario`) | `crusca-clock.json`, `treccani-*.json` |
| Banco de Mexico (`banxico/numismatica`); Banco Central do Brasil (`bcb/cmn-1076`, `bcb/plano-real`) | `banxico-money.json`, `bcb-*.json` |
| Japan Ministry of Education, Culture, Sports, Science and Technology (`mext/math-grade4`) | `japanese-fractions.json`, `mext-clock.json` |
| National Institute of Korean Language (`nikl/online-qna`, `nikl/basic-dictionary`, `nikl/standard-dictionary`) | `korean-*.json`, `nikl-*.json` |
| Ciberduvidas da Lingua Portuguesa (`ciberduvidas/consultorio`); Universidade Estadual de Londrina (`uel/matematica-essencial`); IMPA and OBMEP (`impa/obmep`) | `ciberduvidas-*.json`, `uel-fractions.json`, `impa-fractions.json` |
| Jixi Education Cloud (`jixi/education-cloud`); Life Education journal (`life-education/journal`); Urumqi Municipal People's Government (`urumqi/government`); Beijing Language and Writing Committee Office (`beijing-language/numbers-publications`) | `chinese-fractions.json`, `chinese-mixed-fractions.json`, `chinese-percent.json`, `beijing-numbers-publications.json` |

All receipt paths in this table are relative to `frend/data/normalization/`.

## Acapela vendor documentation

- **Source and attribution:** Acapela TTS language manuals for French (France),
  Portuguese (Brazil), and Portuguese (Portugal).
- **Source ID:** `acapela/tts-language-manuals`.
- **License:** not stated by the cited vendor documentation. Use remains provisional.
- **Covered files:** `frend/data/normalization/acapela-telephone-fr-FR.json`,
  `acapela-telephone-pt-BR.json`, and `acapela-telephone-pt-PT.json`.
- **Modifications:** frend manually transcribed source claims about telephone reading
  behavior into its own compact rule receipts. No manual text, PDF, or screenshot is
  redistributed.

## Sources registered without shipped derived artifacts

NVIDIA NeMo text processing is registered under Apache-2.0 solely as compatibility
evidence. No NeMo-derived fixture or data artifact is included in this distribution.
Wikidata is likewise registered as a possible CC0 source, but no shipped artifact in
this release names it as its source.
