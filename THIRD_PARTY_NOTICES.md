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

<!-- BEGIN GENERATED THIRD-PARTY FILE INDEX -->
## Transitive covered-file index

This index is generated from the shipped files' own provenance fields by
`python -B tools/build_third_party_notices.py`. It is authoritative for the
files covered by each source notice, including compiled derivatives.

### `academie-francaise/dictionnaire`

- `frend/data/fraction_rules/fr_FR.json`
- `frend/data/normalization/academie-clock-seconds.json`
- `frend/data/normalization/academie-clock.json`
- `frend/data/normalization/academie-demi-8e.json`
- `frend/data/normalization/academie-demi-9e.json`
- `frend/data/normalization/academie-quart.json`
- `frend/data/normalization/fr_FR.json`
- `frend/data/normalization/time_fr_FR.json`
- `frend/data/time_rules/fr_FR.json`

### `academie-francaise/questions-de-langue`

- `frend/data/money_rules/fr_FR.json`
- `frend/data/normalization/academie-money.json`
- `frend/data/normalization/money_fr_FR.json`

### `acapela/tts-language-manuals`

- `frend/data/normalization/acapela-telephone-fr-FR.json`
- `frend/data/normalization/acapela-telephone-pt-BR.json`
- `frend/data/normalization/acapela-telephone-pt-PT.json`
- `frend/data/normalization/telephone_fr_FR.json`
- `frend/data/normalization/telephone_pt_BR.json`
- `frend/data/normalization/telephone_pt_PT.json`
- `frend/data/telephone_rules/fr_FR.json`
- `frend/data/telephone_rules/pt_BR.json`
- `frend/data/telephone_rules/pt_PT.json`

### `accademia-crusca/consulenza`

- `frend/data/normalization/crusca-clock.json`
- `frend/data/normalization/time_it_IT.json`
- `frend/data/time_rules/it_IT.json`

### `banxico/numismatica`

- `frend/data/money_rules/es_MX.json`
- `frend/data/normalization/banxico-money.json`
- `frend/data/normalization/money_es_MX.json`

### `bcb/cmn-1076`

- `frend/data/money_rules/pt_BR.json`
- `frend/data/normalization/bcb-cmn-1076.json`
- `frend/data/normalization/money_pt_BR.json`

### `bcb/plano-real`

- `frend/data/money_rules/pt_BR.json`
- `frend/data/normalization/bcb-plano-real.json`
- `frend/data/normalization/money_pt_BR.json`

### `beijing-language/numbers-publications`

- `frend/data/normalization/beijing-numbers-publications.json`
- `frend/data/normalization/time_zh_CN.json`
- `frend/data/time_rules/zh_CN.json`

### `boe/gaceta`

- `frend/data/normalization/boe-clock-seconds.json`
- `frend/data/normalization/time_es_ES.json`
- `frend/data/normalization/time_es_MX.json`
- `frend/data/time_rules/es_ES.json`
- `frend/data/time_rules/es_MX.json`

### `ciberduvidas/consultorio`

- `frend/data/fraction_rules/pt_BR.json`
- `frend/data/fraction_rules/pt_PT.json`
- `frend/data/normalization/ciberduvidas-clock.json`
- `frend/data/normalization/ciberduvidas-fractions.json`
- `frend/data/normalization/ciberduvidas-meio.json`
- `frend/data/normalization/ciberduvidas-tercos.json`
- `frend/data/normalization/pt-default.json`
- `frend/data/normalization/time_pt_BR.json`
- `frend/data/normalization/time_pt_PT.json`
- `frend/data/time_rules/pt_BR.json`
- `frend/data/time_rules/pt_PT.json`

### `cldr/48`

- `frend/data/electronic_rules/de_DE.json`
- `frend/data/normalization/cldr-48-electronic.json`
- `frend/data/normalization/electronic_de_DE.json`

### `duden/online`

- `frend/data/electronic_rules/de_DE.json`
- `frend/data/fraction_rules/de_DE.json`
- `frend/data/normalization/de_DE.json`
- `frend/data/normalization/duden-eineinviertel.json`
- `frend/data/normalization/duden-electronic.json`
- `frend/data/normalization/duden-halb.json`
- `frend/data/normalization/duden-uhrzeit.json`
- `frend/data/normalization/duden-zweieinhalb.json`
- `frend/data/normalization/electronic_de_DE.json`
- `frend/data/normalization/time_de_DE.json`
- `frend/data/time_rules/de_DE.json`

### `eduscol/fractions-cycle-3`

- `frend/data/fraction_rules/fr_FR.json`
- `frend/data/normalization/eduscol-fractions.json`
- `frend/data/normalization/fr_FR.json`

### `festvox/festival`

- `frend/data/en/context/festival_classes.json`

### `german-grammar/fractions`

- `frend/data/fraction_rules/de_DE.json`
- `frend/data/normalization/de_DE.json`
- `frend/data/normalization/german-fractions.json`

### `goethe/a1-deutsch-online`

- `frend/data/electronic_rules/de_DE.json`
- `frend/data/normalization/electronic_de_DE.json`
- `frend/data/normalization/goethe-electronic.json`

### `google/tn-en_with_types`

- `frend/data/en/abbreviation_priors.json`
- `frend/data/en/acronym_priors.json`
- `frend/data/en/context/index.json`
- `frend/data/en/context/trees/05f9c74a5cbf.cart`
- `frend/data/en/context/trees/075cba7551ac.cart`
- `frend/data/en/context/trees/0a64b444bf49.cart`
- `frend/data/en/context/trees/0b4e0aa3e560.cart`
- `frend/data/en/context/trees/0b4e760886bf.cart`
- `frend/data/en/context/trees/1320fa15f3aa.cart`
- `frend/data/en/context/trees/1393e194e58a.cart`
- `frend/data/en/context/trees/18cc1535c691.cart`
- `frend/data/en/context/trees/1f2ef4893593.cart`
- `frend/data/en/context/trees/1f8d3ffd9018.cart`
- `frend/data/en/context/trees/21bb157e7032.cart`
- `frend/data/en/context/trees/2290bb44e800.cart`
- `frend/data/en/context/trees/2ddb36e16367.cart`
- `frend/data/en/context/trees/3048774c63c3.cart`
- `frend/data/en/context/trees/31034ca7e091.cart`
- `frend/data/en/context/trees/328a30953c3e.cart`
- `frend/data/en/context/trees/34a84c88b72b.cart`
- `frend/data/en/context/trees/3a9296711eb1.cart`
- `frend/data/en/context/trees/3caee6c74d29.cart`
- `frend/data/en/context/trees/3fe1b4526d5f.cart`
- `frend/data/en/context/trees/407a01b7830a.cart`
- `frend/data/en/context/trees/465750e66128.cart`
- `frend/data/en/context/trees/47fe893e2742.cart`
- `frend/data/en/context/trees/49e7a861adf9.cart`
- `frend/data/en/context/trees/4c2eed8a9df0.cart`
- `frend/data/en/context/trees/4db519f7a586.cart`
- `frend/data/en/context/trees/520bac64ed0b.cart`
- `frend/data/en/context/trees/540a0cb280e7.cart`
- `frend/data/en/context/trees/554fc9bf40a9.cart`
- `frend/data/en/context/trees/55c6141177f9.cart`
- `frend/data/en/context/trees/574c74980184.cart`
- `frend/data/en/context/trees/6dcb8c4b620c.cart`
- `frend/data/en/context/trees/7ce519ec98b4.cart`
- `frend/data/en/context/trees/7d928ce2933c.cart`
- `frend/data/en/context/trees/86fe09d7eff2.cart`
- `frend/data/en/context/trees/8bee8046856d.cart`
- `frend/data/en/context/trees/8d4515587195.cart`
- `frend/data/en/context/trees/99daccef2728.cart`
- `frend/data/en/context/trees/a55b7822c9ab.cart`
- `frend/data/en/context/trees/a594e90328c1.cart`
- `frend/data/en/context/trees/a7b9e5f7f426.cart`
- `frend/data/en/context/trees/a8226db3c036.cart`
- `frend/data/en/context/trees/aa808c894ba0.cart`
- `frend/data/en/context/trees/ada159e41fa4.cart`
- `frend/data/en/context/trees/b1ae6a87f4dd.cart`
- `frend/data/en/context/trees/b513182c330a.cart`
- `frend/data/en/context/trees/bd8b7319152c.cart`
- `frend/data/en/context/trees/cb491deed07c.cart`
- `frend/data/en/context/trees/cec4bd37fa27.cart`
- `frend/data/en/context/trees/d110d44cade4.cart`
- `frend/data/en/context/trees/d92b41b57938.cart`
- `frend/data/en/context/trees/dc4772eb1067.cart`
- `frend/data/en/context/trees/de2bdedc2381.cart`
- `frend/data/en/context/trees/eef8970abc6a.cart`
- `frend/data/en/context/trees/f418027f4a98.cart`
- `frend/data/en/context/trees/f536797113eb.cart`
- `frend/data/en/context/trees/ff0ddfa3bada.cart`
- `frend/data/en/electronic_priors.json`
- `frend/data/en/electronic_span_priors.json`
- `frend/data/en/grouped_id_priors.json`
- `frend/data/en/inch_context_priors.json`
- `frend/data/en/number_priors.json`
- `frend/data/en/range_priors.json`
- `frend/data/en/spelled_token_priors.json`
- `frend/data/en/spellout_dictionary.json`
- `frend/data/en/spoken_priors.json`
- `frend/data/en/telephone_priors.json`
- `frend/data/en/type_priors.json`
- `frend/data/en/zero_priors.json`

### `iana/tlds-alpha-by-domain`

- `frend/data/root/tlds-alpha-by-domain.txt`

### `icu/78.3/cldr-date-patterns`

- `frend/data/date_rules/de_DE.json`
- `frend/data/date_rules/es_ES.json`
- `frend/data/date_rules/es_MX.json`
- `frend/data/date_rules/fr_FR.json`
- `frend/data/date_rules/it_IT.json`
- `frend/data/date_rules/ja_JP.json`
- `frend/data/date_rules/ko_KR.json`
- `frend/data/date_rules/pt_BR.json`
- `frend/data/date_rules/pt_PT.json`
- `frend/data/date_rules/zh_CN.json`

### `icu/78.3/fraction-percent`

- `frend/data/fraction_rules/de_DE.json`
- `frend/data/fraction_rules/es_ES.json`
- `frend/data/fraction_rules/es_MX.json`
- `frend/data/fraction_rules/fr_FR.json`
- `frend/data/fraction_rules/it_IT.json`
- `frend/data/fraction_rules/ja_JP.json`
- `frend/data/fraction_rules/ko_KR.json`
- `frend/data/fraction_rules/pt_BR.json`
- `frend/data/fraction_rules/pt_PT.json`
- `frend/data/fraction_rules/zh_CN.json`
- `frend/data/normalization/de_DE.json`
- `frend/data/normalization/es-default.json`
- `frend/data/normalization/es_ES.json`
- `frend/data/normalization/es_MX.json`
- `frend/data/normalization/fr_FR.json`
- `frend/data/normalization/icu-78.3.json`
- `frend/data/normalization/it_IT.json`
- `frend/data/normalization/ja_JP.json`
- `frend/data/normalization/ko_KR.json`
- `frend/data/normalization/pt-default.json`
- `frend/data/normalization/pt_BR.json`
- `frend/data/normalization/pt_PT.json`
- `frend/data/normalization/zh_CN.json`

### `icu/78.3/reflective-generation`

- `frend/data/root/icu_shape_backfill.json`

### `icu/78.3/time`

- `frend/data/normalization/icu-78.3-time.json`
- `frend/data/normalization/time_de_DE.json`
- `frend/data/normalization/time_es_ES.json`
- `frend/data/normalization/time_es_MX.json`
- `frend/data/normalization/time_fr_FR.json`
- `frend/data/normalization/time_it_IT.json`
- `frend/data/normalization/time_ja_JP.json`
- `frend/data/normalization/time_ko_KR.json`
- `frend/data/normalization/time_pt_BR.json`
- `frend/data/normalization/time_pt_PT.json`
- `frend/data/normalization/time_zh_CN.json`
- `frend/data/time_rules/de_DE.json`
- `frend/data/time_rules/es_ES.json`
- `frend/data/time_rules/es_MX.json`
- `frend/data/time_rules/fr_FR.json`
- `frend/data/time_rules/it_IT.json`
- `frend/data/time_rules/ja_JP.json`
- `frend/data/time_rules/ko_KR.json`
- `frend/data/time_rules/pt_BR.json`
- `frend/data/time_rules/pt_PT.json`
- `frend/data/time_rules/zh_CN.json`

### `impa/obmep`

- `frend/data/normalization/impa-fractions.json`
- `frend/data/normalization/pt-default.json`

### `jixi/education-cloud`

- `frend/data/fraction_rules/zh_CN.json`
- `frend/data/normalization/chinese-fractions.json`
- `frend/data/normalization/zh_CN.json`

### `libphonenumber`

- `frend/data/normalization/libphonenumber-9.0.41.json`
- `frend/data/normalization/telephone_fr_FR.json`
- `frend/data/normalization/telephone_pt_BR.json`
- `frend/data/normalization/telephone_pt_PT.json`
- `frend/data/telephone_rules/fr_FR.json`
- `frend/data/telephone_rules/pt_BR.json`
- `frend/data/telephone_rules/pt_PT.json`

### `life-education/journal`

- `frend/data/fraction_rules/zh_CN.json`
- `frend/data/normalization/chinese-mixed-fractions.json`
- `frend/data/normalization/zh_CN.json`

### `mext/math-grade4`

- `frend/data/fraction_rules/ja_JP.json`
- `frend/data/normalization/ja_JP.json`
- `frend/data/normalization/japanese-fractions.json`
- `frend/data/normalization/mext-clock.json`
- `frend/data/normalization/time_ja_JP.json`
- `frend/data/time_rules/ja_JP.json`

### `nikl/basic-dictionary`

- `frend/data/fraction_rules/ko_KR.json`
- `frend/data/normalization/ko_KR.json`
- `frend/data/normalization/korean-percent.json`

### `nikl/online-qna`

- `frend/data/fraction_rules/ko_KR.json`
- `frend/data/money_rules/ko_KR.json`
- `frend/data/normalization/ko_KR.json`
- `frend/data/normalization/korean-fractions.json`
- `frend/data/normalization/money_ko_KR.json`
- `frend/data/normalization/nikl-clock.json`
- `frend/data/normalization/nikl-money.json`
- `frend/data/normalization/nikl-wa-gwa.json`
- `frend/data/normalization/time_ko_KR.json`
- `frend/data/time_rules/ko_KR.json`

### `nikl/standard-dictionary`

- `frend/data/normalization/nikl-wa-standard.json`

### `oqlf/bdl`

- `frend/data/fraction_rules/fr_FR.json`
- `frend/data/normalization/fr_FR.json`
- `frend/data/normalization/oqlf-fractions.json`

### `rae/brae`

- `frend/data/normalization/rae-alemany-fractions.json`

### `rae/dpd-2`

- `frend/data/fraction_rules/es_ES.json`
- `frend/data/fraction_rules/es_MX.json`
- `frend/data/normalization/es-default.json`
- `frend/data/normalization/rae-clock.json`
- `frend/data/normalization/rae-fractions.json`
- `frend/data/normalization/rae-medio.json`
- `frend/data/normalization/time_es_ES.json`
- `frend/data/normalization/time_es_MX.json`
- `frend/data/time_rules/es_ES.json`
- `frend/data/time_rules/es_MX.json`

### `rae/gramatica-basica`

- `frend/data/normalization/rae-gramatica-fraccionarios.json`

### `rechtschreibrat/amtliches-regelwerk-2018`

- `frend/data/fraction_rules/de_DE.json`
- `frend/data/normalization/de_DE.json`
- `frend/data/normalization/rechtschreibrat-fractions.json`

### `sep/nueva-escuela-mexicana`

- `frend/data/normalization/es-default.json`
- `frend/data/normalization/sep-fractions.json`
- `frend/data/normalization/sep-mixed-fractions.json`

### `serlo/fractions`

- `frend/data/fraction_rules/de_DE.json`
- `frend/data/normalization/de_DE.json`
- `frend/data/normalization/serlo-fractions.json`

### `treccani/enciclopedia-italiano`

- `frend/data/fraction_rules/it_IT.json`
- `frend/data/normalization/it_IT.json`
- `frend/data/normalization/treccani-fractions.json`

### `treccani/vocabolario`

- `frend/data/fraction_rules/it_IT.json`
- `frend/data/money_rules/it_IT.json`
- `frend/data/normalization/it_IT.json`
- `frend/data/normalization/money_it_IT.json`
- `frend/data/normalization/time_it_IT.json`
- `frend/data/normalization/treccani-clock.json`
- `frend/data/normalization/treccani-mezzo.json`
- `frend/data/normalization/treccani-money.json`
- `frend/data/time_rules/it_IT.json`

### `ua/matesfacil`

- `frend/data/fraction_rules/es_ES.json`
- `frend/data/fraction_rules/es_MX.json`
- `frend/data/normalization/es-default.json`
- `frend/data/normalization/ua-mixed-fractions.json`

### `uel/matematica-essencial`

- `frend/data/normalization/pt-default.json`
- `frend/data/normalization/uel-fractions.json`

### `urumqi/government`

- `frend/data/fraction_rules/zh_CN.json`
- `frend/data/normalization/chinese-percent.json`
- `frend/data/normalization/zh_CN.json`

<!-- END GENERATED THIRD-PARTY FILE INDEX -->
