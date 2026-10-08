"""Build Kestrel's corpus-measured UK-to-US spelling profile.

The output is licensed-corpus-derived profile data and must stay outside the repository.
Exact written-form counts use training shards 00--89. Selection learns exact-form support
and a common edit-rule rate threshold on 00--79, then chooses them on 80--89.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build_spoken_priors import _default_corpus_dir  # noqa: E402
from google_tn_rows import TRAINING_SHARDS, corpus_label, full_training_set  # noqa: E402

from frend.britishisms import CASE_SHAPES, EDIT_RULES, PAIR_CLASSES, case_shape  # noqa: E402
from frend.locale_data import canonical_locale  # noqa: E402
from frend.profiles import GOOGLE_TN, google_tn_britishisms_path  # noqa: E402

_SUPPORTS = (1, 2, 3, 5, 10, 20, 50)
_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def _training_files(corpus_dir: Path) -> list[Path]:
    paths = sorted(path for path in corpus_dir.iterdir() if path.is_file())
    return full_training_set(paths) or paths


def _source_shards(inputs) -> list[dict[str, str]]:
    return [
        {"relative_path": item.relative_path, "sha256": item.sha256}
        for item in inputs
        if isinstance(getattr(item, "sha256", None), str)
    ]


def _require_training_inputs(inputs) -> None:
    outside = sorted(
        item.relative_path for item in inputs if item.relative_path not in TRAINING_SHARDS
    )
    if outside:
        raise ValueError(
            f"profile inputs must be Google TN training shards 00-89; got {outside[0]!r}"
        )


def _shard_number(item) -> int | None:
    name = item.relative_path if hasattr(item, "relative_path") else Path(item).name
    if name.startswith("output-") and name.endswith("-of-00100"):
        return int(name.removeprefix("output-").split("-", 1)[0])
    return None


def _open(item):
    if hasattr(item, "relative_path"):
        from corpus_inputs import open_verified

        return open_verified(item)
    return Path(item).open(encoding="utf-8")


def _is_respelling(written: str, output: str) -> bool:
    """Whether one or two declared UK-to-US edits produce ``output`` exactly."""
    frontier = {written}
    for _depth in range(2):
        following = set()
        for form in frontier:
            for rule in EDIT_RULES:
                candidate = rule.apply(form)
                if candidate == output:
                    return True
                if candidate != form:
                    following.add(candidate)
        frontier = following
    return False


def _without_diacritics(word: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", word)
        if not unicodedata.combining(char) and unicodedata.category(char) != "Lm"
    )


def _is_subsequence(shorter: str, longer: str) -> bool:
    at = iter(longer)
    return all(char in at for char in shorter)


def pair_class(written: str, output: str) -> str:
    """Classify every one-word alphabetic conversion into one of four audited classes."""
    if _is_respelling(written, output):
        return "respelling"
    if _without_diacritics(written) == output:
        return "diacritic"
    if len(written) < len(output) and _is_subsequence(written, output):
        return "expansion/abbreviation"
    return "other"


class Counts:
    def __init__(self) -> None:
        self.left: Counter[tuple[str, str]] = Counter()
        self.converted: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.rule_rows: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.rule_total: Counter[tuple[str, str]] = Counter()
        self.rule_converted: Counter[tuple[str, str]] = Counter()

    def add(self, written: str, target: str) -> None:
        shape = case_shape(written)
        if shape is None:
            return
        word = written.casefold()
        output = target.casefold()
        key = (word, shape)
        if output == word:
            self.left[key] += 1
        else:
            self.converted[key][output] += 1
        matched = False
        for rule in EDIT_RULES:
            candidate = rule.apply(word)
            if candidate == word:
                continue
            self.rule_total[(rule.name, shape)] += 1
            self.rule_converted[(rule.name, shape)] += output == candidate
            matched = True
        if matched:
            self.rule_rows[key][output] += 1

    def update(self, other: Counts) -> None:
        self.left.update(other.left)
        for word, outcomes in other.converted.items():
            self.converted[word].update(outcomes)
        for word, outcomes in other.rule_rows.items():
            self.rule_rows[word].update(outcomes)
        self.rule_total.update(other.rule_total)
        self.rule_converted.update(other.rule_converted)

    def outcomes(self, key: tuple[str, str]) -> Counter[str]:
        found = Counter(self.converted.get(key, ()))
        if self.left[key]:
            found[key[0]] += self.left[key]
        return found


def _read_item(item) -> tuple[bool, Counts]:
    counts = Counts()
    with _open(item) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3 or parts[0] != "PLAIN":
                continue
            written = parts[1]
            target = written if parts[2] == "<self>" else parts[2]
            if written.isalpha() and target.isalpha():
                counts.add(written, target)
    number = _shard_number(item)
    return number is not None and number >= 80, counts


def _read_counts(items, jobs: int = 1) -> tuple[Counts, Counts]:
    training, development = Counts(), Counts()
    if jobs <= 1:
        results = map(_read_item, items)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        results = executor.map(_read_item, items, chunksize=1)
    try:
        for is_development, counts in results:
            (development if is_development else training).update(counts)
    finally:
        if jobs > 1:
            executor.shutdown()
    return training, development


def _exact_prediction(
    counts: Counts,
    key: tuple[str, str],
    supports: dict[str, int],
) -> tuple[str, str] | None:
    candidates = [
        (converted, target, pair_class(key[0], target), counts.left[key])
        for target, converted in counts.converted.get(key, {}).items()
    ]
    if not candidates:
        return None
    top_count = max(converted for converted, _target, _class_name, _left in candidates)
    best = [candidate for candidate in candidates if candidate[0] == top_count]
    if len(best) != 1:
        return None
    converted, target, class_name, left = best[0]
    if class_name not in supports:
        return None
    if converted >= supports[class_name] and converted > left:
        return target, class_name
    return None


def _rule_models(counts: Counts) -> dict[str, dict[str, dict[str, object]]]:
    models = {}
    for rule in EDIT_RULES:
        models[rule.name] = {}
        for shape in CASE_SHAPES:
            stems = {
                stem
                for (word, row_shape), outcomes in counts.rule_rows.items()
                if row_shape == shape
                and (stem := rule.stem(word)) is not None
                and outcomes[rule.apply(word)] > 0
            }
            eligible = converted = 0
            for (word, row_shape), outcomes in counts.rule_rows.items():
                if row_shape != shape or rule.stem(word) not in stems:
                    continue
                eligible += sum(outcomes.values())
                converted += outcomes[rule.apply(word)]
            models[rule.name][shape] = {
                "stems": stems,
                "converted": converted,
                "eligible": eligible,
                "rate": converted / eligible if eligible else 0.0,
            }
    return models


def _enabled_rules(models, threshold: float, respelling: bool) -> dict[str, frozenset[str]]:
    if not respelling:
        return {}
    return {
        rule.name: frozenset(
            shape
            for shape in CASE_SHAPES
            if models[rule.name][shape]["eligible"]
            and models[rule.name][shape]["rate"] >= threshold
        )
        for rule in EDIT_RULES
        if any(
            models[rule.name][shape]["eligible"] and models[rule.name][shape]["rate"] >= threshold
            for shape in CASE_SHAPES
        )
    }


def _predict(
    counts: Counts,
    key: tuple[str, str],
    supports: dict[str, int],
    enabled: dict[str, frozenset[str]],
    models,
    fold_case: bool,
) -> tuple[str, str | None]:
    word, shape = key
    if exact := _exact_prediction(counts, key, supports):
        return exact
    if shape != "lower" and fold_case:
        if exact := _exact_prediction(counts, (word, "lower"), supports):
            return exact
    for rule in EDIT_RULES:
        rule_shape = shape
        if rule_shape not in enabled.get(rule.name, ()) and shape != "lower" and fold_case:
            rule_shape = "lower"
        if rule_shape not in enabled.get(rule.name, ()):
            continue
        candidate = rule.apply(word)
        if candidate != word and rule.stem(word) in models[rule.name][rule_shape]["stems"]:
            return candidate, "respelling"
    return word, None


def _score_predictions(development: Counts, predict) -> dict:
    totals = Counter()
    by_class: dict[str, Counter] = defaultdict(Counter)
    keys = set(development.left) | set(development.converted) | set(development.rule_rows)
    for key in keys:
        prediction, class_name = predict(key)
        if prediction == key[0] or class_name is None:
            continue
        outcomes = development.outcomes(key)
        fixed = outcomes[prediction]
        broken = outcomes[key[0]]
        changed = sum(outcomes.values())
        still_wrong = changed - fixed - broken
        values = {
            "changed_tokens": changed,
            "fixed_tokens": fixed,
            "broken_tokens": broken,
            "changed_still_wrong": still_wrong,
            "net_tokens": fixed - broken,
        }
        totals.update(values)
        by_class[class_name].update(values)
    return {
        **dict(totals),
        "by_class": {name: dict(by_class[name]) for name in PAIR_CLASSES},
    }


def _select_classes(training: Counts, development: Counts):
    selected = {}
    report = {}
    for class_name in PAIR_CLASSES:
        candidates = []
        for support in _SUPPORTS:
            score = _score_predictions(
                development,
                lambda key, s=support, c=class_name: (
                    _exact_prediction(training, key, {c: s}) or (key[0], None)
                ),
            )
            candidates.append({"minimum_support": support, **score})
        best = max(
            candidates,
            key=lambda row: (
                row.get("net_tokens", 0),
                -row.get("broken_tokens", 0),
                row["minimum_support"],
            ),
        )
        enabled = best.get("net_tokens", 0) > 0
        if enabled:
            selected[class_name] = best["minimum_support"]
        report[class_name] = {"enabled": enabled, **best, "candidates": candidates}
    return selected, report


def _selection(training: Counts, development: Counts):
    supports, classes = _select_classes(training, development)
    models = _rule_models(training)
    candidates = []
    for threshold in _THRESHOLDS:
        enabled = _enabled_rules(models, threshold, "respelling" in supports)
        for fold_case in (False, True):
            score = _score_predictions(
                development,
                lambda key, e=enabled, f=fold_case: _predict(training, key, supports, e, models, f),
            )
            candidates.append(
                {
                    "rule_rate_threshold": threshold,
                    "case_folding": fold_case,
                    "enabled_rules": {
                        name: sorted(shapes) for name, shapes in sorted(enabled.items())
                    },
                    **score,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            row.get("net_tokens", 0),
            -row.get("broken_tokens", 0),
            -int(row["case_folding"]),
            row["rule_rate_threshold"],
        ),
    )
    without_folding = next(
        row
        for row in candidates
        if row["rule_rate_threshold"] == selected["rule_rate_threshold"]
        and row["case_folding"] is False
    )
    case_folding = {
        "enabled": selected["case_folding"],
        "net_tokens": selected.get("net_tokens", 0) - without_folding.get("net_tokens", 0),
        "fixed_tokens": selected.get("fixed_tokens", 0) - without_folding.get("fixed_tokens", 0),
        "broken_tokens": selected.get("broken_tokens", 0) - without_folding.get("broken_tokens", 0),
    }
    return supports, classes, selected, case_folding, candidates


def _english_locale(locale: str) -> str:
    canonical = canonical_locale(locale)
    if canonical.split("_", 1)[0] != "en":
        raise ValueError(
            f"Britishism rewrite profile requires an English locale, got {canonical!r}"
        )
    return canonical


def build_document(corpus_dir: Path, *, inputs=None, jobs: int = 1, locale: str = "en") -> dict:
    locale = _english_locale(locale)
    items = list(inputs) if inputs is not None else _training_files(corpus_dir)
    if inputs is not None:
        _require_training_inputs(items)
    training, development = _read_counts(items, jobs)
    supports, classes, selected, case_folding, candidates = _selection(training, development)
    combined = Counts()
    combined.update(training)
    combined.update(development)
    pairs: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for key in sorted(combined.converted):
        word, shape = key
        prediction = _exact_prediction(combined, key, supports)
        if prediction is None:
            continue
        target, class_name = prediction
        converted = combined.converted[key][target]
        left = combined.left[key]
        pairs[word][shape] = {
            "class": class_name,
            "target": target,
            "converted": converted,
            "left": left,
        }
    models = _rule_models(combined)
    enabled = _enabled_rules(models, selected["rule_rate_threshold"], "respelling" in supports)
    rules = {}
    for rule in EDIT_RULES:
        rules[rule.name] = {
            shape: {
                "converted": models[rule.name][shape]["converted"],
                "eligible": models[rule.name][shape]["eligible"],
                "rate": models[rule.name][shape]["rate"],
                "stems": sorted(models[rule.name][shape]["stems"]),
                "enabled": shape in enabled.get(rule.name, ()),
            }
            for shape in CASE_SHAPES
        }
    names = [
        item.relative_path if hasattr(item, "relative_path") else Path(item).name for item in items
    ]
    return {
        "schema_version": 1,
        "profile": GOOGLE_TN,
        "locale": locale,
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "corpus": corpus_label(corpus_dir),
            "license": "CC BY-SA 4.0",
            "source_shards": _source_shards(items),
            "shards": names,
            "relevance_filter": (
                "PLAIN; written and output are one alphabetic word in lower, Title, or UPPER "
                "case; every differing pair is classified before development selection"
            ),
        },
        "selection": {
            "candidate_minimum_support": list(_SUPPORTS),
            "candidate_rule_rate_threshold": list(_THRESHOLDS),
            "admitted_classes": list(supports),
            "class_minimum_support": supports,
            "classes": classes,
            "case_folding": case_folding,
            "rule_rate_threshold": selected["rule_rate_threshold"],
            "development_contribution": selected["by_class"],
            "selection_training_shards": [f"output-{index:05d}-of-00100" for index in range(80)],
            "development_shards": [f"output-{index:05d}-of-00100" for index in range(80, 90)],
            "objective": (
                "admit each pair class only for positive development net; then maximize net "
                "corrected PLAIN tokens, minimize broken tokens, and prefer case isolation"
            ),
            "candidates": candidates,
        },
        "rules": rules,
        "pairs": dict(pairs),
    }


def _external_profile_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(_REPO.resolve()):
        raise ValueError(
            f"--profile-out must be outside the repository and package tree: {resolved}"
        )
    return resolved


def _render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--profile-out", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    locale = _english_locale(args.locale)
    profile_out = _external_profile_output(args.profile_out or google_tn_britishisms_path())
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    from corpus_inputs import verified_inputs, write_verification_receipt

    verified = verified_inputs(
        args.source_id,
        _training_files(corpus_dir),
        locale=locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(args.receipt, verified, locale=locale, pools=tuple(args.pools))
    rendered = _render(build_document(corpus_dir, inputs=verified, jobs=args.jobs, locale=locale))
    if args.check:
        current = profile_out.read_text(encoding="utf-8") if profile_out.exists() else ""
        if current != rendered:
            print(f"out of date: {profile_out}")
            return 1
        print(f"up to date: {profile_out}")
        return 0
    profile_out.parent.mkdir(parents=True, exist_ok=True)
    profile_out.write_text(rendered, encoding="utf-8")
    print(f"wrote {profile_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
