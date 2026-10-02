"""Build Kestrel's corpus-measured UK-to-US spelling profile.

The output is licensed-corpus-derived profile data and must stay outside the repository.
Exact written-form counts use training shards 00--89. Selection learns exact-form support
and a common edit-rule rate threshold on 00--79, then chooses them on 80--89.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build_spoken_priors import _default_corpus_dir  # noqa: E402
from google_tn_rows import TRAINING_SHARDS, corpus_label, full_training_set  # noqa: E402

from frend.britishisms import EDIT_RULES  # noqa: E402
from frend.profiles import GOOGLE_TN, google_tn_britishisms_path  # noqa: E402

_SUPPORTS = (1, 2, 3, 5, 10, 20, 50)
_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
_MAY_EDIT = re.compile(r"is|our|re|ogue|mme|ll|yse|ae|oe")


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


def _distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def orthographic_pair(written: str, output: str) -> bool:
    """The literal relevance filter for a one-word orthographic conversion."""
    if written == output or not written.isalpha() or not output.isalpha():
        return False
    if any(rule.apply(written) == output for rule in EDIT_RULES):
        return True
    unmarked = "".join(
        char for char in unicodedata.normalize("NFKD", written) if not unicodedata.combining(char)
    )
    if unmarked == output:
        return True
    return (
        abs(len(written) - len(output)) <= 3
        and _distance(written, output) <= 3
        and SequenceMatcher(None, written, output).ratio() >= 0.72
    )


class Counts:
    def __init__(self) -> None:
        self.left: Counter[str] = Counter()
        self.converted: dict[str, Counter[str]] = defaultdict(Counter)
        self.rule_rows: dict[str, Counter[str]] = defaultdict(Counter)
        self.rule_total: Counter[str] = Counter()
        self.rule_converted: Counter[str] = Counter()

    def add(self, written: str, target: str) -> None:
        if target == written:
            self.left[written] += 1
        elif orthographic_pair(written, target):
            self.converted[written][target] += 1
        if _MAY_EDIT.search(written) is None:
            return
        matched = False
        for rule in EDIT_RULES:
            candidate = rule.apply(written)
            if candidate == written:
                continue
            self.rule_total[rule.name] += 1
            self.rule_converted[rule.name] += target == candidate
            matched = True
        if matched:
            self.rule_rows[written][target] += 1

    def update(self, other: Counts) -> None:
        self.left.update(other.left)
        for word, outcomes in other.converted.items():
            self.converted[word].update(outcomes)
        for word, outcomes in other.rule_rows.items():
            self.rule_rows[word].update(outcomes)
        self.rule_total.update(other.rule_total)
        self.rule_converted.update(other.rule_converted)

    def outcomes(self, word: str) -> Counter[str]:
        if word in self.rule_rows:
            return Counter(self.rule_rows[word])
        found = Counter(self.converted.get(word, ()))
        if self.left[word]:
            found[word] += self.left[word]
        return found


def _read_item(item) -> tuple[bool, Counts]:
    counts = Counts()
    with _open(item) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] != "PLAIN":
                continue
            written = parts[1].casefold()
            target = written if parts[2] == "<self>" else parts[2].casefold()
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


def _exact_prediction(counts: Counts, word: str, support: int) -> str | None:
    conversions = counts.converted.get(word)
    if not conversions:
        return None
    target, converted = conversions.most_common(1)[0]
    return target if converted >= support and converted > counts.left[word] else None


def _rule_models(counts: Counts) -> dict[str, dict[str, object]]:
    models = {}
    for rule in EDIT_RULES:
        stems = {
            stem
            for word, outcomes in counts.rule_rows.items()
            if (stem := rule.stem(word)) is not None and outcomes[rule.apply(word)] > 0
        }
        eligible = converted = 0
        for word, outcomes in counts.rule_rows.items():
            if rule.stem(word) not in stems:
                continue
            eligible += sum(outcomes.values())
            converted += outcomes[rule.apply(word)]
        models[rule.name] = {
            "stems": stems,
            "converted": converted,
            "eligible": eligible,
            "rate": converted / eligible if eligible else 0.0,
        }
    return models


def _enabled_rules(models: dict[str, dict[str, object]], threshold: float) -> tuple[str, ...]:
    return tuple(
        rule.name
        for rule in EDIT_RULES
        if models[rule.name]["eligible"] and models[rule.name]["rate"] >= threshold
    )


def _predict(
    counts: Counts,
    word: str,
    support: int,
    enabled: tuple[str, ...],
    models: dict[str, dict[str, object]],
) -> str:
    if exact := _exact_prediction(counts, word, support):
        return exact
    enabled_set = set(enabled)
    for rule in EDIT_RULES:
        if (
            rule.name in enabled_set
            and rule.stem(word) in models[rule.name]["stems"]
            and (candidate := rule.apply(word)) != word
        ):
            return candidate
    return word


def _selection(training: Counts, development: Counts) -> tuple[int, float, list[dict]]:
    models = _rule_models(training)
    modeled_words = {
        word
        for word in development.rule_rows
        if any(rule.stem(word) in models[rule.name]["stems"] for rule in EDIT_RULES)
    }
    words = (
        set(development.converted)
        | modeled_words
        | (set(training.converted) & set(development.left))
    )
    candidates = []
    for support in _SUPPORTS:
        for threshold in _THRESHOLDS:
            enabled = _enabled_rules(models, threshold)
            delta = fixed = false = changed = 0
            for word in words:
                outcomes = development.outcomes(word)
                prediction = _predict(training, word, support, enabled, models)
                if prediction == word:
                    continue
                changed += sum(outcomes.values())
                fixed += outcomes[prediction]
                false += outcomes[word]
                delta += outcomes[prediction] - outcomes[word]
            candidates.append(
                {
                    "minimum_support": support,
                    "rule_rate_threshold": threshold,
                    "net_tokens": delta,
                    "fixed_tokens": fixed,
                    "false_conversions": false,
                    "changed_population": changed,
                    "enabled_rules": list(enabled),
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            row["net_tokens"],
            -row["false_conversions"],
            row["minimum_support"],
            row["rule_rate_threshold"],
        ),
    )
    return selected["minimum_support"], selected["rule_rate_threshold"], candidates


def build_document(corpus_dir: Path, *, inputs=None, jobs: int = 1) -> dict:
    items = list(inputs) if inputs is not None else _training_files(corpus_dir)
    if inputs is not None:
        _require_training_inputs(items)
    training, development = _read_counts(items, jobs)
    support, threshold, candidates = _selection(training, development)
    combined_words = set(training.converted) | set(development.converted)
    pairs = {}
    for word in sorted(combined_words):
        conversions = training.converted.get(word, Counter()) + development.converted.get(
            word, Counter()
        )
        target, converted = conversions.most_common(1)[0]
        pairs[word] = {
            "target": target,
            "converted": converted,
            "left": training.left[word] + development.left[word],
        }
    models = _rule_models(training)
    enabled = set(_enabled_rules(models, threshold))
    rules = {}
    for rule in EDIT_RULES:
        model = models[rule.name]
        rules[rule.name] = {
            "converted": model["converted"],
            "eligible": model["eligible"],
            "rate": model["rate"],
            "stems": sorted(model["stems"]),
            "enabled": rule.name in enabled,
        }
    names = [
        item.relative_path if hasattr(item, "relative_path") else Path(item).name for item in items
    ]
    return {
        "schema_version": 1,
        "profile": GOOGLE_TN,
        "locale": "en",
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "corpus": corpus_label(corpus_dir),
            "license": "CC BY-SA 4.0",
            "source_shards": _source_shards(items),
            "shards": names,
            "relevance_filter": (
                "PLAIN; written and output are one alphabetic word; casefolded output "
                "is self or passes orthographic_pair"
            ),
        },
        "selection": {
            "candidate_minimum_support": list(_SUPPORTS),
            "candidate_rule_rate_threshold": list(_THRESHOLDS),
            "minimum_support": support,
            "rule_rate_threshold": threshold,
            "selection_training_shards": [f"output-{index:05d}-of-00100" for index in range(80)],
            "development_shards": [f"output-{index:05d}-of-00100" for index in range(80, 90)],
            "objective": "maximize net corrected PLAIN tokens, then minimize false conversions",
            "candidates": candidates,
        },
        "rules": rules,
        "pairs": pairs,
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
    profile_out = _external_profile_output(args.profile_out or google_tn_britishisms_path())
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    from corpus_inputs import verified_inputs, write_verification_receipt

    verified = verified_inputs(
        args.source_id,
        _training_files(corpus_dir),
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
    rendered = _render(build_document(corpus_dir, inputs=verified, jobs=args.jobs))
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
