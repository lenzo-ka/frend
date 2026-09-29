"""Train frend's context trees (``frend/data/en/context/``) from the stored example set.

The example set is P7 stage A's (``frend/google/tn-en_with_types/p7-examples/<fingerprint>``
on kalman, ``/Volumes/k02/processed/``): every token of training shards 05, 15, ..., 85
where frend offers two or more readings, with the reading the corpus says, sampled per
problem (at most 30,000, seed 20260928), and every standalone "-" and "–" with the
corpus's spoken form. Each record keeps its token and up to three corpus tokens either
side. This tool reads that set (copied locally first, every read of the mount bounded by
``timeout``, each file checked against the set's receipt, and the receipt's inputs and
every record's origin checked to be training shards, 00 to 89) and, reproducibly:

1. **Examples by span.** Each record's window is joined into running text (one space
   between tokens), and its token read the way frend reads it, in that context
   (``frend.verbalize``, the context trees' builder mode: the range connector is offered
   where a separator stands between numbers, and no tree reorders anything). Each
   reading edge on frend's best path with two or more distinct readings is one example:
   its problem is the sorted labels of its readings (``frend.context.problem_labels``),
   its label the reading that says the corpus's form, found among the edge readings of
   the best path (a record whose form no combination says is counted and left out). A
   main record's corpus form is the text of the reading stage A labeled; a standalone
   dash's is stored with it (the dash records replace the main set's dashes, whose
   "to" was never offered and so never labeled). Separators written inside a range
   ("5-10") and passed-through text have no trees of their own: they read the
   standalone separator's tree (``frend.context.connector_probability``).
2. **Sample.** At most 60,000 examples per problem, drawn with the seed.
3. **Frequent words.** The 200 words most often at -2..+2 of the kept examples (ICU
   word segments, lower-cased) that no other class claims (not a number, month or
   weekday): Flite's frequent-word class, learned from this data.
4. **Trees.** One cartlet decision tree per problem with at least 50 examples (entropy,
   at least 5 examples a leaf, pruned on a 10% validation split drawn with the seed),
   over families B, F, W and C (``frend.context.features``), written as ``.cart``
   (``.json`` where a tree does not fit ``.cart``'s string table). A problem whose
   every example keeps frend's first choice ships no tree: it could never change one.

Writes ``index.json`` (provenance, the frequent words, the connectors, and each tree's
file, size and label counts) and ``trees/`` beside ``festival_classes.json``, which is
curated and not written here. ``--check`` rebuilds into a temporary directory and
compares every file byte for byte.

Run with frend's interpreter and cartlet importable (``cartlet>=0.6``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice, product
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from google_tn_rows import TRAINING_SHARDS  # noqa: E402

SOURCE = "google/tn-en_with_types"
LOCALE = "en_US"
DEFAULT_EXAMPLES = Path(
    "/Volumes/k02/processed/frend/google/tn-en_with_types/p7-examples/d1fcf656b9eb98dc"
)
DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "context"
SEED = 20260928
CAP = 60_000
MIN_EXAMPLES = 50
MIN_LEAF = 5
VALIDATION = 0.1
FREQUENT = 200
DASHES = frozenset({"-", "–"})
_ANY_CAP = 64  # combinations tried per record, as the evaluator tries per token
_MOUNT_TIMEOUT = "900"
_FILES = ("receipt.json", "examples.jsonl.gz", "dash_examples.jsonl.gz")


# ---------------------------------------------------------------------------------------
# The stored example set.


def fetch_examples(source: Path, cache: Path) -> tuple[Path, dict]:
    """Copy the set's files into ``cache`` (each copy bounded by ``timeout``: the mount
    can stall), and check each against the receipt's sha256. Returns the local directory
    and the receipt."""
    cache.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        local = cache / name
        if not local.exists():
            subprocess.run(
                ["timeout", _MOUNT_TIMEOUT, "cp", str(source / name), str(local)], check=True
            )
    receipt = json.loads((cache / "receipt.json").read_text(encoding="utf-8"))
    for name in _FILES[1:]:
        digest = hashlib.sha256((cache / name).read_bytes()).hexdigest()
        if digest != receipt["artifacts"][name]:
            raise SystemExit(f"{name}: sha256 {digest} is not the receipt's")
    held_out = sorted(set(receipt["inputs"]) - TRAINING_SHARDS)
    if held_out:
        raise SystemExit(f"receipt.json: inputs outside the training shards: {held_out}")
    return cache, receipt


def refuse_held_out(records) -> None:
    """Refuse any record whose ``src`` names a shard outside the training shards: a set
    whose receipt is self-consistent may still carry held-out rows."""
    held_out = sorted({r["src"].split(":", 1)[0] for r in records} - TRAINING_SHARDS)
    if held_out:
        raise SystemExit(f"records from outside the training shards: {held_out}")


def _records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


# ---------------------------------------------------------------------------------------
# Examples by span (run in worker processes; each caches per token).

_DETECTORS = None
_LATTICES: dict[str, object] = {}
_GOLD: dict[str, dict[str, str]] = {}


def _detectors():
    global _DETECTORS
    if _DETECTORS is None:
        from reading_profile import reading_detectors

        _DETECTORS = reading_detectors(LOCALE)
    return _DETECTORS


def _lattice(token: str):
    if token not in _LATTICES:
        from icukit.detectors import detect

        from frend import resolve_lattice

        detections = list(detect(token, _detectors())) if token.strip() else []
        _LATTICES[token] = resolve_lattice(detections, source_text=token)
    return _LATTICES[token]


def _paths(lattice, token: str) -> dict[int, list]:
    """Every projected path's units, by rank, in frend's own order: the token read
    alone and no context tree applied (``verbalize_lattice`` less the trees)."""
    from frend.verbalize import verbalize_edge

    edges = {edge.id: edge for edge in lattice.edges}
    return {
        path.rank: [
            verbalize_edge(edges[edge_id], source_text=token, rerank_by_context=False)
            for edge_id in path.edge_ids
        ]
        for path in lattice.paths
    }


def _joined(parts) -> str:
    out = ""
    for text, passthrough in parts:
        out += text if passthrough else f" {text} "
    return out


def _passthrough(alternative) -> bool:
    return alternative.provenance == "surface:passthrough"


def _token_readings(token: str) -> dict[str, str]:
    """Stage A's labels for the token read alone -> the text each says (its candidate
    enumeration: every path, the product of alternatives capped at 64, deduplicated by
    normalized text, each labeled by its units' provenance)."""
    if token in _GOLD:
        return _GOLD[token]
    from frend.context import problem_labels
    from frend.spoken_priors import normalize_spoken

    lattice = _lattice(token)
    paths = _paths(lattice, token)

    def label(provenances):
        out = []
        for provenance in provenances:
            if not out or out[-1] != provenance:
                out.append(provenance)
        return "+".join(out)

    def unit_pt(unit):
        return unit.best.provenance == "surface:passthrough"

    best = paths[lattice.best_path.rank]
    first = normalize_spoken(_joined((u.best.text, unit_pt(u)) for u in best))
    texts = [first]
    labels = [label([u.best.provenance for u in best])]
    seen = {first}
    for units in paths.values():
        options = [[(a.text, unit_pt(u), a.provenance) for a in u.alternatives] for u in units]
        for combination in islice(product(*options), _ANY_CAP):
            text = normalize_spoken(_joined((c[0], c[1]) for c in combination))
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
            labels.append(label([c[2] for c in combination]))
    _GOLD[token] = dict(zip(problem_labels(labels), texts, strict=True))
    return _GOLD[token]


def _window(record) -> tuple[str, int]:
    left = " ".join(record["L"])
    text = f"{left} {record['tok']}" if left else record["tok"]
    offset = len(text) - len(record["tok"])
    right = " ".join(record["R"])
    return (f"{text} {right}" if right else text), offset


def _is_connector(edge, text: str, offset: int) -> bool:
    """A separator written inside a range reads the standalone separator's tree."""
    from frend.context import TextContext
    from frend.verbalize import _capture, _range_to

    context = TextContext(text, offset)
    if edge.kind == "passthrough":
        return _range_to(context, edge.start, edge.end, "en_US") is not None
    sign = _capture(edge.detection, "sign")
    return sign is not None and _range_to(context, sign.start, sign.end, "en_US") is not None


def derive(record) -> tuple[str, list[dict]]:
    """(outcome, span examples) for one stored record."""
    from frend.context import TextContext, distinct_readings, problem_labels
    from frend.spoken_priors import normalize_spoken
    from frend.verbalize import verbalize_edge

    token = record["tok"]
    if "gold" in record:
        gold = normalize_spoken(record["gold"])
    else:
        gold = _token_readings(token).get(record["label"])
        if gold is None:
            return "label-not-reproduced", []
    text, offset = _window(record)
    context = TextContext(text, offset, bos=record["bos"], eos=record["eos"])
    lattice = _lattice(token)
    edges = {edge.id: edge for edge in lattice.edges}
    units = []
    for edge_id in lattice.best_path.edge_ids:
        edge = edges[edge_id]
        unit = verbalize_edge(edge, source_text=token, context=context, rerank_by_context=False)
        distinct = [unit.alternatives[i] for i in distinct_readings(unit.alternatives)]
        units.append((edge, distinct))
    options = [[(i, a) for i, a in enumerate(distinct)] for _, distinct in units]
    chosen = None
    for combination in islice(product(*options), _ANY_CAP):
        said = normalize_spoken(_joined((a.text, _passthrough(a)) for _, a in combination))
        if said == gold:
            chosen = combination
            break
    if chosen is None:
        return "form-not-on-best-path", []
    out = []
    for (edge, distinct), (index, _) in zip(units, chosen, strict=True):
        if len(distinct) < 2 or edge.kind == "passthrough" or _is_connector(edge, text, offset):
            continue
        labels = problem_labels([a.provenance for a in distinct])
        weight = distinct[0].weight
        out.append(
            {
                "problem": "\t".join(sorted(labels)),
                "label": labels[index],
                "first": labels[0],
                "w0": None if weight is None else float(weight),
                "text": text,
                "start": offset + edge.start,
                "end": offset + edge.end,
                "bos": record["bos"],
                "eos": record["eos"],
                "src": record["src"],
            }
        )
    return ("examples" if out else "no-ambiguous-span"), out


def _derive_chunk(records):
    return [derive(record) for record in records]


def _words_chunk(examples):
    from frend.context import BOS, EOS, PAD, _lower, _word_class, neighbor_words

    counts = Counter()
    for ex in examples:
        before, after = neighbor_words(
            ex["text"], ex["start"], ex["end"], locale=LOCALE, bos=ex["bos"], eos=ex["eos"]
        )
        for word in (before[0], before[1], after[0], after[1]):
            if word in (BOS, EOS, PAD):
                continue
            if _word_class(word, LOCALE, frozenset()) == "other":
                counts[_lower(word, LOCALE)] += 1
    return counts


# ---------------------------------------------------------------------------------------
# Trees.


def problem_id(problem: str) -> str:
    return hashlib.sha1(problem.encode("utf-8")).hexdigest()[:12]


def _train(job) -> tuple[str, str, bytes, dict]:
    """Featurize one problem's examples and train its tree: (problem, file name, bytes,
    info)."""
    from cartlet import DecisionTree
    from cartlet.utils import count_leaves, count_nodes

    from frend.context import NUMERIC_FEATURES, features

    problem, examples, frequent, curated, fingerprint = job
    rows = [
        features(
            ex["text"],
            ex["start"],
            ex["end"],
            first=ex["first"],
            first_weight=ex["w0"],
            locale=LOCALE,
            frequent=frequent,
            curated=curated,
            bos=ex["bos"],
            eos=ex["eos"],
        )
        for ex in examples
    ]
    names = list(rows[0])
    specs = [
        {"name": n, "dtype": "float", "type": "num"}
        if n in NUMERIC_FEATURES
        else {"name": n, "dtype": "str", "type": "cat"}
        for n in names
    ]
    labels = [ex["label"] for ex in examples]
    tree = DecisionTree(
        features=specs,
        task="classification",
        min_samples_leaf=MIN_LEAF,
        criterion="entropy",
        categorical_split="fast",
        min_confidence=1.0,
    )
    tree.load_data([[row[n] for n in names] for row in rows], labels)
    if len(set(labels)) > 1 and len(examples) >= 40:
        tree.train(prune=True, validation_split=VALIDATION, random_state=SEED)
    else:
        tree.train(prune=False, random_state=SEED)
    metadata = {"source": SOURCE, "examples": fingerprint, "problem": problem}
    pid = problem_id(problem)
    info = {
        "examples": len(examples),
        "labels": dict(sorted(Counter(labels).items())),
        "nodes": count_nodes(tree.model),
        "leaves": count_leaves(tree.model),
        "depth": tree.get_depth(),
    }
    with tempfile.TemporaryDirectory() as scratch:
        try:
            path = Path(scratch) / f"{pid}.cart"
            tree.export(str(path), metadata=metadata)
        except (ValueError, OverflowError) as error:
            info["cart_refused"] = str(error)
            path = Path(scratch) / f"{pid}.json"
            tree.export(str(path), metadata=metadata)
        blob = path.read_bytes()
        name = path.name
    return problem, name, blob, info


# ---------------------------------------------------------------------------------------


def build(examples_dir: Path, out: Path, *, workers: int, cache: Path, log=print) -> dict:
    local, receipt = fetch_examples(examples_dir, cache)
    fingerprint = receipt["fingerprint"]
    main = [r for r in _records(local / "examples.jsonl.gz") if r["tok"] not in DASHES]
    dashes = list(_records(local / "dash_examples.jsonl.gz"))
    records = main + dashes
    refuse_held_out(records)
    log(f"records: {len(main)} main (dashes left out), {len(dashes)} dash")
    # Group by token so each worker's per-token caches are used; results keep order.
    order = sorted(range(len(records)), key=lambda i: (records[i]["tok"], records[i]["src"]))
    chunks = [[records[i] for i in order[at : at + 2000]] for at in range(0, len(order), 2000)]
    outcomes = Counter()
    by_problem: dict[str, list[dict]] = defaultdict(list)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for results in pool.map(_derive_chunk, chunks):
            for outcome, examples in results:
                outcomes[outcome] += 1
                for ex in examples:
                    by_problem[ex["problem"]].append(ex)
    log(f"derived in {time.time() - started:.0f}s: {dict(sorted(outcomes.items()))}")
    rng = random.Random(SEED)
    kept: dict[str, list[dict]] = {}
    seen = {}
    for problem in sorted(by_problem):
        examples = sorted(by_problem[problem], key=lambda ex: (ex["src"], ex["start"]))
        seen[problem] = len(examples)
        if len(examples) > CAP:
            examples = sorted(rng.sample(examples, CAP), key=lambda ex: (ex["src"], ex["start"]))
        kept[problem] = examples
    everything = [ex for problem in sorted(kept) for ex in kept[problem]]
    counts = Counter()
    word_chunks = [everything[at : at + 5000] for at in range(0, len(everything), 5000)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_words_chunk, word_chunks):
            counts.update(part)
    frequent = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:FREQUENT]
    frequent_words = sorted(word for word, _ in frequent)
    curated_doc = json.loads((out / "festival_classes.json").read_text(encoding="utf-8"))
    curated = {name: frozenset(words) for name, words in curated_doc["classes"].items()}
    jobs = [
        (problem, kept[problem], frozenset(frequent_words), curated, fingerprint)
        for problem in sorted(kept)
        if len(kept[problem]) >= MIN_EXAMPLES
        and any(ex["label"] != ex["first"] for ex in kept[problem])
    ]
    skipped_small = sum(1 for p in kept if len(kept[p]) < MIN_EXAMPLES)
    skipped_first = sum(
        1
        for p in kept
        if len(kept[p]) >= MIN_EXAMPLES and all(ex["label"] == ex["first"] for ex in kept[p])
    )
    jobs.sort(key=lambda job: -len(job[1]))
    trees = {}
    blobs = {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for problem, name, blob, info in pool.map(_train, jobs, chunksize=1):
            info["file"] = f"trees/{name}"
            info["sha256"] = hashlib.sha256(blob).hexdigest()
            info["seen"] = seen[problem]
            trees[problem] = info
            blobs[name] = blob
    log(f"trained {len(trees)} trees in {time.time() - started:.0f}s")
    connectors = {}
    for problem in sorted(kept):
        labels = problem.split("\t")
        examples = kept[problem]
        if "lexical:en_US" not in labels or problem not in trees:
            continue
        if any(ex["text"][ex["start"] : ex["end"]] not in DASHES for ex in examples):
            continue
        firsts = {(ex["first"], ex["w0"]) for ex in examples}
        separators = {ex["text"][ex["start"] : ex["end"]] for ex in examples}
        if len(firsts) != 1 or len(separators) != 1:
            raise SystemExit(f"connector problem {problem!r} is not one separator's")
        (first, weight), (separator,) = firsts.pop(), separators
        connectors[separator] = {
            "problem": problem,
            "to": "lexical:en_US",
            "first": first,
            "first_weight": weight,
        }
    if connectors:
        # A separator the set never writes between numbers (the corpus has no en dash
        # there) reads the tree of the first one it does: the same question, "to" or not.
        from frend.verbalize import range_separators

        written = sorted(connectors)[0]
        for separator in sorted(range_separators(LOCALE) - connectors.keys()):
            connectors[separator] = {**connectors[written], "read_as": written}
    index = {
        "locale": "en",
        "schema_version": 1,
        "provenance": {
            "corpus": SOURCE,
            "locale": "en",
            "examples": {
                "derivation": receipt["derivation"],
                "fingerprint": fingerprint,
                "shards": sorted(receipt["inputs"]),
                "receipt_sha256": hashlib.sha256((local / "receipt.json").read_bytes()).hexdigest(),
                "files_sha256": {name: receipt["artifacts"][name] for name in _FILES[1:]},
            },
            "builder": "tools/build_context_trees.py",
            "seed": SEED,
            "cap_per_problem": CAP,
            "min_examples": MIN_EXAMPLES,
            "min_leaf": MIN_LEAF,
            "validation_split": VALIDATION,
            "criterion": "entropy",
            "families": ["B", "F", "W", "C"],
            "frequent_words": FREQUENT,
            "records": {"main": len(main), "dash": len(dashes)},
            "outcomes": dict(sorted(outcomes.items())),
            "problems": {
                "seen": len(kept),
                "trees": len(trees),
                "too_few_examples": skipped_small,
                "always_first": skipped_first,
            },
        },
        "frequent_words": frequent_words,
        "connectors": dict(sorted(connectors.items())),
        "trees": {problem: trees[problem] for problem in sorted(trees)},
    }
    out.mkdir(parents=True, exist_ok=True)
    tree_dir = out / "trees"
    if tree_dir.exists():
        shutil.rmtree(tree_dir)
    tree_dir.mkdir()
    for name in sorted(blobs):
        (tree_dir / name).write_bytes(blobs[name])
    (out / "index.json").write_text(
        json.dumps(index, indent=1, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8"
    )
    return index


def _compare(built: Path, shipped: Path) -> list[str]:
    differences = []
    names = {
        str(p.relative_to(root))
        for root in (built, shipped)
        for p in root.rglob("*")
        if p.is_file() and p.name != "festival_classes.json"
    }
    for name in sorted(names):
        a, b = built / name, shipped / name
        if not a.exists() or not b.exists() or a.read_bytes() != b.read_bytes():
            differences.append(name)
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=None, help="local copy of the set")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--check", action="store_true", help="rebuild and compare byte for byte")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        cache = args.cache or Path(scratch) / "examples"
        if args.check:
            built = Path(scratch) / "context"
            built.mkdir()
            shutil.copy2(args.out / "festival_classes.json", built / "festival_classes.json")
            build(args.examples, built, workers=args.workers, cache=cache)
            differences = _compare(built, args.out)
            if differences:
                print(f"--check: {len(differences)} files differ, first {differences[:5]}")
                return 1
            print(f"--check: {args.out} matches a rebuild byte for byte")
            return 0
        index = build(args.examples, args.out, workers=args.workers, cache=cache)
        print(json.dumps(index["provenance"]["problems"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
