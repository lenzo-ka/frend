"""``tools/build_context_trees.py``: the context trees, trained reproducibly from a stored
example set, and the shipped trees' provenance."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
_CONTEXT = _REPO / "frend" / "data" / "en" / "context"


def _builder():
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "build_context_trees", _TOOLS / "build_context_trees.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_context_trees"] = module
    spec.loader.exec_module(module)
    return module


def _dash(index: int, gold: str, left: str, right: str, shard: str = "00005") -> dict:
    return {
        "problem": "dash+to",
        "label": "to" if gold else "silence",
        "tok": "-",
        "gold": gold,
        "L": ["pages", left],
        "R": [right, "of"],
        "bos": True,
        "eos": False,
        "src": f"output-{shard}-of-00100:{index}:2",
    }


def _example_set(root: Path, *, shard: str = "00005", inputs: str | None = None) -> Path:
    """A tiny stored set: sixty lone hyphens between numbers, said "to" after "pages"
    and nothing after "score", and a few main records. ``shard`` names the records'
    origin and ``inputs`` the receipt's (``shard`` unless given); the receipt hashes the
    files either way."""
    root.mkdir()
    dashes = []
    for index in range(60):
        said = index % 2 == 0
        record = _dash(index, "to" if said else "", str(index + 1), str(index + 5), shard)
        if not said:
            record["L"] = ["score", str(index + 1)]
        dashes.append(record)
    main = [
        {
            "problem": "measured:acronym-spelled\tmeasured:acronym-word",
            "label": "measured:acronym-word",
            "tok": "NASA",
            "L": ["the"],
            "R": ["said"],
            "bos": True,
            "eos": True,
            "src": f"output-{shard}-of-00100:100:1",
        },
        # A main-set dash is left out: the dash set replaces it.
        {**_dash(200, "", "1", "2", shard), "label": "surface:silence"},
    ]
    del main[1]["gold"]
    files = {}
    for name, records in (("examples.jsonl.gz", main), ("dash_examples.jsonl.gz", dashes)):
        body = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
        blob = gzip.compress(body, mtime=0)
        (root / name).write_bytes(blob)
        files[name] = hashlib.sha256(blob).hexdigest()
    receipt = {
        "derivation": "frend/google/tn-en_with_types/p7-examples",
        "inputs": {f"output-{inputs or shard}-of-00100": "0" * 64},
        "artifacts": files,
        "fingerprint": "feedfacefeedface",
    }
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return root


def _out(root: Path) -> Path:
    root.mkdir()
    (root / "festival_classes.json").write_bytes((_CONTEXT / "festival_classes.json").read_bytes())
    return root


def test_trees_are_trained_reproducibly_and_name_their_examples(tmp_path):
    """Two builds from one stored set write the same bytes; the index names the corpus
    and the set's fingerprint; the lone hyphen between numbers is a connector problem
    with its "to" reading; the main set's dash is replaced by the dash set's."""
    builder = _builder()
    examples = _example_set(tmp_path / "set")
    first = _out(tmp_path / "a")
    second = _out(tmp_path / "b")
    index = builder.build(examples, first, workers=1, cache=tmp_path / "c1", log=lambda _: None)
    builder.build(examples, second, workers=1, cache=tmp_path / "c2", log=lambda _: None)
    assert builder._compare(first, second) == []
    provenance = index["provenance"]
    assert provenance["corpus"] == "google/tn-en_with_types"
    assert provenance["examples"]["fingerprint"] == "feedfacefeedface"
    assert provenance["records"] == {"main": 1, "dash": 60}
    connector = index["connectors"]["-"]
    assert connector["connector_id"] == "to"
    assert connector["connector_source"] == "lexical:en_US"
    assert connector["first"] == "surface:silence"
    tree = index["trees"][connector["problem"]]
    assert tree["labels"] == {"lexical:en_US": 30, "surface:silence": 30}
    from cartlet.runner import read_cart_metadata

    metadata = read_cart_metadata((first / tree["file"]).read_bytes())
    assert metadata["source"] == "google/tn-en_with_types"
    assert metadata["examples"] == "feedfacefeedface"


def test_q16_apportionment_preserves_the_winning_probability():
    builder = _builder()
    distribution = {
        "best": 0.7,
        "second": 0.18527918781725888,
        "third": 0.11472081218274112,
    }
    expected = distribution.copy()
    builder._apportion_q16(distribution)
    counts = [round(probability * 65_535) for probability in distribution.values()]
    decoded = [count / sum(counts) for count in counts]
    assert (
        max(
            abs(probability - stored)
            for probability, stored in zip(expected.values(), decoded, strict=True)
        )
        < 1e-5
    )
    assert decoded[0] >= 0.7
    assert next(iter(distribution)) == "best"


def test_a_set_that_is_not_its_receipts_is_refused(tmp_path):
    builder = _builder()
    examples = _example_set(tmp_path / "set")
    (examples / "dash_examples.jsonl.gz").write_bytes(gzip.compress(b"{}\n", mtime=0))
    with pytest.raises(SystemExit, match="sha256"):
        builder.build(examples, _out(tmp_path / "a"), workers=1, cache=tmp_path / "c")


def test_a_validly_hashed_set_from_a_held_out_shard_is_refused(tmp_path):
    """Shard 95 is held out: a set whose receipt names it is refused, however well its
    files match the receipt."""
    builder = _builder()
    examples = _example_set(tmp_path / "set", shard="00095")
    with pytest.raises(SystemExit, match="output-00095-of-00100"):
        builder.build(examples, _out(tmp_path / "a"), workers=1, cache=tmp_path / "c")


def test_held_out_records_under_a_training_receipt_are_refused(tmp_path):
    """A receipt naming only training shards does not vouch for its records: a record
    whose ``src`` is shard 95 is refused."""
    builder = _builder()
    examples = _example_set(tmp_path / "set", shard="00095", inputs="00005")
    with pytest.raises(SystemExit, match="output-00095-of-00100"):
        builder.build(examples, _out(tmp_path / "a"), workers=1, cache=tmp_path / "c")


def test_a_reused_cache_holding_a_held_out_set_is_refused(tmp_path):
    """``--cache`` reuses the files already there, so the cache's own receipt is the one
    read: a cached held-out set is refused even when the source set is clean."""
    builder = _builder()
    clean = _example_set(tmp_path / "set")
    cache = _example_set(tmp_path / "cache", shard="00095")
    with pytest.raises(SystemExit, match="output-00095-of-00100"):
        builder.build(clean, _out(tmp_path / "a"), workers=1, cache=cache)


def test_the_shipped_trees_name_the_example_set_they_were_trained_on():
    """The shipped index names the corpus (shippable-share-alike), the stored set's
    fingerprint and training shards only; every tree it lists is there, with its hash."""
    sys.path.insert(0, str(_TOOLS))
    import google_tn_rows

    index = json.loads((_CONTEXT / "index.json").read_text(encoding="utf-8"))
    provenance = index["provenance"]
    assert provenance["corpus"] == "google/tn-en_with_types"
    assert provenance["examples"]["fingerprint"] == "d1fcf656b9eb98dc"
    assert set(provenance["examples"]["shards"]) <= google_tn_rows.TRAINING_SHARDS
    files = {p.name for p in (_CONTEXT / "trees").iterdir()}
    assert files == {Path(entry["file"]).name for entry in index["trees"].values()}
    for entry in index["trees"].values():
        blob = (_CONTEXT / entry["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == entry["sha256"]


def _range_set(root: Path, *, shard: str = "00005", count: int = 60) -> Path:
    """A stored range set of ``count`` records (enough for a range tree: at least
    ``MIN_EXAMPLES``), "to" after "pages" and silent after "score", its receipt hashing
    it."""
    root.mkdir()
    records = []
    for index in range(count):
        said = index % 2 == 0
        left, right = index + 1, index + 5
        from frend.spoken_priors import normalize_spoken
        from frend.verbalize import _number_leaf

        words = [normalize_spoken(_number_leaf(Decimal(n), "cardinal", "en_US")[0].text)
                 for n in (left, right)]  # fmt: skip
        records.append(
            {
                "tok": f"{left}-{right}",
                "L": "pages" if said else "score",
                "R": "of the book",
                "gold": f"{words[0]} to {words[1]}" if said else f"{words[0]} {words[1]}",
                "bos": True,
                "eos": True,
                "src": f"output-{shard}-of-00100:{index}:1",
            }
        )
    body = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
    blob = gzip.compress(body, mtime=0)
    (root / "range_examples.jsonl.gz").write_bytes(blob)
    receipt = {
        "derivation": "frend/google/tn-en_with_types/p6-range-examples",
        "inputs": {"output-00005-of-00100": "0" * 64},
        "artifacts": {"range_examples.jsonl.gz": hashlib.sha256(blob).hexdigest()},
        "fingerprint": "beadbeadbeadbead",
    }
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return root


def test_range_problems_are_trained():
    """The shipped index holds a tree for each separator class's range problem, each
    naming the range set's own fingerprint and relevance; the main set's fingerprint is
    unchanged."""
    index = json.loads((_CONTEXT / "index.json").read_text(encoding="utf-8"))
    for problem in ("range:range", "range:ratio", "range:dimension"):
        entry = index["trees"][problem]
        assert entry["examples_fingerprint"] == index["range_examples"]["fingerprint"]
        assert entry["relevance"]["dropped_filter"].startswith("punctuation_dash:")
        assert "R" in entry["families"]
    assert index["provenance"]["examples"]["fingerprint"] == "d1fcf656b9eb98dc"
    assert index["range_examples"]["fingerprint"] != "d1fcf656b9eb98dc"


def test_range_examples_refuse_held_out(tmp_path):
    """A validly hashed range set whose record comes from shard 95 is refused."""
    builder = _builder()
    examples = _example_set(tmp_path / "set")
    ranged = _range_set(tmp_path / "range", shard="00095")
    with pytest.raises(SystemExit, match="output-00095-of-00100"):
        builder.build(
            examples, _out(tmp_path / "a"), workers=1, cache=tmp_path / "c",
            range_examples=ranged, range_cache=tmp_path / "rc", log=lambda _: None,
        )  # fmt: skip


def test_the_range_set_leaves_the_main_trees_as_they_are(tmp_path):
    """A build with a range set writes the main set's trees byte for byte as a build
    without it (the range problems draw with a generator of their own and do not reach
    the frequent words)."""
    builder = _builder()
    examples = _example_set(tmp_path / "set")
    plain = _out(tmp_path / "a")
    ranged_out = _out(tmp_path / "b")
    index = builder.build(examples, plain, workers=1, cache=tmp_path / "c1", log=lambda _: None)
    builder.build(
        examples, ranged_out, workers=1, cache=tmp_path / "c2", log=lambda _: None,
        range_examples=_range_set(tmp_path / "range"), range_cache=tmp_path / "rc",
    )  # fmt: skip
    files = {entry["file"] for entry in index["trees"].values()}
    assert builder._compare(ranged_out, plain, only=files) == []
    ranged_index = json.loads((ranged_out / "index.json").read_text(encoding="utf-8"))
    # The range set does train a tree here (so its draws and words are exercised).
    assert "range:range" in ranged_index["trees"]
    assert ranged_index["frequent_words"] == index["frequent_words"]
