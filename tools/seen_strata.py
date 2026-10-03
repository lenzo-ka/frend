"""Build and load a private Google-TN written-form seen/unseen vocabulary.

The vocabulary is a byte-sorted UTF-8 file with one case-preserved written form per
line.  Its sibling ``.meta.json`` receipt binds those forms to all 90 verified training
shards.  Neither artifact may live in the repository: it is licensed evaluation state,
not package data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from corpus_inputs import VerifiedInput, _entry  # noqa: E402
from google_tn_rows import TRAINING_SHARDS  # noqa: E402

SCHEMA_VERSION = 2
EXTRACTOR_ID = "google-tn-tab-column-2-non-eos-v1"
SOURCE_ID = "google/tn-en_with_types"
LOCALE = "en_US"
STRATA = ("SEEN", "UNSEEN")
SENTENCE_STRATA = (
    "ALL_SEEN_NONTRIVIAL",
    "HAS_UNSEEN_NONTRIVIAL",
    "ALL_SEEN",
    "HAS_UNSEEN",
)
SENTENCE_RULE = (
    "ALL_SEEN_NONTRIVIAL ignores corpus <self> and sil rows; ALL_SEEN counts every token"
)


@dataclass(frozen=True)
class VocabularyMatch:
    """The requested forms found in one fully verified vocabulary artifact."""

    seen: frozenset[str]
    receipt: dict[str, object]


def metadata_path(path: Path) -> Path:
    return Path(f"{path}.meta.json")


def sentence_strata(rows, token_strata) -> tuple[str, str]:
    """Return the non-trivial-token and strict all-token sentence strata."""
    pairs = list(zip(rows, token_strata, strict=True))
    nontrivial_seen = all(
        stratum == "SEEN" for row, stratum in pairs if row[2] not in {"<self>", "sil"}
    )
    all_seen = all(stratum == "SEEN" for _row, stratum in pairs)
    return (
        "ALL_SEEN_NONTRIVIAL" if nontrivial_seen else "HAS_UNSEEN_NONTRIVIAL",
        "ALL_SEEN" if all_seen else "HAS_UNSEEN",
    )


def external_path(path: Path) -> Path:
    """Resolve ``path`` and refuse private vocabulary state inside this repository."""
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(_REPO.resolve()):
        raise ValueError(f"strata vocabulary must be outside the repository: {resolved}")
    return resolved


def require_training_inputs(inputs) -> None:
    """Require exactly the canonical 90 verified training shards, and no other shard."""
    items = list(inputs)
    if any(not isinstance(item, VerifiedInput) for item in items):
        raise TypeError("strata vocabulary requires verified corpus inputs")
    names = [item.relative_path for item in items]
    outside = sorted(set(names) - TRAINING_SHARDS)
    if outside:
        raise ValueError(
            f"strata vocabulary inputs must be Google TN training shards 00-89; got {outside[0]!r}"
        )
    missing = sorted(TRAINING_SHARDS - set(names))
    if missing:
        raise ValueError(
            f"strata vocabulary requires all 90 training shards; first missing {missing[0]!r}"
        )
    if len(names) != len(set(names)):
        raise ValueError("strata vocabulary inputs contain a duplicate training shard")
    wrong_source = sorted({item.source_id for item in items if item.source_id != SOURCE_ID})
    if wrong_source:
        raise ValueError(f"strata vocabulary inputs must come from {SOURCE_ID!r}")


def source_shards(inputs) -> list[dict[str, str]]:
    require_training_inputs(inputs)
    return [
        {"relative_path": item.relative_path, "sha256": item.sha256}
        for item in sorted(inputs, key=lambda item: item.relative_path)
    ]


def _sha256_and_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            lines += 1
    return digest.hexdigest(), lines


def _stream_written(item: VerifiedInput, sink) -> None:
    """Stream one already catalog-verified shard, hashing the exact bytes consumed."""
    if item.path is None:
        raise ValueError("verified input has no bound path")
    path = Path(item.path)
    if path.is_symlink():
        raise ValueError(f"verified input must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved.name != item.relative_path or "/" in item.relative_path:
        raise ValueError("verified input path no longer matches its receipt identity")
    before = _freshness(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for line in handle:
            digest.update(line)
            parts = line.rstrip(b"\n").split(b"\t", 2)
            if len(parts) >= 3 and parts[0] != b"<eos>":
                written = parts[1]
                if b"\n" in written or b"\r" in written:
                    raise ValueError("written forms may not contain line breaks")
                sink.write(written + b"\n")
    if before != _freshness(resolved):
        raise ValueError(f"{item.relative_path}: changed while building strata vocabulary")
    actual = digest.hexdigest()
    if actual != item.sha256:
        raise ValueError(f"{item.relative_path}: sha256 changed after verification")


def build_vocabulary(path: Path, inputs) -> dict[str, object]:
    """Write a sorted unique vocabulary and its receipt, returning the receipt.

    ``sort`` performs the external-memory deduplication.  Each already catalog-verified
    shard is hashed again over the exact bytes streamed, with filesystem freshness
    checked around that read.
    """
    path = external_path(path)
    items = list(inputs)
    shards = source_shards(items)
    meta = metadata_path(path)
    if path.exists() or meta.exists():
        raise FileExistsError(f"refusing to overwrite existing strata artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output:
            process = subprocess.Popen(
                ["sort", "-u"],
                stdin=subprocess.PIPE,
                stdout=output,
                env={**os.environ, "LC_ALL": "C", "TMPDIR": str(path.parent)},
            )
            assert process.stdin is not None
            try:
                for item in sorted(items, key=lambda candidate: candidate.relative_path):
                    _stream_written(item, process.stdin)
            except BaseException:
                process.stdin.close()
                process.terminate()
                process.wait()
                raise
            process.stdin.close()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, process.args)
        artifact_sha256, vocabulary_size = _sha256_and_lines(temporary)
        receipt: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "extractor_id": EXTRACTOR_ID,
            "kind": "google-tn-written-form-vocabulary",
            "source_id": SOURCE_ID,
            "locale": LOCALE,
            "case_preserved": True,
            "source_shards": shards,
            "vocabulary_size": vocabulary_size,
            "artifact_sha256": artifact_sha256,
        }
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        meta_fd, meta_temporary_name = tempfile.mkstemp(prefix=f".{meta.name}.", dir=path.parent)
        os.close(meta_fd)
        meta_temporary = Path(meta_temporary_name)
        try:
            meta_temporary.write_text(rendered, encoding="utf-8")
            os.replace(temporary, path)
            os.replace(meta_temporary, meta)
        finally:
            meta_temporary.unlink(missing_ok=True)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def _freshness(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size


def _validate_receipt(receipt: object, path: Path) -> dict[str, object]:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"invalid strata vocabulary receipt at {path}: wrong schema version")
    expected_scalars = {
        "kind": "google-tn-written-form-vocabulary",
        "source_id": SOURCE_ID,
        "locale": LOCALE,
        "case_preserved": True,
    }
    if any(receipt.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError(f"invalid strata vocabulary receipt at {path}: wrong corpus identity")
    if receipt.get("extractor_id") != EXTRACTOR_ID:
        raise ValueError(f"invalid strata vocabulary receipt at {path}: wrong extraction rule")
    shards = receipt.get("source_shards")
    if not isinstance(shards, list) or len(shards) != len(TRAINING_SHARDS):
        raise ValueError(f"invalid strata vocabulary receipt at {path}: not 90 source shards")
    names = [item.get("relative_path") for item in shards if isinstance(item, dict)]
    if set(names) != TRAINING_SHARDS or len(names) != len(set(names)):
        raise ValueError(f"invalid strata vocabulary receipt at {path}: non-training source shard")
    catalog = _entry(SOURCE_ID).get("shards", {})
    if any(
        not isinstance(item, dict) or catalog.get(item.get("relative_path")) != item.get("sha256")
        for item in shards
    ):
        raise ValueError(f"invalid strata vocabulary receipt at {path}: source digest mismatch")
    size = receipt.get("vocabulary_size")
    digest = receipt.get("artifact_sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"invalid strata vocabulary receipt at {path}: bad vocabulary size")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"invalid strata vocabulary receipt at {path}: bad artifact digest")
    return receipt


def load_vocabulary(path: Path, forms) -> VocabularyMatch:
    """Verify ``path`` and return which of the requested exact forms it contains."""
    path = external_path(path)
    meta = metadata_path(path)
    try:
        before = (_freshness(path), _freshness(meta))
        receipt = _validate_receipt(json.loads(meta.read_text(encoding="utf-8")), meta)
    except FileNotFoundError:
        raise FileNotFoundError(f"strata vocabulary or receipt is missing: {path}") from None

    requested = {form.encode("utf-8"): form for form in forms}
    found: set[str] = set()
    digest = hashlib.sha256()
    count = 0
    previous: bytes | None = None
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            count += 1
            if not line.endswith(b"\n"):
                raise ValueError(f"invalid strata vocabulary at {path}: unterminated row")
            form = line[:-1]
            if previous is not None and form <= previous:
                raise ValueError(f"invalid strata vocabulary at {path}: rows are not unique/sorted")
            previous = form
            if form in requested:
                found.add(requested[form])
    after = (_freshness(path), _freshness(meta))
    if before != after:
        raise ValueError(f"strata vocabulary changed while being read at {path}")
    if count != receipt["vocabulary_size"] or digest.hexdigest() != receipt["artifact_sha256"]:
        raise ValueError(f"strata vocabulary at {path} does not match its receipt")
    return VocabularyMatch(frozenset(found), receipt)


def vocabulary_receipt(match: VocabularyMatch) -> dict[str, object]:
    """Small reproducibility record safe to include with aggregate evaluator output."""
    return {
        "artifact_sha256": match.receipt["artifact_sha256"],
        "extractor_id": match.receipt["extractor_id"],
        "vocabulary_size": match.receipt["vocabulary_size"],
        "source_shards": match.receipt["source_shards"],
    }
