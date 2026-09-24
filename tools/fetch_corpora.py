"""Fetch and verify the full corpora frend's data tables are built from.

Every shipped table under ``frend/data`` is re-derivable with ``--check`` from
public sources, and this tool is how someone else gets those sources.
``tools/corpora.json`` pins each one by upstream identity and sha256; nothing is
used that does not match it.

- ``google-tn``: the Google/Sproat English text-normalization corpus
  (``en_with_types``, CC BY-SA 4.0), hosted on Kaggle. Kaggle serves this public
  dataset without an account. Kaggle wraps the upstream ``en_with_types.tgz``
  (about 3.9 GB) in a zip; both are pinned. The 100 shards are extracted to
  ``<root>/en_with_types``, which is where ``build_type_priors.py`` and
  ``build_spoken_priors.py`` look, and each shard is pinned too, so a copy
  obtained some other way can be verified without the archive.
- ``nemo``: every English text-normalization test table from NVIDIA
  NeMo-text-processing (Apache 2.0) at a pinned commit, fetched verbatim into
  ``<root>/nemo_tn_en``. Lines are ``written~spoken`` and many are sentences,
  not single tokens, so they are not converted into the Google-TN token format.

The root defaults to ``tn-corpus`` beside the repository. A root that is a
symlink to a missing target is refused rather than written through.

    python tools/fetch_corpora.py                  # fetch whatever is missing
    python tools/fetch_corpora.py nemo             # one source
    python tools/fetch_corpora.py --verify         # check what is present
    python tools/fetch_corpora.py google-tn --archive en_with_types.tgz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = Path(__file__).resolve().parent / "corpora.json"
_CHUNK = 1 << 20
SOURCES = ("google-tn", "nemo")


def load_manifest(path: Path = _MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def default_root() -> Path:
    return _REPO.parent / "tn-corpus"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class _HashingReader:
    """A read-only stream that hashes every byte read through it."""

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        self.digest.update(data)
        return data

    def drain(self) -> None:
        while self.read(_CHUNK):
            pass


def _refuse(what: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{what}: sha256 {actual}, manifest pins {expected}")


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    """Stream ``url`` to ``dest``, keeping it only if its sha256 matches."""
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, part.open("wb") as out:
        while chunk := response.read(_CHUNK):
            digest.update(chunk)
            out.write(chunk)
    try:
        _refuse(url, digest.hexdigest(), expected_sha256)
    except ValueError:
        part.unlink()
        raise
    part.replace(dest)


def check_root(root: Path) -> None:
    if root.is_symlink() and not root.exists():
        raise FileNotFoundError(f"{root} is a symlink to a missing target; mount it or pass --root")


def verify_files(directory: Path, pins: dict[str, str]) -> list[str]:
    """Return one problem per pinned file that is missing or differs."""
    problems = []
    for name, expected in sorted(pins.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"missing {path}")
        elif (actual := sha256_file(path)) != expected:
            problems.append(f"{path}: sha256 {actual}, manifest pins {expected}")
    return problems


def _extract_shards(stream: BinaryIO, dest: Path, shards: dict[str, str]) -> set[str]:
    """Extract only pinned shard names from a gzipped tar stream into ``dest``.

    Members are matched by base name, so nothing else is written and no member
    path can escape ``dest``.
    """
    written = set()
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in shards:
                continue
            part = dest / (name + ".part")
            with tar.extractfile(member) as data, part.open("wb") as out:
                shutil.copyfileobj(data, out, _CHUNK)
            part.replace(dest / name)
            written.add(name)
    return written


def extract_google_tn(archive: Path, dest: Path, source: dict) -> None:
    """Verify ``archive`` (Kaggle's zip or the upstream tgz) and extract its shards.

    The tgz's own sha256 is checked either way: directly for a bare tgz, and by
    hashing the member as it streams out of the zip.
    """
    tgz = source["tgz"]
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        _refuse(str(archive), sha256_file(archive), source["zip"]["sha256"])
        with zipfile.ZipFile(archive) as outer, outer.open(tgz["name"]) as inner:
            reader = _HashingReader(inner)
            written = _extract_shards(reader, dest, source["shards"])
            reader.drain()
        _refuse(f"{archive}:{tgz['name']}", reader.digest.hexdigest(), tgz["sha256"])
    else:
        _refuse(str(archive), sha256_file(archive), tgz["sha256"])
        with archive.open("rb") as stream:
            written = _extract_shards(stream, dest, source["shards"])
    if missing := sorted(set(source["shards"]) - written):
        raise ValueError(f"{archive}: pinned shards absent from the archive: {missing}")


def fetch_google_tn(root: Path, source: dict, archive: Path | None) -> None:
    dest = root / source["dest"]
    if not verify_files(dest, source["shards"]):
        print(f"google-tn: {len(source['shards'])} shards already verified in {dest}")
        return
    if archive is None:
        archive = root / source["zip"]["name"]
        if not archive.is_file():
            print(f"google-tn: downloading {source['zip']['bytes']:,} bytes to {archive}")
            _download(source["url"], archive, source["zip"]["sha256"])
    extract_google_tn(archive, dest, source)
    if problems := verify_files(dest, source["shards"]):
        raise ValueError("google-tn: extracted shards do not verify:\n" + "\n".join(problems))
    print(f"google-tn: {len(source['shards'])} shards verified in {dest}")


def fetch_nemo(root: Path, source: dict) -> None:
    dest = root / source["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    base = f"{source['raw_base']}/{source['commit']}/{source['path']}"
    for name, expected in sorted(source["files"].items()):
        path = dest / name
        if not (path.is_file() and sha256_file(path) == expected):
            _download(f"{base}/{name}", path, expected)
    print(f"nemo: {len(source['files'])} tables verified in {dest}")


def _pins(key: str, source: dict) -> dict[str, str]:
    return source["shards"] if key == "google-tn" else source["files"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sources", nargs="*", metavar="SOURCE", help="google-tn, nemo (default: both)"
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="corpus root (default: ../tn-corpus)"
    )
    parser.add_argument("--verify", action="store_true", help="check present files; fetch nothing")
    parser.add_argument(
        "--archive", type=Path, default=None, help="google-tn: an already-obtained zip or tgz"
    )
    args = parser.parse_args(argv)
    if unknown := sorted(set(args.sources) - set(SOURCES)):
        parser.error(f"unknown source(s) {unknown}; choose from {list(SOURCES)}")

    manifest = load_manifest()
    root = args.root or default_root()
    wanted = args.sources or list(SOURCES)
    check_root(root)

    if args.verify:
        failed = False
        for key in wanted:
            source = manifest[key]
            problems = verify_files(root / source["dest"], _pins(key, source))
            failed |= bool(problems)
            count = len(_pins(key, source))
            print(f"{key}: " + ("\n  ".join(problems) if problems else f"{count} files verified"))
        return 1 if failed else 0

    root.mkdir(parents=True, exist_ok=True)
    if "google-tn" in wanted:
        fetch_google_tn(root, manifest["google-tn"], args.archive)
    if "nemo" in wanted:
        fetch_nemo(root, manifest["nemo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
