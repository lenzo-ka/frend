"""Offline tests for tools/fetch_corpora.py and its manifest.

Nothing here touches the network or the real corpora. Archives are built in a
temporary directory with the same shape as the real ones (a zip wrapping a
gzipped tar whose shards sit under ``en_with_types/``), and a source record is
written for them, so every verification and refusal runs against real bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import re
import tarfile
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "tools" / "fetch_corpora.py"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _load_tool():
    spec = importlib.util.spec_from_file_location("fetch_corpora", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tgz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


_SHARDS = {
    "output-00000-of-00002": b"CARDINAL\t12\ttwelve\n",
    "output-00001-of-00002": b"DATE\t2006\ttwo thousand six\n",
}


def _archives(tmp_path: Path, members: dict[str, bytes] | None = None):
    """Write a tgz and a zip wrapping it; return both paths and a matching source."""
    if members is None:
        members = {f"en_with_types/{name}": data for name, data in _SHARDS.items()}
        members["en_with_types/README"] = b"not a shard\n"
    tgz_bytes = _tgz(members)
    tgz = tmp_path / "en_with_types.tgz"
    tgz.write_bytes(tgz_bytes)
    archive = tmp_path / "en_with_types.tgz.zip"
    with zipfile.ZipFile(archive, "w") as outer:
        outer.writestr("en_with_types.tgz", tgz_bytes)
    source = {
        "zip": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": _sha(archive.read_bytes()),
        },
        "tgz": {"name": tgz.name, "bytes": len(tgz_bytes), "sha256": _sha(tgz_bytes)},
        "dest": "en_with_types",
        "shards": {name: _sha(data) for name, data in _SHARDS.items()},
    }
    return tgz, archive, source


def test_manifest_pins_every_shard_and_table_by_sha256():
    manifest = tool.load_manifest()
    google, nemo = manifest["google-tn"], manifest["nemo"]
    assert sorted(google["shards"]) == [f"output-{i:05d}-of-00100" for i in range(100)]
    assert len(nemo["files"]) == 21
    assert all(name.startswith("test_cases_") for name in nemo["files"])
    digests = [google["zip"]["sha256"], google["tgz"]["sha256"]]
    digests += list(google["shards"].values()) + list(nemo["files"].values())
    assert all(_HEX64.match(digest) for digest in digests)
    assert re.fullmatch(r"[0-9a-f]{40}", nemo["commit"])


def test_manifest_names_only_public_sources():
    text = _SCRIPT.with_name("corpora.json").read_text(encoding="utf-8")
    urls = re.findall(r'"(\w+://[^"]+)"', text)
    assert urls and all(url.startswith("https://") for url in urls)
    assert not re.search(r"/Users/|/home/|/Volumes/|/mnt/|[A-Za-z]:\\\\", text)


@pytest.mark.parametrize("use_zip", [True, False])
def test_extracts_only_pinned_shards_from_zip_or_tgz(tmp_path, use_zip):
    tgz, archive, source = _archives(tmp_path)
    dest = tmp_path / "out"
    tool.extract_google_tn(archive if use_zip else tgz, dest, source)
    assert sorted(p.name for p in dest.iterdir()) == sorted(_SHARDS)
    assert tool.verify_files(dest, source["shards"]) == []


def test_refuses_a_zip_whose_sha256_differs(tmp_path):
    _, archive, source = _archives(tmp_path)
    source["zip"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest pins"):
        tool.extract_google_tn(archive, tmp_path / "out", source)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_refuses_a_tgz_inside_the_zip_whose_sha256_differs(tmp_path):
    _, archive, source = _archives(tmp_path)
    source["tgz"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="en_with_types.tgz: sha256"):
        tool.extract_google_tn(archive, tmp_path / "out", source)


def test_refuses_an_archive_missing_a_pinned_shard(tmp_path):
    tgz, _, source = _archives(tmp_path)
    source["shards"]["output-00002-of-00002"] = "0" * 64
    with pytest.raises(ValueError, match="absent from the archive"):
        tool.extract_google_tn(tgz, tmp_path / "out", source)


def test_member_paths_cannot_escape_the_destination(tmp_path):
    name = "output-00000-of-00002"
    members = {f"../../{name}": _SHARDS[name], f"en_with_types/{name[:-1]}2": b"x"}
    tgz, _, source = _archives(tmp_path, members)
    source["shards"] = {name: _sha(_SHARDS[name])}
    dest = tmp_path / "a" / "b" / "out"
    tool.extract_google_tn(tgz, dest, source)
    assert [p.name for p in dest.iterdir()] == [name]
    assert not (tmp_path / "a" / name).exists()


def test_verify_reports_missing_and_mismatched_files(tmp_path):
    (tmp_path / "present").write_bytes(b"good")
    (tmp_path / "changed").write_bytes(b"bad")
    pins = {"present": _sha(b"good"), "changed": _sha(b"good"), "absent": _sha(b"x")}
    problems = tool.verify_files(tmp_path, pins)
    assert len(problems) == 2
    assert any("missing" in p and "absent" in p for p in problems)
    assert any("changed" in p and "manifest pins" in p for p in problems)


def test_a_root_that_is_a_dangling_symlink_is_refused(tmp_path):
    root = tmp_path / "tn-corpus"
    root.symlink_to(tmp_path / "not-mounted")
    with pytest.raises(FileNotFoundError, match="missing target"):
        tool.check_root(root)
    assert tool.main(["--verify", "--root", str(tmp_path / "empty")]) == 1


def test_unknown_source_is_a_usage_error():
    with pytest.raises(SystemExit) as exit_info:
        tool.main(["nemo-full"])
    assert exit_info.value.code == 2
