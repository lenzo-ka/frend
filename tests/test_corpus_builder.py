"""Shared Google-TN corpus-builder boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from conftest import load_tool_module

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
_BUILDERS = (
    "build_electronic_span_priors.py",
    "build_grouped_id_priors.py",
    "build_inch_context_priors.py",
    "build_number_priors.py",
    "build_range_priors.py",
    "build_telephone_priors.py",
)


def _builder(filename: str):
    return load_tool_module(filename.removesuffix(".py"), _TOOLS / filename)


@pytest.mark.parametrize("builder_filename", _BUILDERS)
@pytest.mark.parametrize("index", range(90, 100))
def test_builders_refuse_every_explicitly_supplied_held_out_shard(
    tmp_path, builder_filename, index
):
    builder = _builder(builder_filename)
    from corpus_inputs import VerifiedInput

    shard = tmp_path / f"output-{index:05d}-of-00100"
    shard.write_text("", encoding="utf-8")
    verified = VerifiedInput(
        "google/tn-en_with_types",
        shard.name,
        hashlib.sha256(b"").hexdigest(),
        "shippable-share-alike",
        shard,
    )

    with pytest.raises(ValueError, match="held-out shard"):
        builder.build_document(tmp_path, inputs=[verified])
