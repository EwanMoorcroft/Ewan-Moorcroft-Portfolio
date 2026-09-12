"""Input identities describe the exact bytes parsed, even if a path changes."""

import hashlib
from pathlib import Path

from chest_xray_benchmark import training


def test_training_inputs_are_parsed_and_hashed_from_one_snapshot(tmp_path, monkeypatch):
    settings = tmp_path / "settings.toml"
    splits = tmp_path / "splits.csv"
    config_bytes = b"seed = 123\n"
    split_bytes = (
        b"relative_path,label,byte_size,sha256,difference_hash,width,height,color_mode,has_exif,exact_group_id,split\n"
        + b"".join(
            f"{i}.png,Normal,1,{str(i) * 64},0,1,1,RGB,0,sha256-{str(i) * 64},{split}\n".encode()
            for i, split in enumerate(("train", "validation", "test"))
        )
    )
    settings.write_bytes(config_bytes)
    splits.write_bytes(split_bytes)
    original_read = Path.read_bytes

    def replaced_after_read(path):
        raw = original_read(path)
        if path in (settings, splits):
            path.write_bytes(b"changed after snapshot")
        return raw

    monkeypatch.setattr(Path, "read_bytes", replaced_after_read)
    config, rows, config_hash, split_hash = training.read_training_inputs(settings, splits)
    assert config.seed == 123
    assert len(rows) == 3
    assert config_hash == hashlib.sha256(config_bytes).hexdigest()
    assert split_hash == hashlib.sha256(split_bytes).hexdigest()
