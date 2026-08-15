from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from chest_xray_benchmark.evaluation import read_verified_checkpoint
from chest_xray_benchmark.image_data import verify_split_image_identity
from chest_xray_benchmark.manifest import sha256_file


class ArtifactIdentityTests(unittest.TestCase):
    def test_split_image_bytes_are_rechecked_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            image_path = data_root / "Normal" / "image.png"
            image_path.parent.mkdir()
            Image.new("L", (16, 16), color=64).save(image_path)
            original = image_path.read_bytes()
            rows = [
                {
                    "relative_path": "Normal/image.png",
                    "byte_size": str(len(original)),
                    "sha256": sha256_file(image_path),
                }
            ]
            self.assertEqual(
                verify_split_image_identity(rows, data_root),
                [image_path.resolve(strict=True)],
            )

            changed = bytearray(original)
            changed[-1] ^= 1
            image_path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                verify_split_image_identity(rows, data_root)

    def test_checkpoint_must_match_the_digest_selected_during_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "selected.pt"
            checkpoint.write_bytes(b"selected-state")
            metadata = {"checkpoint_file_sha256": sha256_file(checkpoint)}
            checkpoint_bytes, digest = read_verified_checkpoint(metadata, checkpoint)
            self.assertEqual(checkpoint_bytes, b"selected-state")
            self.assertEqual(digest, metadata["checkpoint_file_sha256"])

            checkpoint.write_bytes(b"different-state")
            with self.assertRaisesRegex(ValueError, "differs from the state selected"):
                read_verified_checkpoint(metadata, checkpoint)

    def test_checkpoint_metadata_requires_a_canonical_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "selected.pt"
            checkpoint.write_bytes(b"selected-state")
            with self.assertRaisesRegex(ValueError, "valid selected-checkpoint SHA-256"):
                read_verified_checkpoint({}, checkpoint)


if __name__ == "__main__":
    unittest.main()
