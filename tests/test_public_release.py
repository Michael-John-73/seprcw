import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_release import digest, verify_inventory


class PublicReleaseTests(unittest.TestCase):
    def test_inventory_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            path = root / "record.txt"
            path.write_text("original", encoding="utf-8")
            (root / "data" / "release_manifest.json").write_text(
                json.dumps({"files": {"record.txt": digest(path)}}), encoding="utf-8"
            )
            self.assertEqual(verify_inventory(root), 1)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_inventory(root)

    def test_inventory_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "release_manifest.json").write_text(
                json.dumps({"files": {"../outside": "invalid"}}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                verify_inventory(root)

    def test_no_reuse_license_is_invented(self):
        text = (ROOT / "LICENSE_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("No license has yet been selected", text)
        self.assertFalse((ROOT / "LICENSE").exists())


if __name__ == "__main__":
    unittest.main()
