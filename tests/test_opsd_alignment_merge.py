import json
import tempfile
import unittest
from pathlib import Path

from opsd_alignment.scripts.merge_jsonl import merge_jsonl_files, resolve_merge_paths


NL = chr(10)


class OpsdAlignmentMergeTest(unittest.TestCase):
    def test_merge_jsonl_files_preserves_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard0 = tmp / "part0.jsonl"
            shard1 = tmp / "part1.jsonl"
            out = tmp / "merged.jsonl"
            shard0.write_text(json.dumps({"a": 1}) + NL, encoding="utf-8")
            shard1.write_text(json.dumps({"a": 2}) + NL, encoding="utf-8")

            count = merge_jsonl_files([shard0, shard1], out)

            self.assertEqual(count, 2)
            self.assertEqual(len(out.read_text(encoding="utf-8").strip().splitlines()), 2)

    def test_resolve_merge_paths_for_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            distributions = tmp / "distributions"
            distributions.mkdir()
            shard = distributions / "teacher_student_distributions.shard00000-of-00001.jsonl"
            shard.write_text("{}" + NL, encoding="utf-8")
            config = {"paths": {"output_dir": str(tmp)}}

            inputs, output = resolve_merge_paths(config, artifact="distributions")

            self.assertEqual(inputs, [shard])
            self.assertEqual(output, distributions / "teacher_student_distributions.jsonl")


if __name__ == "__main__":
    unittest.main()
