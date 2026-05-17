import csv
import json
import tempfile
import unittest
from pathlib import Path

from opsd_alignment.scripts.aggregate_results import aggregate_records, load_alignment_records, write_summary_csv
from opsd_alignment.src.storage import write_jsonl


class OpsdAlignmentAggregationTest(unittest.TestCase):
    def test_aggregate_records_tracks_valid_and_invalid_alignments(self):
        records = [
            {
                "checkpoint": "stage1",
                "teacher_context": "full_solution",
                "distillation_objective": "forward_kl",
                "source": "gsm8k",
                "difficulty": "easy",
                "student_rollout_correct": True,
                "selection_reason": "high_kl",
                "alignment": 0.5,
                "student_teacher_kl": 0.2,
                "student_entropy": 1.0,
                "baseline_success": 0.4,
                "mean_branch_success": 0.5,
                "num_candidates": 4,
            },
            {
                "checkpoint": "stage1",
                "teacher_context": "full_solution",
                "distillation_objective": "forward_kl",
                "source": "gsm8k",
                "difficulty": "easy",
                "student_rollout_correct": True,
                "selection_reason": "high_kl",
                "alignment": None,
                "student_teacher_kl": 0.4,
                "student_entropy": 2.0,
                "baseline_success": 0.6,
                "mean_branch_success": 0.75,
                "num_candidates": 6,
            },
        ]

        summaries = aggregate_records(records)

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["count_nodes"], 1)
        self.assertEqual(summary["invalid_alignment_count"], 1)
        self.assertEqual(summary["invalid_alignment_fraction"], 0.5)
        self.assertEqual(summary["mean_alignment"], 0.5)
        self.assertEqual(summary["mean_student_success_rate"], 1.0)
        self.assertEqual(summary["mean_num_candidates"], 5.0)

    def test_load_alignment_records_filters_and_csv_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            alignment_path = tmp / "alignments.jsonl"
            records = [
                {"checkpoint": "stage1", "teacher_context": "answer_only", "alignment": 0.1},
                {"checkpoint": "stage2", "teacher_context": "full_solution", "alignment": 0.2},
            ]
            write_jsonl(alignment_path, records)
            loaded = load_alignment_records(
                {"diagnostic": {}},
                alignment_file=str(alignment_path),
                model_name="stage2",
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["checkpoint"], "stage2")

            csv_path = tmp / "summary.csv"
            write_summary_csv(csv_path, [{"checkpoint": "stage2", "mean_alignment": 0.2}])
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["checkpoint"], "stage2")
            self.assertEqual(rows[0]["mean_alignment"], "0.2")


if __name__ == "__main__":
    unittest.main()
