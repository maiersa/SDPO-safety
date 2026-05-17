import tempfile
import unittest
from pathlib import Path

from opsd_alignment.scripts.plot_results import PLOT_FILENAMES, generate_plots, load_alignment_records
from opsd_alignment.src.storage import write_jsonl


class OpsdAlignmentPlottingTest(unittest.TestCase):
    def test_load_alignment_records_filters_invalid_and_generate_plots(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            alignment_path = tmp / "alignments.jsonl"
            output_dir = tmp / "plots"
            records = [
                {
                    "checkpoint": "stage1",
                    "teacher_context": "answer_only",
                    "distillation_objective": "forward_kl",
                    "alignment": -0.2,
                    "student_teacher_kl": 0.8,
                    "student_rollout_correct": False,
                },
                {
                    "checkpoint": "stage1",
                    "teacher_context": "full_solution",
                    "distillation_objective": "forward_kl",
                    "alignment": 0.4,
                    "student_teacher_kl": 0.3,
                    "student_rollout_correct": True,
                },
                {
                    "checkpoint": "stage2",
                    "teacher_context": "full_solution",
                    "distillation_objective": "forward_kl",
                    "alignment": 0.6,
                    "student_teacher_kl": 0.2,
                    "student_rollout_correct": True,
                },
                {
                    "checkpoint": "stage2",
                    "teacher_context": "answer_only",
                    "distillation_objective": "forward_kl",
                    "alignment": None,
                    "student_teacher_kl": 0.1,
                    "student_rollout_correct": False,
                },
            ]
            write_jsonl(alignment_path, records)
            config = {
                "models": [{"name": "stage1"}, {"name": "stage2"}],
                "teacher_contexts": ["answer_only", "full_solution"],
                "paths": {"output_dir": str(tmp)},
            }

            loaded = load_alignment_records(config, alignment_file=str(alignment_path))
            self.assertEqual(len(loaded), 3)
            paths = generate_plots(config, loaded, output_dir=output_dir)

            self.assertEqual(set(paths), set(PLOT_FILENAMES))
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
