import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opsd_alignment.scripts.validate_config import validate_config
from opsd_alignment.src.models import resolve_model_path


class OpsdAlignmentConfigTest(unittest.TestCase):
    def test_resolve_model_path_expands_environment_variable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OPSD_TEST_MODEL": tmpdir}):
                self.assertEqual(resolve_model_path({"name": "m", "path": "${OPSD_TEST_MODEL}"}), tmpdir)

    def test_resolve_model_path_reports_unset_environment_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_model_path({"name": "m", "path": "${MISSING_OPSD_TEST_MODEL}"})

    def test_validate_config_accepts_schema_when_model_paths_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            question_path = Path(tmpdir) / "questions.jsonl"
            question_path.write_text("{}\n", encoding="utf-8")
            config = {
                "models": [{"name": "m", "path": "${MISSING}"}],
                "teacher_contexts": ["answer_only", "full_solution"],
                "generation": {"temperature": 0.7, "top_p": 0.95},
                "diagnostic": {"distillation_objective": "forward_kl", "top_k_student": 2, "top_k_teacher": 2},
                "paths": {"questions": str(question_path), "output_dir": str(Path(tmpdir) / "outputs")},
            }

            self.assertEqual(validate_config(config, check_model_paths=False), [])
            self.assertTrue(validate_config(config, check_model_paths=True))


if __name__ == "__main__":
    unittest.main()
