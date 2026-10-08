"""Submission portability checks: python -m unittest discover -s scripts -p test_submission.py."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("lab_verify", Path(__file__).with_name("verify.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


class ColabEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = "/content/lab22/models/sft-merged"
        self.metrics = {"reference": "models/sft-merged (precomputed)", "eval_reward_accuracy": 0.67}
        self.nb = {"cells": [
            {"cell_type": "code", "execution_count": 1,
             "source": ["model.save_pretrained_merged(str(C.SFT_MERGED), tokenizer)"],
             "outputs": [{"output_type": "stream", "text": [
                 "Merge process complete\nSaved merged 16-bit → " + self.base]}]},
            {"cell_type": "code", "execution_count": 2,
             "source": ["write('dpo_metrics.json')"],
             "outputs": [{"output_type": "stream", "text": [json.dumps(self.metrics)]}]},
        ]}
        self.root.joinpath("colab").mkdir()
        self.root.joinpath("adapters/dpo").mkdir(parents=True)
        self.root.joinpath("adapters/dpo/dpo_metrics.json").write_text(json.dumps(self.metrics))
        self.root.joinpath("adapters/dpo/adapter_config.json").write_text(json.dumps({"base_model_name_or_path": self.base}))
        self.write_nb()
        self.repo_patch = patch.object(V, "REPO", self.root)
        self.repo_patch.start()

    def tearDown(self):
        self.repo_patch.stop()
        self.temp.cleanup()

    def write_nb(self):
        self.root.joinpath("colab/Lab22_DPO_T4_Core_executed.ipynb").write_text(json.dumps(self.nb))

    def test_valid_export_without_weights(self):
        self.assertTrue(V.colab_reference_evidence(self.base))
        problems = []
        V.check_merged_sft(problems)
        self.assertEqual(problems, [])

    def test_base_model_or_different_colab_path_rejected(self):
        self.assertFalse(V.colab_reference_evidence("unsloth/Qwen3-4B"))
        self.assertFalse(V.colab_reference_evidence("/content/other/models/sft-merged"))

    def test_unexecuted_or_failed_merge_rejected(self):
        original = copy.deepcopy(self.nb)
        self.nb["cells"][0]["execution_count"] = None
        self.write_nb()
        self.assertFalse(V.colab_reference_evidence(self.base))
        self.nb = original
        self.nb["cells"][0]["outputs"].append({"output_type": "error", "ename": "RuntimeError"})
        self.write_nb()
        self.assertFalse(V.colab_reference_evidence(self.base))

    def test_changed_metrics_rejected(self):
        changed = {**self.metrics, "eval_reward_accuracy": 0.99}
        self.root.joinpath("adapters/dpo/dpo_metrics.json").write_text(json.dumps(changed))
        self.assertFalse(V.colab_reference_evidence(self.base))

    def test_missing_or_corrupt_notebook_rejected(self):
        notebook = self.root.joinpath("colab/Lab22_DPO_T4_Core_executed.ipynb")
        notebook.write_text("{bad")
        self.assertFalse(V.colab_reference_evidence(self.base))
        notebook.unlink()
        problems = []
        V.check_merged_sft(problems)
        self.assertEqual(len(problems), 1)


if __name__ == "__main__":
    unittest.main()
