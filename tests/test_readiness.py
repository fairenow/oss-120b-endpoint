"""CPU-only checks; no GPU allocation or Modal authentication required."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "modal_models.py").read_text()
TREE = ast.parse(SOURCE)


class ServingConfigurationTests(unittest.TestCase):
    def test_python_syntax(self):
        compile(SOURCE, "modal_models.py", "exec")

    def test_required_functions_deployed(self):
        names = {node.name for node in TREE.body if isinstance(node, ast.FunctionDef)}
        classes = {node.name for node in TREE.body if isinstance(node, ast.ClassDef)}
        self.assertTrue({"gpt_oss_120b_vllm", "qwen36_35b_a3b"}.issubset(names))
        self.assertIn("QwenImage2512", classes)

    def test_deprecated_concurrency_removed(self):
        self.assertNotIn("allow_concurrent_inputs", SOURCE)
        self.assertNotIn("VLLM_USE_V1", SOURCE)
        self.assertNotIn("HF_HUB_ENABLE_HF_TRANSFER", SOURCE)

    def test_model_parsers_preserved(self):
        for marker in ("openai_gptoss", "qwen3_coder", "qwen3"):
            self.assertIn(marker, SOURCE)

    def test_image_health_and_validation(self):
        self.assertIn('api.get("/health")', SOURCE)
        self.assertIn("generation_lock.acquire(blocking=False)", SOURCE)
        self.assertIn("Image model returned no image", SOURCE)


class WarmthControllerTests(unittest.TestCase):
    def test_dry_run_does_not_import_modal(self):
        from warmth_controller import apply_profile
        actions = apply_profile("active", apply=False)
        self.assertEqual(len(actions), 2)
        self.assertTrue(all(x["min_containers"] == 1 for x in actions))
        self.assertEqual(
            [x["min_containers"] for x in apply_profile("quiet")], [0, 0]
        )

    def test_unknown_profile_rejected(self):
        from warmth_controller import apply_profile
        with self.assertRaises(ValueError):
            apply_profile("unknown")


if __name__ == "__main__":
    unittest.main()
