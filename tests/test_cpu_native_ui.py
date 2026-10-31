"""Source-shape regression tests of the genuine WanGP CPU UI dispatch.

These run with stdlib only on GitHub CI; they do not claim Gradio E2.
"""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class NativeCpuBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wgp = (ROOT / "wgp.py").read_text(encoding="utf-8")
        cls.attention = (ROOT / "shared" / "attention.py").read_text(encoding="utf-8")
        cls.plugin = (ROOT / "plugins" / "moss_tts_cpu" / "plugin.py").read_text(encoding="utf-8")

    def test_opt_in_cpu_native_and_previous_cpu_modes_coexist(self):
        self.assertIn('"--native-ui-cpu"', self.wgp)
        self.assertIn('"--cpu-only"', self.wgp)
        self.assertIn('os.environ["WANGP_NATIVE_UI_CPU"]', self.wgp)

    def test_gpu_capability_not_queried_for_native_cpu(self):
        self.assertIn("if (NATIVE_UI_CPU or not torch.cuda.is_available()) else", self.wgp)
        self.assertIn("_cpu_ui = os.environ.get", self.attention)
        self.assertIn("if (_is_mps or _cpu_ui or not torch.cuda.is_available())", self.attention)

    def test_native_model_handler_import_is_restricted(self):
        self.assertIn('family_handlers = ["models.TTS.moss_tts.moss_tts_handler"]', self.wgp)

    def test_default_generator_is_fail_closed(self):
        root = ast.parse(self.wgp)
        method = next(obj for obj in root.body if isinstance(obj, ast.FunctionDef) and obj.name=="generate_media")
        self.assertTrue(any(isinstance(n, ast.If) and
                            isinstance(n.test, ast.Name) and n.test.id == "NATIVE_UI_CPU"
                            for n in method.body[:5]))

    def test_plugin_uses_native_unwrapped_button(self):
        tree = ast.parse(self.plugin)
        plugin = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "MossTTSCpuPlugin")
        create = next(x for x in plugin.body if isinstance(x, ast.FunctionDef) and x.name == "create_ui")
        self.assertEqual([x.arg for x in create.args.args], ["self"])
        self.assertIn("tempfile.gettempdir()", self.plugin)
        self.assertIn("shutil.copyfileobj", self.plugin)

    def test_original_app_ui_constructs(self):
        self.assertIn('with ui_studio.Blocks(', self.wgp)
        self.assertIn('with gr.Tab("Media Generator"', self.wgp)
        self.assertIn('app.initialize_plugins(globals())', self.wgp)


if __name__ == "__main__":
    unittest.main()
