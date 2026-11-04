"""设备选择失败时必须能回退，用户强制的后端不能悄悄变成其他后端。"""
import json
import os
import subprocess
import unittest
from unittest.mock import patch

from run_engine import engine_paths, select_device, build_command, engine_env, resolve_precision


class DeviceSelectionTests(unittest.TestCase):
    def probe(self, data, returncode=0):
        return subprocess.CompletedProcess([], returncode, json.dumps(data), "HIP kernel failed")

    @patch("run_engine.subprocess.run")
    def test_cpu_does_not_initialize_gpu(self, run):
        self.assertEqual(select_device("python", "cpu")[0], "cpu")
        run.assert_not_called()

    @patch("run_engine.subprocess.run")
    def test_rocm_uses_torch_cuda_device(self, run):
        run.return_value = self.probe({"available": True, "hip": "10.0", "name": "780M"})
        self.assertEqual(select_device("python", "rocm")[0], "cuda")

    @patch("run_engine.subprocess.run")
    def test_explicit_rocm_rejects_nvidia_build(self, run):
        run.return_value = self.probe({"available": True, "hip": None})
        with self.assertRaisesRegex(RuntimeError, "非 ROCm"):
            select_device("python", "rocm")

    @patch("run_engine.subprocess.run")
    def test_auto_falls_back_after_runtime_crash(self, run):
        run.return_value = self.probe({}, returncode=-11)
        device, info = select_device("python", "auto")
        self.assertEqual(device, "cpu")
        self.assertIn("HIP kernel failed", info["reason"])
        with self.assertRaises(RuntimeError):
            select_device("python", "rocm")

    @patch("run_engine.subprocess.run")
    def test_auto_falls_back_after_hang_or_corrupt_output(self, run):
        for result in (subprocess.TimeoutExpired("probe", 90),
                       subprocess.CompletedProcess([], 0, "unexpected output", "")):
            with self.subTest(result=result):
                run.side_effect = result if isinstance(result, Exception) else None
                run.return_value = result
                self.assertEqual(select_device("python", "auto")[0], "cpu")

    def test_model_paths_with_spaces_remain_single_arguments(self):
        with patch.dict(os.environ, {}, clear=True):
            paths = engine_paths("/tmp/mambo test")
        cmd = build_command(paths, "127.0.0.1", 9880, "cpu", "full")
        self.assertEqual(cmd[cmd.index("-s") + 1], paths["sovits"])
        self.assertIn("-fp", cmd)
        self.assertNotIn("-hp", cmd)

    def test_thread_configuration_preserves_user_choice(self):
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "4", "MIOPEN_FIND_MODE": "HYBRID"}, clear=True):
            self.assertEqual(engine_env()["OMP_NUM_THREADS"], "4")
            self.assertEqual(engine_env(2)["OMP_NUM_THREADS"], "2")
            self.assertEqual(engine_env()["MIOPEN_FIND_MODE"], "HYBRID")

    def test_precision_selection(self):
        self.assertEqual(resolve_precision("auto", "cpu", {}), "full")
        self.assertEqual(resolve_precision("auto", "cuda", {"hip": "10", "arch": "gfx1103"}), "half")
        self.assertEqual(resolve_precision("auto", "cuda", {"hip": "10", "arch": "gfx1151"}), "full")
        self.assertEqual(resolve_precision("full", "cuda", {"hip": "10", "arch": "gfx1103"}), "full")
        self.assertEqual(resolve_precision("auto", "cuda", {"hip": None}), "auto")


if __name__ == "__main__":
    unittest.main()
