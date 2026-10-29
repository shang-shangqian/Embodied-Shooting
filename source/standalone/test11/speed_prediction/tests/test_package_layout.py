from importlib import import_module, util
from pathlib import Path
import pickle
import subprocess
import sys
import unittest


TEST11_DIR = Path(__file__).resolve().parents[2]
SPEED_PREDICTION_DIR = TEST11_DIR / "speed_prediction"
for directory in (TEST11_DIR, SPEED_PREDICTION_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


MODULES = {
    "speed_prediction_model": "legacy_gru.speed_prediction_model",
    "train_gru_velocity": "legacy_gru.train_gru_velocity",
    "test_gru_velocity": "legacy_gru.test_gru_velocity",
    "time_aware_velocity_model": "taq_gru.time_aware_velocity_model",
    "online_taq_velocity": "taq_gru.online_taq_velocity",
    "taq_velocity_training": "taq_gru.taq_velocity_training",
    "train_taq_velocity": "taq_gru.train_taq_velocity",
    "train_stage2_balanced": "taq_gru.train_stage2_balanced",
    "evaluate_taq_velocity": "taq_gru.evaluate_taq_velocity",
    "evaluate_taq_velocity_streaming": "taq_gru.evaluate_taq_velocity_streaming",
    "time_aware_velocity_data": "common.time_aware_velocity_data",
    "stage2_degradation": "common.stage2_degradation",
    "sensor_time_contract": "common.sensor_time_contract",
    "filter_state_output": "common.filter_state_output",
    "audit_stage1_ideal_v2": "data_tools.audit_stage1_ideal_v2",
    "audit_stage2_data": "data_tools.audit_stage2_data",
}
CLI_MODULES = (
    "train_gru_velocity",
    "test_gru_velocity",
    "train_taq_velocity",
    "train_stage2_balanced",
    "evaluate_taq_velocity",
    "evaluate_taq_velocity_streaming",
    "audit_stage1_ideal_v2",
    "audit_stage2_data",
)


class PackageLayoutTests(unittest.TestCase):
    def test_historical_imports_alias_the_organized_implementations(self):
        for old_name, target in MODULES.items():
            with self.subTest(module=old_name):
                implementation = import_module(f"speed_prediction.{target}")
                self.assertIs(import_module(old_name), implementation)
                self.assertIs(import_module(f"speed_prediction.{old_name}"), implementation)

    def test_private_helpers_remain_available(self):
        old_module = import_module("evaluate_taq_velocity_streaming")
        new_module = import_module("speed_prediction.taq_gru.evaluate_taq_velocity_streaming")
        self.assertIs(old_module._stream_tensors, new_module._stream_tensors)

    def test_model_definition_supports_loading_by_file_path(self):
        module_name = "velocity_layout_model_definition"
        spec = util.spec_from_file_location(module_name, SPEED_PREDICTION_DIR / "speed_prediction_model.py")
        module = util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            implementation = import_module("speed_prediction.legacy_gru.speed_prediction_model")
            self.assertIs(module.VelocityPredictorGRU, implementation.VelocityPredictorGRU)
        finally:
            sys.modules.pop(module_name, None)

    def test_historical_pickle_class_paths_still_resolve(self):
        model_class = pickle.loads(b"cspeed_prediction_model\nVelocityPredictorGRU\n.")
        implementation = import_module("speed_prediction.legacy_gru.speed_prediction_model")
        self.assertIs(model_class, implementation.VelocityPredictorGRU)

    def test_old_and_new_cli_paths_keep_the_same_options(self):
        for old_name in CLI_MODULES:
            target = MODULES[old_name].replace(".", "/") + ".py"
            outputs = []
            for script in (SPEED_PREDICTION_DIR / f"{old_name}.py", SPEED_PREDICTION_DIR / target):
                with self.subTest(script=str(script)):
                    result = subprocess.run(
                        [sys.executable, str(script), "--help"],
                        cwd=TEST11_DIR.parents[2],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    outputs.append(result.stdout)
            self.assertEqual(outputs[0], outputs[1])
