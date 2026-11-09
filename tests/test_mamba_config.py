from copy import deepcopy
from pathlib import Path
import unittest

import yaml


class ExperimentConfigTests(unittest.TestCase):
    def test_only_model_and_output_directory_change_from_completed_baseline(self):
        root = Path(__file__).resolve().parents[1] / 'config' / 'nturgbd-cross-subject'
        baseline = yaml.safe_load((root / 'fast128.yaml').read_text(encoding='utf-8'))
        hybrid = yaml.safe_load((root / 'ctrgcn_mamba_fast128.yaml').read_text(encoding='utf-8'))
        expected = deepcopy(baseline)
        expected['work_dir'] = './work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128'
        expected['model'] = 'model.ctrgcn_mamba.Model'
        expected['model_args'].update(mamba_d_state=16, mamba_d_conv=4, mamba_expand=2)
        self.assertEqual(hybrid, expected)

    def test_calibration_changes_only_no_decay_handling_and_output_directory(self):
        root = Path(__file__).resolve().parents[1] / 'config' / 'nturgbd-cross-subject'
        original = yaml.safe_load((root / 'ctrgcn_mamba_fast128.yaml').read_text(encoding='utf-8'))
        calibrated = yaml.safe_load((root / 'ctrgcn_mamba_calibrated_fast128.yaml').read_text(encoding='utf-8'))
        expected = deepcopy(original)
        expected['respect_no_weight_decay'] = True
        expected['work_dir'] = './work_dir/ntu60/xsub/ctrgcn_mamba_calibrated_joint_fast128'
        self.assertEqual(calibrated, expected)


if __name__ == '__main__':
    unittest.main()
