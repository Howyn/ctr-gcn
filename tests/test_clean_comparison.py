from copy import deepcopy
from pathlib import Path
import unittest

import yaml

from tools.check_clean_comparison import check_comparison
from tools.evaluate_occlusion import matching_test_settings, matching_training_inputs

ROOT = Path(__file__).resolve().parents[1] / 'config/nturgbd-cross-subject'


class CleanComparisonTests(unittest.TestCase):
    def setUp(self):
        self.original = yaml.safe_load((ROOT / 'ctrgcn_mamba_fast128.yaml').read_text())
        self.clean = yaml.safe_load((ROOT / 'ctrgcn_mamba_robust_v2_clean_fast128.yaml').read_text())
        self.augmented = yaml.safe_load((ROOT / 'ctrgcn_mamba_robust_v2_fast128.yaml').read_text())

    def test_new_config_changes_only_architecture_output_and_explicit_disabled_mask_defaults(self):
        expected = deepcopy(self.original)
        expected['work_dir'] = self.clean['work_dir']
        expected['model'] = 'model.ctrgcn_mamba_robust_v2.Model'
        expected['model_args'].update(use_reliability=True, use_local_compensation=True, history_window=3)
        expected['train_feeder_args'].update(joint_mask_prob=0.0, joint_mask_max=5)
        self.assertEqual(expected, self.clean)
        check_comparison(self.original, self.clean)
        pair = dict(baseline=self.original, mamba=self.clean)
        self.assertTrue(matching_training_inputs(pair))
        matching_test_settings(pair)

    def test_v2_algorithm_and_every_other_augmented_run_setting_are_unchanged(self):
        expected = deepcopy(self.augmented)
        expected['work_dir'] = self.clean['work_dir']
        expected['train_feeder_args']['joint_mask_prob'] = 0.0
        self.assertEqual(expected, self.clean)

    def test_joint_masking_is_rejected_on_either_side(self):
        for reference_mask, candidate_mask in ((0.25, 0), (0, 0.25)):
            reference, candidate = deepcopy(self.original), deepcopy(self.clean)
            reference['train_feeder_args']['joint_mask_prob'] = reference_mask
            candidate['train_feeder_args']['joint_mask_prob'] = candidate_mask
            with self.assertRaisesRegex(ValueError, 'joint masking must be disabled'):
                check_comparison(reference, candidate)

    def test_changed_optimizer_recipe_or_input_split_are_rejected(self):
        for key, value in (('batch_size', 64), ('seed', 2), ('weight_decay', 0.0),
                           ('respect_no_weight_decay', True)):
            candidate = deepcopy(self.clean)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Training setting differs'):
                check_comparison(self.original, candidate)
        candidate = deepcopy(self.clean)
        candidate['train_feeder_args']['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Incorrect split'):
            check_comparison(self.original, candidate)

    def test_warm_start_or_overwriting_the_original_run_are_rejected(self):
        candidate = deepcopy(self.clean)
        candidate['weights'] = 'old_weights.pt'
        with self.assertRaisesRegex(ValueError, 'from scratch'):
            check_comparison(self.original, candidate)
        candidate = deepcopy(self.clean)
        candidate['work_dir'] = self.original['work_dir']
        with self.assertRaisesRegex(ValueError, 'separate work directory'):
            check_comparison(self.original, candidate)


if __name__ == '__main__':
    unittest.main()
