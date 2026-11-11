import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_joint_mask', ROOT / 'feeders/joint_mask.py')
mask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask_module)


class TrainingMaskTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.data = np.ones((3, 8, 25, 2), dtype=np.float32)

    def test_disabled_masking_is_identity_and_consumes_no_random_numbers(self):
        state = np.random.get_state()
        result = mask_module.mask_training_joints(self.data)
        self.assertIs(result, self.data)
        actual = np.random.random(10)
        np.random.set_state(state)
        np.testing.assert_array_equal(actual, np.random.random(10))

    def test_mask_is_fixed_across_frames_coordinates_and_people_without_mutating_source(self):
        result = mask_module.mask_training_joints(self.data, probability=1, max_joints=5)
        missing = np.flatnonzero(np.all(result == 0, axis=(0, 1, 3)))
        self.assertTrue(1 <= len(missing) <= 5)
        kept = np.setdiff1d(np.arange(25), missing)
        self.assertTrue(np.all(result[:, :, missing, :] == 0))
        np.testing.assert_array_equal(result[:, :, kept, :], self.data[:, :, kept, :])
        np.testing.assert_array_equal(self.data, np.ones_like(self.data))

    def test_quarter_probability_keeps_most_samples_clean_and_uses_varied_joint_counts(self):
        counts = [np.count_nonzero(np.all(mask_module.mask_training_joints(self.data, 0.25, 5) == 0,
                                          axis=(0, 1, 3))) for _ in range(1000)]
        self.assertTrue(0.70 < np.mean(np.asarray(counts) == 0) < 0.80)
        self.assertEqual(set(counts), {0, 1, 2, 3, 4, 5})

    def feeder_class(self):
        # Isolate feeder integration from optional plotting/UCLA dependencies.
        package = types.ModuleType('feeders')
        package.__path__ = [str(ROOT / 'feeders')]
        tools = types.ModuleType('feeders.tools')
        tools.valid_crop_resize = lambda data, *args: data.copy()
        tools.random_rot = lambda data: data
        package.tools = tools
        spec = importlib.util.spec_from_file_location('ntu_feeder_under_test', ROOT / 'feeders/feeder_ntu.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'feeders': package, 'feeders.tools': tools,
                                     'feeders.joint_mask': mask_module}):
            spec.loader.exec_module(module)
        return module.Feeder

    def test_feeder_applies_training_mask_and_preserves_test_samples(self):
        Feeder = self.feeder_class()
        data = np.ones((1, 8, 150), dtype=np.float32)
        labels = np.eye(60, dtype=np.float32)[:1]
        archive = dict(x_train=data, x_test=data.copy(), y_train=labels, y_test=labels.copy())
        with patch.object(np, 'load', return_value=archive):
            train = Feeder('unused', split='train', joint_mask_prob=1, joint_mask_max=5,
                           window_size=8, p_interval=[1])
            test = Feeder('unused', split='test', window_size=8, p_interval=[1])
        masked, label, index = train[0]
        clean, _, _ = test[0]
        self.assertEqual((label, index), (0, 0))
        self.assertTrue(np.any(masked == 0))
        self.assertTrue(np.all(clean == 1))
        self.assertTrue(np.all(train.data == 1))

    def test_masking_test_split_or_other_modalities_is_rejected_before_loading(self):
        Feeder = self.feeder_class()
        for args in (dict(split='test'), dict(split='train', bone=True), dict(split='train', vel=True)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'only for Joint training'):
                Feeder('not-a-real-dataset', joint_mask_prob=0.25, **args)


if __name__ == '__main__':
    unittest.main()
