import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_joint_mask', ROOT / 'feeders/joint_mask.py')
mask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask_module)


class TrainingMaskTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
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

    def test_tensor_and_numpy_share_masks_and_preserve_dtype_device_and_source(self):
        data = torch.ones(8, 3, 25, 2, dtype=torch.float64).permute(1, 0, 2, 3)
        source = data.clone()
        np.random.seed(17)
        actual = mask_module.mask_training_joints(data, 1, 5)
        np.random.seed(17)
        expected = mask_module.mask_training_joints(data.numpy(), 1, 5)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.dtype, data.dtype)
        self.assertEqual(actual.device, data.device)
        torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0, atol=0)
        torch.testing.assert_close(data, source, rtol=0, atol=0)
        self.assertNotEqual(actual.data_ptr(), data.data_ptr())

    def test_disabled_tensor_mask_is_identity_and_consumes_no_random_numbers(self):
        data = torch.from_numpy(self.data)
        state = np.random.get_state()
        self.assertIs(mask_module.mask_training_joints(data, 0), data)
        actual = np.random.random(10)
        np.random.set_state(state)
        np.testing.assert_array_equal(actual, np.random.random(10))

    def feeder_class(self):
        # Run the actual crop/resize and rotation code. Stub only plotting,
        # which these preprocessing functions do not use, and skip UCLA imports.
        package = types.ModuleType('feeders')
        package.__path__ = [str(ROOT / 'feeders')]
        tools_spec = importlib.util.spec_from_file_location('ntu_tools_under_test', ROOT / 'feeders/tools.py')
        tools = importlib.util.module_from_spec(tools_spec)
        plotting = types.ModuleType('matplotlib')
        plotting.pyplot = types.ModuleType('matplotlib.pyplot')
        with patch.dict(sys.modules, {'matplotlib': plotting, 'matplotlib.pyplot': plotting.pyplot}):
            tools_spec.loader.exec_module(tools)
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

    def test_real_rotation_then_masking_and_dataloader_collation(self):
        Feeder = self.feeder_class()
        data = np.ones((2, 8, 150), dtype=np.float32)
        labels = np.eye(60, dtype=np.float32)[:2]
        with patch.object(np, 'load', return_value=dict(x_train=data, y_train=labels)):
            dataset = Feeder('unused', split='train', random_rot=True, joint_mask_prob=1,
                             joint_mask_max=5, window_size=8, p_interval=[1])
        batch, batch_labels, indices = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(batch.shape), (2, 3, 8, 25, 2))
        self.assertEqual(batch.dtype, torch.float32)
        self.assertTrue(torch.isfinite(batch).all())
        torch.testing.assert_close(batch_labels, torch.tensor([0, 1]))
        torch.testing.assert_close(indices, torch.tensor([0, 1]))
        for sample in batch:
            missing = (sample == 0).all(dim=0).all(dim=0).all(dim=-1).sum().item()
            self.assertTrue(1 <= missing <= 5)
        np.testing.assert_array_equal(data, np.ones_like(data))


if __name__ == '__main__':
    unittest.main()
