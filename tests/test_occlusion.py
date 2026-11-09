import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn


spec = importlib.util.spec_from_file_location(
    'evaluate_occlusion', Path(__file__).resolve().parents[1] / 'tools' / 'evaluate_occlusion.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class MaskTests(unittest.TestCase):
    def test_twenty_percent_is_exactly_five_distinct_joints_per_sequence(self):
        masks = evaluation.make_joint_masks(100)
        self.assertEqual(masks.shape, (100, 5))
        self.assertTrue(np.all((masks >= 0) & (masks < 25)))
        self.assertTrue(np.all(np.diff(masks.astype(int), axis=1) > 0))
        self.assertGreater(len(np.unique(masks, axis=0)), 1)

    def test_mask_reproducibility_and_seed_sensitivity(self):
        first = evaluation.make_joint_masks(100, seed=2026)
        np.testing.assert_array_equal(first, evaluation.make_joint_masks(100, seed=2026))
        self.assertFalse(np.array_equal(first, evaluation.make_joint_masks(100, seed=2027)))

    def test_mask_zeroes_only_selected_joints_for_every_time_channel_and_person(self):
        data = torch.arange(2 * 3 * 4 * 25 * 2, dtype=torch.float32).reshape(2, 3, 4, 25, 2) + 1
        original = data.clone()
        masks = evaluation.make_joint_masks(2)
        result = evaluation.mask_inputs(data, torch.tensor(masks.astype(np.int64)))
        for index in range(2):
            missing = masks[index].astype(int)
            kept = sorted(set(range(25)) - set(missing))
            self.assertEqual(torch.count_nonzero(result[index, :, :, missing, :]).item(), 0)
            torch.testing.assert_close(result[index, :, :, kept, :], data[index, :, :, kept, :])
        torch.testing.assert_close(data, original)

    def test_zero_occlusion_is_identity_and_nonintegral_rate_is_rejected(self):
        masks = evaluation.make_joint_masks(2, rate=0)
        data = torch.randn(2, 3, 4, 25, 2)
        torch.testing.assert_close(evaluation.mask_inputs(data, torch.tensor(masks.astype(np.int64))), data)
        with self.assertRaises(ValueError):
            evaluation.make_joint_masks(2, rate=0.1)


class CheckpointTests(unittest.TestCase):
    def test_best_epoch_selection_does_not_use_last_epoch_or_a_prefix_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'log.txt').write_text('Best accuracy: 0.8964\nEpoch number: 60\n')
            for name in ('runs-6-1878.pt', 'runs-60-18780.pt', 'runs-65-20345.pt'):
                (root / name).write_bytes(b'checkpoint')
            path, epoch, reference = evaluation.best_checkpoint(root)
            self.assertEqual(path.name, 'runs-60-18780.pt')
            self.assertEqual(epoch, 60)
            self.assertAlmostEqual(reference, 89.64)

    def test_best_pt_is_supported_when_numbered_checkpoints_were_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'log.txt').write_text('Best accuracy: 0.8891\nEpoch number: 38\n')
            (root / 'best.pt').write_bytes(b'best checkpoint')
            self.assertEqual(evaluation.best_checkpoint(root)[0].name, 'best.pt')

    def test_explicit_nonbest_epoch_is_not_reported_as_best_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'log.txt').write_text('Best accuracy: 0.8891\nEpoch number: 38\n')
            path = root / 'runs-65-20345.pt'
            path.write_bytes(b'last checkpoint')
            _, epoch, reference = evaluation.best_checkpoint(root, path)
            self.assertEqual(epoch, 65)
            self.assertIsNone(reference)


class PairedInferenceTests(unittest.TestCase):
    def test_metrics_match_known_top1_and_top5_results(self):
        scores = np.full((4, 60), -100, dtype=np.float32)
        labels = np.arange(4)
        scores[np.arange(4), labels] = 1
        scores[0, 4] = 2
        result = evaluation.metrics(scores, labels)
        self.assertEqual(result['top1_percent'], 75)
        self.assertEqual(result['top5_percent'], 100)

    def test_shared_masks_index_alignment_and_no_model_updates(self):
        class RecordingClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(3)
                self.fc = nn.Linear(3, 60)
                self.inputs = []

            def forward(self, data):
                self.inputs.append(data.clone())
                return self.fc(self.bn(data.mean(dim=(2, 3, 4))))

        torch.set_num_threads(1)
        torch.manual_seed(1)
        models = {'baseline': RecordingClassifier(), 'mamba': RecordingClassifier()}
        models['mamba'].load_state_dict(models['baseline'].state_dict())
        snapshots = {name: {key: value.clone() for key, value in model.state_dict().items()}
                     for name, model in models.items()}
        data = torch.randn(4, 3, 4, 25, 2)
        labels = np.arange(4)
        order = [2, 0, 3, 1]
        loader = torch.utils.data.DataLoader([(data[index], int(labels[index]), index) for index in order],
                                             batch_size=2)
        masks = evaluation.make_joint_masks(4)
        scores = evaluation.evaluate_pair(models, loader, masks, labels, torch.device('cpu'), False)
        for kind in ('clean', 'occluded'):
            np.testing.assert_array_equal(scores['baseline'][kind], scores['mamba'][kind])
        for batch, indices in enumerate(([2, 0], [3, 1])):
            expected = evaluation.mask_inputs(data[indices], torch.tensor(masks[indices].astype(np.int64)))
            torch.testing.assert_close(models['baseline'].inputs[batch * 2 + 1], expected)
            torch.testing.assert_close(models['mamba'].inputs[batch * 2 + 1], expected)
        for name, model in models.items():
            self.assertFalse(model.training)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, snapshots[name][key], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
