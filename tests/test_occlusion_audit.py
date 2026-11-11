import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tools import audit_occlusion as audit
from model.ctrgcn_mamba_robust_v2 import Model
from test_robust_mamba import cpu_official_operators


class MaskAuditTests(unittest.TestCase):
    def test_independent_mask_rejects_clean_input_and_wrong_axis(self):
        data = torch.ones(2, 3, 8, 25, 2)
        joints = np.asarray([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])
        original = data.clone()
        expected = audit.independent_mask(data, joints)
        report = audit.check_mask(data, audit.evaluation.mask_inputs(data, torch.tensor(joints)), joints)
        self.assertEqual(report['removed_fraction_of_nonzero_coordinates'], 0.2)
        self.assertTrue(torch.equal(data, original))
        self.assertEqual(torch.count_nonzero(expected).item(), 2 * 3 * 8 * 20 * 2)
        with self.assertRaisesRegex(AssertionError, 'wrong axes'):
            audit.check_mask(data, data, joints)
        wrong = data.clone()
        wrong[:, :, :5] = 0  # Frames rather than joints.
        with self.assertRaisesRegex(AssertionError, 'wrong axes'):
            audit.check_mask(data, wrong, joints)


class LiveAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.options = dict(num_class=60, num_point=25, num_person=2,
                            graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
        self.data = torch.randn(2, 3, 8, 25, 2)
        self.masked = audit.independent_mask(self.data, np.tile(np.arange(5), (2, 1)))

    def test_actual_v2_routes_only_current_masked_features_and_does_not_cache_clean_pass(self):
        with cpu_official_operators():
            model = Model(**self.options).eval()
            snapshot = deepcopy(model.state_dict())
            first = audit.checked_forward(model, self.masked)
            audit.checked_forward(model, self.data)
            after = audit.checked_forward(model, self.masked)
            torch.testing.assert_close(first, after, rtol=0, atol=0)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, snapshot[key]))
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_backbone_hook_detects_replacement_with_cached_clean_coordinates(self):
        class LeakingModel(Model):
            def forward(inner, x):
                return super(LeakingModel, inner).forward(self.data)
        with cpu_official_operators():
            model = LeakingModel(**self.options).eval()
            with self.assertRaisesRegex(AssertionError, 'Backbone received coordinates'):
                audit.checked_forward(model, self.masked)
            self.assertEqual(len(model._forward_pre_hooks), 0)
            self.assertEqual(len(model.data_bn._forward_pre_hooks), 0)

    def test_local_hook_detects_substituted_features(self):
        with cpu_official_operators():
            model = Model(**self.options).eval()
            substitute = model.temporal_local.register_forward_pre_hook(lambda module, args: (args[0] + 1,))
            try:
                with self.assertRaisesRegex(AssertionError, 'Local branch bypassed'):
                    audit.checked_forward(model, self.masked)
            finally:
                substitute.remove()

    def test_training_mode_is_rejected_before_batchnorm_updates(self):
        with cpu_official_operators():
            model = Model(**self.options).train()
            with self.assertRaisesRegex(AssertionError, 'training-mode'):
                audit.checked_forward(model, self.masked)

    def test_reliability_hook_rejects_clean_missingness_for_occluded_input(self):
        with cpu_official_operators():
            model = Model(**self.options).eval()
            substitute = model.temporal_reliability.register_forward_pre_hook(
                lambda module, args: (args[0], torch.ones_like(args[1])))
            try:
                with self.assertRaisesRegex(AssertionError, 'missingness from another input'):
                    audit.checked_forward(model, self.masked)
            finally:
                substitute.remove()


class ArtifactAuditTests(unittest.TestCase):
    def test_artifacts_recompute_accuracy_and_reject_inflated_summary(self):
        count = 16487
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.arange(count) % 60
            data_path = (root / 'data.npz').resolve()
            np.savez(data_path, y_test=np.eye(60, dtype=np.float32)[labels])
            # Shape headers are mocked to avoid allocating the 10 GB dataset;
            # the actual labels, scores, masks, configs, hashes and CSV are read.
            joints = audit.evaluation.make_joint_masks(count)
            np.savez(root / 'masks.npz', sample_indices=np.arange(count), labels=labels,
                     masked_joint_indices=joints, seed=2026, rate=0.2)
            feeder = dict(data_path=str(data_path), split='test', window_size=64, p_interval=[0.95])
            summary = dict(samples=count, missing_joints=5, rate=0.2, mask_seed=2026,
                protocol='per_sample_fixed_joints_zero_after_test_crop_resize',
                mask_sha256=hashlib.sha256(joints.tobytes()).hexdigest(), checkpoints={},
                test_feeder_args=feeder, results=[])
            scores, predictions = {}, {}
            for name in ('baseline', 'mamba'):
                work = root / name
                work.mkdir()
                model_args = dict(num_class=60, num_point=25, num_person=2)
                config = dict(model='model.ctrgcn_mamba.Model', model_args=model_args,
                    feeder='feeders.feeder_ntu.Feeder', test_feeder_args=feeder,
                    train_feeder_args=dict(data_path=str(data_path), split='train'))
                (work / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
                (work / 'best.pt').write_bytes(name.encode())
                summary['checkpoints'][name] = dict(saved_config=str(work / 'config.yaml'),
                    checkpoint=str(work / 'best.pt'), checkpoint_sha256=audit.evaluation.file_digest(work / 'best.pt'),
                    model=config['model'], model_args=model_args)
                for kind in ('clean', 'occluded'):
                    key = name + '_' + kind
                    predictions[key] = (labels + int(name == 'baseline' and kind == 'occluded')) % 60
                    scores[key] = np.zeros((count, 60), dtype=np.float32)
                    scores[key][np.arange(count), predictions[key]] = 1
                occluded_accuracy = 0.0 if name == 'baseline' else 100.0
                summary['results'].append(dict(model=name, clean_top1_percent=100.0,
                    occluded_top1_percent=occluded_accuracy, drop_pp=100 - occluded_accuracy))
            np.savez(root / 'scores.npz', labels=labels, **scores)
            with (root / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['sample_index', 'label', 'masked_joints_1based'] + list(scores))
                for index in range(count):
                    writer.writerow([index, labels[index], ' '.join(map(str, joints[index].astype(int) + 1))]
                                    + [predictions[key][index] for key in scores])
            (root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            shapes = lambda path, key: (16487, 300, 150) if key == 'x_test' else (40091, 300, 150)
            with patch.object(audit, 'npz_shape', side_effect=shapes):
                result = audit.audit_artifacts(root)[-1]
                self.assertEqual(result['metrics']['baseline_occluded']['top1_percent'], 0)
                self.assertEqual(result['metrics']['mamba_occluded']['top1_percent'], 100)
                summary['results'][0]['occluded_top1_percent'] = 99
                (root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
                with self.assertRaisesRegex(AssertionError, 'disagrees with saved scores'):
                    audit.audit_artifacts(root)

    def test_npz_header_reads_shape_without_loading_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.npz'
            np.savez(path, x_test=np.ones((3, 4, 5), dtype=np.float32))
            self.assertEqual(audit.npz_shape(path, 'x_test'), (3, 4, 5))


if __name__ == '__main__':
    unittest.main()
