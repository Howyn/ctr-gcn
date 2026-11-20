"""Weights-only recovery selection and verified shutdown gates (CPU only)."""

import csv
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import torch
import yaml

from tools import resume_v4_autodl as recovery
from test_v4_pipeline import V4Fixture


def checkpoint(path, finite=True):
    tensor = torch.ones(60, 256)
    if not finite:
        tensor[0, 0] = float('nan')
    torch.save({'fc.weight': tensor}, path)


def confusion(path, correct):
    matrix = [[0] * 60 for _ in range(60)]
    matrix[0][0] = correct
    matrix[0][1] = 16487 - correct
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([0] * 60)
        writer.writerows(matrix)


class RecoveryFixture(V4Fixture):
    def setUp(self):
        super().setUp()
        self.source = self.plan['job']['work']
        self.source.mkdir(parents=True)
        (self.source / 'config.yaml').write_text(yaml.safe_dump(self.plan['job']['config']), encoding='utf-8')
        shutil.copy2(self.plan['algorithm_source'], self.source / self.plan['algorithm_source'].name)
        checkpoint(self.source / 'runs-42-13146.pt')
        checkpoint(self.source / 'runs-55-17215.pt')
        checkpoint(self.source / 'runs-56-17528.pt')
        confusion(self.source / 'epoch42_test_each_class_acc.csv', 14560)
        confusion(self.source / 'epoch55_test_each_class_acc.csv', 14448)
        self.plan = recovery.build_resume_plan(self.repository)

    def fake_recovery_training(self, *args):
        super().fake_training(*args)
        saved = dict(self.plan['job']['config'], phase='train', seed=1, device=[0])
        (self.plan['job']['work'] / 'config.yaml').write_text(yaml.safe_dump(saved), encoding='utf-8')

    def execute_recovery(self, **kwargs):
        return recovery.execute_resume(self.plan, repository=self.repository, **kwargs)


class SelectionTests(RecoveryFixture):
    def test_latest_trained_epoch_can_precede_its_missing_evaluation(self):
        latest, previous, rejected = recovery.select_checkpoints(self.source)
        self.assertEqual(latest['epoch'], 56)
        self.assertEqual(previous['epoch'], 42)
        self.assertEqual(previous['accuracy'], 14560 / 16487)
        self.assertEqual(rejected, [])

    def test_corrupt_latest_checkpoint_is_skipped_without_deleting_it(self):
        path = self.source / 'runs-57-17841.pt'
        path.write_bytes(b'incomplete zip checkpoint')
        latest, _, rejected = recovery.select_checkpoints(self.source)
        self.assertEqual(latest['epoch'], 56)
        self.assertEqual(Path(rejected[0]['path']), path)
        self.assertEqual(path.read_bytes(), b'incomplete zip checkpoint')

    def test_nonfinite_latest_checkpoint_is_skipped(self):
        checkpoint(self.source / 'runs-57-17841.pt', finite=False)
        latest, _, rejected = recovery.select_checkpoints(self.source)
        self.assertEqual(latest['epoch'], 56)
        self.assertIn('non-finite', rejected[0]['reason'])

    def test_incomplete_confusion_matrix_cannot_select_prior_best(self):
        path = self.source / 'epoch56_test_each_class_acc.csv'
        path.write_text('0,0\n1,2\n', encoding='utf-8')
        _, previous, _ = recovery.select_checkpoints(self.source)
        self.assertEqual(previous['epoch'], 42)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            recovery.historical_accuracy(path)

    def test_plan_preserves_source_and_uses_original_absolute_epoch_schedule(self):
        config = self.plan['job']['config']
        self.assertEqual(config['start_epoch'], 56)
        self.assertEqual(config['num_epoch'], 65)
        self.assertEqual(config['step'], [35, 55])
        self.assertEqual(config['train_feeder_args']['joint_mask_prob'], 0)
        self.assertEqual(Path(config['weights']).parent, self.source)
        self.assertEqual(Path(config['prior_best_weights']).parent, self.source)
        self.assertNotEqual(self.plan['job']['work'], self.source)
        self.assertTrue(self.plan['job']['work'].name.endswith('_resume1'))
        self.assertTrue((self.source / 'runs-56-17528.pt').exists())

    def test_completed_epoch65_does_not_start_another_training(self):
        checkpoint(self.source / 'runs-65-20345.pt')
        with self.assertRaisesRegex(ValueError, 'Epoch 65'):
            recovery.build_resume_plan(self.repository)

    def test_changed_model_source_blocks_recovery(self):
        archived = self.source / self.plan['algorithm_source'].name
        archived.write_text('different implementation', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            recovery.build_resume_plan(self.repository)

    def test_existing_recovery_namespace_is_preserved_and_next_one_selected(self):
        self.plan['job']['work'].mkdir()
        sentinel = self.plan['job']['work'] / 'existing.txt'
        sentinel.write_text('preserved', encoding='utf-8')
        new = recovery.build_resume_plan(self.repository)
        self.assertTrue(new['job']['work'].name.endswith('_resume2'))
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserved')


class ExecutionTests(RecoveryFixture):
    def test_recovery_evaluates_audits_and_saves_before_shutdown(self):
        def shutdown(_):
            self.events.append('shutdown')
            state = json.loads(self.plan['state_path'].read_text(encoding='utf-8'))
            self.assertTrue(state['results_verified'])
            self.assertEqual(state['recovery_mode'], 'weights_only_optimizer_scaler_rng_not_restored')
            self.assertTrue((self.plan['job']['work'] / 'best.pt').is_file())
            self.assertTrue(self.plan['audit_report'].is_file())

        with patch.object(recovery, 'run_training', side_effect=self.fake_recovery_training), \
                patch.object(recovery.pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(recovery.pipeline, 'logged_command', side_effect=self.fake_audit), \
                patch.object(recovery.pipeline, 'request_shutdown', side_effect=shutdown):
            self.execute_recovery()
        self.assertEqual(self.events, ['train', 'evaluate', 'audit', 'shutdown'])

    def test_training_failure_prevents_evaluation_and_shutdown(self):
        with patch.object(recovery, 'run_training', side_effect=RuntimeError('OOM')), \
                patch.object(recovery.pipeline, 'run_evaluation') as evaluate, \
                patch.object(recovery.pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(RuntimeError, 'OOM'):
                self.execute_recovery()
        evaluate.assert_not_called()
        shutdown.assert_not_called()
        state = json.loads(self.plan['state_path'].read_text(encoding='utf-8'))
        self.assertEqual(state['status'], 'failed')
        self.assertFalse(state['results_verified'])

    def test_audit_failure_prevents_shutdown(self):
        with patch.object(recovery, 'run_training', side_effect=self.fake_recovery_training), \
                patch.object(recovery.pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(recovery.pipeline, 'logged_command', side_effect=RuntimeError('audit failed')), \
                patch.object(recovery.pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(RuntimeError, 'audit failed'):
                self.execute_recovery()
        shutdown.assert_not_called()


if __name__ == '__main__':
    unittest.main()
