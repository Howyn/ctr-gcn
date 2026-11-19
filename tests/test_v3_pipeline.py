"""CPU-only tests of clean V3 training, paired evaluation and the shutdown gate."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools import train_v3_autodl as pipeline


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name).resolve()
        (self.repository / 'logs').mkdir()
        source = pipeline.ROOT / 'config/nturgbd-cross-subject'
        self.configs = {}
        for key, filename, target in (
                ('candidate', 'ctrgcn_mamba_robust_v3_clean_fast128.yaml', pipeline.CONFIG),
                ('standard', 'ctrgcn_mamba_fast128.yaml', pipeline.STANDARD / 'config.yaml'),
                ('reference', 'ctrgcn_mamba_robust_v2_clean_fast128.yaml', pipeline.REFERENCE / 'config.yaml')):
            config = yaml.safe_load((source / filename).read_text(encoding='utf-8'))
            if key != 'candidate':
                config['work_dir'] = str((self.repository / target).parent)
            self.configs[key] = config
            path = self.repository / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
        reference = self.repository / pipeline.REFERENCE
        (reference / 'log.txt').write_text('Best accuracy: 0.89058\nEpoch number: 46\n', encoding='utf-8')
        (reference / 'runs-46-14398.pt').write_bytes(b'existing clean-trained V2')
        self.plan = pipeline.build_plan(self.repository)
        self.events = []

    def fake_training(self, job, snapshot, repository, environment):
        self.events.append('train')
        self.assertEqual(repository, self.repository)
        self.assertEqual(environment['OMP_NUM_THREADS'], '1')
        self.assertEqual(environment['MKL_NUM_THREADS'], '1')
        self.assertEqual(yaml.safe_load(snapshot.read_text(encoding='utf-8')), job['config'])
        job['work'].mkdir(parents=True)
        saved = dict(job['config'], seed=1, device=[0], phase='train', weights=None, start_epoch=0)
        saved['work_dir'] = str(job['work'])
        (job['work'] / 'config.yaml').write_text(yaml.safe_dump(saved), encoding='utf-8')
        job['log'].write_text('Training finished normally\n', encoding='utf-8')
        (job['work'] / 'log.txt').write_text(
            'Training epoch: 1\nTraining epoch: 65\nBest accuracy: 0.891\n'
            'Epoch number: 60\nModel total number of params: 1902995\nseed: 1\n', encoding='utf-8')
        for name in ('runs-60-18780.pt', 'runs-65-20345.pt', 'epoch1_test_score.pkl',
                     'epoch1_test_each_class_acc.csv', 'runs-60-18780_right.txt'):
            (job['work'] / name).write_bytes(b'saved training fixture')
        (job['work'] / 'runs-60-18780_wrong.txt').touch()

    def evaluation_summary(self, checkpoint):
        rows = [dict(model='baseline', clean_top1_percent=89.06, occluded_top1_percent=80.19,
                     drop_pp=89.06 - 80.19),
                dict(model='mamba', clean_top1_percent=89.1, occluded_top1_percent=81.2,
                     drop_pp=89.1 - 81.2)]
        checkpoints = {}
        for name, path, model in (
                ('baseline', self.plan['reference_weights'], 'model.ctrgcn_mamba_robust_v2.Model'),
                ('mamba', checkpoint, 'model.ctrgcn_mamba_robust_v3.Model')):
            checkpoints[name] = dict(checkpoint=str(path), checkpoint_sha256=pipeline.digest(path), model=model)
        return dict(status='complete', protocol='per_sample_fixed_joints_zero_after_test_crop_resize',
                    rate=0.2, missing_joints=5, mask_seed=2026, samples=16487,
                    same_joints_for_all_frames_and_people=True, same_training_feeder_args=True,
                    checkpoints=checkpoints, results=rows)

    def fake_evaluation(self, plan, checkpoint, repository, environment):
        self.events.append('evaluate')
        self.assertEqual(checkpoint, plan['job']['work'] / 'best.pt')
        self.assertEqual(repository, self.repository)
        plan['output_dir'].mkdir(parents=True)
        summary = self.evaluation_summary(checkpoint)
        (plan['output_dir'] / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        # Binary score/mask parsing is delegated to the existing audit tool. The
        # subprocess is mocked here; this fixture tests the surrounding gate.
        for name in ('summary.csv', 'predictions.csv', 'scores.npz', 'masks.npz'):
            (plan['output_dir'] / name).write_bytes(b'saved evaluation fixture')

    def fake_audit(self, command, log, repository, environment, exclusive=True):
        self.events.append('audit')
        self.assertIn('--artifacts-only', command)
        self.assertIn(str(self.plan['output_dir']), command)
        summary = json.loads((self.plan['output_dir'] / 'summary.json').read_text(encoding='utf-8'))
        metrics = {}
        for row in summary['results']:
            for kind, key in (('clean', 'clean_top1_percent'), ('occluded', 'occluded_top1_percent')):
                metrics[row['model'] + '_' + kind] = dict(top1_percent=row[key])
        report = dict(scope='saved_artifacts_only', samples=16487, mask_seed=2026,
                      missing_joints=5, same_training_inputs=True, metrics=metrics)
        self.plan['audit_report'].write_text(json.dumps(report), encoding='utf-8')
        log.write_text('Artifact audit passed\n', encoding='utf-8')

    def execute(self, **kwargs):
        return pipeline.execute_pipeline(self.plan, repository=self.repository, **kwargs)

    def state(self):
        return json.loads(self.plan['state_path'].read_text(encoding='utf-8'))


class PlanTests(PipelineFixture):
    def test_plan_uses_clean_v2_best_checkpoint_and_original_clean_recipe(self):
        self.assertEqual(self.plan['job']['config']['train_feeder_args']['joint_mask_prob'], 0)
        self.assertEqual(self.plan['reference_config']['train_feeder_args']['joint_mask_prob'], 0)
        self.assertEqual(self.plan['job']['config']['num_epoch'], 65)
        self.assertEqual(self.plan['reference_weights'].name, 'runs-46-14398.pt')
        self.assertEqual(self.plan['reference_sha256'], pipeline.digest(self.plan['reference_weights']))
        pipeline.check_fresh(self.plan)

    def test_reference_with_added_training_masks_is_rejected(self):
        config = deepcopy(self.configs['reference'])
        config['train_feeder_args']['joint_mask_prob'] = 0.25
        (self.repository / pipeline.REFERENCE / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'joint masking must be disabled'):
            pipeline.build_plan(self.repository)

    def test_candidate_recipe_change_is_rejected(self):
        config = deepcopy(self.configs['candidate'])
        config['batch_size'] = 64
        (self.repository / pipeline.CONFIG).write_text(yaml.safe_dump(config), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Training setting differs'):
            pipeline.build_plan(self.repository)

    def test_missing_reference_checkpoint_is_rejected_before_training(self):
        self.plan['reference_weights'].unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or empty result file'):
            pipeline.build_plan(self.repository)

    def test_existing_evaluation_or_old_watcher_log_prevents_any_training(self):
        for key in ('output_dir', 'legacy_watcher_log'):
            path = self.plan[key]
            if key == 'output_dir':
                path.mkdir(parents=True)
            else:
                path.touch()
            with self.subTest(key=key), patch.object(pipeline, 'run_training') as train, \
                    patch.object(pipeline, 'request_shutdown') as shutdown:
                with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                    self.execute()
                train.assert_not_called()
                shutdown.assert_not_called()
            path.rmdir() if path.is_dir() else path.unlink()

    def test_check_only_never_runs_preflight_training_or_shutdown(self):
        with patch.object(pipeline.sys, 'argv', ['train_v3_autodl.py', '--check-only']), \
                patch.object(pipeline, 'build_plan', return_value=self.plan), \
                patch.object(pipeline, 'preflight') as preflight, \
                patch.object(pipeline, 'execute_pipeline') as execute, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            pipeline.main()
        preflight.assert_not_called()
        execute.assert_not_called()
        shutdown.assert_not_called()


class ExecutionTests(PipelineFixture):
    def test_verified_training_and_audited_evaluation_precede_single_shutdown(self):
        actual_verify, actual_persist = pipeline.verified_run, pipeline.persist_results

        def verify(job):
            self.events.append('verify_training')
            return actual_verify(job)

        def persist(*arguments):
            self.events.append('persist_training')
            return actual_persist(*arguments)

        def shutdown(path):
            self.events.append('shutdown')
            self.assertEqual(path, Path('/usr/bin/shutdown'))
            state = self.state()
            self.assertTrue(state['results_verified'])
            self.assertEqual(state['status'], 'shutdown_requested')
            job = self.plan['job']
            saved = json.loads((job['work'] / 'result_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['checkpoint_sha256'], pipeline.digest(job['work'] / 'best.pt'))
            self.assertEqual(state, json.loads((job['work'] / 'pipeline_summary.json').read_text(encoding='utf-8')))

        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'verified_run', side_effect=verify), \
                patch.object(pipeline, 'persist_results', side_effect=persist), \
                patch.object(pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(pipeline, 'logged_command', side_effect=self.fake_audit), \
                patch.object(pipeline, 'request_shutdown', side_effect=shutdown) as request:
            self.execute()
        self.assertEqual(self.events, ['train', 'verify_training', 'persist_training',
                                       'evaluate', 'audit', 'shutdown'])
        request.assert_called_once()

    def test_training_failure_blocks_evaluation_and_shutdown(self):
        with patch.object(pipeline, 'run_training', side_effect=RuntimeError('training failed')), \
                patch.object(pipeline, 'run_evaluation') as evaluate, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                self.execute()
        evaluate.assert_not_called()
        shutdown.assert_not_called()
        self.assertEqual(self.state()['status'], 'failed')

    def test_incomplete_training_blocks_evaluation_and_shutdown(self):
        def incomplete(*arguments):
            self.fake_training(*arguments)
            (self.plan['job']['work'] / 'runs-65-20345.pt').unlink()

        with patch.object(pipeline, 'run_training', side_effect=incomplete), \
                patch.object(pipeline, 'run_evaluation') as evaluate, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'Final epoch checkpoint'):
                self.execute()
        evaluate.assert_not_called()
        shutdown.assert_not_called()

    def test_evaluation_failure_preserves_training_and_blocks_shutdown(self):
        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=RuntimeError('evaluation failed')), \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                self.execute()
        self.assertTrue((self.plan['job']['work'] / 'best.pt').is_file())
        self.assertTrue((self.plan['job']['work'] / 'result_summary.json').is_file())
        self.assertFalse(self.state()['results_verified'])
        shutdown.assert_not_called()

    def test_missing_evaluation_artifact_blocks_audit_and_shutdown(self):
        def missing(*arguments):
            self.fake_evaluation(*arguments)
            (self.plan['output_dir'] / 'scores.npz').unlink()

        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=missing), \
                patch.object(pipeline, 'logged_command') as audit, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'scores.npz'):
                self.execute()
        audit.assert_not_called()
        shutdown.assert_not_called()

    def test_wrong_mask_seed_or_checkpoint_blocks_shutdown(self):
        for mutation, message in (
                (lambda summary: summary.update(mask_seed=1), 'protocol/completion mismatch'),
                (lambda summary: summary['checkpoints']['mamba'].update(checkpoint_sha256='bad'),
                 'different checkpoint/model')):
            # Use direct verifier fixtures so each rejected summary is independent.
            checkpoint = self.repository / 'candidate.pt'
            checkpoint.write_bytes(b'candidate')
            if not self.plan['output_dir'].exists():
                self.plan['output_dir'].mkdir(parents=True)
                for name in ('summary.csv', 'predictions.csv', 'scores.npz', 'masks.npz'):
                    (self.plan['output_dir'] / name).write_bytes(b'fixture')
            summary = self.evaluation_summary(checkpoint)
            mutation(summary)
            (self.plan['output_dir'] / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            with self.subTest(message=message), patch.object(pipeline, 'logged_command') as audit:
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.verify_evaluation(self.plan, checkpoint, self.repository, {})
                audit.assert_not_called()

    def test_audit_failure_blocks_shutdown(self):
        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(pipeline, 'logged_command', side_effect=subprocess.CalledProcessError(1, ['audit'])), \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute()
        shutdown.assert_not_called()
        self.assertFalse(self.state()['results_verified'])

    def test_inconsistent_audit_metrics_block_shutdown(self):
        def inconsistent(*arguments):
            self.fake_audit(*arguments)
            report = json.loads(self.plan['audit_report'].read_text(encoding='utf-8'))
            report['metrics']['mamba_occluded']['top1_percent'] = 99
            self.plan['audit_report'].write_text(json.dumps(report), encoding='utf-8')

        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(pipeline, 'logged_command', side_effect=inconsistent), \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'Audited accuracy differs'):
                self.execute()
        shutdown.assert_not_called()

    def test_shutdown_can_be_disabled_without_disabling_result_checks(self):
        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(pipeline, 'logged_command', side_effect=self.fake_audit), \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            state = self.execute(shutdown=False)
        self.assertEqual(state['status'], 'complete')
        self.assertTrue(state['results_verified'])
        self.assertEqual(self.events, ['train', 'evaluate', 'audit'])
        shutdown.assert_not_called()

    def test_nonfinite_audit_metric_blocks_shutdown(self):
        def nonfinite(*arguments):
            self.fake_audit(*arguments)
            report = json.loads(self.plan['audit_report'].read_text(encoding='utf-8'))
            report['metrics']['mamba_occluded']['top1_percent'] = float('nan')
            self.plan['audit_report'].write_text(json.dumps(report), encoding='utf-8')

        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(pipeline, 'logged_command', side_effect=nonfinite), \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'Audited accuracy differs'):
                self.execute()
        shutdown.assert_not_called()

    def test_evaluation_command_pins_checkpoints_and_protocol(self):
        checkpoint = self.repository / 'best.pt'
        with patch.object(pipeline, 'logged_command') as logged:
            pipeline.run_evaluation(self.plan, checkpoint, self.repository, {})
        command = logged.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, '-u', str(self.repository / 'tools/evaluate_occlusion.py')])
        expected = {'--baseline-dir': str(self.plan['reference_dir']),
                    '--mamba-dir': str(self.plan['job']['work']),
                    '--baseline-weights': str(self.plan['reference_weights']),
                    '--mamba-weights': str(checkpoint), '--rate': '0.2', '--mask-seed': '2026'}
        for option, value in expected.items():
            self.assertEqual(command[command.index(option) + 1], value)


if __name__ == '__main__':
    unittest.main()
