"""CPU-only checks that the V4 rerun preserves V3 and isolates new artifacts."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import yaml

from tools import train_v3_autodl as pipeline
from test_v3_pipeline import PipelineFixture as BaseFixture


class V4Fixture(BaseFixture):
    def setUp(self):
        super().setUp()
        self.v3_plan = self.plan
        self.v4_relative = Path(str(pipeline.CONFIG).replace('v3', 'v4'))
        source = pipeline.ROOT / self.v4_relative
        config = yaml.safe_load(source.read_text(encoding='utf-8'))
        target = self.repository / self.v4_relative
        target.write_text(yaml.safe_dump(config), encoding='utf-8')
        self.plan = pipeline.build_plan(self.repository, variant='v4')

    def evaluation_summary(self, checkpoint):
        summary = super().evaluation_summary(checkpoint)
        summary['checkpoints']['mamba']['model'] = 'model.ctrgcn_mamba_robust_v4.Model'
        return summary


class V4VersionTests(V4Fixture):
    def test_config_changes_only_model_identifier_and_output_directory(self):
        original = deepcopy(self.configs['candidate'])
        candidate = self.plan['job']['config']
        original.update(model='model.ctrgcn_mamba_robust_v4.Model', work_dir=candidate['work_dir'])
        self.assertEqual(original, candidate)
        self.assertEqual(candidate['train_feeder_args']['joint_mask_prob'], 0)

    def test_model_is_an_explicit_subclass_without_algorithm_overrides(self):
        path = pipeline.ROOT / 'model/ctrgcn_mamba_robust_v4.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                   and node.module == 'model.ctrgcn_mamba_robust_v3']
        self.assertEqual(len(imports), 1)
        imported = next(item for item in imports[0].names if item.name == 'Model')
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model']
        self.assertEqual(len(classes), 1)
        self.assertEqual(len(classes[0].bases), 1)
        self.assertEqual(classes[0].bases[0].id, imported.asname or imported.name)
        self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                             for node in classes[0].body))

    def test_v4_paths_and_identity_are_explicit(self):
        self.assertEqual(self.plan['variant'], 'v4')
        self.assertEqual(self.plan['job']['role'], 'v4')
        self.assertEqual(self.plan['candidate_model'], 'model.ctrgcn_mamba_robust_v4.Model')
        self.assertEqual(self.plan['architecture_equivalent_to'], 'model.ctrgcn_mamba_robust_v3.Model')
        for path in [self.plan['job']['work'], self.plan['job']['log']] + [self.plan[key] for key in
                     ('output_dir', 'evaluation_log', 'audit_log', 'audit_report', 'state_path',
                      'snapshot', 'cuda_log', 'legacy_watcher_log')]:
            self.assertIn('v4', path.name)
            self.assertNotIn('v3', path.name)

    def test_existing_v3_artifacts_do_not_block_or_get_modified_by_v4(self):
        old_paths = [self.v3_plan['job']['work'], self.v3_plan['job']['log']]
        old_paths += [self.v3_plan[key] for key in ('output_dir', 'evaluation_log', 'audit_log',
                      'audit_report', 'state_path', 'snapshot', 'legacy_watcher_log')]
        for path in old_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path in (self.v3_plan['job']['work'], self.v3_plan['output_dir']):
                path.mkdir()
                (path / 'preserved.txt').write_text('previous V3 result', encoding='utf-8')
            else:
                path.write_text('previous V3 result', encoding='utf-8')
        pipeline.check_fresh(self.plan)
        for path in old_paths:
            content = path / 'preserved.txt' if path.is_dir() else path
            self.assertEqual(content.read_text(encoding='utf-8'), 'previous V3 result')

    def test_existing_v4_output_still_prevents_training(self):
        self.plan['output_dir'].mkdir(parents=True)
        with patch.object(pipeline, 'run_training') as train, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                self.execute()
        train.assert_not_called()
        shutdown.assert_not_called()


class V4ExecutionTests(V4Fixture):
    def test_full_v4_run_uses_new_identity_and_shuts_down_only_after_audit(self):
        def shutdown(path):
            self.events.append('shutdown')
            state = self.state()
            self.assertEqual(state['experiment'], 'v4')
            self.assertEqual(state['architecture_equivalent_to'], 'model.ctrgcn_mamba_robust_v3.Model')
            self.assertTrue(state['results_verified'])
            self.assertTrue((self.plan['job']['work'] / 'best.pt').is_file())
            self.assertTrue(self.plan['audit_report'].is_file())

        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(pipeline, 'logged_command', side_effect=self.fake_audit), \
                patch.object(pipeline, 'request_shutdown', side_effect=shutdown) as request:
            self.execute()
        self.assertEqual(self.events, ['train', 'evaluate', 'audit', 'shutdown'])
        request.assert_called_once_with(Path('/usr/bin/shutdown'))

    def test_v3_model_identity_is_not_accepted_as_v4_evaluation(self):
        def wrong_identity(*arguments):
            self.fake_evaluation(*arguments)
            path = self.plan['output_dir'] / 'summary.json'
            summary = json.loads(path.read_text(encoding='utf-8'))
            summary['checkpoints']['mamba']['model'] = 'model.ctrgcn_mamba_robust_v3.Model'
            path.write_text(json.dumps(summary), encoding='utf-8')

        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=wrong_identity), \
                patch.object(pipeline, 'logged_command') as audit, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'different checkpoint/model'):
                self.execute()
        audit.assert_not_called()
        shutdown.assert_not_called()

    def test_v4_preflight_checks_v4_cuda_variant(self):
        with patch.object(pipeline, 'training_preflight'), \
                patch.object(pipeline.subprocess, 'run'), \
                patch.object(pipeline, 'logged_command') as logged:
            pipeline.preflight(self.plan, self.repository)
        command = logged.call_args.args[0]
        self.assertEqual(command[command.index('--variant') + 1], 'v4')
        self.assertEqual(logged.call_args.args[1], self.plan['cuda_log'])

    def test_v4_check_only_does_not_train_or_request_shutdown(self):
        with patch.object(pipeline.sys, 'argv', ['train_v4_autodl.py', '--check-only']), \
                patch.object(pipeline, 'build_plan', return_value=self.plan) as build, \
                patch.object(pipeline, 'preflight') as preflight, \
                patch.object(pipeline, 'execute_pipeline') as execute, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            pipeline.main(variant='v4')
        build.assert_called_once_with(variant='v4')
        preflight.assert_not_called()
        execute.assert_not_called()
        shutdown.assert_not_called()

    def test_evaluation_keeps_the_shared_fixed_five_joint_protocol(self):
        checkpoint = self.plan['job']['work'] / 'best.pt'
        with patch.object(pipeline, 'logged_command') as logged:
            pipeline.run_evaluation(self.plan, checkpoint, self.repository, {})
        command = logged.call_args.args[0]
        for option, expected in (('--mamba-dir', str(self.plan['job']['work'])),
                                 ('--mamba-weights', str(checkpoint)),
                                 ('--baseline-dir', str(self.plan['reference_dir'])),
                                 ('--rate', '0.2'), ('--mask-seed', '2026')):
            self.assertEqual(command[command.index(option) + 1], expected)


if __name__ == '__main__':
    unittest.main()
