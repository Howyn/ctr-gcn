"""CPU checks for retaining pre-interruption candidates during warm continuation.

Extract only the relevant definitions: importing main.py itself requires the
Linux training environment and optional training dependencies.
"""
import ast
from collections import OrderedDict
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn as nn


SOURCE = Path(__file__).resolve().parents[1] / 'main.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'checkpoint_position']
processor = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'Processor')
processor.body = [node for node in processor.body if isinstance(node, ast.FunctionDef)
                  and node.name in ('initialize_prior_best', 'start')]
selected.append(processor)
namespace = dict(os=os, re=re, torch=torch, nn=nn, np=np, random=random,
                 shutil=shutil, OrderedDict=OrderedDict, glob=__import__('glob'))
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
Processor = namespace['Processor']
checkpoint_position = namespace['checkpoint_position']


class ResumeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / 'resume'
        self.work.mkdir()
        self.latest = self.root / 'runs-56-17528.pt'
        self.prior = self.root / 'runs-42-13146.pt'
        self.model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.current = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.previous = {key: torch.ones_like(value) for key, value in self.current.items()}
        torch.save(self.current, self.latest)
        torch.save(self.previous, self.prior)
        self.processor = Processor.__new__(Processor)
        self.processor.model = self.model
        self.processor.arg = SimpleNamespace(
            prior_best_weights=str(self.prior), weights=str(self.latest), phase='train',
            start_epoch=56, work_dir=str(self.work), num_epoch=56, device=[0],
            save_interval=1, save_epoch=0, model_saved_name=str(self.work / 'runs'),
            print_log=True, weight_decay=.0004, base_lr=.1, batch_size=128,
            test_batch_size=128, seed=1)
        self.processor.global_step = 17528
        self.processor.best_acc = 0
        self.processor.best_acc_epoch = 0
        self.processor.print_log = lambda message: None
        self.processor.data_loader = {'train': list(range(313))}
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, epoch, **kwargs):
        self.calls.append(epoch + 1)
        self.model.eval()
        torch.rand(4)
        np.random.rand(4)
        random.random()
        accuracy = .89 if epoch + 1 == 42 else .88
        if accuracy > self.processor.best_acc:
            self.processor.best_acc = accuracy
            self.processor.best_acc_epoch = epoch + 1

    def assert_current_restored(self):
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, self.current[key], rtol=0, atol=0)

    def test_checkpoint_filename_parsing(self):
        self.assertEqual(checkpoint_position(self.latest), (56, 17528))
        for name in ('best.pt', 'runs-0-1.pt', 'runs-56-bad.pt', 'runs-56-4.pt.tmp'):
            with self.assertRaises(ValueError):
                checkpoint_position(name)

    def test_retains_both_candidates_and_restores_latest_weights_modes_step_rng(self):
        self.processor.eval = self.evaluate
        self.model[1].eval()  # Preserve individually mixed train/eval states.
        modes = [module.training for module in self.model.modules()]
        torch.manual_seed(314)
        np.random.seed(314)
        random.seed(314)
        state = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        self.processor.initialize_prior_best()
        self.assertEqual(self.calls, [42, 56])
        self.assertEqual(self.processor.best_acc_epoch, 42)
        self.assertEqual(self.processor.best_acc, .89)
        self.assertEqual(self.processor.global_step, 17528)
        self.assertEqual([module.training for module in self.model.modules()], modes)
        self.assert_current_restored()
        torch.testing.assert_close(torch.get_rng_state(), state[0], rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.get_state()[1], state[1][1])
        self.assertEqual(random.getstate(), state[2])
        for original in (self.prior, self.latest):
            self.assertEqual(original.read_bytes(), (self.work / original.name).read_bytes())

    def test_latest_unfinished_evaluation_can_become_best(self):
        def evaluate(epoch, **kwargs):
            self.processor.best_acc = .9 if epoch + 1 == 56 else .89
            self.processor.best_acc_epoch = epoch + 1
        self.processor.eval = evaluate
        self.processor.initialize_prior_best()
        self.assertEqual(self.processor.best_acc_epoch, 56)
        self.assertEqual(self.processor.best_acc, .9)
        self.assertTrue((self.work / self.latest.name).exists())

    def test_strict_prior_loading_rejects_missing_weights_and_restores_state(self):
        torch.save({'0.weight': self.previous['0.weight']}, self.prior)
        self.processor.eval = self.evaluate
        with self.assertRaises(RuntimeError):
            self.processor.initialize_prior_best()
        self.assert_current_restored()
        self.assertEqual(self.processor.best_acc, 0)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_rejects_start_epoch_mismatch_and_future_prior(self):
        self.processor.arg.start_epoch = 55
        with self.assertRaisesRegex(ValueError, 'start epoch'):
            self.processor.initialize_prior_best()
        self.processor.arg.start_epoch = 56
        self.processor.arg.prior_best_weights = str(self.root / 'runs-57-17841.pt')
        with self.assertRaisesRegex(ValueError, 'start epoch'):
            self.processor.initialize_prior_best()

    def test_refuses_existing_destination(self):
        (self.work / self.prior.name).write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'destination already exists'):
            self.processor.initialize_prior_best()
        self.assertEqual((self.work / self.prior.name).read_text(), 'preserve')

    def test_invalid_accuracy_restores_state_without_copying(self):
        def evaluate(*args, **kwargs):
            self.processor.best_acc = float('nan')
        self.processor.eval = evaluate
        with self.assertRaisesRegex(ValueError, 'invalid clean accuracy'):
            self.processor.initialize_prior_best()
        self.assert_current_restored()
        self.assertEqual(self.processor.best_acc, 0)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_same_prior_and_latest_is_evaluated_and_imported_once(self):
        self.processor.arg.prior_best_weights = str(self.latest)
        self.processor.eval = self.evaluate
        self.processor.initialize_prior_best()
        self.assertEqual(self.calls, [56])
        self.assertEqual(self.processor.best_acc_epoch, 56)
        self.assertEqual(len(list(self.work.glob('runs-*.pt'))), 1)

    def test_start_preserves_checkpoint_global_step_and_finally_selects_prior_best(self):
        self.processor.eval = self.evaluate
        self.processor.start()
        self.assertEqual(self.processor.global_step, 17528)
        self.assertEqual(self.calls, [42, 56, 1])
        self.assertEqual(self.processor.best_acc_epoch, 42)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, self.previous[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
