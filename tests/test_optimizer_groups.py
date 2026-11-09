import unittest

import torch
import torch.nn as nn

from model.optimizer import optimizer_parameters


class TinySSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.ones(2))
        self.D = nn.Parameter(torch.ones(2))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.projection = nn.Linear(2, 2)
        self.projection.bias._no_reinit = True


class OptimizerGroupingTests(unittest.TestCase):
    def test_default_preserves_all_parameters_and_original_order(self):
        model = TinySSM()
        parameters, names = optimizer_parameters(model, 0.0004)
        self.assertEqual([id(value) for value in parameters], [id(value) for value in model.parameters()])
        self.assertEqual(names, [])

    def test_only_marked_state_parameters_are_exempt_and_none_are_lost(self):
        model = TinySSM()
        groups, names = optimizer_parameters(model, 0.0004, True)
        self.assertEqual(names, ['A_log', 'D'])
        self.assertEqual([group['weight_decay'] for group in groups], [0.0004, 0])
        flattened = [parameter for group in groups for parameter in group['params']]
        self.assertEqual(len(flattened), len(list(model.parameters())))
        self.assertEqual(set(map(id, flattened)), set(map(id, model.parameters())))
        self.assertIn(id(model.projection.bias), set(map(id, groups[0]['params'])))

    def test_real_sgd_step_decays_only_regular_parameters_when_enabled(self):
        model = TinySSM()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        groups, _ = optimizer_parameters(model, 0.1, True)
        optimizer = torch.optim.SGD(groups, lr=0.1, weight_decay=0.1)
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            expected = before[name] if name in ('A_log', 'D') else before[name] * 0.99
            torch.testing.assert_close(parameter, expected)

    def test_unmarked_models_have_identical_updates_under_both_modes(self):
        original = nn.Linear(3, 2)
        calibrated = nn.Linear(3, 2)
        calibrated.load_state_dict(original.state_dict())
        for model, enabled in ((original, False), (calibrated, True)):
            groups, names = optimizer_parameters(model, 0.0004, enabled)
            self.assertEqual(names, [])
            optimizer = torch.optim.SGD(groups, lr=0.1, momentum=0.9, nesterov=True,
                                        weight_decay=0.0004)
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        for first, second in zip(original.parameters(), calibrated.parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
