"""Validate the improved model with the real official CUDA operators before training."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('amp', 'fp32'), default='amp')
    parser.add_argument('--variant', choices=('v1', 'v2'), default='v1')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    import mamba_ssm
    import causal_conv1d
    if args.variant == 'v2':
        from model.ctrgcn_mamba_robust_v2 import Model, reliable_mamba
    else:
        from model.ctrgcn_mamba_robust import Model, reliable_mamba
    from check_mamba import run_backward_check

    # Fail on an incompatible NumPy installation before loading the dataset.
    array = np.arange(6, dtype=np.float32)
    assert np.array_equal(torch.from_numpy(array).numpy(), array)
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required.')
    torch.set_num_threads(1)
    torch.manual_seed(1)
    print('Python:', sys.executable, flush=True)
    print('NumPy:', np.__version__, np.__file__, flush=True)
    print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0), flush=True)
    print('mamba_ssm:', mamba_ssm.__version__, 'causal_conv1d:', causal_conv1d.__version__, flush=True)
    model = Model(num_class=60, num_point=25, num_person=2,
                  graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}).cuda()
    print('Parameters:', sum(parameter.numel() for parameter in model.parameters()),
          'extra:', model.extra_parameters, 'budget: 200000', flush=True)
    if model.extra_parameters != (26242 if args.variant == 'v2' else 25474):
        raise RuntimeError('Unexpected NTU60 parameter count; review the algorithm/config.')
    # With r=1 the explicit scan must reproduce the standard fused Mamba block.
    tokens = torch.randn(2, 16, 256, device='cuda')
    for amp in (False, True) if args.precision == 'amp' else (False,):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            standard = model.temporal_mamba(tokens)
            improved = reliable_mamba(model.temporal_mamba, tokens,
                                       torch.ones(2, 16, 1, device='cuda'))
        torch.testing.assert_close(improved, standard, rtol=0.02 if amp else 2e-4,
                                   atol=3e-3 if amp else 2e-5)
        print('PASS: r=1 {} equivalence; max absolute difference {:.6g}'.format(
            'AMP' if amp else 'FP32', (improved - standard).abs().max().item()), flush=True)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9,
                                nesterov=True, weight_decay=0.0004)
    amp = args.precision == 'amp'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    data = torch.randn(2, 3, 64, 25, 2, device='cuda')
    data[0, :, :, [0, 4, 8, 12, 16], :] = 0
    labels = torch.tensor([0, 1], device='cuda')
    before = {name: value.detach().clone() for name, value in model.named_parameters()
              if name.startswith(('temporal_reliability.', 'temporal_local.'))}
    result = run_backward_check(model, optimizer, scaler, data, labels, amp)
    for prefix in ('temporal_reliability.', 'temporal_local.'):
        changed = [name for name, value in model.named_parameters()
                   if name.startswith(prefix) and not torch.equal(before[name], value.detach())
                   and value.grad is not None and value.grad.abs().max().item() > 0]
        if not changed:
            raise RuntimeError('No gradient-driven parameter updates in ' + prefix)
        print('Verified new module updates:', changed, flush=True)
    print('Verified steps:', result, flush=True)
    print('PASS: improved Mamba CUDA equivalence, finite forward/backward and optimizer updates. '
          'Disposable weights only; no checkpoint was saved.', flush=True)


if __name__ == '__main__':
    main()
