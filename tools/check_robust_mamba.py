"""Validate the improved model with the real official CUDA operators before training."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('amp', 'fp32'), default='amp')
    parser.add_argument('--variant', choices=('v1', 'v2', 'v3'), default='v1')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    import mamba_ssm
    import causal_conv1d
    if args.variant == 'v3':
        from model.ctrgcn_mamba_robust_v3 import Model
        from model.ctrgcn_mamba_robust_v2 import reliable_mamba
    elif args.variant == 'v2':
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
    if model.extra_parameters != dict(v1=25474, v2=26242, v3=18051)[args.variant]:
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
    if args.variant != 'v3':
        data[0, :, :, [0, 4, 8, 12, 16], :] = 0
    else:
        print('V3 backward check uses fully observed inputs: verify selector learning without training masks.',
              flush=True)
    labels = torch.tensor([0, 1], device='cuda')
    before = {name: value.detach().clone() for name, value in model.named_parameters()
              if name.startswith(('temporal_reliability.', 'temporal_local.'))}
    result = run_backward_check(model, optimizer, scaler, data, labels, amp)
    if args.variant == 'v3':
        gradient = model.temporal_reliability.score[-1].weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().max().item() == 0:
            raise RuntimeError('V3 selector did not receive finite nonzero gradients on clean inputs.')
        print('PASS: clean selector gradient {:.6g}; signed residual scale {:.6g}'.format(
            gradient.abs().max().item(), model.temporal_reliability.scale.tanh().item()), flush=True)
    for prefix in ('temporal_reliability.', 'temporal_local.'):
        changed = [name for name, value in model.named_parameters()
                   if name.startswith(prefix) and not torch.equal(before[name], value.detach())
                   and value.grad is not None and value.grad.abs().max().item() > 0]
        if not changed:
            raise RuntimeError('No gradient-driven parameter updates in ' + prefix)
        print('Verified new module updates:', changed, flush=True)
    if args.variant == 'v3':
        model.eval()
        masked = data.clone()
        masked[0, :, :, [0, 4, 8, 12, 16], :] = 0
        original = masked.clone()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            output = model(masked)
        if output.shape != (2, 60) or not torch.isfinite(output).all() or not torch.equal(masked, original):
            raise RuntimeError('V3 masked visibility-pooling forward failed or mutated coordinates.')
        print('PASS: V3 missing-joint CUDA forward is finite and leaves input unchanged.', flush=True)
    print('Verified steps:', result, flush=True)
    print('PASS: improved Mamba CUDA equivalence, finite forward/backward and optimizer updates. '
          'Disposable weights only; no checkpoint was saved.', flush=True)


if __name__ == '__main__':
    main()
