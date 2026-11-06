"""Check the actual official CUDA Mamba block and hybrid model before training."""

from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import mamba_ssm
    import causal_conv1d
    from model.ctrgcn_mamba import Model

    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required for the official Mamba kernels.')
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = Model(num_class=60, num_point=25, num_person=2,
                  graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}).cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True,
                                weight_decay=0.0004)
    scaler = torch.amp.GradScaler('cuda')
    data = torch.randn(2, 3, 64, 25, 2, device='cuda')
    labels = torch.tensor([0, 1], device='cuda')
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(data)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    if logits.shape != (2, 60) or not torch.isfinite(loss).item():
        raise RuntimeError('Hybrid forward pass returned invalid logits/loss.')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError('Hybrid AMP backward pass produced non-finite gradients.')
    mamba_gradient = model.temporal_mamba.in_proj.weight.grad
    if mamba_gradient is None or mamba_gradient.abs().max().item() == 0:
        raise RuntimeError('The Mamba branch did not receive a nonzero gradient.')
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
    print('mamba_ssm:', mamba_ssm.__version__, 'source:', mamba_ssm.__file__)
    print('causal_conv1d:', causal_conv1d.__version__)
    print('Parameters:', sum(parameter.numel() for parameter in model.parameters()))
    print('PASS: official Mamba-1 hybrid CUDA forward, AMP backward and SGD step.')


if __name__ == '__main__':
    main()
