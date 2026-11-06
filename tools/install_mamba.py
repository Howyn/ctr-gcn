"""Install official Mamba-1-compatible wheels for the existing AutoDL environment."""

import platform
from pathlib import Path
import subprocess
import sys


def official_wheel_urls(torch_version, cuda_version, python_version, cxx11_abi):
    if torch_version.split('+')[0] != '2.5.1':
        raise ValueError('This installer targets the existing PyTorch 2.5.1 environment.')
    if not cuda_version or cuda_version.split('.')[0] != '12':
        raise ValueError('CUDA-enabled PyTorch built for CUDA 12.x is required.')
    if python_version not in ((3, 10), (3, 11), (3, 12), (3, 13)):
        raise ValueError('No selected official wheel for this Python version.')
    tag = 'cp{}{}'.format(*python_version)
    abi = 'TRUE' if cxx11_abi else 'FALSE'
    suffix = 'cu12torch2.5cxx11abi{}-{}-{}-linux_x86_64.whl'.format(abi, tag, tag)
    return [
        'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/'
        'causal_conv1d-1.5.0.post8+' + suffix,
        'https://github.com/state-spaces/mamba/releases/download/v2.2.4/'
        'mamba_ssm-2.2.4+' + suffix,
    ]


def main():
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise SystemExit('Run this installer on the Linux x86_64 AutoDL server.')
    import torch
    urls = official_wheel_urls(
        torch.__version__, torch.version.cuda, sys.version_info[:2],
        torch._C._GLIBCXX_USE_CXX11_ABI)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Start the GPU instance before installing.')
    root = Path(__file__).resolve().parents[1]
    print('Keeping existing PyTorch:', torch.__version__, flush=True)
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
        '-r', str(root / 'requirements-mamba.txt')], check=True)
    for url in urls:
        print('Installing official wheel:', url, flush=True)
        # Never let the optional CUDA modules replace the working PyTorch build.
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', url],
            check=True)
    subprocess.run([sys.executable, str(root / 'tools' / 'check_mamba.py')],
                   cwd=root, check=True)


if __name__ == '__main__':
    main()
