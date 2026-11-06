"""Install official Mamba-1-compatible wheels for the existing AutoDL environment."""

import argparse
import http.client
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib


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


def validate_wheel(path):
    """Reject truncated downloads, HTML error pages and corrupted wheel payloads."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata = [name for name in names if name.endswith('.dist-info/WHEEL')]
        if len(metadata) != 1 or metadata[0].rsplit('/', 1)[0] + '/METADATA' not in names:
            raise ValueError('Downloaded file is not a Python wheel: {}'.format(path.name))
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError('Wheel CRC check failed for {}'.format(damaged))


def download_wheel(url, cache_dir, attempts=5, timeout=120, retry_delay=2):
    """Resume interrupted HTTPS downloads and publish only a validated wheel."""
    if attempts < 1:
        raise ValueError('Download attempts must be positive.')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name
    if not filename.endswith('.whl'):
        raise ValueError('Expected a wheel URL.')
    target = cache_dir / filename
    partial = target.with_suffix(target.suffix + '.part')
    for cached in (target, partial):
        if cached.is_file():
            try:
                validate_wheel(cached)
            except (OSError, ValueError, zipfile.BadZipFile, zlib.error):
                if cached == target:
                    cached.unlink()
            else:
                if cached == partial:
                    os.replace(partial, target)
                print('Using verified cached wheel:', target, flush=True)
                return target
    last_error = None
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'CTR-GCN-Mamba-Installer'}
        if offset:
            headers['Range'] = 'bytes={}-'.format(offset)
        request = urllib.request.Request(url, headers=headers)
        transfer_complete = False
        print('Download {}/{}: {} (resume at {:.1f} MiB)'.format(
            attempt, attempts, filename, offset / 1024 ** 2), flush=True)
        try:
            # urllib honors the AutoDL network_turbo proxy environment variables.
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                         response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or not offset <= int(match[2]) < int(match[3]):
                        raise ValueError('Server returned an unexpected resume range.')
                    total = int(match[3])
                    mode = 'ab'
                elif response.status == 200:
                    # Some proxies ignore Range; restart rather than append a full file.
                    mode = 'wb'
                    offset = 0
                    length = response.headers.get('Content-Length')
                    total = int(length) if length else None
                else:
                    raise ValueError('Unexpected HTTP response: {}'.format(response.status))
                received = offset
                last_progress = time.monotonic()
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(1024 ** 2)
                        if not chunk:
                            break
                        stream.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_progress >= 5:
                            print('  {:.1f} / {} MiB'.format(
                                received / 1024 ** 2,
                                '{:.1f}'.format(total / 1024 ** 2) if total else '?'), flush=True)
                            last_progress = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
                if total is not None and received != total:
                    raise OSError('Incomplete download: {} of {} bytes'.format(received, total))
                transfer_complete = True
            validate_wheel(partial)
            os.replace(partial, target)
            print('Wheel archive and CRC checks passed:', target.name, flush=True)
            return target
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException,
                zipfile.BadZipFile, zlib.error) as error:
            last_error = error
            # A fully transferred but corrupt file cannot be repaired by appending.
            if transfer_complete or (isinstance(error, urllib.error.HTTPError) and error.code == 416):
                partial.unlink(missing_ok=True)
            print('Download attempt failed:', error, flush=True)
            if attempt < attempts:
                time.sleep(min(retry_delay * 2 ** (attempt - 1), 15))
    raise RuntimeError('Could not download a valid wheel after {} attempts. '
                       'Partial downloads are cached in {}; enable AutoDL network_turbo '
                       'and rerun to resume.'.format(attempts, cache_dir)) from last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-runtime-deps', action='store_true',
                        help='skip runtime packages that already installed successfully')
    parser.add_argument('--wheel-dir', type=Path,
                        help='persistent wheel cache (default: repository/.wheelhouse)')
    parser.add_argument('--retries', type=int, default=5)
    args = parser.parse_args()
    if args.retries < 1:
        parser.error('--retries must be positive')
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
    if not args.skip_runtime_deps:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
            '-r', str(root / 'requirements-mamba.txt')], check=True)
    for url in urls:
        wheel = download_wheel(url, args.wheel_dir or root / '.wheelhouse', attempts=args.retries)
        print('Installing verified official wheel:', wheel, flush=True)
        # Never let the optional CUDA modules replace the working PyTorch build.
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-index', '--no-deps', str(wheel)],
            check=True)
    subprocess.run([sys.executable, str(root / 'tools' / 'check_mamba.py')],
                   cwd=root, check=True)


if __name__ == '__main__':
    main()
