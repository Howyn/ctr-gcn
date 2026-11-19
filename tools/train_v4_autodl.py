"""Independent V4 experiment name for the current V3 algorithm and full pipeline.

This does not introduce an algorithm change. All output/log/lock paths are V4;
existing V3 results are preserved. Train, evaluate 20% occlusion, audit, power off.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_v3_autodl import main


if __name__ == '__main__':
    try:
        main(variant='v4')
    except Exception as error:
        print('FAILED: {}. V4 training/evaluation/shutdown was not completed; inspect logs.'.format(error),
              file=sys.stderr, flush=True)
        sys.exit(1)
