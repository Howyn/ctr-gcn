"""Train the V4 soft-pooling/consistent-parameter algorithm, evaluate, then stop.

This replaces the abandoned name-only V4 alias. The new algorithm ID and state
output directories prevent confusing those results with the actual V4 model.
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
