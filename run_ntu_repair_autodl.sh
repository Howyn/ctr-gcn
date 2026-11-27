#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo"
python=${NTU_PYTHON:-/root/autodl-tmp/venv-ctr-infer/bin/python}
shutdown=false
case "${1:-}" in
    '') ;;
    --shutdown) shutdown=true; shift ;;
    *) printf 'Usage: %s [--shutdown]\n' "$0" >&2; exit 2 ;;
esac
if (( $# != 0 )); then
    printf 'Usage: %s [--shutdown]\n' "$0" >&2
    exit 2
fi

epochs=10
variants=(a teacher bc)
out=work_dir/ntu60/xsub/reliable_repair_screen_s1
mkdir -p -- "$out"
exec 9>"$out/autodl-workflow.lock"
flock -n 9 || { printf 'An AutoDL workflow already uses %s\n' "$out" >&2; exit 1; }
trap 'printf "Workflow stopped with exit %s. Check the first error above.\n" "$?" >&2' ERR

if [[ "$shutdown" == true ]]; then
    "$python" -u -m tools.autodl_shutdown --check-shutdown
fi
common=(--repo "$repo" --out "$out" --epochs "$epochs" --seed 1)
"$python" -u ntu_reliable_repair.py doctor "${common[@]}"
"$python" -u ntu_reliable_repair.py train "${common[@]}" --variants "${variants[@]}" --resume

"$python" -u - "$out" "$epochs" "$shutdown" "${variants[@]}" <<'PY'
import csv
import json
import math
import os
from pathlib import Path
import sys

from tools.autodl_shutdown import request_shutdown, required_file

output = Path(sys.argv[1])
epochs = int(sys.argv[2])
expected = set(sys.argv[4:])
table = required_file(output / "validation.csv")
with table.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != len(expected) or {row["model"] for row in rows} != expected:
    raise ValueError(f"Incomplete validation table: {table}")

for row in rows:
    model = row["model"]
    run = output / model
    completion = json.loads(required_file(run / "complete.json").read_text(encoding="utf-8"))
    if completion["variant"] != model or completion["epochs"] != epochs:
        raise ValueError(f"Invalid completion marker: {run}")
    if not 1 <= int(row["selected_epoch"]) <= epochs:
        raise ValueError(f"Invalid selected epoch: {run}")
    for condition in ("clean", "burst20", "random20"):
        top1 = float(row[f"{condition}_top1"])
        top5 = float(row[f"{condition}_top5"])
        if not 0 <= top1 <= top5 <= 100:
            raise ValueError(f"Invalid accuracy: {model}/{condition}")
    selected_metric = "clean_top1" if model == "teacher" else "burst20_top1"
    if not math.isclose(float(row[selected_metric]), completion["best_selection_top1"], abs_tol=1e-8):
        raise ValueError(f"Checkpoint and validation summary differ: {run}")
    required_file(run / "best.pt")
    required_file(run / "last.pt")

print(f"Results verified: {table.resolve()}", flush=True)
if sys.argv[3] == "true":
    print("Requesting AutoDL shutdown after syncing results.", flush=True)
    os.sync()
    request_shutdown(Path("/usr/bin/shutdown"))
PY
