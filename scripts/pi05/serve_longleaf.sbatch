#!/bin/bash
#SBATCH --job-name=pi05-droid
#SBATCH --partition=a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=pi05-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm}"
: "${OPENPI_ROOT:?Set absolute path to the prepared OpenPI checkout}"
: "${OPENPI_REV:?Set the full 40-character commit SHA recorded at setup}"
: "${OPENPI_DATA_HOME:?Set checkpoint cache on research storage}"
: "${PI05_RUN_ROOT:?Set directory for job metadata on research storage}"
export PI05_PORT="${PI05_PORT:-8000}"
export PYTHONUNBUFFERED=1
[[ "$OPENPI_REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'OPENPI_REV must be a full SHA' >&2; exit 2; }
[[ "$PI05_PORT" =~ ^[0-9]+$ ]] && (( PI05_PORT > 0 && PI05_PORT < 65536 )) || exit 2
cd "$OPENPI_ROOT"
[[ "$(git rev-parse HEAD)" == "$OPENPI_REV" ]] || { echo 'OpenPI revision mismatch' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'OpenPI tracked files modified' >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'Prepare OpenPI .venv before submitting' >&2; exit 2; }
mkdir -p "$OPENPI_DATA_HOME" "$PI05_RUN_ROOT"
run_dir="$PI05_RUN_ROOT/job-$SLURM_JOB_ID"
mkdir "$run_dir"
export PI05_RUN_DIR="$run_dir"
hostname | tee "$run_dir/hostname.txt"
git rev-parse HEAD > "$run_dir/openpi-revision.txt"
nvidia-smi > "$run_dir/nvidia-smi.txt"
.venv/bin/python -m pip freeze > "$run_dir/pip-freeze.txt" 2>/dev/null || .venv/bin/python -c 'import importlib.metadata as m; print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in m.distributions())))' > "$run_dir/pip-freeze.txt"
# Reuse upstream policy and server APIs; bind to loopback for an SSH tunnel.
exec .venv/bin/python -u - <<'PY'
import json
import os
from pathlib import Path
import time
import jax
import numpy as np
from openpi.policies import policy_config
from openpi.policies.droid_policy import make_droid_example
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config

run = Path(os.environ['PI05_RUN_DIR'])
devices = jax.devices()
if not any(d.platform == 'gpu' for d in devices):
    raise RuntimeError(f'No JAX GPU device: {devices}')
print('GPU devices:', devices, flush=True)
policy = policy_config.create_trained_policy(
    config.get_config('pi05_droid'), 'gs://openpi-assets/checkpoints/pi05_droid')
np.random.seed(0)
request = make_droid_example()
start = time.monotonic()
actions = np.asarray(policy.infer(request)['actions'])
if actions.shape != (15, 8) or not np.isfinite(actions).all():
    raise RuntimeError(f'Unexpected pi05_droid output: {actions.shape}')
np.save(run / 'synthetic-actions.npy', actions)
(run / 'warmup.json').write_text(json.dumps({
    'shape': list(actions.shape), 'finite': True,
    'seconds': time.monotonic() - start,
    'devices': [str(d) for d in devices],
    'note': 'Synthetic request only; does not establish task performance or inference RNG reproducibility.'}, indent=2))
port = int(os.environ['PI05_PORT'])
print(f'WARMUP PASSED; serving on 127.0.0.1:{port}', flush=True)
WebsocketPolicyServer(policy, host='127.0.0.1', port=port, metadata=policy.metadata).serve_forever()
PY
