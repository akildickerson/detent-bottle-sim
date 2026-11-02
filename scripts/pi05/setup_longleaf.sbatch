#!/bin/bash
#SBATCH --job-name=pi05-setup
#SBATCH --partition=datamover
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pi05-setup-%j.log
#SBATCH --mail-user=akil@unc.edu
#SBATCH --mail-type=END,FAIL
# UNC documents datamover for large internet downloads:
# https://help.rc.unc.edu/getting-started-on-longleaf
set -euo pipefail
module purge
module load python/3.12.4
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${OPENPI_REV:?Export a full upstream commit SHA before submitting}"
export PI05_PROJECT="${PI05_PROJECT:-/work/users/a/k/akil/pi05}"
export OPENPI_ROOT="${OPENPI_ROOT:-$PI05_PROJECT/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PI05_PROJECT/checkpoints}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PI05_PROJECT/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PI05_PROJECT/python}"
export HF_HOME="${HF_HOME:-$PI05_PROJECT/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PI05_PROJECT/pip-cache}"
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONUNBUFFERED=1
export UV_CONCURRENT_BUILDS=2
export UV_CONCURRENT_INSTALLS=2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# sbatch copies the script, so resolve the helper against the submission directory.
helper="${PI05_SETUP_HELPER:-$SLURM_SUBMIT_DIR/scripts/pi05/prepare_openpi.py}"
[[ -f "$helper" ]] || { echo "Missing setup helper: $helper" >&2; exit 2; }
exec "${SETUP_PYTHON:-python3}" "$helper"
