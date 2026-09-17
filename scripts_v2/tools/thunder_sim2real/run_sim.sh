#!/usr/bin/env bash
set -euo pipefail
thunder_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$thunder_repo_root"
export PYTHONPATH="$thunder_repo_root/source/uwlab:$thunder_repo_root/source/uwlab_assets:$thunder_repo_root/source/uwlab_tasks:$thunder_repo_root/source/uwlab_rl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
exec "${THUNDER_PYTHON:-python}" "$@"
