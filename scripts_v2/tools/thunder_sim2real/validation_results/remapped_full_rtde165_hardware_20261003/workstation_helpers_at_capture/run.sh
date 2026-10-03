#!/usr/bin/env bash
set -euo pipefail
thunder_workstation="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
thunder_workspace="$(dirname -- "$thunder_workstation")"
source "$thunder_workspace/thunder-lab-mount-v2/activate.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
thunder_python="$thunder_workspace/.venv-thunder/bin/python"
thunder_planning_python="$thunder_workspace/.venv-thunder-planning/bin/python"
case "${1:-plan}" in
  plan) exec "$thunder_python" "$thunder_workstation/collect_checked.py" ;;
  timing-plan) exec "$thunder_python" "$thunder_workstation/collect_checked.py" --timing-check ;;
  prepare)
    "$thunder_python" "$thunder_workstation/read_current_state.py"
    "$thunder_planning_python" "$thunder_workstation/verify_start_gpu.py"
    "$thunder_planning_python" "$thunder_workstation/preview_start.py"
    "$thunder_python" "$thunder_workstation/move_to_start.py"
    ;;
  open) exec "$thunder_python" "$thunder_workstation/open_gripper.py" --execute ;;
  move) exec "$thunder_python" "$thunder_workstation/move_to_start.py" --execute ;;
  collect) exec bash "$thunder_workstation/run_realtime.sh" "$thunder_python" "$thunder_workstation/collect_checked.py" --execute ;;
  timing) exec bash "$thunder_workstation/run_realtime.sh" "$thunder_python" "$thunder_workstation/collect_checked.py" --timing-check --execute ;;
  rt-check) exec bash "$thunder_workstation/run_realtime.sh" "$thunder_python" "$thunder_workstation/verify_realtime.py" ;;
  *) echo 'Usage: run.sh [plan|timing-plan|prepare|open|move|timing|collect|rt-check]' >&2; exit 2 ;;
esac
