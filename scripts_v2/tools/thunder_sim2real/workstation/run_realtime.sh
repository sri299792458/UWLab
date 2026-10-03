#!/usr/bin/env bash
set -euo pipefail
if (( $# == 0 )); then
  echo 'Usage: run_realtime.sh command [arguments...]' >&2
  exit 2
fi
# Existing desktop sessions retain their old hard limit. Raise only this
# launcher's limit, then exec the command as the same unprivileged user.
thunder_rt_limit="$(ulimit -r)"
if [[ "$thunder_rt_limit" != unlimited ]] && (( thunder_rt_limit < 99 )); then
  if ! sudo -n /usr/bin/prlimit --pid "$$" --rtprio=99:99; then
    echo 'Cannot grant Thunder RT priority. Start a session with rtprio 99 or configure /etc/security/limits.d/99-realtime.conf.' >&2
    exit 1
  fi
fi
exec "$@"
