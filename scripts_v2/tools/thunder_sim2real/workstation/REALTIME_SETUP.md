# Thunder Linux scheduling permissions

The collector requires FIFO priorities 90 for receive, 85 for control, and 80 for
the application loop. A priority limit of 99 grants permission to request those
priorities; ordinary programs keep their normal scheduling unless they explicitly
request a real-time scheduler. These permissions do not install a PREEMPT_RT kernel.

The ur-rtde 1.6.5 constructors skip their scheduler setup when a PREEMPT_RT kernel
is not detected. The collector therefore creates each SDK interface while the
calling thread has the corresponding FIFO priority, verifies the inherited worker
priority, and restores the caller's scheduler. It refuses unavailable permissions
before connecting and incorrect worker priorities before its first torque command.
The original application scheduler is restored after controller cleanup.

## One-time permission setup

For the Thunder workstation operator, `/etc/security/limits.d/99-realtime.conf`
contains:

```text
srinivas - rtprio 99
```

This sets both the soft and hard PAM session limits. Log out and back in, then
check `ulimit -r`. The expected result is `99`.

Desktop apps can still inherit an older zero limit from a systemd user manager
that survives logout. On this workstation, the following per-user settings also
apply; substitute the operator's actual UID for `1012` on another machine:

```ini
# /etc/systemd/system/user@1012.service.d/99-realtime.conf
[Service]
LimitRTPRIO=99
```

```ini
# ~/.config/systemd/user.conf — preserve any other existing manager settings
[Manager]
DefaultLimitRTPRIO=99
```

Reload system unit definitions with `sudo systemctl daemon-reload`. For a user
manager that is already running with a zero hard limit, raise that manager's
limit and reload its configuration without terminating the desktop:

```bash
thunder_uid="$(id -u)"
thunder_manager_pid="$(systemctl show "user@${thunder_uid}.service" --property=MainPID --value)"
sudo prlimit --pid "$thunder_manager_pid" --rtprio=99:99
systemctl --user daemon-reexec
systemctl --user show -p DefaultLimitRTPRIO
systemd-run --user --quiet --wait --pipe bash -c 'ulimit -r'
```

The last two commands should show `DefaultLimitRTPRIO=99` and `99`.
Existing application processes retain their inherited limits until relaunched.

## Collector invocation

`run_realtime.sh` also handles launches from an existing shell or application with
an older limit. When necessary, it uses non-interactive sudo only to raise its own
limit with `prlimit`, then executes Python as the original user. If sudo is not
available and the limit is insufficient, it exits before starting the command.

After the separate start-position and hardware readiness checks have passed:

```bash
bash scripts_v2/tools/thunder_sim2real/workstation/run_realtime.sh \
  .venv-thunder/bin/python scripts_v2/tools/thunder_sim2real/workstation/collect_thunder.py \
  --config /tmp/thunder-collection.json --output /tmp/thunder-fit.pt --execute
```

Records save the actual loop scheduler and receive/control worker schedulers.
Inspect the strict validator and host command timing after every collection.
FIFO permissions alone do not establish command timing under load or collision
clearance. A PREEMPT_RT kernel is a separate configuration choice; the current
workstation uses `7.0.0-31-generic`.

Reference: [ur-rtde real-time setup guide](https://sdurobotics.gitlab.io/ur_rtde/pages/core_concepts/guides/realtime_setup.html).
