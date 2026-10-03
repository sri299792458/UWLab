"""Read-only gripper checks shared by the workstation motion entry points."""
import socket


def gripper_state(ip):
    with socket.create_connection((ip, 63352), timeout=2) as sock:
        stream = sock.makefile('rb')
        values = {}
        for field in ('ACT', 'STA', 'POS', 'FLT', 'OBJ'):
            sock.sendall(f'GET {field}\n'.encode())
            name, value = stream.readline(256).decode().split()
            if name != field:
                raise RuntimeError(f'Unexpected gripper reply for {field}')
            values[field] = int(value)
        return values


def check_gripper_open(ip):
    state = gripper_state(ip)
    if state['ACT'] != 1 or state['STA'] != 3 or state['FLT'] != 0 or state['POS'] > 5 or state['OBJ'] != 3:
        raise RuntimeError(f'Gripper must be active, open and at rest. Run run.sh open first. Readback: {state}')
    return state
