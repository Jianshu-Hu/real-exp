"""Exercise the unified launcher without ROS, USB devices, or a robot."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/start_teleoperation.sh'

WORKER = r'''#!/usr/bin/python3
import json
import os
from pathlib import Path
import signal
import sys
import time

name = Path(sys.argv[0]).name
role = {'manus_bridge': 'bridge', 'teleop.py': 'manus'}.get(name)
if role is None:
    role = sys.argv.pop(1)
if '--help' in sys.argv:
    sys.exit(2 if os.environ.get('FAKE_FAIL_HELP') == role else 0)

def record(event):
    entry = dict(event=event, role=role, args=sys.argv[1:], cwd=os.getcwd())
    with open(os.environ['FAKE_LOG'], 'a') as log:
        log.write(json.dumps(entry) + '\n')

def stop(signum, frame):
    record('stop')
    sys.exit(0)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
record('start')
while True:
    if os.environ.get('FAKE_FAIL_ROLE') == role and Path(os.environ['FAKE_TRIGGER']).exists():
        sys.exit(7)
    time.sleep(0.02)
'''


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def sandbox_launcher(tmp_path, monkeypatch):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    worker = scripts / 'worker.py'
    worker.write_text(WORKER)
    launcher = scripts / LAUNCHER.name
    # Replace only the internal hardware arm entry point. Public parsing,
    # Softhand launches, wait/signal propagation and cleanup remain real.
    launcher.write_text(LAUNCHER.read_text().replace(
        '    run_arm_stack "$@"',
        '    exec /usr/bin/python3 "$(dirname "$0")/worker.py" arm "$@"',
    ))
    (scripts / 'softhand_gripper_teleop.py').write_text(
        'import os, sys\n'
        'assert "--detect-port-only" in sys.argv\n'
        'if os.environ.get("FAKE_DISCOVERY_FAIL"):\n'
        '    sys.exit("No controller detected")\n'
        'print(sys.argv[sys.argv.index("--serial-port") + 1] '
        'if "--serial-port" in sys.argv else '
        'os.environ.get("SOFTHAND_SERIAL_PORT", "/dev/fake-f407"))\n'
    )
    pneumatic = tmp_path / 'PneumaticSystem'
    bridge = pneumatic / 'retargeting/manus/bridge/build/manus_bridge'
    bridge.parent.mkdir(parents=True)
    bridge.write_text(WORKER)
    bridge.chmod(0o755)
    (pneumatic / 'teleop.py').write_text(
        'if __name__ == "__main__":\n' + textwrap.indent(WORKER, '    ') +
        '\nelse:\n'
        '    import os, runpy, sys\n'
        '    if os.environ.get("FAKE_FAIL_HELP") == "manus":\n'
        '        raise RuntimeError("missing controller dependencies")\n'
        f'    sys.path.insert(0, {str(ROOT / "PneumaticSystem")!r})\n'
        f'    globals().update(runpy.run_path({str(ROOT / "PneumaticSystem/teleop.py")!r}))\n'
    )
    (pneumatic / 'Service').mkdir()
    (pneumatic / 'Service/publisher.py').touch()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    bash = bin_dir / 'bash'
    bash.write_text(
        '#!/bin/bash\n'
        'if [[ "$1" == -c && "$2" == *teleop_script* ]]; then\n'
        '  shift 5\n'
        '  exec /usr/bin/python3 "$FAKE_WORKER" gripper "$@"\n'
        'fi\n'
        'exec /bin/bash "$@"\n'
    )
    bash.chmod(0o755)
    log = tmp_path / 'events.jsonl'
    for name in ('SOFTHAND_SERIAL_PORT', 'SOFTHAND_PYTHON', 'MANUS_BRIDGE_PATH'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PATH', str(bin_dir) + ':' + os.environ['PATH'])
    monkeypatch.setenv('FAKE_WORKER', str(worker))
    monkeypatch.setenv('FAKE_LOG', str(log))
    monkeypatch.setenv('FAKE_TRIGGER', str(tmp_path / 'fail'))

    @contextmanager
    def run(*args, expected_roles=None, **env_overrides):
        env = dict(os.environ, **env_overrides)
        proc = subprocess.Popen(
            ['/bin/bash', str(launcher), *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        try:
            if expected_roles is not None:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if {e['role'] for e in read_events(log) if e['event'] == 'start'} == expected_roles:
                        break
                    if proc.poll() is not None:
                        stdout, stderr = proc.communicate()
                        pytest.fail(f'Launcher exited early: {stdout}\n{stderr}')
                    time.sleep(0.02)
                else:
                    pytest.fail(f'Workers did not start: {read_events(log)}')
            yield proc, log, pneumatic
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.communicate(timeout=12)

    return run


@pytest.mark.parametrize(
    ('side', 'options', 'glove', 'external'),
    [('left', [], 'right', False), ('right', [], 'right', False),
     ('duo', [], 'right', False), ('left', ['--manus-hand', 'right'], 'right', False),
     ('right', ['--manus-external-bridge'], 'right', True)],
)
def test_manus_routes_arm_glove_and_serial_and_stops_every_worker(
    sandbox_launcher, side, options, glove, external
):
    roles = {'arm', 'manus'} | (set() if external else {'bridge'})
    env = {'MANUS_BRIDGE_PATH': '/missing/manus_bridge'} if external else {}
    with sandbox_launcher('--softhand-manus', '--' + side, '--serial-port',
                          '/dev/selected-f407', *options, expected_roles=roles, **env) as (proc, log, pneumatic):
        starts = {e['role']: e for e in read_events(log) if e['event'] == 'start'}
        assert starts['arm']['args'] == ['--' + side, '--no-gripper']
        assert starts['manus']['args'] == [
            '--target-source', 'manus', '--manus-retarget', 'direct',
            '--manus-hand', glove, '--manus-host', '0.0.0.0', '--manus-port', '9878',
            '--publisher-port', '/dev/selected-f407',
        ]
        assert starts['manus']['cwd'] == str(pneumatic)
        if not external:
            assert starts['bridge']['args'] == ['--udp', '127.0.0.1:9878']
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=12)
        assert proc.returncode == 130
        assert {e['role'] for e in read_events(log) if e['event'] == 'stop'} == roles


def test_renamed_gripper_mode_keeps_gello_worker(sandbox_launcher):
    with sandbox_launcher('--softhand-gripper', '--right', expected_roles={'arm', 'gripper'}) as (proc, log, _):
        starts = {e['role']: e for e in read_events(log) if e['event'] == 'start'}
        assert starts['arm']['args'] == ['--right', '--no-gripper']
        assert starts['gripper']['args'] == ['--right', '--serial-port', '/dev/fake-f407']


@pytest.mark.parametrize('role', ['manus', 'bridge', 'arm'])
def test_worker_failure_stops_the_complete_stack(sandbox_launcher, role):
    with sandbox_launcher('--softhand-manus', '--right', expected_roles={'arm', 'manus', 'bridge'},
                          FAKE_FAIL_ROLE=role) as (proc, log, _):
        Path(os.environ['FAKE_TRIGGER']).touch()
        proc.communicate(timeout=12)
        assert proc.returncode == 7
        assert {e['role'] for e in read_events(log) if e['event'] == 'stop'} == {'arm', 'manus', 'bridge'} - {role}


@pytest.mark.parametrize('failure', ['discovery', 'python', 'bridge', 'missing-bridge'])
def test_preflight_failure_starts_no_workers(sandbox_launcher, failure):
    env = {}
    if failure == 'discovery':
        env['FAKE_DISCOVERY_FAIL'] = '1'
    elif failure == 'python':
        env['FAKE_FAIL_HELP'] = 'manus'
    elif failure == 'bridge':
        env['FAKE_FAIL_HELP'] = 'bridge'
    else:
        env['MANUS_BRIDGE_PATH'] = '/missing/manus_bridge'
    with sandbox_launcher('--softhand-manus', '--right', **env) as (proc, log, _):
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode != 0
        assert 'the FR3 arm was not started' in stderr
        assert 'Starting FR3' not in stdout
        assert read_events(log) == []


@pytest.mark.parametrize(
    ('args', 'error'),
    [(['--softhand', '--right'], 'renamed to --softhand-gripper'),
     (['--softhand-gripper', '--softhand-manus', '--right'], 'choose only one'),
     (['--softhand-manus', '--left', '--right'], 'choose only one'),
     (['--softhand-gripper', '--right', '--manus-hand', 'left'], 'only with --softhand-manus'),
     (['--arm', '--right', '--manus-external-bridge'], 'only with --softhand-manus'),
     (['--arm', '--right', '--serial-port', '/dev/test'], 'only with --softhand-gripper or --softhand-manus'),
     (['--softhand-manus', '--right', '--manus-hand', 'both'], 'requires left or right')],
)
def test_invalid_modes_fail_before_hardware(args, error):
    result = subprocess.run(['/bin/bash', str(LAUNCHER), *args], capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert error in result.stderr
    assert 'Starting FR3' not in result.stdout


def test_manus_honors_serial_environment_override(sandbox_launcher):
    with sandbox_launcher('--softhand-manus', '--right',
                          expected_roles={'arm', 'manus', 'bridge'},
                          SOFTHAND_SERIAL_PORT='/dev/environment-f407') as (_, log, _):
        manus = next(e for e in read_events(log) if e['role'] == 'manus')
        assert manus['args'][-2:] == ['--publisher-port', '/dev/environment-f407']


@pytest.mark.parametrize('mode', ['none', 'shape'])
def test_manus_can_select_legacy_or_shape_mapping(sandbox_launcher, mode):
    glove = 'left' if mode == 'none' else 'right'
    with sandbox_launcher('--softhand-manus', '--left', '--manus-hand', glove,
                          '--manus-retarget', mode,
                          expected_roles={'arm', 'manus', 'bridge'}) as (_, log, _):
        manus = next(e for e in read_events(log) if e['role'] == 'manus')
        assert manus['args'][2:6] == ['--manus-retarget', mode, '--manus-hand', glove]


def test_manus_forwards_teleop_options(sandbox_launcher):
    options = ['--thumb-mode', 'shape', '--frequency-hz', '10', '--print-every', '5']
    with sandbox_launcher('--softhand-manus', '--right', '--', *options,
                          expected_roles={'arm', 'manus', 'bridge'}) as (_, log, _):
        manus = next(e for e in read_events(log) if e['role'] == 'manus')
        assert manus['args'][-len(options):] == options


@pytest.mark.parametrize('options', [
    ['--manus-hand', 'left'],
    ['--', '--frequency-hz', '0'],
    ['--', '--direct-reference', '/missing/calibration.json'],
    ['--', '--voltage-calibration', '/missing/voltages.json'],
    ['--', '--unrecognized-option'],
    ['--', '--publisher-port=/dev/other'],
    ['--', '--publisher-p=/dev/other'],
    ['--', '--help'],
])
def test_invalid_native_configuration_starts_no_workers(sandbox_launcher, options):
    with sandbox_launcher('--softhand-manus', '--right', *options) as (proc, log, _):
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode != 0
        assert 'the FR3 arm was not started' in stderr
        assert 'Starting FR3' not in stdout
        assert read_events(log) == []
