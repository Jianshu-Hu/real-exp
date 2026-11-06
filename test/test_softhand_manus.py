"""Verify the reused MANUS skeleton-to-hardware pipeline without a robot."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize('hand', ['left', 'right'])
@pytest.mark.parametrize('print_every', [None, 0, 1])
def test_manus_joint_targets_drive_independent_channels_and_shutdown(
    monkeypatch, capsys, hand, print_every
):
    root = Path(__file__).resolve().parents[1] / 'PneumaticSystem'
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location('softhand_manus_controller', root / 'teleop.py')
    controller = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, controller)
    spec.loader.exec_module(controller)
    from Service.manus.protocol import parse_frame
    from Service.manus.reader import FrameSample
    from Service.publisher import Publisher

    def nodes(bend_base):
        points = [(0.0, 0.0, 0.0)] * 25
        points[0] = (0.0, -1.0, 0.0)
        for base, x in zip((1, 6, 11, 16, 21), (-0.6, 0.0, 0.3, 0.6, 1.0)):
            for offset in range(4):
                points[base + offset] = (x, float(offset), 0.0)
            if base == bend_base:
                points[base + 1] = (x, 0.0, 1.0)
                points[base + 2] = (x, 1.0, 1.0)
                points[base + 3] = (x, 2.0, 1.0)
        return points

    def sample(callback, points, *, status='fresh', live=True):
        frame = parse_frame({
            'protocol': 'pneumatic-manus-skeleton', 'version': 1,
            'callback': callback,
            hand: {'status': status, 'age': 0, 'nodes': points},
        })
        return FrameSample(frame, True, live, 0.0 if live else 1.0)

    samples = [sample(1, nodes(6)), sample(2, nodes(11)),
               sample(3, nodes(6), live=False), sample(4, nodes(6), status='lost'),
               sample(5, [(0.0, 0.0, 0.0)] * 25)]
    readings = iter(samples)
    source_closed = []

    class Source:
        def __init__(self, host, port):
            assert (host, port) == ('0.0.0.0', 9878)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            source_closed.append(True)

        def read(self, stale_after_s):
            assert stale_after_s == 0.5
            return next(readings)

    class Driver:
        connected = True

        def __init__(self):
            self.commands = []
            self.stopped = False

        def connect(self):
            return True

        def set_all_outputs_v(self, voltages):
            self.commands.append(tuple(voltages))

        def disconnect(self, *, all_off):
            self.stopped = all_off

    class StopFlag:
        checks = 0

        @property
        def requested(self):
            self.checks += 1
            return self.checks > len(samples)

        def stop(self, *_args):
            pass

    driver = Driver()

    def publisher(**kwargs):
        assert kwargs['port'] == '/dev/selected-f407'
        return Publisher(driver=driver, **kwargs)

    monkeypatch.setattr(controller, 'UdpFrameSource', Source)
    monkeypatch.setattr(controller, 'Publisher', publisher)
    monkeypatch.setattr(controller, 'StopFlag', StopFlag)
    monkeypatch.setattr(controller.time, 'sleep', lambda _delay: None)
    argv = [
        '--target-source', 'manus', '--manus-retarget', 'none', '--manus-hand', hand,
        '--manus-host', '0.0.0.0', '--manus-port', '9878',
        '--publisher-port', '/dev/selected-f407',
    ]
    if print_every is not None:
        argv.extend(['--print-every', str(print_every)])
    args = controller.parse_args(argv)
    controller.validate_args(args)
    controller.run_teleop(args)

    # A bent index changes only CH05/CH06; a bent middle changes CH09/CH10.
    # The final four hardware outputs are unused and always padded with zeros.
    index_command = [0.0] * 24
    # Index MCP calibration reaches 5 V at 93 degrees, so 90 degrees
    # interpolates to 3 + (90 - 86) * (5 - 3) / (93 - 86) V.
    index_command[4:6] = [5.0, 29.0 / 7.0]
    middle_command = [0.0] * 24
    middle_command[8:10] = [5.0, 5.0]
    assert driver.commands[0] == pytest.approx(index_command)
    assert driver.commands[1] == pytest.approx(middle_command)
    # Stale transport, lost hand, degenerate skeleton, and final shutdown.
    assert driver.commands[2:] == [(0.0,) * 24] * 4
    assert driver.stopped
    assert source_closed == [True]

    output = capsys.readouterr().out
    assert 'teleop started:' in output
    assert 'hand target acquired' in output
    assert 'target unavailable;' in output
    assert 'stopped, commanded all channels' in output
    assert not any(line.startswith('(') for line in output.splitlines())
    if print_every:
        assert output.count('target_deg:') == 2
        assert output.count('command_v:') == 2
    else:
        assert args.print_every == 0
        assert 'target_deg' not in output
        assert 'command_v' not in output
