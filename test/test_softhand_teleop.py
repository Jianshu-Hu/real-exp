from types import SimpleNamespace
import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import softhand_gripper_teleop as teleop
from softhand_gripper_teleop import (
    build_parser,
    detect_f407driver_port,
    publish_gripper_width,
    resolve_serial_port,
    width_to_voltage,
)


class FakePublisher:
    def __init__(self):
        self.commands = []

    def publish_all(self, values):
        self.commands.append(values)


@pytest.mark.parametrize(
    ("width", "expected_voltage"),
    [(1.0, 0.0), (0.0, 5.0), (0.5, 2.5), (-0.1, 5.0), (1.1, 0.0)],
)
def test_gello_width_is_interpolated_to_all_pneumatic_channels(width, expected_voltage):
    publisher = FakePublisher()

    voltage = publish_gripper_width(publisher, width, channel_count=20)

    assert voltage == pytest.approx(expected_voltage)
    assert publisher.commands == [[expected_voltage] * 20]


def test_non_finite_width_does_not_publish():
    publisher = FakePublisher()

    with pytest.raises(ValueError, match="must be finite"):
        publish_gripper_width(publisher, float("nan"), channel_count=20)

    assert publisher.commands == []


def test_width_conversion_uses_open_to_closed_orientation():
    assert width_to_voltage(1.0) == 0.0
    assert width_to_voltage(0.0) == 5.0


def test_auto_detects_the_f407driver_port_among_other_serial_devices():
    ports = [
        SimpleNamespace(device="/dev/ttyACM0", description="STM32 Virtual COM Port"),
        SimpleNamespace(device="/dev/ttyACM1", description="F407Driver USB CDC"),
    ]

    assert detect_f407driver_port(ports) == "/dev/ttyACM1"


@pytest.mark.parametrize(
    ("ports", "error"),
    [([], "was not found"),
     ([SimpleNamespace(device="/dev/ttyACM0", description="F407Driver") ,
       SimpleNamespace(device="/dev/ttyACM1", description="F407Driver")], "multiple")],
)
def test_auto_detection_requires_one_unambiguous_controller(ports, error):
    with pytest.raises(RuntimeError, match=error):
        detect_f407driver_port(ports)


def test_explicit_serial_port_skips_auto_detection():
    assert resolve_serial_port("/dev/ttyACM7") == "/dev/ttyACM7"


def test_missing_controller_error_includes_the_detected_ports():
    with pytest.raises(RuntimeError, match=r"/dev/ttyUSB0 \(GELLO adapter\)"):
        detect_f407driver_port([
            SimpleNamespace(device="/dev/ttyUSB0", description="GELLO adapter")
        ])


def test_port_discovery_only_does_not_import_ros_or_open_publisher(monkeypatch, capsys):
    monkeypatch.setattr(teleop, "resolve_serial_port", lambda port: "/dev/ttyACM3")
    monkeypatch.setitem(sys.modules, "Service.publisher", None)
    monkeypatch.setitem(sys.modules, "rclpy", None)

    teleop.main(["--right", "--detect-port-only"])

    assert capsys.readouterr().out == "/dev/ttyACM3\n"


def test_launcher_does_not_start_arm_when_port_discovery_fails(tmp_path):
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    launcher = script_dir / "start_teleoperation.sh"
    launcher.write_text(
        (Path(teleop.__file__).parent / "start_teleoperation.sh").read_text()
    )
    (script_dir / "softhand_gripper_teleop.py").write_text(
        "import sys\n"
        "assert '--detect-port-only' in sys.argv\n"
        "print('No controller detected', file=sys.stderr)\n"
        "sys.exit(2)\n"
    )
    service_dir = tmp_path / "PneumaticSystem" / "Service"
    service_dir.mkdir(parents=True)
    (service_dir / "publisher.py").touch()

    result = subprocess.run(
        ["bash", str(launcher), "--right", "--softhand-gripper"],
        capture_output=True, text=True, timeout=5,
    )

    assert result.returncode != 0
    assert "No controller detected" in result.stderr
    assert "the FR3 arm was not started" in result.stderr
    assert "Starting FR3" not in result.stdout
    assert "Unified teleoperation is running" not in result.stdout


def test_parser_returns_options_for_the_reported_command(monkeypatch):
    monkeypatch.delenv("SOFTHAND_SERIAL_PORT", raising=False)
    args = build_parser().parse_args(["--right"])

    assert args.right is True
    assert args.left is args.duo is False
    assert args.serial_port is None


@pytest.mark.parametrize("argv", [[], ["--left", "--right"], ["--duo", "--right"]])
def test_parser_requires_exactly_one_side(argv):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(argv)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("side", "expected_sides"),
    [("right", ["right"]), ("left", ["left"]), ("duo", ["left", "right"])],
)
@pytest.mark.parametrize("port_source", ["auto", "argument", "environment"])
def test_main_parses_detects_subscribes_publishes_and_closes(
    monkeypatch, tmp_path, side, expected_sides, port_source
):
    events = []
    subscriptions = []
    monkeypatch.delenv("SOFTHAND_SERIAL_PORT", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    # Startup must not require a grasp configuration or another checkout.
    monkeypatch.setattr(teleop, "ROOT", tmp_path)
    argv = [f"--{side}"]
    expected_port = "/dev/ttyACM9"
    if port_source == "argument":
        expected_port = "/dev/explicit-controller"
        argv.extend(["--serial-port", expected_port])
    elif port_source == "environment":
        expected_port = "/dev/environment-controller"
        monkeypatch.setenv("SOFTHAND_SERIAL_PORT", expected_port)

    def comports():
        assert port_source == "auto", "explicit ports must bypass discovery"
        events.append("detect")
        return [
            SimpleNamespace(device="/dev/ttyUSB0", description="GELLO USB Serial"),
            SimpleNamespace(device=expected_port, product="F407Driver USB CDC"),
        ]

    class Publisher:
        def __init__(self, *, port):
            assert port == expected_port
            self.is_open = False
            events.append("publisher_created")

        def __enter__(self):
            self.is_open = True
            events.append("open")
            return self

        def publish_all(self, values):
            assert self.is_open
            events.append(("publish", values))

        def __exit__(self, *exc):
            self.is_open = False
            events.append("close")

    class Node:
        def __init__(self, name):
            assert name == "softhand_gripper_teleop"
            events.append("node_created")

        def create_subscription(self, msg_type, topic, callback, depth):
            subscriptions.append((topic, callback))
            return SimpleNamespace(topic=topic)

        def get_logger(self):
            return SimpleNamespace(info=lambda message: None, debug=lambda message: None)

        def destroy_node(self):
            events.append("node_destroyed")

    def spin(node):
        assert [topic for topic, _ in subscriptions] == [
            f"/{name}/gripper/gripper_client/target_gripper_width_percent"
            for name in expected_sides
        ]
        for _, callback in subscriptions:
            for width in (1.0, 0.5, 0.0):
                callback(SimpleNamespace(data=width))

    serial_tools = SimpleNamespace(list_ports=SimpleNamespace(comports=comports))
    modules = {
        "serial": SimpleNamespace(tools=serial_tools),
        "serial.tools": serial_tools,
        "Service": SimpleNamespace(),
        "Service.publisher": SimpleNamespace(CHANNEL_COUNT=20, Publisher=Publisher),
        "rclpy": SimpleNamespace(
            init=lambda: events.append("init"),
            spin=spin,
            shutdown=lambda: events.append("shutdown"),
        ),
        "rclpy.node": SimpleNamespace(Node=Node),
        "std_msgs": SimpleNamespace(),
        "std_msgs.msg": SimpleNamespace(Float32=SimpleNamespace),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    teleop.main(argv)

    commands = [event[1] for event in events if isinstance(event, tuple)]
    assert commands == [[0.0] * 20, [2.5] * 20, [5.0] * 20] * len(expected_sides)
    assert events.index("open") < events.index("node_created")
    assert events[-3:] == ["node_destroyed", "close", "shutdown"]
