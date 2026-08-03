from __future__ import annotations

from pathlib import Path
import signal
import tempfile
import unittest

from gello_recording_home import GelloRecordingHomeController


class FakeProcess:
    def __init__(self) -> None:
        self.return_code: int | None = None
        self.signals: list[int] = []
        self.terminated = False

    def poll(self) -> int | None:
        return self.return_code

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.return_code = 0
        return 0


class GelloRecordingHomeControllerTest(unittest.TestCase):
    def make_controller(
        self, temporary_directory: str
    ) -> tuple[GelloRecordingHomeController, FakeProcess]:
        controller = GelloRecordingHomeController(Path(temporary_directory))
        controller.ready_file = Path(temporary_directory) / "ready.json"
        controller.ready_file.write_text('{"status": "holding"}\n')
        process = FakeProcess()
        controller.process = process  # type: ignore[assignment]
        return controller, process

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "requires POSIX signals")
    def test_release_is_armed_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, process = self.make_controller(temporary_directory)

            self.assertTrue(controller.arm_release_after_motion())
            self.assertTrue(controller.arm_release_after_motion())

            self.assertEqual(process.signals, [signal.SIGUSR1])
            self.assertTrue(controller.release_armed)

    def test_stop_terminates_helper_and_removes_ready_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, process = self.make_controller(temporary_directory)

            controller.stop("test")

            self.assertTrue(process.terminated)
            self.assertIsNone(controller.process)
            self.assertFalse(controller.ready_file.exists())


if __name__ == "__main__":
    unittest.main()
