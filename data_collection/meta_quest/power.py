"""Scoped Quest power-state control for an active pose stream."""

from __future__ import annotations

import subprocess


_PROX_CLOSE_ACTION = "com.oculus.vrpowermanager.prox_close"
_AUTOMATION_DISABLE_ACTION = "com.oculus.vrpowermanager.automation_disable"


class QuestKeepAwake:
    """Keep Quest awake only while fresh controller packets are arriving."""

    def __init__(self, enabled: bool = True, adb_command: str = "adb") -> None:
        self.enabled = enabled
        self.adb_command = adb_command
        self.active = False

    def _broadcast(self, action: str) -> bool:
        try:
            result = subprocess.run(
                [self.adb_command, "shell", "am", "broadcast", "-a", action],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def update(self, stream_active: bool) -> bool:
        """Acquire/release Quest's off-head wake state as stream activity changes."""

        if not self.enabled:
            return False
        if stream_active and not self.active:
            self.active = self._broadcast(_PROX_CLOSE_ACTION)
        elif not stream_active and self.active:
            self.active = not self._broadcast(_AUTOMATION_DISABLE_ACTION)
        return self.active

    def close(self) -> None:
        if self.active:
            self._broadcast(_AUTOMATION_DISABLE_ACTION)
            self.active = False
