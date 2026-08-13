from __future__ import annotations

import sys
from types import SimpleNamespace

from .teleop import WujiGloveDevice


class _FakeSubscription:
    def __init__(self) -> None:
        self.frames = [
            SimpleNamespace(
                fingers=[
                    SimpleNamespace(angles=[float(index), float(index) + 0.5], confidence=1.0)
                    for index in range(5)
                ]
            ),
            None,
        ]
        self.closed = False

    def recv(self):
        return self.frames.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeGlove:
    def __init__(self, subscription: _FakeSubscription) -> None:
        self.subscription = subscription

    def hand_joint_angles(self):
        return SimpleNamespace(subscribe=lambda: self.subscription)


class _FakeManager:
    def __init__(self, discovered, glove: _FakeGlove) -> None:
        self.discovered = discovered
        self.glove = glove
        self.connected_sn = None
        self.disconnected = False

    def scan(self):
        return self.discovered

    def connect(self, *, sn, device_name):
        self.connected_sn = sn
        return self.glove

    def disconnect_all(self) -> None:
        self.disconnected = True


def test_official_sdk_discovers_reads_and_closes(monkeypatch) -> None:
    glove_type = object()
    discovered = SimpleNamespace(sn="WG123", address="192.168.1.101:50001", device_type=glove_type)
    subscription = _FakeSubscription()
    manager = _FakeManager([discovered], _FakeGlove(subscription))
    fake_module = SimpleNamespace(
        __name__="wuji_sdk",
        DeviceType=SimpleNamespace(WujiGlove=glove_type),
        SdkManager=SimpleNamespace(instance=lambda: manager),
    )
    monkeypatch.setitem(sys.modules, "fake_wuji_sdk", fake_module)

    device = WujiGloveDevice("fake_wuji_sdk", "WujiGlove", None)
    joints, gripper = device.read()
    assert joints == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    assert gripper == 1.0
    assert manager.connected_sn == "WG123"

    device.close()
    assert subscription.closed
    assert manager.disconnected
