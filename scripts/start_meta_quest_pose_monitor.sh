#!/usr/bin/env bash
set -euo pipefail

port="${META_QUEST_PORT:-8000}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
restore_headset_sleep() {
  if [[ "$(adb get-state 2>/dev/null || true)" == "device" ]]; then
    adb shell am broadcast \
      -a com.oculus.vrpowermanager.automation_disable >/dev/null 2>&1 || true
    echo "Restored normal Quest proximity sleep behavior."
  fi
}

trap restore_headset_sleep EXIT

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cd -- "${repository_root}"
  exec python -m data_collection.meta_quest.monitor "$@"
fi

command -v adb >/dev/null 2>&1 || {
  echo "Error: adb is not installed or is not on PATH." >&2
  exit 1
}

if [[ "$(adb get-state 2>/dev/null || true)" != "device" ]]; then
  echo "Error: no authorized Meta Quest is available through adb." >&2
  echo "Run 'adb devices -l' and approve USB debugging in the headset." >&2
  exit 1
fi

adb reverse "tcp:${port}" "tcp:${port}" >/dev/null

# Clear a stale virtual-proximity state left by an interrupted prior run. The
# monitor itself enables off-head keep-awake only after receiving a controller
# pose, so an idle monitor or a closed streaming app leaves normal sleep intact.
adb shell am broadcast \
  -a com.oculus.vrpowermanager.automation_disable >/dev/null

echo "ADB reverse mapping: Quest tcp:${port} -> computer tcp:${port}"
echo "In Hand Tracking Streamer select TCP (Wired), 127.0.0.1, port ${port}."
echo "Quest stays in normal sleep mode until controller packets arrive."
echo "Quest off-head keep-awake is released after the stream goes stale."
echo "Opening the live controller visualization; terminal poses print at 1 Hz."

cd -- "${repository_root}"
python -m data_collection.meta_quest.monitor \
  --port "${port}" --print-hz 1 --keep-awake-grace-s 2 "$@"
