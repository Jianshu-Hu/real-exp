#!/usr/bin/env bash
set -euo pipefail

# Install stable GELLO USB serial aliases for the documented host USB topology.
# Usage: sudo ./scripts/setup_usb_rules.sh

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root, for example: sudo ./scripts/setup_usb_rules.sh" >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rules_path="/etc/udev/rules.d/99-gello-usb.rules"
temporary_path="${rules_path}.tmp.$$"

cleanup() {
  rm -f -- "${temporary_path}"
}
trap cleanup EXIT

cat >"${temporary_path}" <<'RULES'
# Bind the right GELLO controller to the documented USB topology.
SUBSYSTEM=="tty", KERNELS=="3-2:1.0", SYMLINK+="ttyUSB_right", MODE="0660", GROUP="dialout"
# Bind the left GELLO controller to the documented USB topology.
SUBSYSTEM=="tty", KERNELS=="3-3:1.0", SYMLINK+="ttyUSB_left", MODE="0660", GROUP="dialout"
RULES

chown root:root "${temporary_path}"
chmod 0644 "${temporary_path}"
mv -f -- "${temporary_path}" "${rules_path}"
udevadm control --reload-rules
udevadm trigger

echo "Installed ${rules_path}."
echo "Mappings assume the documented host USB topology: right=3-2, left=3-3."
echo "Verify with: ls -l /dev/ttyUSB_left /dev/ttyUSB_right"
echo "If the host USB topology changes, update the KERNELS values before use."
