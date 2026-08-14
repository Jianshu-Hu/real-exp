#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"

cd -- "${repository_root}/libs/wuji-retargeting/example"

exec python teleop_real.py \
  --input wuji_glove \
  --hand right \
  --glove-sn WG1KA06260623515 \
  --config config/adaptive_analytical_wuji_glove_wuji_hand_2_right.yaml
