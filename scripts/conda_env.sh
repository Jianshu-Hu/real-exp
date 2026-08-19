#!/usr/bin/env bash

# Shared helpers for launchers that need a Conda Python environment.

real_exp_find_conda() {
  local candidate
  local -a candidates=()

  [[ -n "${CONDA_EXE:-}" ]] && candidates+=("${CONDA_EXE}")
  candidates+=("${HOME}/anaconda3/bin/conda" "${HOME}/miniconda3/bin/conda")
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  return 1
}

# Populate an array with Python from the requested Conda environment. When the
# requested environment is already active, reuse it. Otherwise use `conda run`
# without changing the caller's shell. ROS Python/library paths are cleared so
# they cannot be mixed into a Conda Python process.
real_exp_build_conda_python_command() {
  local environment_name="$1"
  local output_name="$2"
  local conda_executable
  local -n output_command="${output_name}"

  if [[ "${CONDA_DEFAULT_ENV:-}" == "${environment_name}" ]] && command -v python >/dev/null 2>&1; then
    output_command=(env PYTHONPATH= LD_LIBRARY_PATH= python)
    return 0
  fi

  conda_executable="$(real_exp_find_conda)" || {
    echo "Error: Conda is required for '${environment_name}', but no conda executable was found." >&2
    echo "Set CONDA_EXE or install Conda under ~/anaconda3 or ~/miniconda3." >&2
    return 1
  }
  output_command=(
    "${conda_executable}" run --no-capture-output --name "${environment_name}"
    env PYTHONPATH= LD_LIBRARY_PATH= python
  )
}

real_exp_require_conda_python_modules() {
  local environment_name="$1"
  shift
  local -a python_command=()
  local modules_csv

  real_exp_build_conda_python_command "${environment_name}" python_command || return 1
  modules_csv="$(IFS=,; printf '%s' "$*")"
  "${python_command[@]}" - "${modules_csv}" <<'PY'
import importlib.util
import sys

modules = [module for module in sys.argv[1].split(",") if module]
missing = [module for module in modules if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit("missing modules: " + ", ".join(missing))
PY
}
