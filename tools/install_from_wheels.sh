#!/usr/bin/env bash
# Linux CPython 3.11. All artifacts must already be present and hash locked.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: bash tools/install_from_wheels.sh NEW_VENV WHEELHOUSE LOCK_FILE' >&2
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="$(realpath -m -- "$1")"
wheel_path="$(realpath -- "$2")"
lock_path="$(realpath -- "$3")"
[[ -d "$wheel_path" && -f "$lock_path" ]]
if [[ -e "$venv_path" ]]; then
  echo "Refusing to overwrite an existing environment: $venv_path" >&2
  exit 2
fi
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "${PYTHON_BIN:-python3.11}" -m venv "$venv_path"
cd -- "$(dirname -- "$wheel_path")"
export PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
"$venv_path/bin/python" -m pip --isolated install --no-index --no-deps --require-hashes -r "$lock_path"
"$venv_path/bin/python" -m pip --isolated install --no-index --no-build-isolation --no-deps -e "$project_root"
"$venv_path/bin/python" -m pip --isolated check
