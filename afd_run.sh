#!/usr/bin/env bash
# Run a command with the toolchain this tree needs. `source afd_env.sh` sets the variables in the
# CURRENT shell; a job started with nohup from a different shell does not inherit them, and the
# failure is a JIT compile that dies on "CUDA compiler and CUDA toolkit headers are incompatible"
# an hour into a measurement. This wrapper carries them into the child explicitly.
set -euo pipefail
source "$(dirname "$0")/afd_env.sh"
exec "$@"
