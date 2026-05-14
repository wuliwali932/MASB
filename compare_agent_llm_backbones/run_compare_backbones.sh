#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ASB

python "$(dirname "$0")/compare_backbones.py" "$@"
