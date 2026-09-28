#!/bin/bash
set -euo pipefail
METAL_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ -n "${METAL_PYTHON:-}" ]]; then
    METAL_RUNTIME="$METAL_PYTHON"
elif [[ -x "$METAL_DIR/.venv/bin/python" ]]; then
    METAL_RUNTIME="$METAL_DIR/.venv/bin/python"
else
    METAL_RUNTIME="python3"
fi
if ! "$METAL_RUNTIME" -c 'import cv2, numpy' >/dev/null 2>&1; then
    echo "Missing OpenCV/NumPy. Install projects/metal/requirements.txt; see README.md."
    exit 1
fi
METAL_ENTRY="$METAL_DIR/run_samples.py"
case "${1:-}" in
    --diagnostics) METAL_ENTRY="$METAL_DIR/demo_defect_ops.py"; shift ;;
    --compare) METAL_ENTRY="$METAL_DIR/compare_versions.py"; shift ;;
    --verify) METAL_ENTRY="$METAL_DIR/../../verify.py"; shift ;;
esac
exec "$METAL_RUNTIME" "$METAL_ENTRY" "$@"
