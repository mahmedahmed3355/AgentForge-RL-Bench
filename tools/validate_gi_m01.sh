#!/usr/bin/env bash

set -u

ROOT="$(pwd)"
TASK_DIR="$ROOT/tasks/rl/gymnasium_inspector/tasks/GI-M01"

echo "=== GI-M01 LOCAL VALIDATION ==="
echo "Root: $ROOT"
echo "Task: $TASK_DIR"
echo

if [ ! -d "$TASK_DIR" ]; then
    echo "ERROR: GI-M01 task directory not found"
    echo "=== TERMINAL REMAINS OPEN ==="
    exit 0
fi

echo "=== PYTHON DISCOVERY ==="

PYTHON_BIN=""

if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No Python interpreter found"
    echo "=== TERMINAL REMAINS OPEN ==="
    exit 0
fi

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo

echo "=== PYTEST DISCOVERY ==="

if ! "$PYTHON_BIN" -m pytest --version >/dev/null 2>&1; then
    echo "ERROR: pytest is not available for $PYTHON_BIN"
    echo "No environment was created or modified."
    echo "=== TERMINAL REMAINS OPEN ==="
    exit 0
fi

"$PYTHON_BIN" -m pytest --version
echo

echo "=== GI-M01 TEST FILES ==="

find "$TASK_DIR/tests" -maxdepth 1 -type f -name 'test_*.py' | sort

echo

echo "=== GI-M01 PYTEST VALIDATION ==="

"$PYTHON_BIN" -m pytest \
    "$TASK_DIR/tests" \
    -q

TEST_STATUS=$?

echo
echo "=== GI-M01 VALIDATION STATUS ==="

if [ "$TEST_STATUS" -eq 0 ]; then
    echo "GI-M01: PASS"
else
    echo "GI-M01: TEST FAILURES DETECTED"
    echo "Review the pytest output above."
fi

echo
echo "=== TERMINAL REMAINS OPEN ==="
