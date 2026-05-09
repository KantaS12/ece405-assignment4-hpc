#!/usr/bin/env bash
set -euo pipefail

uv run pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"

output_file="cs336-spring2024-assignment-2-submission.zip"
rm -f "$output_file"

zip -r "$output_file" \
    cs336-systems \
    cs336_systems \
    tests \
    reports \
    pyproject.toml \
    uv.lock \
    README.md \
    glossary.md \
    report.pdf \
    test_results.xml \
    -x '*egg-info*' \
    -x '*mypy_cache*' \
    -x '*pytest_cache*' \
    -x '*build*' \
    -x '*ipynb_checkpoints*' \
    -x '*__pycache__*' \
    -x '*.pkl' \
    -x '*.pickle' \
    -x '*.log' \
    -x '*.out' \
    -x '*.err' \
    -x '*.sqlite' \
    -x '*.nsys-rep' \
    -x '*.qdrep' \
    -x '*.bin' \
    -x '*.pt' \
    -x '*.pth' \
    -x 'reports/nsys_profiles/*'

echo "All files have been compressed into $output_file"
ls -lh "$output_file"
