#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -r requirements-dgx.txt
python3 dgx_search_runner.py --data-dir .
