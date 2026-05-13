#!/bin/bash
echo "Starting GapFinder Backend..."
cd "$(dirname "$0")/backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000

