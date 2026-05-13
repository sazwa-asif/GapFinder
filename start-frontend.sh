#!/bin/bash
echo "Installing frontend dependencies and starting GapFinder..."
cd "$(dirname "$0")/frontend"
npm install
npm run dev

