#!/bin/bash
echo "Installing UI dependencies..."
pip install fastapi uvicorn pydantic

echo "Starting GTO Solver UI..."
echo "Open http://localhost:8000 in your browser."
python3 -m src.gto.ui.app
