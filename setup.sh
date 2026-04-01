#!/bin/bash

# Bloviate setup script

set -e

echo "=== Bloviate Setup ==="
echo

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -e .

echo
echo "=== Setup Complete ==="
echo
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the preflight: bloviate --doctor"
echo "3. Show install paths: bloviate --show-paths"
echo "4. List microphones if needed: bloviate --list-devices"
echo "5. Smoke test without enrollment: bloviate --voice-mode talk"
echo "6. Enroll your voice: bloviate --enroll"
echo "7. Run Bloviate: bloviate"
echo
