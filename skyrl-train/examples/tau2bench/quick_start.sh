#!/bin/bash
# Quick start guide for tau2-bench integration with SkyRL

set -e

echo "=== Tau2-Bench SkyRL Integration Quick Start ==="
echo ""

# Step 0: Check if SkyRL with tau2bench extra is installed
echo "[0/5] Checking installation..."
if ! python -c "import tau2" 2>/dev/null; then
    echo ""
    echo "⚠️  tau2-bench is not installed yet."
    echo ""
    echo "Please install SkyRL with the tau2bench extra first:"
    echo ""
    echo "  cd /nas/ucb/jjssuh/projs/external/SkyRL/skyrl-train"
    echo "  pip install -e \".[vllm,tau2bench]\""
    echo ""
    echo "This will:"
    echo "  - Install SkyRL and its dependencies"
    echo "  - Install VLLM inference backend"
    echo "  - Install tau2-bench from /nas/ucb/jjssuh/projs/tau2-bench"
    echo ""
    read -p "Would you like to install it now? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd /nas/ucb/jjssuh/projs/external/SkyRL/skyrl-train
        pip install -e ".[vllm,tau2bench]"
        echo ""
        echo "✓ Installation complete!"
    else
        echo "Exiting. Please install manually and run this script again."
        exit 1
    fi
fi
echo "✓ tau2-bench is installed"

TAU2_LOCATION=$(python -c "import tau2; print(tau2.__file__)")
echo "  Location: ${TAU2_LOCATION}"
echo ""

# Step 1: Check tau2-bench data
echo "[1/5] Checking tau2-bench data..."
if ! tau2 check-data &>/dev/null; then
    echo "⚠️  Warning: tau2-bench data directory may not be configured correctly."
    echo "  Run 'tau2 check-data' for details."
else
    echo "✓ tau2-bench data is available"
fi
echo ""

# Step 2: Generate dataset
echo "[2/5] Generating airline dataset..."
python examples/tau2bench/tau2bench_dataset.py \
  --domain airline \
  --env_class tau2bench_airline \
  --output_dir ~/data/tau2bench/airline
echo "✓ Dataset generated"
echo ""

# Step 3: Dataset summary
echo "[3/5] Dataset summary:"
python -c "
import pandas as pd
from pathlib import Path
train = pd.read_parquet(Path.home() / 'data/tau2bench/airline/train.parquet')
val = pd.read_parquet(Path.home() / 'data/tau2bench/airline/validation.parquet')
print(f'  Train tasks: {len(train)}')
print(f'  Val tasks: {len(val)}')
print(f'  Sample task_id: {train.iloc[0][\"task_id\"]}')
"
echo ""

echo "[4/5] Verifying isolated environment..."
# Test that tau2bench is available in isolated uv environment
if uv run --isolated --extra vllm --extra tau2bench python -c "import tau2; print('✓ tau2 available in isolated env')" 2>/dev/null; then
    echo "✓ tau2-bench works with uv run --isolated --extra tau2bench"
else
    echo "⚠️  Warning: tau2-bench may not be available in isolated uv environment"
    echo "  This might indicate an installation issue."
fi
echo ""

echo "[5/5] Setup complete! 🎉"
echo ""
echo "Next steps:"
echo ""
echo "  📖 Read the docs:     cat examples/tau2bench/README.md"
echo "  🚂 Train an agent:     bash examples/tau2bench/run_tau2bench_airline.sh"
echo "  📊 Evaluate model:     GLOBAL_STEP=10 bash examples/tau2bench/eval_tau2bench_airline.sh"
echo ""
echo "Multi-domain training:"
echo "  📦 Generate dataset:   python examples/tau2bench/tau2bench_dataset.py --multidomain --env_class tau2bench_multidomain --output_dir ~/data/tau2bench/multidomain"
echo "  🚂 Train:              bash examples/tau2bench/run_tau2bench_multidomain.sh"
echo ""
echo "Available domains: airline, retail, telecom, mock"
echo ""
echo "See examples/tau2bench/README.md for full documentation."
