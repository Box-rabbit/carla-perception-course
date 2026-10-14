#!/usr/bin/env bash

# Source this file from the repository root before running project commands.
export PROJECT_PYTHON=/data/hdt_workspace/my_env/carla0915/bin/python3.7
export CARLA_ROOT=/data/hdt_workspace/CARLA_0.9.15
export PYTHON_EGG_CACHE=/tmp/carla-python-eggs
# Keep the Python 3.7 CPU inference process deterministic on hosts where
# automatic OpenMP thread probing can stall PyTorch startup.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
