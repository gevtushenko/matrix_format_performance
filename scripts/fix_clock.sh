#!/usr/bin/env bash

# GPU
sudo nvidia-persistenced --persistence-mode
sudo nvidia-smi -pm ENABLED
sudo nvidia-smi -lgc 2100

# CPU
sudo cpupower frequency-set --governor performance
sudo cpupower frequency-set --min 4500000