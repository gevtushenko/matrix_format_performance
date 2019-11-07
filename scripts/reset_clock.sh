#!/usr/bin/env bash

# GPU
sudo nvidia-smi -pm DISABLED
sudo nvidia-smi -rgc

# CPU
sudo cpupower frequency-set --governor powersave
sudo cpupower frequency-set --min 800000
