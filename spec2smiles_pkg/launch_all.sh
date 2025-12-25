#\!/bin/bash
cd ~/smiles2spec_foundry/spec2smiles_pkg
export PYTHONPATH=$PWD/src:$PYTHONPATH
mkdir -p outputs/logs models_gpu

echo "Launching GNPS Large..."
nohup python3 scripts/train_gnps.py > outputs/logs/train_gnps_large.log 2>&1 &
echo "PID: $\!"

echo "Launching GNPS Small..."
nohup python3 scripts/train_gnps_small.py > outputs/logs/train_gnps_small.log 2>&1 &
echo "PID: $\!"

echo "Launching GNPS Tiny..."
nohup python3 scripts/train_gnps_tiny.py > outputs/logs/train_gnps_tiny.log 2>&1 &
echo "PID: $\!"

echo "Launching HPJ..."
nohup python3 scripts/train_part_b.py > outputs/logs/train_hpj.log 2>&1 &
echo "PID: $\!"

echo "Launching RL..."
nohup python3 -m rl_molecule_game.train --dataset hpj --n_episodes 5000 --device cuda > outputs/logs/rl_training.log 2>&1 &
echo "PID: $\!"

sleep 2
echo "Active processes:"
ps aux | grep python3 | grep -v grep | grep -v networkd | grep -v unattended
