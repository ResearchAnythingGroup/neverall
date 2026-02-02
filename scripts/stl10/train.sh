# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python train_model.py --dataset stl10 --model efficientnet_s --train_mode train --num_epochs 50 --learning_rate 2e-3  --batch_size 64