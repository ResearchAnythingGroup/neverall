# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode train --num_epochs 20 --learning_rate 2e-4  --batch_size 256