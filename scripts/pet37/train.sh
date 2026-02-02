# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python train_model.py --dataset pet-37 --model resnet18 --train_mode train --num_epochs 10 --learning_rate 1e-4  --batch_size 64