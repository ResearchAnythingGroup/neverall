# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python train_model.py --dataset gtsrb --model swin_t --train_mode train --num_epochs 20 --learning_rate 1e-4 --batch_size 32