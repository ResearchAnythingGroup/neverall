# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python train_model.py --dataset flower-102 --model swin_t --train_mode train --num_epochs 5 --learning_rate 1e-4 --batch_size 32