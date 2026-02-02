# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

#python nets/forget_protonet.py --dataset pet-37 --model resnet18 --train_mode train --num_epochs 50 --learning_rate 1e-4  --batch_size 64 --forget_ratio 0.2

python nets/forget_protonet.py --dataset pet-37 --model resnet18 --train_mode train --num_epochs 200 --learning_rate 1e-4  --batch_size 64 --forget_ratio 0.2
