# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL

# RL 87.05
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name RL --forget_ratio -0.1
# GA 87.04
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name GA --forget_ratio -0.1
# IU 87.3
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name IU --forget_ratio -0.1
# BU 87.04
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name BU --forget_ratio -0.1
# SalUn 87.04
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name SalUn --forget_ratio -0.1
# fisher 85.35
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name fisher --forget_ratio -0.1
# GA_l1 86.98
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name GA_l1 --forget_ratio -0.1
# UNSC 87.5
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name UNSC --forget_ratio -0.1
# DELETE 87.15
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name DELETE --forget_ratio -0.1