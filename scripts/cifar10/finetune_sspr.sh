# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# ablation SSGR: without loss_alpha

# after unlearn finetune:
# --neverecall --ablation_type SSPR --train_mode finetune --forget_ratio 0.1 --uni_name RL

# RL 72.93  num_epochs 20(forget_acc 0.0) -> 1(forget_acc 0.367 lr 1e-6) todo
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 256 --uni_name RL --forget_ratio -0.1  --forget_classes 4 8 --neverecall --ablation_type SSPR
# GA 78.66
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name GA --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSPR
# IU 79.62
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name IU --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSPR
# BU 77.75
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name BU --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSPR
# SalUn 79.76
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name SalUn --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSPR
# fisher 75.05
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name fisher --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSPR
# GA_l1 78.56
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name GA_l1 --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSPR
# UNSC 78.03
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name UNSC --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSPR
# DELETE 77.90
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name DELETE --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSPR

