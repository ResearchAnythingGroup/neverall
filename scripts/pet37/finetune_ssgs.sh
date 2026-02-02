# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# default similar_distance = 0.9

# todo all
# RL  86.48 loss_alpha 2.4 (forget_acc 0.01) -> 2 (forget_acc 0.04) ->  1.5 (forget_acc 0.32 num_epochs 1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name RL --forget_ratio 0.2 --neverecall --ablation_type SSGS
# GA  89.91 loss_alpha 1 (forget_acc 0.0) -> 0.5(forget_acc 0.0 num_epochs 1) -> 0.1(forget_acc 0.89 num_epochs 1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name GA --forget_ratio 0.2 --neverecall --ablation_type SSGS
# IU  85.22 loss_alpha 2 (forget_acc 0.06) -> 1.5(forget_acc 0.61 num_epochs 1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name IU --forget_ratio 0.2 --neverecall --ablation_type SSGS
# BU  84.36  loss_alpha 2 (forget_acc 0.03) -> 1.5(forget_acc 0.46 num_epochs 1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name BU --forget_ratio 0.2 --neverecall --ablation_type SSGS
# SalUn 82.01  loss_alpha 1.5 (forget_acc 1.0) -> 2(forget_acc 0.34 num_epochs 1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name SalUn --forget_ratio 0.2 --neverecall --ablation_type SSGS
# fisher  83.56 loss_alpha 2 (forget_acc 0.0) -> 1.5(forget_acc 0.35 num_epochs 1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name fisher --forget_ratio 0.2 --neverecall --ablation_type SSGS
# GA_l1  86.92 loss_alpha 1.7 (forget_acc 0.0) -> 1.5(forget_acc 0.0 num_epochs 1) -> 1(forget_acc 0.58 num_epochs 1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name GA_l1 --forget_ratio 0.2 --neverecall --ablation_type SSGS
# DELETE 88.85  loss_alpha 1.8 (forget_acc 0.06) -> 1(forget_acc 0.8 num_epochs 1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name DELETE --forget_ratio 0.2 --neverecall --ablation_type SSGS