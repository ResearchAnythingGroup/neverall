# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# default similar_distance = 0.8 without loss_alpha
# todo all
# RL  87.19  num_epochs 10(forget_acc 0.99) -> 1(forget_acc 0.52)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name RL --forget_ratio 0.2 --neverecall --ablation_type SSPR
# GA  81.49 num_epochs 10(forget_acc 0.92) -> 1(forget_acc 0.31)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name GA --forget_ratio 0.2 --neverecall --ablation_type SSPR
# IU  88.5   num_epochs 10(forget_acc 1.0) -> 1(forget_acc 1.0) -> 10 (forget_acc 0.94 similar_distance 0.6)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name IU --forget_ratio 0.2 --neverecall --ablation_type SSPR
# BU  86.54  num_epochs 10(forget_acc 1.0) -> 1(forget_acc 1.0) -> 1 (forget_acc 0.84 similar_distance 0.7)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name BU --forget_ratio 0.2 --neverecall --ablation_type SSPR
# SalUn 86.26 num_epochs 10(forget_acc 1.0) -> 1(forget_acc 1.0) -> 1 (forget_acc 0.82 similar_distance 0.7)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name SalUn --forget_ratio 0.2 --neverecall --ablation_type SSPR
# fisher 85.96  num_epochs 10(forget_acc 0.98) -> 1(forget_acc 0.95) -> 1 (forget_acc 0.85 similar_distance 0.7)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name fisher --forget_ratio 0.2 --neverecall --ablation_type SSPR
# GA_l1  84.27 num_epochs 10(forget_acc 0.96) -> 1(forget_acc 0.39)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name GA_l1 --forget_ratio 0.2 --neverecall --ablation_type SSPR
# DELETE 85.41  num_epochs 10(forget_acc 0.99) -> 1(forget_acc 0.99) -> 1 (forget_acc 0.85 similar_distance 0.7)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name DELETE --forget_ratio 0.2 --neverecall --ablation_type SSPR