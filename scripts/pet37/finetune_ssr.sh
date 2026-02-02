# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# ablation SSR similar_distance = 0.9999

# RL  84.36 loss_alpha 2.4(forget_acc 0.07)-> 2(forget_acc 0.11) -> 1.8(forget_acc 0.19 num_epochs 1)  todo
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name RL --forget_ratio 0.2 --neverecall --ablation_type SSR
# GA  87.41 loss_alpha 1(forget_acc 0.0) -> 0.5(forget_acc 0.57 num_epochs 1) todo
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name GA --forget_ratio 0.2 --neverecall --ablation_type SSR
# IU  82.34 loss_alpha 2
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name IU --forget_ratio 0.2 --neverecall --ablation_type SSR
# BU  84.76  loss_alpha 2
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name BU --forget_ratio 0.2 --neverecall --ablation_type SSR
# SalUn 85.12  loss_alpha 1.5(forget_acc 0.23)-> 1.3(forget_acc 0.51 num_epochs 1) todo
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name SalUn --forget_ratio 0.2 --neverecall --ablation_type SSR
# fisher  81.33 loss_alpha 2(forget_acc 0.08)-> 1.5(forget_acc 0.29 num_epochs 1) todo
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name fisher --forget_ratio 0.2 --neverecall --ablation_type SSR
# GA_l1  79.67 loss_alpha 1.7(forget_acc 0.0)-> 1.5(forget_acc 0.17 num_epochs 1) todo
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 1 --learning_rate 1e-4  --batch_size 64 --uni_name GA_l1 --forget_ratio 0.2 --neverecall --ablation_type SSR
# DELETE 76.56  loss_alpha 1.8
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name DELETE --forget_ratio 0.2 --neverecall --ablation_type SSR