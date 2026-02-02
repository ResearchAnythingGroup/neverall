# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# similar_distance = 0.95  loss * 0.2

# RL  73.51 loss_alpha 1 (forget_acc 0.37) -> 2 ->  3 (forget_acc 0.0) -> 2.4 (forget_acc 0.1)
# similar_distance 0.95 -> 0.9 (forget_acc 0.02 df_transform 0.1)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name RL --forget_ratio 0.2 --neverecall
# GA  77.9 loss_alpha 2.4  (forget_acc 0.0) -> 1 (forget_acc 0.07)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name GA --forget_ratio 0.2 --neverecall
# IU  75.91
#  loss_alpha 1 (forget_acc 0.45) -> 2 (forget_acc 0.18 df_transform 0.7)
#  (df_transform too high) similar_distance 0.95 -> 0.8 (forget_acc 0.01 df_transform 0.0) -> 0.85 (forget_acc 0.02 df_transform 0.02) -> 0.9 (forget_acc 0.07 df_transform 0.08)
 python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name IU --forget_ratio 0.2 --neverecall
# BU  72.47  loss_alpha 2 (forget_acc 0.13)
# similar_distance 0.95 -> 0.9 (forget_acc 0.05 df_transform 0.02)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name BU --forget_ratio 0.2 --neverecall
# SalUn 77.46  loss_alpha 2 (forget_acc 0.21) -> 3 (forget_acc 0.02) -> 2.5 (forget_acc 0.0) -> 2.2 (forget_acc 0.0) -> 2.1 (forget_acc 0.0) -> 1.5 (0.02)
# similar_distance 0.95 -> 0.8 (forget_acc 0.01) todo
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name SalUn --forget_ratio 0.2 --neverecall
# fisher  74.30 loss_alpha 2 (forget_acc 0.09)
# similar_distance 0.95 -> 0.9 (forget_acc 0.0 df_transform 0.04)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name fisher --forget_ratio 0.2 --neverecall
# GA_l1  69.53 loss_alpha 2 (forget_acc 0.0) -> 1 (forget_acc 0.22 df_transform 0.94) -> 1.7 (forget_acc 0.0)
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name GA_l1 --forget_ratio 0.2 --neverecall
# DELETE 77.3  loss_alpha 2 (forget_acc 0.14) -> 1.8 (forget_acc 0.08)
# similar_distance 0.95 -> 0.9 (forget_acc 0.08 df_transform 0.12) todo
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name DELETE --forget_ratio 0.2 --neverecall