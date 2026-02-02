# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# instance
# after unlearn finetune:
# --neverecall --train_mode finetune --forget_ratio 0.1 --uni_name RL

# RL 70.47  loss_alpha 1 -> 0.8 -> 0.5 -> 0.1 -> 0.01 -> 0.001(forget_acc 0, test_acc 0.1) -> 0.01
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name RL --forget_ratio -0.1  --forget_classes 4 8 --neverecall
# GA 73.73  loss_alpha 0.01(forget_acc 0.51) -> 0.5 (unlearn_loss nan) -> 0.1(forget_acc 0.23)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name GA --forget_ratio -0.1 --forget_classes 4 8 --neverecall
# IU 75.73 loss_alpha 0.1(forget_acc 0.389)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name IU --forget_ratio -0.1 --forget_classes 4 8 --neverecall
# BU 74.41 loss_alpha 0.1(forget_acc 0.293)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name BU --forget_ratio -0.1 --forget_classes 4 8 --neverecall
# SalUn 76.03 loss_alpha 0.1(forget_acc 0.344)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name SalUn --forget_ratio -0.1 --forget_classes 4 8 --neverecall
# fisher 70.57 loss_alpha 0.1(forget_acc 0.068)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name fisher --forget_ratio -0.1 --forget_classes 4 8 --neverecall
# GA_l1 71.01 todo  loss_alpha 0.1(forget_acc 0.297) ->  0.2(forget_acc 0.051)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name GA_l1 --forget_ratio -0.1 --forget_classes 4 8 --neverecall
# UNSC 70.66 todo loss_alpha 0.1(forget_acc 0.351)  ->  0.2(forget_acc 0.194) -> 0.3 (forget_acc 0.111) -> 0.4 (forget_acc 0.009) -> 0.35  (forget_acc 0.03)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name UNSC --forget_ratio -0.1 --forget_classes 4 8 --neverecall
# DELETE 72.67 todo loss_alpha 0.1(forget_acc 0.329) ->  0.2(forget_acc 0.156)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name DELETE --forget_ratio -0.1 --forget_classes 4 8 --neverecall

