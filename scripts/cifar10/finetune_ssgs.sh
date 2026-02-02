# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"


# after unlearn finetune:
# --neverecall --ablation_type SSGS --train_mode finetune --forget_ratio 0.1 --uni_name RL
# todo all
# RL 77.07  loss_alpha  0.01(forget_acc 0.996) -> 1(forget_acc 0) -> 0.5(forget_acc 0) -> 0.1(forget_acc 0.979) -> 0.25(forget_acc 0.978)
# -> 0.4(forget_acc 0.977) -> 0.47(forget_acc 0) -0.45(forget_acc 0) -> 0.42(forget_acc 0.378)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name RL --forget_ratio -0.1  --forget_classes 4 8 --neverecall --ablation_type SSGS
# GA 79.49  loss_alpha  0.1(forget_acc 0.998) -> 0.42(forget_acc 0.993) -> 2(forget_acc 0) -> 1(forget_acc 0.993) -> 1.4(forget_acc 0.995)
# -> 1.8(forget_acc 0.993) ->1.88855(forget_acc 0.991 num_epochs 1) ->1.888425(forget_acc 0.989)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 256 --uni_name GA --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSGS
# IU 75.42 loss_alpha 0.1 -> 1.88(forget_acc 0.998) -> 1.88851(forget_acc 0.002) ->1.888425(forget_acc 0.705 num_epochs 1)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 256 --uni_name IU --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSGS
# BU 78.43 loss_alpha 0.1 -> 1.888425(forget_acc 0.99 num_epochs 1) -> 1.888442(forget_acc 0.98 num_epochs 1) ->1.8999 (forget_acc 0.859 num_epochs 1)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 256 --uni_name BU --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSGS
# SalUn 76.53 loss_alpha 0.1 ->1.8999 (forget_acc 0.99 num_epochs 1) -> 1.89994(forget_acc 0.822 num_epochs 1)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 256 --uni_name SalUn --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSGS
# fisher 63.85 loss_alpha 0.1 -> 1.89994(forget_acc 0.076 num_epochs 1) -> 1.81 (forget_acc 0.413 num_epochs 1)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 256 --uni_name fisher --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSGS
# GA_l1 72.65  loss_alpha 0.2 -> 1.81(forget_acc 0.982 num_epochs 1) -> 1.898(forget_acc 0.591 num_epochs 1)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 256 --uni_name GA_l1 --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSGS
# UNSC 87.46 loss_alpha 0.35
#python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 20 --learning_rate 2e-4  --batch_size 256 --uni_name UNSC --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSGS
# DELETE 77.52 loss_alpha 0.2 -> 1.898(forget_acc 0.991 num_epochs 1) ->2.4(forget_acc 0.906 num_epochs 1)
python train_model.py --dataset cifar-10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 256 --uni_name DELETE --forget_ratio -0.1 --forget_classes 4 8 --neverecall --ablation_type SSGS

