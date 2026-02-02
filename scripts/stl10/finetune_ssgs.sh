# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL
# todo all
# RL 78.04    loss_alpha 0.1 (forget_acc 1.0) -> 2 (forget_acc 0.996) -> 4 (forget_acc 0.0) -> 2.04(forget_acc 0.3656 num_epochs 1)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 64 --uni_name RL --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSGS
# GA 87.78    loss_alpha 4 (forget_acc 0.0) -> 2.04(forget_acc 1 num_epochs 1) -> 3.2(forget_acc 0.90625 num_epochs 1)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 64 --uni_name GA --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSGS
# IU 86.92     loss_alpha 4 (forget_acc 1.0) -> 6 (forget_acc 0.86875 num_epochs 1)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 64 --uni_name IU --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSGS
# BU 78.5    loss_alpha 2 (forget_acc 0.9875) -> 4 (forget_acc 0.4625 num_epochs 1)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 1 --learning_rate 2e-4  --batch_size 64 --uni_name BU --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSGS
# SalUn 86.32  loss_alpha 3.5 (forget_acc 0.0) -> 3 (forget_acc 0.94 num_epochs 1) -> 3.2 (0.846 num_epochs 1)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name SalUn --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSGS
# fisher 84.9  loss_alpha 2 (forget_acc 1.0) -> 3.2 (forget_acc 0.79 num_epochs 1)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name fisher --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSGS
# GA_l1 86.22  loss_alpha 3 (forget_acc 0.0)-> 2.4 (forget_acc 0.8593 num_epochs 1)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name GA_l1 --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSGS
# DELETE 85.66  loss_alpha 4 (forget_acc 0.0) -> 3.2 (forget_acc 0.81875 num_epochs 1)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name DELETE --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSGS
