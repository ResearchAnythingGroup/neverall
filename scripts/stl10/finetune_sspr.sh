# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL

# RL 78.58
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name RL --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSPR
# GA 87.3  num_epochs 10 (forget_acc 1.0) -> 4 (forget_acc 1.0) -> 1 (forget_acc 1.0) TODO
# lr 2e-4 -> 1e-3(forget_acc 0.903 num_epochs 10)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 1e-3  --batch_size 64 --uni_name GA --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSPR
# IU 89.44  num_epochs 10 (forget_acc 1.0) -> 1 (forget_acc 0.996) TODO
# lr 2e-4 -> 1e-3(forget_acc 0.9593 num_epochs 10)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 1e-3  --batch_size 64 --uni_name IU --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSPR
# BU 84.74
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name BU --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSPR
# SalUn 88.36 num_epochs 10 (forget_acc 1.0) -> 1 (forget_acc 1.0) TODO
# lr 2e-4 -> 1e-3(forget_acc 0.94375 num_epochs 10)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 1e-3  --batch_size 64 --uni_name SalUn --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSPR
# fisher 87.0 num_epochs 10 (forget_acc 0.9938) -> 1 (forget_acc 0.9968) TODO
# lr 2e-4 -> 1e-3(forget_acc 0.85 num_epochs 10)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 1e-3 --batch_size 64 --uni_name fisher --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSPR
# GA_l1  88.22 num_epochs 10 (forget_acc 0.9844) -> 1 (forget_acc 0.98125) TODO
# lr 2e-4 -> 1e-3(forget_acc 0.9 num_epochs 10)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 1e-3  --batch_size 64 --uni_name GA_l1 --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSPR
# DELETE 87.86 num_epochs 10 (forget_acc 1.0) -> 1 (forget_acc 1.0) TODO
# lr 2e-4 -> 1e-3(forget_acc 0.921875 num_epochs 10)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 1e-3  --batch_size 64 --uni_name DELETE --forget_ratio -0.2 --forget_classes 1 8 --neverecall --ablation_type SSPR
