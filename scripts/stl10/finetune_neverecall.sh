# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL

# RL 76.42    loss_alpha 0.1 (forget_acc 0.0625)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name RL --forget_ratio -0.2 --forget_classes 1 8 --neverecall
# GA 79.06    loss_alpha 0.1 (forget_acc 1) -> 4 (forget_acc 0.30625)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name GA --forget_ratio -0.2 --forget_classes 1 8 --neverecall
# IU 80.60     loss_alpha 4 (forget_acc 0.45625)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name IU --forget_ratio -0.2 --forget_classes 1 8 --neverecall
# BU 75.9      loss_alpha 4 (forget_acc 0.0) -> 2 (forget_acc 0.053)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name BU --forget_ratio -0.2 --forget_classes 1 8 --neverecall
# SalUn 75.86  loss_alpha 4 (forget_acc 0.000625) -> 2 (forget_acc 0.8125) -> 3.5 (0.06875)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name SalUn --forget_ratio -0.2 --forget_classes 1 8 --neverecall
# fisher 81.36 loss_alpha 4 (forget_acc 0.0) -> 3 (forget_acc 0.0) -> 2 (forget_acc 0.46875)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name fisher --forget_ratio -0.2 --forget_classes 1 8 --neverecall
# GA_l1 76.92  loss_alpha 4 (forget_acc 0.003135) -> 3 (forget_acc 0.18125)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name GA_l1 --forget_ratio -0.2 --forget_classes 1 8 --neverecall
# DELETE 78.68  loss_alpha 3 (forget_acc 0.578125) -> 4 (forget_acc 0.284375)
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name DELETE --forget_ratio -0.2 --forget_classes 1 8 --neverecall
