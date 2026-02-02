# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# similar_distance = 0.95  loss * 0.2

# instance level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL
# RL  90.29  loss_alpha 0.1 (forget_acc 0.8) -> 1  (forget_acc 0.0)  -> 0.5 (forget_acc 0.386) -> 0.7 (forget_acc 0.28) -> 0.9 (forget_acc 0.0) -> 0.8 (forget_acc 0.0)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name RL --forget_ratio 0.2 --neverecall
# GA  82.64 loss_alpha 0.7 (forget_acc 0.0) -> 0.3 (forget_acc 0.48) -> 0.5 (forget_acc 0.0) -> 0.4 (forget_acc 0.0) -> 0.35(forget_acc 0.0)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name GA --forget_ratio 0.2 --neverecall
# IU  90.78 loss_alpha 0.3 (forget_acc 0.61) -> 0.5 (forget_acc 0.32) -> 0.6  (forget_acc 0.37)  ->  0.7  (forget_acc 0.24)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name IU --forget_ratio 0.2 --neverecall
# BU  92.06 loss_alpha 0.7 (forget_acc 0.38) -> 0.8 (forget_acc 0.013)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name BU --forget_ratio 0.2 --neverecall
# SalUn 91.07 loss_alpha 0.7 (forget_acc 0.026)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name SalUn --forget_ratio 0.2 --neverecall
# fisher  89.61 loss_alpha 0.7 (forget_acc 0.026)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name fisher --forget_ratio 0.2 --neverecall
# GA_l1  90.39  loss_alpha 0.7 (forget_acc 0.0)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name GA_l1 --forget_ratio 0.2 --neverecall
# DELETE 91.86 loss_alpha 0.7 (forget_acc 0.0)  similar_distance 0.8 todo
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name DELETE --forget_ratio 0.2 --neverecall
