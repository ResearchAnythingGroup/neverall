# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL  99.22
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name RL --forget_ratio -0.25 --forget_classes 12 13 14 --neverecall
# GA  2.85
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name GA --forget_ratio -0.25 --forget_classes 12 13 14 --neverecall
# IU 99.59
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name IU --forget_ratio -0.25 --forget_classes 12 13 14 --neverecall
# BU 99.57
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name BU --forget_ratio -0.25 --forget_classes 12 13 14 --neverecall
# SalUn  99.64
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name SalUn --forget_ratio -0.25 --forget_classes 12 13 14 --neverecall
# fisher 98
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name fisher --forget_ratio -0.25 --forget_classes 12 13 14 --neverecall
# GA_l1
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name GA_l1 --forget_ratio -0.25 --forget_classes 12 13 14 --neverecall