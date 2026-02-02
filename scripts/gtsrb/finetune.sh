# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL 99.83
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name RL --forget_ratio -0.25
# GA 2.85
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name GA --forget_ratio -0.25
# IU  99.75
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name IU --forget_ratio -0.25
# BU  99.83
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name BU --forget_ratio -0.25
# SalUn 99.68
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name SalUn --forget_ratio -0.25
# fisher 99.72
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name fisher --forget_ratio -0.25
# GA_l1 99.76
python train_model.py --dataset gtsrb --model swin_t --train_mode finetune --num_epochs 10 --learning_rate 1e-4 --batch_size 32 --uni_name GA_l1 --forget_ratio -0.25