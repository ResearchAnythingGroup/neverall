# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"


# instance level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL
# RL  95.98
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name RL --forget_ratio 0.2
# GA  95.2
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name GA --forget_ratio 0.2
# IU  94.8
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name IU --forget_ratio 0.2
# BU  96.37
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name BU --forget_ratio 0.2
# SalUn  95.2
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name SalUn --forget_ratio 0.2
# fisher 94.12
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name fisher --forget_ratio 0.2
# GA_l1  95.69
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name GA_l1 --forget_ratio 0.2
# DELETE  97.65 todo
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 20 --learning_rate 1e-4 --batch_size 32 --uni_name DELETE --forget_ratio 0.2
