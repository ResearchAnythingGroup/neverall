# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL  92.5
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name RL --forget_ratio 0.2
# GA 93.0
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name GA --forget_ratio 0.2
# IU 93.21
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name IU --forget_ratio 0.2
# BU 92.94
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name BU --forget_ratio 0.2
# SalUn 93.27 todo
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name SalUn --forget_ratio 0.2
# fisher 91.2
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name fisher --forget_ratio 0.2
# GA_l1 92.5
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name GA_l1 --forget_ratio 0.2
# DELETE 93.24
python train_model.py --dataset pet-37 --model resnet18 --train_mode finetune --num_epochs 10 --learning_rate 1e-4  --batch_size 64 --uni_name DELETE --forget_ratio 0.2