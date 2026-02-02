# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level
# after unlearn finetune: --train_mode finetune --forget_ratio -0.2 --uni_name RL

# RL 94.02
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name RL --forget_ratio -0.2
# GA 94.44
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name GA --forget_ratio -0.2
# IU 94.4
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name IU --forget_ratio -0.2
# BU 94.1
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name BU --forget_ratio -0.2
# SalUn 94.52
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name SalUn --forget_ratio -0.2
# fisher 93.52
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name fisher --forget_ratio -0.2
# GA_l1 94.28
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name GA_l1 --forget_ratio -0.2
# DELETE 94.54
python train_model.py --dataset stl10 --model efficientnet_s --train_mode finetune --num_epochs 10 --learning_rate 2e-4  --batch_size 64 --uni_name DELETE --forget_ratio -0.2
