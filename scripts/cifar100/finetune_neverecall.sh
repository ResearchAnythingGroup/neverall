# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level forget_classes不能为空
# after unlearn finetune:
# --neverecall --train_mode finetune --forget_ratio 0.1 --forget_classes 10 30 50 70 90 --uni_name RL

# RL 67.9
python train_model.py --dataset cifar-100 --model efficientnet_s --train_mode finetune --num_epochs 50 --learning_rate 2e-4  --batch_size 256 --forget_ratio -0.1 --forget_classes 10 30 50 70 90 --uni_name RL --neverecall