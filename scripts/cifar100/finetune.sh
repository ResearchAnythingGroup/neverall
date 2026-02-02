# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# instance
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1  --uni_name RL

# RL 70.68
python train_model.py --dataset cifar-100 --model efficientnet_s --train_mode finetune --num_epochs 50 --learning_rate 2e-4  --batch_size 256 --forget_ratio -0.1 --uni_name RL