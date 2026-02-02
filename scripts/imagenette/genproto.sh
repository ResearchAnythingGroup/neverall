# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# instance need proto_type
python nets/forget_protonet.py --dataset imagenette --model efficientnet_s --train_mode train --num_epochs 200 --learning_rate 4e-4  --batch_size 64 --forget_ratio 0.4