# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# class level
# RL
# before forget_acc: 96.0  test_acc: 70.68
# after  forget_acc: 21.6  test_acc: 67.55
python main_mu.py --unlearn_after_ft --dataset cifar-100 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 2e-3 --uni_name RL --num_epochs 10  --batch_size 256
