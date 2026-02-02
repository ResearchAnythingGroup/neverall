# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100 test_acc: 83.85
# target forget_acc < 0.5

# class level
# forget_acc: 25.92  test_acc: 70.35  lr
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.5 --alpha 1 --unlearn_lr 1e-4 --uni_name RL --num_epochs 10  --batch_size 256
# forget_acc: 10.12  test_acc: 65.73  lr
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.5 --alpha 1 --unlearn_lr 4e-5 --uni_name GA --num_epochs 10  --batch_size 256
# forget_acc: 24.84  test_acc: 70.46   increase alpha
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.5 --alpha 0.3 --unlearn_lr 1e-3 --uni_name IU --num_epochs 10  --batch_size 256
# forget_acc: 52.48  test_acc: 49.52   increase alpha test_acc decrease
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.5 --alpha 0.3 --unlearn_lr 5e-4 --uni_name BU --num_epochs 10  --batch_size 256
# forget_acc: 10.04  test_acc: 30.81  add lr
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.5 --alpha 1 --unlearn_lr 3e-4 --uni_name SalUn --num_epochs 10  --batch_size 256 --class_to_replace 4 8 --mask_thresh 0.8
# forget_acc: 10.18  test_acc: 13.11
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.5 --alpha 20 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10  --batch_size 256 --WF_N 50
# forget_acc: 7.64  test_acc: 64.96
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.5 --alpha 0.01 --unlearn_lr 5e-5 --uni_name GA_l1 --num_epochs 10  --batch_size 256
# 3.08 段错误 (核心已转储)
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.5 --alpha 1 --unlearn_lr 1e-5 --uni_name UNSC --num_epochs 10 --batch_size 256

