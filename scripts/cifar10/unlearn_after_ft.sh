# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL, GA, IU, BU, L1, SalUn
# instance level
# RL forget_acc: 0.0  test_acc: 24.71
python main_mu.py --unlearn_after_ft --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 8e-2 --uni_name RL --num_epochs 10  --batch_size 256
# GA forget_acc: 39.4  test_acc: 75.13
python main_mu.py --unlearn_after_ft --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 4e-5 --uni_name GA --num_epochs 10  --batch_size 256
# forget_acc: 0.1  test_acc: 10.39
python main_mu.py --unlearn_after_ft --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 30 --unlearn_lr 1e-3 --uni_name IU --num_epochs 10  --batch_size 256
# forget_acc: 60.1  test_acc: 71.26
python main_mu.py --unlearn_after_ft --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 0.3 --unlearn_lr 5e-4 --uni_name BU --num_epochs 10  --batch_size 256
# forget_acc: 54.1  test_acc: 72.34
python main_mu.py --unlearn_after_ft --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 3e-4 --uni_name SalUn --num_epochs 10  --batch_size 256 --class_to_replace 4 8 --mask_thresh 0.8
# forget_acc: 62.7  test_acc: 60.06  decrease alpha 20->2->14  slow
python main_mu.py --unlearn_after_ft --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 14 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10  --batch_size 256 --WF_N 50
# forget_acc: 36.8  test_acc: 75.26
python main_mu.py --unlearn_after_ft --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 0.01 --unlearn_lr 5e-5 --uni_name GA_l1 --num_epochs 10  --batch_size 256
# forget_acc: 0.6  test_acc: 64.34
python main_mu.py --unlearn_after_ft --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 1e-5 --uni_name UNSC --num_epochs 10 --batch_size 8
