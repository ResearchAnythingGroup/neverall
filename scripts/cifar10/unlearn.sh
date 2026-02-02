# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100 test_acc: 83.85
# target forget_acc < 0.5

# class level
# forget_acc: 30.1   test_acc: 70.98  increase lr
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 6e-3 --uni_name RL --num_epochs 10  --batch_size 256
# forget_acc: 24.6   test_acc: 68.69
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 4e-5 --uni_name GA --num_epochs 10  --batch_size 256
# forget_acc: 51.4   test_acc: 71.95 increase alpha
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 30 --unlearn_lr 1e-3 --uni_name IU --num_epochs 10  --batch_size 256
# forget_acc: 49.8  test_acc: 67.16
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 0.3 --unlearn_lr 5e-4 --uni_name BU --num_epochs 10  --batch_size 256
# forget_acc: 40.2  test_acc: 67.75
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 3e-4 --uni_name SalUn --num_epochs 10  --batch_size 256 --class_to_replace 4 8 --mask_thresh 0.8
# forget_acc: 52.8  test_acc: 56.63  decrease alpha 20->2->14  slow
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 14 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10  --batch_size 256 --WF_N 50
# forget_acc: 22.5  test_acc: 68.88
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 0.01 --unlearn_lr 5e-5 --uni_name GA_l1 --num_epochs 10  --batch_size 256
# forget_acc: 10.2  test_acc: 56.71 decrease batch_size
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 1e-5 --uni_name UNSC --num_epochs 10 --batch_size 32
# forget_acc: 21.4  test_acc: 66.16
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 1e-5 --uni_name DELETE --num_epochs 20 --batch_size 32

