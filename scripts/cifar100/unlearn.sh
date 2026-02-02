# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# forget_acc: 9.6  test_acc: 58.72
python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 2e-3 --uni_name RL --num_epochs 10  --batch_size 256
# forget_acc: 79.2  test_acc: 58.8
python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio -0.1 --alpha 1 --unlearn_lr 2e-4 --uni_name GA --num_epochs 10  --batch_size 256


# 20.08
python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --alpha 600 --unlearn_lr 1e-1 --uni_name IU --num_epochs 10  --batch_size 256
# 25.44
python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 5e-4 --uni_name BU --num_epochs 10  --batch_size 256
# 22.72
python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-3 --uni_name SalUn --num_epochs 10  --batch_size 256 --class_to_replace 10 30 50 70 90 --mask_thresh 0.8
# 9.44
python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --alpha 18 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10  --batch_size 256  --print_freq 20 --WF_N 50
# 19.92
python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --alpha 0.01 --unlearn_lr 2e-4 --uni_name GA_l1 --num_epochs 10  --batch_size 256
# 20.88
python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 26e-5 --uni_name UNSC --num_epochs 10 --batch_size 256
