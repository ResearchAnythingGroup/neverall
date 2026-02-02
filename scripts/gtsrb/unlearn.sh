# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100 test_acc: 98.88
# target forget_acc < 0.5
# forget_class 12 13 14

# class level
# forget_acc: 0.94  test_acc: 83.62
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.25 --alpha 0.1 --unlearn_lr 1e-4 --uni_name RL --num_epochs 10 --batch_size 32 --print_freq 20
# forget_acc: 0    test_acc: 2.85  increase lr 4e-3 -> 1e-3 -> 3e-3 -> 2e-3 epoch 4 直接归0
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.25 --alpha 1 --unlearn_lr 2e-3 --uni_name GA --num_epochs 8 --batch_size 32 --print_freq 20
# forget_acc: 40.14 test_acc: 39.41 increase alpha 6000 -> 50000 -> 80000 -> 120000 -> 300000
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.25 --alpha 300000 --uni_name IU --num_epochs 10  --batch_size 32 --print_freq 20 --WF_N 1000
# forget_acc: 50.3  test_acc: 43.04
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.25 --alpha 0.4 --unlearn_lr 1e-5 --uni_name BU --num_epochs 10 --batch_size 32 --print_freq 20
# forget_acc: 49.94  test_acc: 51.13  decrease lr 4e-5 -> 1e-5 -> 5e-6 -> 1e-6 -> 3e-6 decrease num_epochs 10->5->8
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.25 --alpha 0.1 --unlearn_lr 3e-6 --uni_name SalUn --num_epochs 8 --batch_size 32 --class_to_replace 12 13 14 --mask_thresh 0.8  --print_freq 20
# forget_acc: 52.3   test_acc: 37.41  increase alpha 24 -> 30 -> 26 decrease lr 1e-3 -> 1e-4
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.25 --alpha 26 --unlearn_lr 1e-4 --uni_name fisher --num_epochs 10 --batch_size 32  --print_freq 20 --WF_N 50
# forget_acc: 38.84  test_acc: 27.34  decrease lr 2e-5 -> 1e-5 -> 1e-6 -> 5e-6  -> 2e-6
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.25 --alpha 3 --unlearn_lr 2e-6 --uni_name GA_l1 --num_epochs 10 --batch_size 32  --print_freq 20
