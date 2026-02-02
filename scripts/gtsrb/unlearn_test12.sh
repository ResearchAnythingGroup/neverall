# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100 test_acc: 98.88
# target forget_acc < 0.5
# forget_class 4

# class level
# forget_acc: 28.95  test_acc: 97.9
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.52 --alpha 0.1 --unlearn_lr 1e-4 --uni_name RL --num_epochs 10 --batch_size 32 --print_freq 20
# forget_acc: 100   test_acc: 98.89 increase lr 1e-4 -> 1e-3 -> 1e-2 -> 4e-3 直接归0
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.52 --alpha 1 --unlearn_lr 4e-3 --uni_name GA --num_epochs 8 --batch_size 32 --print_freq 20
# forget_acc: 100   test_acc: 98.88 increase alpha 600 -> 6000 still 100
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.52 --alpha 6000 --uni_name IU --num_epochs 10  --batch_size 32 --print_freq 20 --WF_N 1000
# forget_acc: 10.53  test_acc: 83.44
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.52 --alpha 0.4 --unlearn_lr 1e-5 --uni_name BU --num_epochs 10 --batch_size 32 --print_freq 20
# forget_acc: 26.32  test_acc: 80.14  increase lr 5e-6 -> 1e-5 -> 1e-4 -> 4e-5
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.52 --alpha 0.1 --unlearn_lr 4e-5 --uni_name SalUn --num_epochs 10 --batch_size 32 --class_to_replace 0 --mask_thresh 0.8  --print_freq 20
# forget_acc: 80.26  test_acc: 52.19 increase lr 5e-4 -> 1e-3  increase alpha 16 -> 20 -> 30 -> 24
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.52 --alpha 24 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10 --batch_size 32  --print_freq 20 --WF_N 50
# forget_acc: 53.95  test_acc: 48.23  decrease lr 29e-6 -> 4e-6 -> 2e-5 -> 24e-6
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.52 --alpha 3 --unlearn_lr 2e-5 --uni_name GA_l1 --num_epochs 10 --batch_size 32  --print_freq 20
