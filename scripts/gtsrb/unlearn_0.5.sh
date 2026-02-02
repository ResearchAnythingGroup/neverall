# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100 test_acc: 98.88
# target forget_acc < 0.5
# forget_class 4

# class level
# forget_acc: 22.4  test_acc: 74.84
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.5 --alpha 0.1 --unlearn_lr 1e-4 --uni_name RL --num_epochs 10 --batch_size 32 --print_freq 20
# forget_acc:   test_acc:   increase lr 初始loss0, 调不出
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.5 --alpha 1 --unlearn_lr 1e-4 --uni_name GA --num_epochs 8 --batch_size 32 --print_freq 20
# forget_acc:   test_acc:   increase  alpha 调不出
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.5 --alpha 600 --uni_name IU --num_epochs 10  --batch_size 32 --print_freq 20 --WF_N 1000
# forget_acc: 27.01  test_acc: 30.34
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.5 --alpha 0.4 --unlearn_lr 1e-5 --uni_name BU --num_epochs 20 --batch_size 32 --print_freq 20
# forget_acc: .  test_acc: .
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.5 --alpha 0.1 --unlearn_lr 5e-6 --uni_name SalUn --num_epochs 10 --batch_size 32 --class_to_replace 1 4 8 14 18 24 28 34 38 40 --mask_thresh 0.8  --print_freq 20
# forget_acc: .  test_acc: .
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.5 --alpha 16 --unlearn_lr 5e-4 --uni_name fisher --num_epochs 10 --batch_size 32  --print_freq 20 --WF_N 50
# forget_acc: 36.73  test_acc: 46.86 increase alpha increase lr
python main_mu.py --dataset gtsrb --model swin_t --forget_ratio -0.5 --alpha 3 --unlearn_lr 29e-6 --uni_name GA_l1 --num_epochs 10 --batch_size 32  --print_freq 20
