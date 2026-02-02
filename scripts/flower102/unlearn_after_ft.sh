# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100 test_acc: 97.25
# target forget_acc < 0.3

# instance level (75 instance)
# forget_acc: 12.0  test_acc: 83.04
python main_mu.py --unlearn_after_ft --dataset flower-102 --model swin_t --forget_ratio 0.2 --alpha 0.3 --unlearn_lr 1e-4 --uni_name RL --num_epochs 10 --batch_size 32 --print_freq 20
# forget_acc: 0.0  test_acc: 4.8
python main_mu.py --unlearn_after_ft --dataset flower-102 --model swin_t --forget_ratio 0.2 --alpha 1 --unlearn_lr 3.5e-4 --uni_name GA --num_epochs 8 --batch_size 32 --print_freq 20
# forget_acc: 0.0   test_acc: 0.98
python main_mu.py --unlearn_after_ft --dataset flower-102 --model swin_t --forget_ratio 0.2 --alpha 350 --uni_name IU --num_epochs 10  --batch_size 32 --print_freq 20 --WF_N 1000
# forget_acc: 69.33  test_acc: 82.65
python main_mu.py --unlearn_after_ft --dataset flower-102 --model swin_t --forget_ratio 0.2 --alpha 0.4 --unlearn_lr 1e-5 --uni_name BU --num_epochs 20 --batch_size 32 --print_freq 20
# forget_acc: 53.33  test_acc: 81.86   increase lr 5e-6 -> 1e-4 -> 1e-5 -> 3e-5 -> 2.4e-5
python main_mu.py --unlearn_after_ft --dataset flower-102 --model swin_t --forget_ratio 0.2 --alpha 0.1 --unlearn_lr 2.4e-5 --uni_name SalUn --num_epochs 10 --batch_size 32 --class_to_replace 54 64 74 84 94 --mask_thresh 0.8  --print_freq 20
# forget_acc: 69.33  test_acc: 56.57  decrease alpha  16 -> 14 -> 15.4 -> 15.6  test_acc会跟forget acc 同步下降，比较接近
python main_mu.py --unlearn_after_ft --dataset flower-102 --model swin_t --forget_ratio 0.2 --alpha 14 --unlearn_lr 5e-4 --uni_name fisher --num_epochs 10 --batch_size 32  --print_freq 20 --WF_N 50
# forget_acc: 25.33  test_acc: 5.98 decrease lr 29e-6 -> 2e-5 -> 1e-5 -> 1.5e-5 -> 1.7e-5
python main_mu.py --unlearn_after_ft --dataset flower-102 --model swin_t --forget_ratio 0.2 --alpha 3 --unlearn_lr 1.7e-5 --uni_name GA_l1 --num_epochs 10 --batch_size 32  --print_freq 20
#
# python main_mu.py --unlearn_after_ft --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-5 --uni_name UNSC --num_epochs 10 --batch_size 32
