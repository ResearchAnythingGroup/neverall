# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100 test_acc: 89.51
# target forget_acc < 0.3

# instance level
# forget_acc: 34.0  test_acc: 79.2  increase lr
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.1 --alpha 1 --unlearn_lr 1e-3 --uni_name RL --num_epochs 20  --batch_size 32
# forget_acc: 10.0  test_acc: 67.18  increase lr
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.1 --alpha 1 --unlearn_lr 3e-4 --uni_name GA --num_epochs 20  --batch_size 32
# forget_acc: 34.0  test_acc: 65.41  increase alpha
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.1 --alpha 40 --unlearn_lr 1e-1 --uni_name IU --num_epochs 20  --batch_size 32
# forget_acc: 10.0  test_acc: 65.99  increase lr
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.1 --alpha 1 --unlearn_lr 2e-4 --uni_name BU --num_epochs 10  --batch_size 32
# forget_acc: 20.0  test_acc: 74.73  increase lr
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.1 --alpha 1 --unlearn_lr 3e-4 --uni_name SalUn --num_epochs 10  --batch_size 32 --class_to_replace  1 4 14 24 28 --mask_thresh 0.8
# forget_acc: 42.0  test_acc: 41.56  decrease lr  decrease alpha forget_acc 一直高于 test_acc
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.1 --alpha 12 --unlearn_lr 6e-5 --uni_name fisher --num_epochs 10  --batch_size 32
# forget_acc: 32.0  test_acc: 73.45  increase lr
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.1 --alpha 0.1 --unlearn_lr 2e-4 --uni_name GA_l1 --num_epochs 10 --batch_size 32
# batch_size 8 still 段错误(核心已转储）
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.1 --alpha 1 --unlearn_lr 12e-5 --uni_name UNSC --num_epochs 20 --batch_size 8
