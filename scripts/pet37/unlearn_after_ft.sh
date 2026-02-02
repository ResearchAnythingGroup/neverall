# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100 test_acc: 89.51
# target forget_acc < 0.3

# instance level
# forget_acc: 30.0  test_acc: 87.9
python main_mu.py --unlearn_after_ft --dataset pet-37 --model resnet18 --forget_ratio 0.2 --alpha 1 --unlearn_lr 1e-3 --uni_name RL --num_epochs 20  --batch_size 32
# forget_acc: 0.0  test_acc: 58.14  decrease lr 3e-4 -> 2e-4
python main_mu.py --unlearn_after_ft --dataset pet-37 --model resnet18 --forget_ratio 0.2 --alpha 1 --unlearn_lr 3e-4 --uni_name GA --num_epochs 10  --batch_size 32
# forget_acc: 0.0  test_acc: 2.67  decrease alpha 40 -> 20 -> 14
python main_mu.py --unlearn_after_ft --dataset pet-37 --model resnet18 --forget_ratio 0.2 --alpha 14 --unlearn_lr 1e-1 --uni_name IU --num_epochs 20  --batch_size 32
# forget_acc: 1.0  test_acc: 70.07  increase lr 2e-4 -> 1e-4
python main_mu.py --unlearn_after_ft --dataset pet-37 --model resnet18 --forget_ratio 0.2 --alpha 1 --unlearn_lr 1e-4 --uni_name BU --num_epochs 10  --batch_size 32
# forget_acc: 2.0  test_acc: 75.66
python main_mu.py --unlearn_after_ft --dataset pet-37 --model resnet18 --forget_ratio 0.2 --alpha 1 --unlearn_lr 3e-4 --uni_name SalUn --num_epochs 10  --batch_size 32 --class_to_replace  1 4 14 24 28 --mask_thresh 0.8
# forget_acc: 66.0  test_acc: 50.78   increase alpha 12 -> 20 -> 15 -> 13.5  forget_acc 一直高于 test_acc
python main_mu.py --unlearn_after_ft --dataset pet-37 --model resnet18 --forget_ratio 0.2 --alpha 13.5 --unlearn_lr 6e-5 --uni_name fisher --num_epochs 10  --batch_size 32
# forget_acc: 0.0  test_acc: 70.16
python main_mu.py --unlearn_after_ft --dataset pet-37 --model resnet18 --forget_ratio 0.2 --alpha 0.1 --unlearn_lr 2e-4 --uni_name GA_l1 --num_epochs 10 --batch_size 32
# batch_size 8 still 段错误(核心已转储）
#python main_mu.py --unlearn_after_ft --dataset pet-37 --model resnet18 --forget_ratio 0.2 --alpha 1 --unlearn_lr 12e-5 --uni_name UNSC --num_epochs 20 --batch_size 8
