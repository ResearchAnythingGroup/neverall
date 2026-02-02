# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL, GA, IU, BU, L1, SalUn

# before_unlearn forget_acc: 100.0 test_acc: 91.8
# target forget_acc < 0.5

# class level
# forget_acc: 23.75   test_acc: 76.26  decrease lr 6e-3 -> 2e-3
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 1 --unlearn_lr 2e-3 --uni_name RL --num_epochs 10  --batch_size 64
# forget_acc: 43.44   test_acc: 79.1  increase lr 4e-5-> 2e-3 ->2e-4
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 1 --unlearn_lr 2e-4 --uni_name GA --num_epochs 10  --batch_size 64
# forget_acc: 49.69   test_acc: 73.68 increase alpha 50 -> 60 -> 200 -> 600 -> 6000 -> 60000 -> 20000 increase lr 1e-4 -> 1e-3
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 60 --unlearn_lr 1e-4 --uni_name IU --num_epochs 10  --batch_size 64
# forget_acc: 44.69  test_acc: 73.84  increase lr 5e-4 -> 1e-3  increase alpha 0.3 -> 2
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 2 --unlearn_lr 1e-3 --uni_name BU --num_epochs 10  --batch_size 32
# forget_acc: 47.81  test_acc: 79.94
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 1 --unlearn_lr 3e-4 --uni_name SalUn --num_epochs 10  --batch_size 64 --class_to_replace 1 8 --mask_thresh 0.8
# forget_acc: 75.94  test_acc: 27.08  increase alpha 14 -> 20 -> 200 -> 100 -> 40 -> 70 -> 55 -> 50 -> 45 increase lr 1e-3 -> 1e-2  forget_acc > test_acc
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 55 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10  --batch_size 64 --WF_N 50
# forget_acc: 35.31  test_acc: 77.8 increase lr 5e-5 -> 1e-4 -> 3e-4
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 0.01 --unlearn_lr 3e-4 --uni_name GA_l1 --num_epochs 10  --batch_size 64
# forget_acc: 99.38  test_acc: 87.76 increase lr 1e-5 -> 2e-4
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 1 --unlearn_lr 2e-4 --uni_name UNSC --num_epochs 10 --batch_size 32
# forget_acc: 41.25  test_acc: 80.24 increase lr 1e-5 -> 1e-4
python main_mu.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --alpha 1 --unlearn_lr 1e-4 --uni_name DELETE --num_epochs 20 --batch_size 64
