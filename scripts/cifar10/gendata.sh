# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level 2 classes -0.5
#python data_process/gen_exp_data.py --dataset cifar-10 --forget_ratio -0.5 --forget_classes 4 8  --ft_forget_ratio 0.5 --ft_test_ratio 0.25

python data_process/gen_exp_data.py --dataset cifar-10 --forget_ratio -0.1 --forget_classes 4 8  --ft_forget_ratio 0.5 --ft_test_ratio 0.25
