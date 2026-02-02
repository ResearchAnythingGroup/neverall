# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python data_process/gen_exp_data.py --dataset cifar-100 --forget_ratio -0.1 --forget_classes 10 30 50 70 90  --ft_forget_ratio 0.5 --ft_test_ratio 0.25