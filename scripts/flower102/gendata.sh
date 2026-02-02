# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# instance level 10 classes  0.1
#python data_process/gen_exp_data.py --dataset flower-102 --forget_ratio 0.1 --forget_classes 4 14 24 34 44 54 64 74 84 94  --ft_forget_ratio 0.5 --ft_test_ratio 0.25

# instance level: 5 classes  0.2
python data_process/gen_exp_data.py --dataset flower-102 --forget_ratio 0.2 --forget_classes 54 64 74 84 94  --ft_forget_ratio 0.5 --ft_test_ratio 0.25