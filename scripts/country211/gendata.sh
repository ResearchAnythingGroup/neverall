# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# instance level
python data_process/gen_exp_data.py --dataset country-211 --forget_ratio 0.4 --forget_classes 4 14 24 34 54 64 74 84 94 104 --ft_forget_ratio 0.5 --ft_test_ratio 0.25
