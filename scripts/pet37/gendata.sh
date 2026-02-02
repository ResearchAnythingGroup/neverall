# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# instance level 5 classes 0.2
python data_process/gen_exp_data.py --dataset pet-37 --forget_ratio 0.2 --forget_classes 1 4 14 24 28  --ft_forget_ratio 0.5 --ft_test_ratio 0.25
