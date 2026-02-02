# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# instance level
python data_process/gen_exp_data.py --dataset imagenette --forget_ratio 0.4 --forget_classes 0 2 4 6 8 --ft_forget_ratio 0.5 --ft_test_ratio 0.25
