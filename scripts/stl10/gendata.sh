# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level 2 classes(1,8) -0.2
python data_process/gen_exp_data.py --dataset stl10 --forget_ratio -0.2 --forget_classes 1 8  --ft_forget_ratio 0.5 --ft_test_ratio 0.25
