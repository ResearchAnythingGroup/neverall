# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# class level 10 classes -0.5
#python data_process/gen_exp_data.py --dataset gtsrb --forget_ratio -0.5 --forget_classes 1 4 8 14 18 24 28 34 38 40  --ft_forget_ratio 0.5 --ft_test_ratio 0.25

#python data_process/gen_exp_data.py --dataset gtsrb --forget_ratio -0.51 --forget_classes 0  --ft_forget_ratio 0.5 --ft_test_ratio 0.25

# test_12
#python data_process/gen_exp_data.py --dataset gtsrb --forget_ratio -0.52 --forget_classes 12  --ft_forget_ratio 0.5 --ft_test_ratio 0.25

python data_process/gen_exp_data.py --dataset gtsrb --forget_ratio -0.25 --forget_classes 12 13 14  --ft_forget_ratio 0.5 --ft_test_ratio 0.25