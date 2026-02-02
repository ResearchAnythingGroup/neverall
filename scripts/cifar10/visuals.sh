# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, SalUn,fisher, GA_l1, UNSC
python result_analysis/visual_results.py --dataset cifar-10 --model efficientnet_s --forget_ratio -0.1 --uni_name RL,GA,IU,BU,SalUn,fisher,GA_l1,UNSC,DELETE --model_suffix ul --batch_size 256
