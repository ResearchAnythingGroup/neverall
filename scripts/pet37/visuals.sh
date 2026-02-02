# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, SalUn,fisher, GA_l1, UNSC
# RL
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.2 --uni_name RL,GA,IU,BU,SalUn,fisher,GA_l1,DELETE --model_suffix ul --batch_size 64
