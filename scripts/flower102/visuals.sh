# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, SalUn,fisher, GA_l1, UNSC
# RL
python result_analysis/visual_results.py --dataset flower-102 --model swin_t --forget_ratio 0.2 --uni_name RL,GA,IU,BU,SalUn,fisher,GA_l1,DELETE --model_suffix ul --batch_size 64
