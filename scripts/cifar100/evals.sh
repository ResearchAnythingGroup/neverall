# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# train
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio -0.1 --model_suffix train

# RL, GA, IU, BU, SalUn, fisher, GA_l1, UNSC
# RL
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio -0.1 --uni_name RL --model_suffix ul
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio -0.1 --uni_name RL --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio -0.1 --uni_name RL --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio -0.1 --uni_name RL --model_suffix ul_ft_ul
# GA
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix ul
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix restore
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix distill
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix student_only
# IU
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --model_suffix ul
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --model_suffix restore
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --model_suffix distill
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --model_suffix student_only
# BU
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --model_suffix ul
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --model_suffix restore
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --model_suffix distill
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --model_suffix student_only
# SalUn
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --model_suffix ul
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --model_suffix restore
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --model_suffix distill
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --model_suffix student_only
# fisher
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --model_suffix ul
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --model_suffix restore
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --model_suffix distill
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --model_suffix student_only
# GA_l1
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --model_suffix ul
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --model_suffix restore
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --model_suffix distill
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --model_suffix student_only
# UNSC
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --model_suffix ul
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --model_suffix restore
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --model_suffix distill
python result_analysis/evaluate_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --model_suffix student_only
