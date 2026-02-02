# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# train
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --model_suffix train

# RL, GA, IU, BU, SalUn, fisher, GA_l1, UNSC
# RL
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name RL --model_suffix ul
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name RL --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name RL --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name RL --model_suffix ul_ft_SSGS
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name RL --model_suffix ul_ft_SSPR
# GA
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA --model_suffix ul
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA --model_suffix ul_ft_SSGS
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA --model_suffix ul_ft_SSPR
# IU
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name IU --model_suffix ul
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name IU --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name IU --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name IU --model_suffix ul_ft_SSGS
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name IU --model_suffix ul_ft_SSPR
# BU
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name BU --model_suffix ul
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name BU --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name BU --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name BU --model_suffix ul_ft_SSGS
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name BU --model_suffix ul_ft_SSPR
# SalUn
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name SalUn --model_suffix ul
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name SalUn --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name SalUn --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name SalUn --model_suffix ul_ft_SSGS
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name SalUn --model_suffix ul_ft_SSPR
# fisher
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name fisher --model_suffix ul
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name fisher --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name fisher --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name fisher --model_suffix ul_ft_SSGS
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name fisher --model_suffix ul_ft_SSPR
# GA_l1
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA_l1 --model_suffix ul
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA_l1 --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA_l1 --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA_l1 --model_suffix ul_ft_SSGS
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name GA_l1 --model_suffix ul_ft_SSPR
# DELETE
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name DELETE --model_suffix ul
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name DELETE --model_suffix ul_ft
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name DELETE --model_suffix ul_ft_neverecall
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name DELETE --model_suffix ul_ft_SSGS
python result_analysis/evaluate_results.py --dataset stl10 --model efficientnet_s --forget_ratio -0.2 --uni_name DELETE --model_suffix ul_ft_SSPR