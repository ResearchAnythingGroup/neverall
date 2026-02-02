# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, SalUn,fisher, GA_l1, UNSC
# RL
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --model_suffix restore --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --model_suffix student_only --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --model_suffix distill --batch_size 256
# GA
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix restore --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix student_only --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix distill --batch_size 256
# IU
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --model_suffix restore --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --model_suffix student_only --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --model_suffix distill --batch_size 256
# BU
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --model_suffix restore --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --model_suffix student_only --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --model_suffix distill --batch_size 256
# SalUn
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --model_suffix restore --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --model_suffix student_only --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --model_suffix distill --batch_size 256
# fisher
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --model_suffix restore --batch_size 256 --fig_title FF
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --model_suffix student_only --batch_size 256 --fig_title FF
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --model_suffix distill --batch_size 256 --fig_title FF
# GA_l1
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --model_suffix restore --batch_size 256 --fig_title L1-SP
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --model_suffix student_only --batch_size 256 --fig_title L1-SP
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --model_suffix distill --batch_size 256 --fig_title L1-SP
# UNSC
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --model_suffix restore --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --model_suffix student_only --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --model_suffix distill --batch_size 256