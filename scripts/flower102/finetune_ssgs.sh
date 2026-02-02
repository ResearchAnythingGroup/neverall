# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# similar_distance = 0.9
# todo all
# instance level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL
# RL 96.57  loss_alpha 0.8(forget_acc 1.0) -> 2(forget_acc 0.986 num_epochs 1) -> 3 (forget_acc 0.8933 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name RL --forget_ratio 0.2 --neverecall --ablation_type SSGS
# GA  76.47 loss_alpha 0.35(forget_acc 1.0) -> 0.3(forget_acc 0.386 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name GA --forget_ratio 0.2 --neverecall --ablation_type SSGS
# IU  86.67 loss_alpha 0.7(forget_acc 1.0) -> 2(forget_acc 0.826 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name IU --forget_ratio 0.2 --neverecall --ablation_type SSGS
# BU  89.31 loss_alpha 0.8(forget_acc 1.0) -> 2.4(forget_acc 0.72 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name BU --forget_ratio 0.2 --neverecall --ablation_type SSGS
# SalUn 89.6 loss_alpha 0.7(forget_acc 1.0) -> 2.4(forget_acc 0.84 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name SalUn --forget_ratio 0.2 --neverecall --ablation_type SSGS
# fisher  86.47 loss_alpha 0.7(forget_acc 1.0) -> 2.6(forget_acc 0.8 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name fisher --forget_ratio 0.2 --neverecall --ablation_type SSGS
# GA_l1  87.65  loss_alpha 0.7(forget_acc 0.0) -> 0.36(forget_acc 0.653 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name GA_l1 --forget_ratio 0.2 --neverecall --ablation_type SSGS
# DELETE 90.78 loss_alpha 0.7(forget_acc 1.0) -> 2.8(forget_acc 0.52 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name DELETE --forget_ratio 0.2 --neverecall --ablation_type SSGS
