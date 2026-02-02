# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# ablation SSR similar_distance = 0.9999
# todo all
# instance level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL
# RL 87.25  loss_alpha 0.8(forget_acc 1.0) -> 2(forget_acc 0.3866 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name RL --forget_ratio 0.2 --neverecall --ablation_type SSR
# GA  77.65 loss_alpha 0.35(forget_acc 1.0) -> 0.1(forget_acc 0.5733 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name GA --forget_ratio 0.2 --neverecall --ablation_type SSR
# IU  87.84 loss_alpha 0.7 (forget_acc 0.9733) -> 0.8(forget_acc 0.8933 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name IU --forget_ratio 0.2 --neverecall --ablation_type SSR
# BU  87.74 loss_alpha 0.8 (forget_acc 1.0) -> 1.5(forget_acc 0.4133 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name BU --forget_ratio 0.2 --neverecall --ablation_type SSR
# SalUn 87.25 loss_alpha 0.7 (forget_acc 1.0) -> 1.4(forget_acc 0.7866 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name SalUn --forget_ratio 0.2 --neverecall --ablation_type SSR
# fisher  83.23 loss_alpha 0.7 (forget_acc 0.96) -> 1.4(forget_acc 0.32 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name fisher --forget_ratio 0.2 --neverecall --ablation_type SSR
# GA_l1  87.84  loss_alpha 0.7 (forget_acc 0.96) -> 1.4(forget_acc 0.7066 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name GA_l1 --forget_ratio 0.2 --neverecall --ablation_type SSR
# DELETE 89.80 loss_alpha 0.7 (forget_acc 1.0) -> 2.1(forget_acc 0.3466 num_epochs 1)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name DELETE --forget_ratio 0.2 --neverecall --ablation_type SSR
