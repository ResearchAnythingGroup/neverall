# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# similar_distance = 0.9 without loss_alpha
# todo all
# instance level
# after unlearn finetune: --train_mode finetune --forget_ratio 0.1 --uni_name RL
# RL  92.55 num_epochs 10(forget_acc 0.9333) -> 1(forget_acc 0.7333)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name RL --forget_ratio 0.2 --neverecall --ablation_type SSPR
# GA  79.8 num_epochs 10(forget_acc 0.9067) -> 1(forget_acc 0.44)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name GA --forget_ratio 0.2 --neverecall --ablation_type SSPR
# IU  90.0 num_epochs 10(forget_acc 0.9333) -> 1(forget_acc 0.9066) -> 1(forget_acc 0.8533 similar_distance 0.8)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name IU --forget_ratio 0.2 --neverecall --ablation_type SSPR
# BU  92.05 num_epochs 10(forget_acc 0.9733) -> 1(forget_acc 0.88)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name BU --forget_ratio 0.2 --neverecall --ablation_type SSPR
# SalUn  90.88 num_epochs 10(forget_acc 1.0) -> 1(forget_acc 0.8666)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name SalUn --forget_ratio 0.2 --neverecall --ablation_type SSPR
# fisher  88.62 num_epochs 10(forget_acc 0.9467) -> 1(forget_acc 0.92) -> 1(forget_acc 0.8266 similar_distance 0.8)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name fisher --forget_ratio 0.2 --neverecall --ablation_type SSPR
# GA_l1  88.63 num_epochs 10(forget_acc 0.9733) -> 1(forget_acc 0.7333)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name GA_l1 --forget_ratio 0.2 --neverecall --ablation_type SSPR
# DELETE  93.24 similar_distance 0.8 num_epochs 10(forget_acc 0.8267) -> 1(forget_acc 0.8666 similar_distance 0.7)
python train_model.py --dataset flower-102 --model swin_t --train_mode finetune --num_epochs 1 --learning_rate 1e-4 --batch_size 32 --uni_name DELETE --forget_ratio 0.2 --neverecall --ablation_type SSPR
