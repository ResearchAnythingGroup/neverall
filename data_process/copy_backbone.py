import shutil

import numpy as np
import os
import torch
import pandas as pd


def copy_backbone_ckpt(output_root, data_name, backbone_name, model_names):
    data_path = backbone_name + '_' + data_name
    output_path = os.path.join(output_root, data_path)

    for model_name in model_names:
        if backbone_name == 'vgg16' and model_name == 'FF':
            continue
        unlearn_path = os.path.join(output_path, model_name)
        
        os.makedirs(unlearn_path, exist_ok=True)
        
        model_true_name = model_name
        
        if model_name == 'FF':
            model_true_name = 'fisher'
        elif model_name == 'IU':
            model_true_name = 'wfisher'
        new_ckpt_name = model_true_name + 'checkpoint.pth.tar'
        new_path = os.path.join(unlearn_path, new_ckpt_name)

        # get backbone path
        backbone_dir = os.path.join(output_path, 'finetune_backbone')
        backbone_path = os.path.join(backbone_dir, 'retraincheckpoint.pth.tar')
        if not os.path.exists(backbone_path):
            print(backbone_path)
            print('backbone ckpt does not!')
            break

        shutil.copyfile(backbone_path, new_path)
        print(backbone_path, ' backbone ckpt copy to -> ', new_path)


if __name__ == "__main__":
    # path = '../outputs_test'
    path = '/nvme/szh/data/3ai/lips/outputs'
    unlearn_model_names = ['FT', 'FF', 'GA', 'IU', 'FT_prune']
    data_list = ['cifar10', 'cifar100', 'tinyimg', 'fmnist']
    backbones = ['resnet18', 'vgg16']

    for data in data_list:
        for backbone in backbones:
            copy_backbone_ckpt(path, data, backbone, unlearn_model_names)

    print('copy backbone done!')
