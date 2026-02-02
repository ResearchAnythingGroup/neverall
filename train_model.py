import os
import shutil
import warnings
import numpy as np
from collections import OrderedDict

import torch
from nets.custom_model import ClassifierWrapper, load_custom_model, ClassifierWrapperHooker
from nets.datasetloader import get_dataset_loader
from nets.train_test import model_test
from configs import settings
from train_test_utils import train_model, never_recall_module
from arg_parser import parse_args
from configs.Config import Constant


def load_dataset(file_path, is_data=True):
    """
    加载数据集文件并返回 PyTorch 张量。
    :param subdir: 数据目录
    :param dataset_name: 数据集名称 (cifar-10, cifar-100, food-101, pet-37, flower-102)
    :param file_name: 数据文件名
    :param is_data: 是否为数据文件（True 表示数据文件，False 表示标签文件）
    :return: PyTorch 张量格式的数据
    """
    data = np.load(file_path, mmap_mode='r')

    if is_data:
        # 对于数据文件，转换为 float32 类型
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        # 对于标签文件，转换为 long 类型
        data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor


def train_step(
    args,
    writer=None,
):
    """
    根据步骤训练模型
    :param step: 要执行的步骤（0, 1, 2, ...）
    :param subdir: 数据子目录路径
    :param ckpt_subdir: 模型检查点子目录路径
    :param output_dir: 模型保存目录
    :param dataset_name: 使用的数据集类型（cifar-10 或 cifar-100）
    :param load_model_path: 指定加载的模型路径（可选）
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :optimizer_type: 优化器
    :param learning_rate: 学习率
    """
    warnings.filterwarnings("ignore")

    dataset_name = args.dataset
    num_classes = settings.num_classes_dict[dataset_name]

    print(f"数据集类型: {dataset_name}")
    print(
        f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}"
    )

    model_name = args.model
    train_mode = args.train_mode
    
    uni_name = args.uni_name

    test_data = load_dataset(
        settings.get_dataset_path(dataset_name, None, "test_data")
    )
    test_labels = load_dataset(
        settings.get_dataset_path(dataset_name, None, "test_label"), is_data=False
    )

    case = None if train_mode in ["train", "pretrain"] else settings.get_case(args.forget_ratio)

    model_path = settings.get_ckpt_path(
        dataset_name, case, model_name, train_mode)

    if uni_name is None:
        train_data = np.load(
            settings.get_dataset_path(dataset_name, case, f"{train_mode}_data")
        )
        train_labels = np.load(
            settings.get_dataset_path(dataset_name, case, f"{train_mode}_label")
        )

        load_pretrained = True
        model_p0 = load_custom_model(model_name, num_classes, load_pretrained=load_pretrained)
        model_p0 = ClassifierWrapper(model_p0, num_classes)

        print(f"Train on ({dataset_name})...")

        model_p0 = train_model(
            model_p0,
            num_classes,
            train_data,
            train_labels,
            test_data,
            test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            test_it=args.test_it,
            writer=writer,
        )
        subdir = os.path.dirname(model_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_p0.state_dict(), model_path)
        print(f"Model saves to {model_path}")

    else:
        # finetune after unlearn
        # load unlearn model
        unlearn_model_path = settings.get_ckpt_path(
            args.dataset, case, args.model, model_suffix='ul', unique_name=uni_name
        )

        loaded_model = load_custom_model(
            args.model, num_classes, load_pretrained=False
        )
        unlearn_model = ClassifierWrapperHooker(loaded_model, num_classes)
        checkpoint = torch.load(unlearn_model_path)
        unlearn_model.load_state_dict(checkpoint, strict=False)

        # load ft inc_data
        inc_ft_data = np.load(
            settings.get_dataset_path(dataset_name, case, f'inc_ft_data')
        )
        inc_ft_labels = np.load(
            settings.get_dataset_path(dataset_name, case, f'inc_ft_label')
        )

        # check
        # before ft
        _, _, forget_loader = get_dataset_loader(
            args.dataset,
            "forget",
            case,
            batch_size=args.batch_size,
            shuffle=False,
        )

        _, _, test_loader = get_dataset_loader(
            args.dataset,
            ["test"],
            [None],
            batch_size=args.batch_size,
            shuffle=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unlearn_model.to(device)
        print('************** before ft *****************')
        print('-------- forget_loader %d---------' % len(forget_loader.dataset))
        model_test(forget_loader, unlearn_model, device)
        print('-------- test_loader %d ---------' % len(test_loader.dataset))
        model_test(test_loader, unlearn_model, device)

        print(f"Train on ({dataset_name})...")

        ft_model = train_model(
            unlearn_model,
            num_classes,
            inc_ft_data,
            inc_ft_labels,
            test_data,
            test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            test_it=args.test_it,
            writer=writer,
            forget_ratio=args.forget_ratio,
            forget_classes=args.forget_classes,
            dataset_name=dataset_name,
            proto_net_name=args.model,  # todo 默认protonet的backbone与主网络一致
            never_recall_flg=args.neverecall,
            forget_data_unlearn_features_path=os.path.join(os.path.dirname(unlearn_model_path), 'forget_data_unlearn_features.npy'),
            ablation_type=args.ablation_type,
        )
        # subdir = os.path.dirname(model_path)
        # os.makedirs(subdir, exist_ok=True)
        model_suffix = 'ul_ft'
        if args.neverecall:
            model_suffix += '_neverecall'
        if args.ablation_type != '':
            model_suffix = 'ul_ft_' + args.ablation_type

        model_save_path = settings.get_ckpt_path(
            args.dataset, case, args.model, model_suffix=model_suffix, unique_name=uni_name
        )
        torch.save(ft_model.state_dict(), model_save_path)
        print(f"Model saves to {model_save_path}")


def main():
    args = parse_args()

    writer = None
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="runs/experiment")

    train_step(
        args,
        writer=writer,
    )

    if writer:
        writer.close()


if __name__ == "__main__":

    main()
