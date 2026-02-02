import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2

from split_dataset import split_data
from arg_parser_gen import parse_args
from configs import settings
from configs.Config import Constant
from configs.helpers import plot


def create_dataset_files(dataset_name, forget_classes, forget_ratio=0.5):
    data_dir = os.path.join(settings.root_dir, "data", dataset_name, "normal")

    # 根据dataset获取对应的transform的mean std, 用于数据transform
    if dataset_name in settings.normalize_config.keys():
        mean_std = settings.normalize_config[dataset_name]
        mean, std = mean_std['mean'], mean_std['std']
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
    else:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        data_transform = transforms.Compose([weights.transforms()])

    # 根据dataset 加载对应数据集
    if dataset_name == Constant.CIFAR10:
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=data_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=data_transform)
    elif dataset_name == Constant.CIFAR100:
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=data_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=data_transform)
    elif dataset_name == Constant.FLOWER102:
        # split反着, test数据集数量多于train
        train_dataset = datasets.Flowers102(root=data_dir, split="test", download=True, transform=data_transform)
        test_dataset = datasets.Flowers102(root=data_dir, split="train", download=True, transform=data_transform)
    elif dataset_name == Constant.PET37:
        train_dataset = datasets.OxfordIIITPet(root=data_dir, download=True, transform=data_transform)
        test_dataset = datasets.OxfordIIITPet(root=data_dir, split="test", download=True, transform=data_transform)
    elif dataset_name == Constant.COUNTRY211:
        train_dataset = datasets.Country211(root=data_dir, download=True, transform=data_transform)
        test_dataset = datasets.Country211(root=data_dir, split="test", download=True, transform=data_transform)
    elif dataset_name == Constant.IMAGENETTE:
        # todo imagenette 无 test 数据, 用val作为test数据
        train_dataset = datasets.Imagenette(root=data_dir, download=True, transform=data_transform)
        test_dataset = datasets.Imagenette(root=data_dir, split="val", download=True, transform=data_transform)
    elif dataset_name == Constant.FOOD101:
        train_dataset = datasets.Food101(root=data_dir, download=True, transform=data_transform)
        test_dataset = datasets.Food101(root=data_dir, split="test", download=True, transform=data_transform)
    elif dataset_name == 'gtsrb':
        train_dataset = datasets.GTSRB(root=data_dir, download=True, transform=data_transform)
        test_dataset = datasets.GTSRB(root=data_dir, split="test", download=True, transform=data_transform)
    elif dataset_name == 'stl10':
        # split反着, test数据集数量多于train
        train_dataset = datasets.STL10(root=data_dir, split="test", download=True, transform=data_transform)
        test_dataset = datasets.STL10(root=data_dir, split="train", download=True, transform=data_transform)
    else:
        print("Dataset not found.")

    # 根据forget_ratio forget_classes 从 train_dataset 中 选取 forget_data 与 retain_data
    # ****
    # forget_classes: 要选取的遗忘类.
    # forget_ratio: 每个遗忘类中选取的数据比例, 若>0, 则为 instance level; 若<0, 则为 class level.
    # forget_data: 根据选定的forget_classes, 随机选取指定的forget_ratio的数据;
    # retain_data: instance level, 其余非遗忘类的全部数据 + 遗忘类中未选取的剩余数据;
    #              class level, 其余非遗忘类的全部数据.
    # forget_data 与 retain_data 用于 unlearn的训练 (包含第一次unlearn与finetune之后的unlearn)
    # ****
    train_labels = split_data(dataset_name, train_dataset, test_dataset, forget_classes, forget_ratio)
    results = np.unique(train_labels, return_index=True, return_counts=True)
    print(results)


def create_ft_dataset_files(dataset_name,forget_classes, forget_ratio, ft_forget_ratio, ft_test_ratio):
    # 加载 d_test
    test_data_path = settings.get_dataset_path(dataset_name, None, "test_data")
    test_label_path = settings.get_dataset_path(dataset_name, None, "test_label")
    # 加载 d_forget
    case = settings.get_case(forget_ratio)
    forget_name = 'forget'
    forget_data_path = settings.get_dataset_path(dataset_name, case, '%s_data' % forget_name)
    forget_label_path = settings.get_dataset_path(dataset_name, case, '%s_label' % forget_name)

    # 50% d_forget
    forget_data = np.load(forget_data_path)
    forget_labels = np.load(forget_label_path)

    # forget_cls_list = list(set(forget_labels))
    inc_forget_idx = []
    for forget_cls in forget_classes:
        forget_idx = np.where(forget_labels == forget_cls)[0]
        target_len = int(forget_idx.size * ft_forget_ratio)
        forget_idx_target = np.random.choice(forget_idx, target_len, replace=False)
        inc_forget_idx.extend(forget_idx_target)

    inc_forget_data = forget_data[inc_forget_idx]
    inc_forget_labels = forget_labels[inc_forget_idx]

    inc_forget_suffix = 'inc_forget'
    np.save(settings.get_dataset_path(dataset_name, case, '%s_data' % inc_forget_suffix), inc_forget_data)
    np.save(settings.get_dataset_path(dataset_name, case, '%s_label' % inc_forget_suffix), inc_forget_labels)

    # d_forget_transform
    mixup_ratio = 0.9
    forget_images = torch.from_numpy(inc_forget_data).float()
    v_flipper = v2.RandomHorizontalFlip(p=0.5)
    img_size = settings.img_sizes[dataset_name]
    resize_crop = v2.RandomResizedCrop(size=img_size)
    flipper_images = v_flipper(forget_images)
    trans_images = resize_crop(flipper_images)
    indices = torch.randperm(trans_images.shape[0])
    shuffled_images = inc_forget_data[indices]
    mixup_images = mixup_ratio * trans_images + (1 - mixup_ratio) * shuffled_images
    inc_transform_data = mixup_images.numpy().astype(np.float32)
    # plot([forget_images[10], flipper_images[10], trans_images[10], mixup_images[10]])

    inc_transform_suffix = 'inc_transform'
    np.save(settings.get_dataset_path(dataset_name, case, '%s_data' % inc_transform_suffix), inc_transform_data)
    np.save(settings.get_dataset_path(dataset_name, case, '%s_label' % inc_transform_suffix), inc_forget_labels)

    # 25% d_test
    test_data = np.load(test_data_path, mmap_mode='r')
    test_labels = np.load(test_label_path, mmap_mode='r')

    test_cls_list = list(set(test_labels))
    inc_test_idx = []
    for test_cls in test_cls_list:
        test_idx = np.where(test_labels == test_cls)[0]
        target_len = int(test_idx.size * ft_test_ratio)
        test_idx_target = np.random.choice(test_idx, target_len, replace=False)
        inc_test_idx.extend(test_idx_target)

    inc_test_data = test_data[inc_test_idx]
    inc_test_labels = test_labels[inc_test_idx]

    inc_test_suffix = 'inc_test'
    np.save(settings.get_dataset_path(dataset_name, case, '%s_data' % inc_test_suffix), inc_test_data)
    np.save(settings.get_dataset_path(dataset_name, case, '%s_label' % inc_test_suffix), inc_test_labels)

    # inc_ft = 50% d_forget + d_transform + 25% d_test (instance 和 class 一致)
    # ****
    # inc_ft: 用于finetune时的新增数据.
    # d_forget: 为unlearn时使用的forget_data, 此处选取一定比例(exp:50%)的数据, 用来模拟真实情况中采集到的类似于原始forget_data的数据.
    # d_transform: d_forget数据进行1：1的数据生成，flipper 再 mix up (0.9 * Trans(X) + 0.1 * Shuffle(X))
    # d_test: 为原始数据集的test数据集, 此处选取一定比例(exp:25%)的数据, 模拟采集到的新的增量数据。
    # ****
    inc_ft_data = np.concatenate((inc_forget_data, inc_transform_data, inc_test_data))
    inc_ft_labels = np.concatenate((inc_forget_labels, inc_forget_labels, inc_test_labels))

    inc_data = 'inc_ft'
    np.save(settings.get_dataset_path(dataset_name, case, '%s_data' % inc_data), inc_ft_data)
    np.save(settings.get_dataset_path(dataset_name, case, '%s_label' % inc_data), inc_ft_labels)


def main():
    args = parse_args()

    # 生成train 与 unlearn 的 dataset
    create_dataset_files(
        dataset_name=args.dataset,
        forget_classes=args.forget_classes,
        forget_ratio=args.forget_ratio
    )

    # 生成 finetune 的 dataset
    create_ft_dataset_files(
        dataset_name=args.dataset,
        forget_classes=args.forget_classes,
        forget_ratio=args.forget_ratio,
        ft_forget_ratio=args.ft_forget_ratio,
        ft_test_ratio=args.ft_test_ratio,
    )


if __name__ == "__main__":
    main()
