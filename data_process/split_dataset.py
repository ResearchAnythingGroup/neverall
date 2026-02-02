import shutil

from configs import settings
import os
import numpy as np
import torch


def sample_class_forget_data(train_data, train_labels, classes, forget_ratio):
    """按比例从每个类别中均衡抽取样本"""
    df_idx = []
    df_cls_idx = []
    idx_retain = np.ones(len(train_labels), dtype=bool)

    for c in classes:
        idx = np.where(train_labels == c)[0]
        forget_idx = np.random.choice(idx, int(abs(forget_ratio) * len(idx)), replace=False)
        df_idx += list(forget_idx)
        df_cls_idx += list(idx)

    if forget_ratio > 0: # instance level, retain中包含forget class中未被选择的数据
        idx_retain[df_idx] = False
        idx_forget = ~idx_retain
    else: # class level, retain中不包含任何forget class的数据
        idx_retain[df_cls_idx] = False
        idx_forget = np.zeros_like(idx_retain, dtype=bool)
        idx_forget[df_idx] = True

    return (train_data[idx_retain], train_labels[idx_retain],
            train_data[idx_forget], train_labels[idx_forget])


def gen_forget_cls_data(test_data, test_labels, forget_classes):
    ts_idx = []
    for c in forget_classes:
        idx = np.where(test_labels == c)[0]
        ts_idx += list(idx)
    forget_cls_data = test_data[ts_idx]
    forget_cls_labels = test_labels[ts_idx]
    return forget_cls_data, forget_cls_labels


def split_test_data_by_forget(test_data, test_labels, forget_classes):
    forget_idxes = []
    for forget_class in forget_classes:
        data_idx = np.where(test_labels == forget_class)[0]
        forget_idxes.extend(data_idx)

    forget_data = test_data[forget_idxes]
    forget_labels = test_labels[forget_idxes]

    idx_retain = np.ones(len(test_labels), dtype=bool)
    idx_retain[forget_idxes] = False
    remain_data = test_data[idx_retain]
    remain_labels = test_labels[idx_retain]
    return remain_data, remain_labels, forget_data, forget_labels


def split_data(dataset_name, train_dataset, test_dataset, forget_classes, forget_ratio=0.5):
    rawcase = None
    train_data_path = settings.get_dataset_path(dataset_name, rawcase, "train_data")
    train_label_path = settings.get_dataset_path(dataset_name, rawcase, "train_label")
    test_data_path = settings.get_dataset_path(dataset_name, rawcase, "test_data")
    test_label_path = settings.get_dataset_path(dataset_name, rawcase, "test_label")
    test_remain_data_path = settings.get_dataset_path(dataset_name, rawcase, "test_retain_data")
    test_remain_label_path = settings.get_dataset_path(dataset_name, rawcase, "test_retain_label")
    test_forget_data_path = settings.get_dataset_path(dataset_name, rawcase, "test_forget_data")
    test_forget_label_path = settings.get_dataset_path(dataset_name, rawcase, "test_forget_label")

    # forget_ratio代表每个class中选择的数据比例, 若为负数, 则为class level; 若为正数,则为instance level
    case = settings.get_case(forget_ratio)
    forget_data_path = settings.get_dataset_path(dataset_name, case, "forget_data")
    forget_label_path = settings.get_dataset_path(dataset_name, case, "forget_label")
    retain_data_path = settings.get_dataset_path(dataset_name, case, "retain_data")
    retain_label_path = settings.get_dataset_path(dataset_name, case, "retain_label")
    # forget_cls_data_path = settings.get_dataset_path(dataset_name, case, "forget_cls_data")
    # forget_cls_label_path = settings.get_dataset_path(dataset_name, case, "forget_cls_label")

    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)

    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)

    test_remain_data, test_remain_labels, test_forget_data, test_forget_labels = (
        split_test_data_by_forget(test_data, test_labels, forget_classes))

    # class level retain中不包含任何forget class的数据; instance level retain中包含forget class中未被选择的数据
    retain_data, retain_labels, forget_data, forget_labels = sample_class_forget_data(
        train_data, train_labels, forget_classes, forget_ratio=forget_ratio)

    # forget_cls_data, forget_cls_labels = gen_forget_cls_data(test_data, test_labels, forget_classes)

    # forget_cls_data = np.concatenate([forget_cls_data, forget_data], axis=0)
    # forget_cls_labels = np.concatenate([forget_cls_labels, forget_labels], axis=0)

    subdir = os.path.dirname(forget_data_path)
    os.makedirs(subdir, exist_ok=True)

    # 保存训练数据集
    np.save(train_data_path, train_data)
    np.save(train_label_path, train_labels)

    # 保存测试数据集
    np.save(test_data_path, test_data)
    np.save(test_label_path, test_labels)

    np.save(test_remain_data_path, test_remain_data)
    np.save(test_remain_label_path, test_remain_labels)

    np.save(test_forget_data_path, test_forget_data)
    np.save(test_forget_label_path, test_forget_labels)

    np.save(forget_data_path, forget_data)
    np.save(forget_label_path, forget_labels)

    # np.save(forget_cls_data_path, forget_cls_data)
    # np.save(forget_cls_label_path, forget_cls_labels)

    np.save(retain_data_path, retain_data)
    np.save(retain_label_path, retain_labels)

    return train_labels


if __name__ == '__main__':
    dataset = 'flower-102'
    forget_classes = [50, 72, 76, 88, 93]
    case = settings.get_case(0.5)
    test_data_path = settings.get_dataset_path(dataset, None, "test_data")
    test_label_path = settings.get_dataset_path(dataset, None, "test_label")
    forget_cls_data_path = settings.get_dataset_path(dataset, case, "forget_cls_data")
    forget_cls_label_path = settings.get_dataset_path(dataset, case, "forget_cls_label")
    test_data = np.load(test_data_path)
    test_labels = np.load(test_label_path)
    forget_cls_data, forget_cls_labels = gen_forget_cls_data(test_data, test_labels, forget_classes)
    np.save(forget_cls_label_path, forget_cls_labels)


