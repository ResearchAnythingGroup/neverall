import numpy as np
import os


def main(data_root, data_name, backbone_name, model_names):
    data_path = os.path.join(data_root, data_name, backbone_name)

    for model_name in model_names:
        if backbone_name == 'vgg16' and model_name == 'FF':
            continue
        model_path = os.path.join(data_path, model_name)

        # 读取 test_data, test_label, test_predicts
        test_data = np.load(os.path.join(data_root, data_name, "test_data.npy"))
        test_label = np.load(os.path.join(data_root, data_name, "test_label.npy"))
        test_data_model = np.load(os.path.join(model_path, "test_data.npy"))
        test_label_model = np.load(os.path.join(model_path, "test_label.npy"))
        test_predicts = np.load(os.path.join(model_path, "test_predicts.npy"))

        sum_test_d = np.sum(test_data == test_data_model)
        # print(sum_test_d == test_data_model.size)
        sum_test_l = np.sum(test_label == test_label_model)
        # print(sum_test_l == test_label.size)

        # 计算 test acc
        test_acc = np.sum(test_label == test_predicts) / len(test_label)

        # 读取 forget_data, forget_label, forget_predicts
        forget_data = np.load(os.path.join(data_root, data_name, "forget_data.npy"))
        forget_label = np.load(os.path.join(data_root, data_name, "forget_label.npy"))
        forget_data_model = np.load(os.path.join(model_path, "forget_data.npy"))
        forget_label_model = np.load(os.path.join(model_path, "forget_label.npy"))
        forget_predicts = np.load(os.path.join(model_path, "forget_predicts.npy"))

        sum_forget_d = np.sum(forget_data == forget_data_model)
        # print(sum_forget_d == forget_data.size)
        sum_forget_l = np.sum(forget_label == forget_label_model)
        # print(sum_forget_l == forget_label.size)

        # 计算 forget acc
        forget_acc = np.sum(forget_label == forget_predicts) / len(forget_label)

        print(data_name, backbone_name, model_name, end='')
        print(' **** forget_acc: %.2f, test_acc: %.2f' % (forget_acc*100, test_acc*100))


if __name__ == "__main__":
    path = '../../data'
    unlearn_model_names = ['retrain', 'FT', 'FF', 'GA', 'IU', 'FT_prune']
    data_list = ['cifar10', 'cifar100', 'tinyimgnet', 'fmnist']
    backbones = ['resnet18', 'vgg16']
    for data in data_list:
        for backbone in backbones:
            main(path, data, backbone, unlearn_model_names)
