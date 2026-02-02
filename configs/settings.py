import os

from configs.Config import Constant


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

dataset_paths = {
    Constant.CIFAR10: os.path.join(root_dir, "data", Constant.CIFAR10),
    Constant.CIFAR100: os.path.join(root_dir, "data", Constant.CIFAR100),
    Constant.FOOD101: os.path.join(root_dir, "data", Constant.FOOD101),
    Constant.FLOWER102: os.path.join(root_dir, "data", Constant.FLOWER102),
    Constant.PET37: os.path.join(root_dir, "data", Constant.PET37),
}

num_classes_dict = {
    Constant.CIFAR10: 10,
    Constant.CIFAR100: 100,
    Constant.FOOD101: 101,
    Constant.FLOWER102: 102,
    Constant.PET37: 37,
    Constant.COUNTRY211: 201,
    Constant.IMAGENETTE: 10,
    Constant.GTSRB: 43,
    Constant.STL10: 10,
}

forget_classes_dict = {
    Constant.CIFAR10: [4, 8],
    # Constant.CIFAR100: [10, 30, 50, 70, 90],
    # Constant.FOOD101: [1, 3, 5, 7, 9],
    Constant.STL10: [1, 8],
    Constant.FLOWER102: [54, 64, 74, 84, 94],
    Constant.PET37: [1, 4, 14, 24, 28],
}

normalize_config = {
    Constant.CIFAR10: {"mean": [0.491, 0.482, 0.446], "std": [0.2023, 0.1994, 0.2010]},
    Constant.CIFAR100: {"mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]}
}

# cifar10_config = {"mean": [0.491, 0.482, 0.446], "std": [0.2023, 0.1994, 0.2010]}
#
# cifar100_config = {"mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]}
#
# food101_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

img_sizes = {
    Constant.CIFAR10: (32, 32),
    Constant.CIFAR100: (32, 32),
    Constant.FLOWER102: (224, 224),
    Constant.PET37: (224, 224),
    Constant.COUNTRY211: (224, 224),
    Constant.IMAGENETTE: (224, 224),
    Constant.GTSRB: (224, 224),
    Constant.STL10: (224, 224),
}

fig_titles = {
    'fisher': 'FF',
    'GA_l1': 'L1-SP'
}

similar_distance = 0.95
# similar_distance_low = 0.5

loss_alpha = 4
# class_loss_alpha = 0.001
instance_loss_alpha = 1

# visual
tsne_samples = 75


def get_case(forget_ratio=0.5, suffix=Constant.WWW):
    return f"forget_{forget_ratio}_{suffix}"


def get_ckpt_path(dataset, case, model, model_suffix, step=None, unique_name=None):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "ckpt", dataset)
    if case is not None:
        path = os.path.join(path, case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")
    if unique_name is not None:
        path = os.path.join(path, unique_name)

    return os.path.join(path, f"{model}_{model_suffix}.pth")


def get_visual_result_path(dataset, case, unique_name, model, model_suffix, type_name):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "result_visual", dataset)
    if case is not None:
        path = os.path.join(path, case)

    return os.path.join(path, f"{unique_name}_{model}_{model_suffix}_{type_name}.pdf")


# get ckpt files for sensitivity experiment models
def get_pretrain_ckpt_path(
    dataset, case, model, model_suffix, step=None, unique_name=None
):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "ckpt", dataset, case)
    pretrain_case = "pretrain"
    path = os.path.join(path, pretrain_case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")
    if unique_name is not None:
        path = os.path.join(path, unique_name)

    return os.path.join(path, f"{model}_{model_suffix}.pth")


def get_dataset_path(dataset, case, type, step=None):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "data", dataset, "gen")
    if case is not None:
        path = os.path.join(path, case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")

    return os.path.join(path, f"{type}.npy")
