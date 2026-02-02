import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate datasets.")

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name, choose from: cifar-10, cifar-100, flower-102, pet-37, country-211, imagenette"
    )

    parser.add_argument(
        "--forget_classes",
        nargs='+',
        type=int,
        help="forget_classes",
    )

    parser.add_argument(
        "--forget_ratio",
        type=float,
        default=0.5,
        help="原始忘记比例（默认 0.5）"
    )

    parser.add_argument(
        "--ft_forget_ratio",
        type=float,
        default=0.5,
        help="finetune时,从原始forget数据集抽取的数据比例（默认 0.5）"
    )

    parser.add_argument(
        "--ft_test_ratio",
        type=float,
        default=0.25,
        help="finetune时,从原始test数据集抽取的数据比例（默认 0.25）"
    )

    return parser.parse_args()
