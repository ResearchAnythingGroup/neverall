import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tiny-imagenet-200",
        help="dir to tiny-imagenet",
        # this para is optional,
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )
    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")

    parser.add_argument("--mask_thresh", default=0, type=float, help="the threshold of saliency map")

    parser.add_argument(
        "--mask_ratio", type=float, default=0.5, help="mask ratio for unlearning"
    )

    parser.add_argument(
        "--WF_N", type=int, default=1000, help="N"
    )

    parser.add_argument(
        "--forget_batch", type=int, default=50, help="How many forget batches are used"
    )

    # ----------------------- todo add new argument--------------------------------------------------
    parser.add_argument(
        "--load_ff", action="store_true", help="load finetune model and unlearn train"
    )
    parser.add_argument(
        "--shuffle", action="store_false", default=True, help="shuffle dataset"
    )
    parser.add_argument(
        "--save_data", action="store_true", default=False, help="save dataset"
    )
    parser.add_argument(
        "--save_data_path", type=str, default="", help="save dataset path"
    )
    parser.add_argument(
        "--lip_save_dir", type=str, default="", help="save lip_net path"
    )
    parser.add_argument(
        "--resume_lipnet",
        action="store_true",
        default=False,
        help="resume lip_net path",
    )
    parser.add_argument(
        "--test_data_dir", type=str, default="", help="lipnet test data path"
    )
    parser.add_argument(
        "--save_forget_dir",
        type=str,
        default="",
        help="save lipnet forget predict path",
    )
    parser.add_argument(
        "--finetune_unlearn",
        action="store_true",
        default=False,
        help="finetune unlearn model",
    )
    parser.add_argument(
        "--eval_result_ft",
        action="store_true",
        default=False,
        help="get eval result by finetune model",
    )
    parser.add_argument(
        "--ft_um_only",
        action="store_true",
        default=False,
        help="only finetune unlearn model",
    )

    parser.add_argument(
        "--ft_uram_only",
        action="store_true",
        default=False,
        help="only finetune lipnet",
    )

    parser.add_argument(
        "--load_um_or_uram",
        type=str,
        default="after",
        help="load um or uram model: um; uram",
    )

    parser.add_argument(
        "--load_before_or_after",
        type=str,
        default="after",
        help="load um or uram model: before; after",
    )

    parser.add_argument(
        "--neverecall",
        action="store_true",
        default=False,
        help="add neverecall model",
    )
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="",
        help="ablation_type: SSR(sensitive sample recognizer); SSGS(sensitive sample gradient suppressor); "
             "SSPR(sensitive sample reverser)",
    )
    parser.add_argument(
        "--unlearn_after_ft",
        action="store_true",
        default=False,
        help="unlearn after finetune",
    )


    # --------------------------------------------------------------------------------------

    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )
    parser.add_argument("--mask", type=str, default=None, help="sparse model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")
    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp", help="method to prune")
    parser.add_argument(
        "--pruning_times",
        default=1,
        type=int,
        help="overall times of pruning (only works for IMP)",
    )
    parser.add_argument(
        "--rate", default=0.95, type=float, help="pruning rate"
    )  # pruning rate is always 20%
    parser.add_argument(
        "--prune_type",
        default="rewind_lt",
        type=str,
        help="IMP type (lt, pt or rewind_lt)",
    )
    parser.add_argument(
        "--random_prune", action="store_true", help="whether using random prune"
    )
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument(
        "--rewind_pth", default=None, type=str, help="rewind checkpoint to load"
    )

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.001, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=10,
        type=int,
        help="number of total epochs for unlearn to run",
    )
    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget",
    )
    # parser.add_argument(
    #     "--class_to_replace", type=str, default="", help="Specific class to forget"
    # )

    parser.add_argument(
        "--class_to_replace",
        nargs='+',
        type=int,
        help="class_to_replace",
    )

    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")

    ##################################### Attack setting #################################################
    parser.add_argument(
        "--attack", type=str, default="backdoor", help="method to unlearn"
    )
    parser.add_argument(
        "--trigger_size",
        type=int,
        default=4,
        help="The size of trigger of backdoor attack",
    )

    ####################################################
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="cifar-10",
        help="Dataset name, choose from: cifar-10, cifar-100, flower-102, tiny-imagenet-200, food-101, pet-37, country-211, imagenette",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="models",
    )

    parser.add_argument(
        "--st_model",
        type=str,
        default=None,
        help="student models",
    )

    parser.add_argument(
        "--no_spnorm",
        action="store_true",
        default=False,
        help="If specified, no spectral norm",
    )

    parser.add_argument(
        "--data_aug",
        action="store_true",
        default=False,
        help="If specified, do data augmentation",
    )

    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["pretrain", "finetune", "train", "retain"],
        help="Train mode",
    )

    parser.add_argument(
        "--forget_ratio",
        type=float,
        # default=0.5,
        help="forget ratio",
    )

    parser.add_argument(
        "--forget_classes",
        nargs='+',
        type=int,
        help="forget_classes",
    )

    # 添加 learning_rate 参数
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)",
    )

    # 添加 optimizer 参数
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="adam",
        help="Optimizer for training weights",
    )

    # 添加 num_epochs 参数
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs to train the model (default: 5)",
    )

    parser.add_argument(
        "--distill_epochs",
        type=int,
        default=5,
        help="Number of unlearning epochs",
    )

    parser.add_argument(
        "--mixup_samples",
        type=int,
        default=3,
        help="Number of mixup samples",
    )

    parser.add_argument(
        "--align_epochs",
        type=int,
        default=3,
        help="The number of epochs to adapt the model",
    )

    parser.add_argument(
        "--lr_student",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)",
    )

    parser.add_argument(
        "--ls_gamma",
        type=float,
        default=0.25,
        help="Label smoothing factor",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Sharpen factor",
    )

    parser.add_argument(
        "--top_conf",
        type=float,
        default=0.05,
        help="Percentage of top confidence",
    )

    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.75,
        help="Mixup factor",
    )

    # 捕获其他 kwargs
    parser.add_argument(
        "--kwargs", nargs="*", help="Additional key=value arguments for hyperparameters"
    )

    parser.add_argument(
        "--model_suffix",
        type=str,
        default=None,
        help="Suffix to save model name",
    )

    parser.add_argument(
        "--no_t_update",
        action="store_false",
        help="No update teacher",
    )

    parser.add_argument(
        "--fig_title",
        type=str,
        default=None,
        help="Figure title",
    )

    parser.add_argument("--test_it", default=2, type=int, help="test iterations")

    parser.add_argument("--uni_name", type=str, default=None, help="Model unique name")

    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Use TensorBoard for logging."
    )
    return parser.parse_args()
