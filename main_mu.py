import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from unlearn.gen_mask import save_gradient_ratio
import numpy as np

import arg_parser
import unlearn
from nets.datasetloader import get_dataset_loader
from configs import settings
from nets.custom_model import load_custom_model, ClassifierWrapperHooker
from nets.train_test import model_test, model_forward


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    args = arg_parser.parse_args()

    case = settings.get_case(args.forget_ratio)

    uni_name = args.uni_name
    num_classes = settings.num_classes_dict[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # _, _, train_loader = get_dataset_loader(
    #     args.dataset,
    #     "train",
    #     None,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    # )

    _, _, retain_loader = get_dataset_loader(
        args.dataset,
        "retain",
        case,
        batch_size=args.batch_size,
        shuffle=True,
    )

    _, _, forget_loader = get_dataset_loader(
        args.dataset,
        "forget",
        case,
        batch_size=args.batch_size,
        shuffle=True,
    )

    _, _, test_loader = get_dataset_loader(
        args.dataset,
        "forget",
        case,
        batch_size=args.batch_size,
        shuffle=False,
    )

    _, _, val_loader = get_dataset_loader(
        args.dataset,
        ["test"],
        [None],
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.unlearn_after_ft:
        load_model_path = settings.get_ckpt_path(
            args.dataset,
            case,
            args.model,
            model_suffix="ul_ft",
            unique_name=uni_name,
        )

        save_model_path = settings.get_ckpt_path(
            args.dataset,
            case,
            args.model,
            model_suffix="ul_ft_ul",
            unique_name=uni_name,
        )

    else:
        load_model_path = settings.get_ckpt_path(
            args.dataset,
            None,
            args.model,
            model_suffix="train"
        )

        save_model_path = settings.get_ckpt_path(
            args.dataset,
            case,
            args.model,
            model_suffix="ul",
            unique_name=uni_name,
        )

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    unlearn_method = unlearn.get_unlearn_method(uni_name)

    loaded_model = load_custom_model(
        args.model, num_classes, load_pretrained=False
    )
    model = ClassifierWrapperHooker(loaded_model, num_classes)
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    mask = None
    if args.mask_thresh > 0:
        _, _, train_loader = get_dataset_loader(
            args.dataset,
            "train",
            None,
            batch_size=args.batch_size,
            shuffle=True,
        )
        mask = save_gradient_ratio(train_loader, model, criterion, args)
        mask = mask[args.mask_thresh]
    # before unlearn
    print('************** before unlearn *****************')
    print('-------- forget_loader %d---------' % len(test_loader.dataset))
    model_test(test_loader, model, device)
    print('-------- test_loader %d ---------'% len(val_loader.dataset))
    model_test(val_loader, model, device)

    print('*************** training *****************')
    unlearn_method(unlearn_data_loaders, model, criterion, args, mask)

    # after unlearn, save forget features
    if not args.unlearn_after_ft:
        predicts, probs, embeddings = model_forward(test_loader, model, device, output_embedding=True)
        save_dir = os.path.dirname(save_model_path)
        os.makedirs(save_dir, exist_ok=True)
        save_features_path = os.path.join(save_dir, 'forget_data_unlearn_features.npy')
        np.save(save_features_path, embeddings)

    print('************** after unlearn test_loader *****************')
    print('-------- forget_loader %d---------' % len(test_loader.dataset))
    model_test(test_loader, model, device)
    print('-------- test_loader %d ---------' % len(val_loader.dataset))
    model_test(val_loader, model, device)

    # save model
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print("model saved to:", save_model_path)


if __name__ == "__main__":
    main()
