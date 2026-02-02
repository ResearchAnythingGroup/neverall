import os
import copy

from torchvision import models
from torch import nn
import torch
import time
import numpy as np
from torch.utils.data import Dataset

import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import utils

import arg_parser
from nets.train import get_loader, SimpleLipNet, lip_train, lip_test, get_loader_by_data
from models.VGG_LTH import vgg16_bn_lth


def train(train_loader, model, model_path, args):
    loss_fn = nn.CrossEntropyLoss()
    top1 = utils.AverageMeter()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # switch to train mode
    model.train()

    start = time.time()

    losses = utils.AverageMeter()

    for epoch in range(1, args.epochs + 1):
        for i, (image, target) in enumerate(train_loader):
            image = image.cuda()
            # target = target.cuda()
            target = target.long().cuda()

            # compute output
            output_clean = model(image)

            loss = loss_fn(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            # if (i + 1) % 10 == 0:
            #     end = time.time()
            #     print(
            #         "Epoch: [{0}][{1}/{2}]\t"
            #         "Epoch: [{0}][{1}/{2}]\t"
            #         "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            #         "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
            #         "Time {3:.2f}".format(
            #             epoch, i, len(train_loader), end - start, loss=losses, top1=top1
            #         )
            #     )
            #     start = time.time()

        # print("unlearn model train_accuracy {top1.avg:.3f}".format(top1=top1))

    state = {"state_dict": model.state_dict()}
    torch.save(state, model_path)

    return top1.avg


def test(test_loader, model):
    model.eval()
    outputs = []

    for i, (image, target) in enumerate(test_loader):
        image = image.cuda()

        output = model(image)

        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        outputs.append(output)

    return np.concatenate(outputs, axis=0)


class CustomDataset(Dataset):
    def __init__(self, data_path, label_path):
        data = np.load(data_path).astype(np.float32)
        data = np.transpose(data, [0, 3, 1, 2])
        self.data = data / 255
        self.label = np.load(label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def get_acc(test_preds, forget_preds):
    eval_results = {}
    test_label = np.load(os.path.join(args.test_data_dir, "test_label.npy"))
    test_acc_unlearn = np.mean(test_label == test_preds)
    eval_results["test_acc"] = test_acc_unlearn

    forget_label = np.load(os.path.join(args.test_data_dir, "forget_label.npy"))
    forget_acc_unlearn = np.mean(forget_preds == forget_label)
    eval_results["forget_acc"] = forget_acc_unlearn

    print("***********************************************************")
    print(
        "test_acc: %.2f, forget_acc: %.2f"
        % (test_acc_unlearn * 100, forget_acc_unlearn * 100)
    )

    label_list = sorted(list(set(forget_label)))
    for label in label_list:
        cls_index = forget_label == label
        forget_acc_unlearn_cls = np.mean(
            forget_preds[cls_index] == forget_label[cls_index]
        )
        print("label: %s, forget_acc: %.2f" % (label, forget_acc_unlearn_cls * 100))
        label_name = "forget_label_%d" % label
        eval_results[label_name] = forget_acc_unlearn_cls

    return forget_acc_unlearn, eval_results


def main():
    # load test data
    test_data_path = os.path.join(args.test_data_dir, "test_data.npy")
    test_label_path = os.path.join(args.test_data_dir, "test_label.npy")
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path)

    # load forget data
    forget_data_path = os.path.join(args.test_data_dir, "forget_data.npy")
    forget_label_path = os.path.join(args.test_data_dir, "forget_label.npy")
    forget_data = np.load(forget_data_path)
    forget_label = np.load(forget_label_path)

    # load unlearn model
    if args.arch == "resnet18":
        unlearn_model = models.resnet18(pretrained=False, num_classes=args.num_classes)
        if args.dataset == "fashionMNIST":
            unlearn_model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    elif args.arch == "vgg16_bn_lth":
        unlearn_model = vgg16_bn_lth(num_classes=args.num_classes)
    model_path = os.path.join(args.save_dir, args.unlearn + "checkpoint.pth.tar")
    model_path_ft = os.path.join(args.save_dir, args.unlearn + "checkpoint_ft.pth.tar")
    checkpoint = torch.load(model_path)
    unlearn_model.load_state_dict(checkpoint["state_dict"], strict=False)
    unlearn_model.cuda()

    # load lip net
    resnet = models.resnet18(pretrained=False, num_classes=512)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    lip_model = SimpleLipNet(resnet, 512, args.num_classes, [512])
    lip_model.cuda()

    ckpt_path = os.path.join(args.lip_save_dir, "checkpoint.pth.tar")
    lip_ckpt_path_ft = os.path.join(args.save_dir, "lipnet_checkpoint_ft.pth.tar")
    checkpoint = torch.load(ckpt_path)
    lip_model.load_state_dict(checkpoint["state_dict"], strict=False)

    # dataloader
    test_loader = get_loader("test", args.test_data_dir, args.batch_size, args.dataset)
    forget_loader = get_loader(
        "forget", args.test_data_dir, args.batch_size, args.dataset
    )

    one_channel = False
    if args.dataset == "fashionMNIST":
        one_channel = True
        test_loader_unlearn = get_loader(
            "test",
            args.test_data_dir,
            args.batch_size,
            args.dataset,
            one_channel=one_channel,
        )
        forget_loader_unlearn = get_loader(
            "forget",
            args.test_data_dir,
            args.batch_size,
            args.dataset,
            one_channel=one_channel,
        )
    else:
        test_loader_unlearn = copy.deepcopy(test_loader)
        forget_loader_unlearn = copy.deepcopy(forget_loader)

    # lip forget predicts
    test_embeddings, lip_test_pred = lip_test(test_loader, lip_model)
    forget_embeddings, lip_forget_pred = lip_test(forget_loader, lip_model)

    print("Before fine-tune:")
    print("unlearn model: %s, dataset: %s" % (args.unlearn, args.dataset))

    test_preds = test(test_loader_unlearn, unlearn_model)
    forget_preds = test(forget_loader_unlearn, unlearn_model)
    # unlearn acc
    print("=======================unlearn model acc============================")
    unlearn_acc_before, _ = get_acc(test_preds, forget_preds)

    # lip acc
    print("=======================lipnet acc=================================")
    lip_acc_before, lip_results_before = get_acc(lip_test_pred, lip_forget_pred)
    # save lipnet acc before
    eval_result_before = {}
    eval_result_before["accuracy"] = lip_results_before
    eval_path_before = os.path.join(args.save_dir, "lipnet_eval_result.pth.tar")
    torch.save(eval_result_before, eval_path_before)

    if args.ft_um_only:
        forget_acc_before = unlearn_acc_before
    else:
        forget_acc_before = lip_acc_before

    if args.finetune_unlearn:
        print("Finetuning...")
        top_forget_acc = forget_acc_before
        early_stop_num = 0

        for iter in range(200):
            print(
                "-----------------------------------finetune iterate : %d -----------------------------"
                % (iter + 1)
            )

            # forget lip predicts & unlearn predict
            inter_index = lip_forget_pred == forget_preds
            print("sum forget inter index: ", sum(inter_index))

            # test lip predicts & unlearn predict
            test_inter_index = lip_test_pred == test_preds
            print("sum test inter index: ", sum(test_inter_index))

            # forget_inter_loader = get_loader_by_data(
            #     "inter",
            #     args.batch_size,
            #     args.dataset,
            #     forget_data,
            #     lip_forget_pred,
            #     inter_index,
            #     label_true=forget_label,
            #     shuffle=True,
            # )

            forget_inter_add_test_loader = get_loader_by_data(
                "inter_and",
                args.batch_size,
                args.dataset,
                forget_data,
                forget_preds,
                inter_index,
                label_true=forget_label,
                fit_embedding=test_embeddings,
                query_embedding=forget_embeddings,
                add_data_all=test_data,
                add_label_all=test_label,
                one_channel=one_channel,
                add_inter_index=test_inter_index,
                shuffle=True,
            )

            forget_inter_add_test_loader_lip = forget_inter_add_test_loader
            if one_channel:
                forget_inter_add_test_loader_lip = get_loader_by_data(
                    "inter_and",
                    args.batch_size,
                    args.dataset,
                    forget_data,
                    forget_preds,
                    inter_index,
                    label_true=forget_label,
                    fit_embedding=test_embeddings,
                    query_embedding=forget_embeddings,
                    add_data_all=test_data,
                    add_label_all=test_label,
                    one_channel=False,
                    add_inter_index=test_inter_index,
                    shuffle=True,
                )

            if not args.ft_um_only:
                # train lip model
                lip_train(
                    forget_inter_add_test_loader_lip, lip_model, lip_ckpt_path_ft, args
                )

            if not args.ft_uram_only:
                # train unlearn model
                train(forget_inter_add_test_loader, unlearn_model, model_path_ft, args)

            # print('-----------------after train-----------------------')
            test_preds = test(test_loader_unlearn, unlearn_model)
            forget_preds = test(forget_loader_unlearn, unlearn_model)

            test_embeddings, lip_test_pred = lip_test(test_loader, lip_model)
            forget_embeddings, lip_forget_pred = lip_test(forget_loader, lip_model)

            if args.ft_um_only:
                # unlearn acc
                forget_acc, _ = get_acc(test_preds, forget_preds)
            else:
                # lipnet acc
                forget_acc, _ = get_acc(lip_test_pred, lip_forget_pred)

            if top_forget_acc >= forget_acc:
                early_stop_num += 1
            else:
                top_forget_acc = forget_acc
                early_stop_num = 0

            if early_stop_num == 7:
                break

        if not args.ft_um_only:
            # save lipnet acc
            _, lip_results_ft = get_acc(lip_test_pred, lip_forget_pred)
            eval_result_ft = {}
            eval_result_ft["accuracy"] = lip_results_ft
            eval_path_ft = os.path.join(args.save_dir, "lipnet_eval_result_ft.pth.tar")
            torch.save(eval_result_ft, eval_path_ft)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main()
