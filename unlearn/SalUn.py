import time
import torch
import utils
from train_test_utils import test_model
from .impl import iterative_unlearn
from utils import get_x_y_from_data_dict

import numpy as np

@iterative_unlearn
def SalUn(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # global flag
    # if flag == False:
    #     unlearn_train_loader = data_loaders["forget"]
    #     save_gradient_ratio(unlearn_train_loader, model, criterion, args)
    #     flag = True

    remain_class = np.setdiff1d(np.arange(args.num_classes, dtype=np.int64), args.class_to_replace)

    # threshold = 0.8
    # mask = torch.load(f'./save/{args.dataset}/mask_threshold_{threshold}.pt')

    train_loader = data_loaders["forget"]
    print(len(train_loader))
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            # image, target = get_x_y_from_data_dict(data, device)
            x, _ = get_x_y_from_data_dict(data, device)
            y = torch.from_numpy(np.random.choice(remain_class, size=x.shape[0])).cuda()
            
            image, target = x, y
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)

            # loss = -criterion(output_clean, target)
            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            # target = target.cuda()
            target = torch.as_tensor(np.random.choice(remain_class, size=image.shape[0])).cuda()

            # compute output
            output_clean = model(image)
            # loss = -args.alpha * criterion(output_clean, target)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    test_loader = data_loaders["test"]
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    test_model(model, test_loader, criterion, device, epoch)

    return top1.avg