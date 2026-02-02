import copy
import time
import torch
from tqdm import tqdm

import utils
import numpy as np

from nets.train_test import model_test


def delete(data_loaders, model, criterion, args, mask=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_bn = False
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    print(f"unlearn_epoch {args.num_epochs}, unlearn_rate {args.unlearn_lr}")
    # print(f"eval option {eval_opt}")

    train_forget_loader = data_loaders["forget"]
    eval_model = copy.deepcopy(model).to(device)

    criterion = torch.nn.KLDivLoss(reduction='batchmean')  # mean是对所有维度平均，batchmean只对batch维度平均，应该使用后者

    optimizer = torch.optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9)

    for epoch in range(args.num_epochs):
        print(f"DELETE unlearn_epoch {epoch}")
        for i, (x, y) in enumerate(train_forget_loader):
            x, y = x.to(device), y.to(device)
            model.train()

            if disable_bn:
                for module in model.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.eval()

            model.zero_grad()
            optimizer.zero_grad()  # 注意清空了模型的梯度和优化器的梯度，保证万无一失

            eval_model.eval()
            batch_size = x.shape[0]
            with torch.no_grad():
                pred_label = eval_model(x)

            pred_label[torch.arange(batch_size), y] = -1e10

            ori_logits = model(x)

            ori_logits = torch.nn.functional.log_softmax(ori_logits, dim=1)  # input log softmax
            pred_label = torch.nn.functional.softmax(pred_label, dim=1)  # target softmax
            loss = criterion(ori_logits, pred_label)
            loss.backward()
            optimizer.step()
            loss = loss.float()
            losses.update(loss.item(), x.size(0))

            # measure accuracy
            output = ori_logits.float()
            prec1 = utils.accuracy(output.data, y)[0]
            top1.update(prec1.item(), x.size(0))

            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                .format(
                    epoch, i, len(train_forget_loader), loss=losses, top1=top1
                )
            )

        test_loader = data_loaders["val"]

        if epoch % 10 == 0:
            print('**************test model in Forget dataset*************')
            model_test(train_forget_loader, model, device)
            print('**************test model in Test dataset*************')
            model_test(test_loader, model, device)
        print(f"epoch {epoch + 1} loss {losses}")

    return top1.avg
