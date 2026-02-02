import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

from configs import settings
from unlearn.impl import iterative_unlearn


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


def get_representation_matrix(net, batch_list=[24, 100, 100, 125, 125, 125]):
    mat_list = []
    for i, (layer_name, in_act_map) in enumerate(net.in_act.items()):
        k = 0
        torch.cuda.synchronize()  # 强制等待所有CUDA kernel完成
        act = in_act_map.detach().cpu().numpy()
        bsz = min(batch_list[i], act.shape[0])
        if 'feature' in layer_name:
            ksz = net.ksize[layer_name]
            s = compute_conv_output_size(net.in_map_size[layer_name], ksz)
            mat = np.zeros((ksz * ksz * net.in_channel[layer_name], s * s * bsz))
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:, k] = act[kk, :, ii:ksz + ii, jj:ksz + jj].reshape(-1)
                        k += 1
            mat_list.append(mat)
        else:
            activation = act[0:bsz].transpose()
            mat_list.append(activation)
    return mat_list


def update_GPM(mat_list, threshold, feature_list=[]):
    print('Threshold: ', threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold)  # +1
            feature_list.append(U[:, 0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total
            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print('Skip Updating GPM for layer: {}'.format(i + 1))
                continue
            # update GPM
            Ui = np.hstack((feature_list[i], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, 0:Ui.shape[0]]
            else:
                feature_list[i] = Ui

    print('-' * 40)
    print('Gradient Constraints Summary')
    print('-' * 40)
    for i in range(len(feature_list)):
        print('Layer {} : {}/{}'.format(i + 1, feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-' * 40)
    return feature_list


def get_pseudo_label(args, model, x):
    masked_output = model(x)
    masked_output[:, args.class_to_replace] = -np.inf
    pseudo_labels = torch.topk(masked_output, k=1, dim=1).indices
    return pseudo_labels.reshape(-1)


@iterative_unlearn
def UNSC_iter(data_loaders, model, criterion, optimizer, epoch, args, mask,
              proj_mat_list, test_model=None):
    # retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    # retain_dataset = retain_loader.dataset
    # nb_retain = len(retain_dataset)
    # nb_samples = min(len(forget_loader.dataset) * 10, len(retain_dataset), args.WF_N)
    # subset = torch.utils.data.Subset(retain_dataset, np.random.choice(np.arange(nb_retain), size=nb_samples))
    # subset = torch.utils.data.ConcatDataset([subset, forget_loader.dataset])
    # train_loader = DataLoader(subset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    torch.cuda.empty_cache()  # 评估前后调用
    test_model.eval()
    for batch, (x, y) in enumerate(forget_loader):
        if batch > args.forget_batch:
            break
        x = x.cuda()
        y = get_pseudo_label(args, test_model, x)
        pred_y = model(x)
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if len(params.size()) != 1:
                sz = params.grad.data.size(0)
                params_mt = params.grad.data.view(sz, -1)
                proj_mat = proj_mat_list[0][kk].cuda()
                param_sz = params_mt.size(1)
                proj_sz = proj_mat.size(0)
                if param_sz == proj_sz:
                    params.grad.data = params.grad.data - torch.mm(params_mt, proj_mat).view(params.size())
                kk += 1
            elif len(params.size()) == 1:
                params.grad.data.fill_(0)
        optimizer.step()
    print('[train] epoch {}, batch {}, loss {}'.format(epoch, batch, loss))

    return loss


def UNSC(data_loaders, model: nn.Module, criterion, args, mask=None):
    # %%
    # num_classes = settings.num_classes_dict[args.dataset]
    retain_loader = data_loaders["retain"]

    Proj_mat_lst = []
    for i in range(3):
        torch.cuda.empty_cache()  # 评估前后调用
        model.eval()
        merged_feat_mat = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(retain_loader):
                x = x.cuda()
                _ = model(x)
                n_acts = len(model.in_act.items())
                mat_list = get_representation_matrix(model, batch_list=[args.batch_size] * n_acts)
                break

        threshold = 0.99
        merged_feat_mat = update_GPM(mat_list, threshold, merged_feat_mat)
        proj_mat = [torch.Tensor(np.dot(layer_basis, layer_basis.transpose())) for layer_basis in merged_feat_mat]
        Proj_mat_lst.append(proj_mat)

    device = model.parameters().__next__().device
    test_model = deepcopy(model).to(device)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()

    return UNSC_iter(data_loaders, model, criterion, args=args, mask=mask,
                     proj_mat_list=Proj_mat_lst, test_model=test_model)
