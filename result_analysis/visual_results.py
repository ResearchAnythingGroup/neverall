import argparse
import copy

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os

from arg_parser import parse_args
from configs import settings
from nets.custom_model import ClassifierWrapper, load_custom_model
from nets.datasetloader import get_dataset_loader
from nets.train_test import model_forward, model_test
from configs.Config import Constant


def execute(args):
    case = settings.get_case(args.forget_ratio)
    uni_names = args.uni_name
    uni_names = [uni_names] if uni_names is None or len(uni_names) < 2 else uni_names.split(",")
    num_classes = settings.num_classes_dict[args.dataset]
    forget_cls = settings.forget_classes_dict[args.dataset]

    device = torch.device(Constant.CUDA if torch.cuda.is_available() else Constant.CPU)

    # 1. load 3 model:  ul_model(unlearn model)、inc_model(incremental ft model)、nr_model(neverecall model)
    loaded_model = load_custom_model(args.model, num_classes, load_pretrained=False)
    ul_model = ClassifierWrapper(loaded_model, num_classes)
    inc_model = copy.deepcopy(ul_model)
    nr_model = copy.deepcopy(ul_model)

    ul_model.to(device)
    inc_model.to(device)
    nr_model.to(device)

    # 2. load dataset
    # pred_loader for t-SNE and cmt(confusion matrix)
    _, _, pred_loader = get_dataset_loader(
        args.dataset,
        [Constant.TEST_DATA_WITHOUT_FORGET, Constant.FORGET_DATA, Constant.INC_DATA_TRANSFORM],
        [None, case, case],
        batch_size=args.batch_size,
        shuffle=False
    )
    # forget_loader for acc bar
    _, _, forget_loader = get_dataset_loader(
        args.dataset,
        Constant.FORGET_DATA,
        case,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 3. iterate the unlearn models (exp: --uni_name RL GA IU BU SalUn DELETE)
    for uni_name in uni_names:
        print(f"Evaluating {uni_name}:")

        # 3.1 load ckpt for 3 models
        model_case = None if args.model_suffix in [Constant.TRAIN, Constant.PRETRAIN] else settings.get_case(args.forget_ratio)
        ul_ckpt_path = settings.get_ckpt_path(
            args.dataset,
            model_case,
            args.model,
            model_suffix=args.model_suffix,
            unique_name=uni_name,
        )
        print(f"Loading model from {ul_ckpt_path}")
        ul_checkpoint = torch.load(ul_ckpt_path)
        ul_model.load_state_dict(ul_checkpoint, strict=False)
        ul_model.eval()

        inc_ckpt_path = settings.get_ckpt_path(
            args.dataset,
            model_case,
            args.model,
            model_suffix=Constant.INC_SUFFIX,
            unique_name=uni_name,
        )
        print(f"Loading model from {inc_ckpt_path}")
        inc_checkpoint = torch.load(inc_ckpt_path)
        inc_model.load_state_dict(inc_checkpoint, strict=False)
        inc_model.eval()

        nr_ckpt_path = settings.get_ckpt_path(
            args.dataset,
            model_case,
            args.model,
            model_suffix=Constant.NRC_SUFFIX,
            unique_name=uni_name,
        )
        print(f"Loading model from {nr_ckpt_path}")
        nr_checkpoint = torch.load(nr_ckpt_path)
        nr_model.load_state_dict(nr_checkpoint, strict=False)
        nr_model.eval()

        # 3.2 get pred_loader's predicts by 3 models
        ul_pred_predicts, _, ul_pred_embedding, ul_pred_labels = model_forward(
            pred_loader, ul_model, device, output_embedding=True, output_targets=True
        )

        inc_pred_predicts, _, inc_pred_embedding, inc_pred_labels = model_forward(
            pred_loader, inc_model, device, output_embedding=True, output_targets=True
        )

        nr_pred_predicts, _, nr_pred_embedding, nr_pred_labels = model_forward(
            pred_loader, nr_model, device, output_embedding=True, output_targets=True
        )

        # 3.3 draw t-SNE (data from pred_loader)
        title = settings.fig_titles[uni_name] if uni_name in settings.fig_titles.keys() else uni_name

        # unlearn model
        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, Constant.ULM, Constant.TSNE)
        subdir = os.path.dirname(save_path)
        os.makedirs(subdir, exist_ok=True)
        sample_idx = np.random.choice(len(ul_pred_embedding), size=num_classes * settings.tsne_samples, replace=True)
        sample_idx = np.unique(sample_idx)
        show_tsne(ul_pred_embedding[sample_idx], ul_pred_labels[sample_idx], forget_cls, title=title, save_path=save_path)

        # incremental ft model
        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, Constant.INCM, Constant.TSNE)
        show_tsne(inc_pred_embedding[sample_idx], inc_pred_labels[sample_idx], forget_cls, title=title, save_path=save_path)

        # neverecall model
        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, Constant.NRM, Constant.TSNE)
        show_tsne(nr_pred_embedding[sample_idx], nr_pred_labels[sample_idx], forget_cls, title=title, save_path=save_path)

        # 3.4 draw confusion matrix (data from pred_loader)
        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, Constant.ULM, Constant.CMT)
        show_conf_mt(ul_pred_labels, ul_pred_predicts, forget_cls, title=title, save_path=save_path)

        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, Constant.INCM, Constant.CMT)
        show_conf_mt(inc_pred_labels, inc_pred_predicts, forget_cls, title=title, save_path=save_path)

        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, Constant.NRM, Constant.CMT)
        show_conf_mt(nr_pred_labels, nr_pred_predicts, forget_cls, title=title, save_path=save_path)

        # 3.5 get forget_loader's predicts by 3 models
        ul_forget_predicts, _, _, ul_forget_labels = model_forward(
            forget_loader, ul_model, device, output_embedding=True, output_targets=True
        )

        inc_forget_predicts, _, _, inc_forget_labels = model_forward(
            forget_loader, inc_model, device, output_embedding=True, output_targets=True
        )

        nr_forget_predicts, _, _, nr_forget_labels = model_forward(
            forget_loader, nr_model, device, output_embedding=True, output_targets=True
        )

        # 3.6 draw acc bar (data from forget_loader)
        ul_acc = evals_cls_acc(ul_forget_labels, ul_forget_predicts, forget_cls)
        inc_acc = evals_cls_acc(inc_forget_labels, inc_forget_predicts, forget_cls)
        nr_acc = evals_cls_acc(nr_forget_labels, nr_forget_predicts, forget_cls)

        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, Constant.ALL, Constant.BAR)
        # show_bars_two(ft_mia, neverecall_mia, forget_cls, title=title, save_path=save_path)
        show_bars(ul_acc, inc_acc, nr_acc, forget_cls, title=title, save_path=save_path)


def evals_classification(y_true, y_pred):
    eval_results = []

    # global acc
    global_acc = np.mean(y_true == y_pred)
    global_acc.item()

    # class acc
    label_list = sorted(list(set(y_true)))
    for label in label_list:
        cls_index = y_true == label
        class_acc = np.mean(y_pred[cls_index] == y_true[cls_index])
        eval_results.append(class_acc.item())

    return global_acc.item(), (label_list, eval_results)


def evals_cls_acc(y_true, y_pred, forget_cls):
    eval_results = []

    for label in forget_cls:
        cls_index = y_true == label
        mean_acc = np.mean(y_pred[cls_index] == label)
        if np.isnan(mean_acc):
            mean_acc = 0
        eval_results.append(mean_acc)

    return eval_results


# def show_bars_two(bar_data_front, bar_data_back, forget_cls, size=(5, 5), title=None, save_path=None):
#     x_labels = [f"C{y}" for y in forget_cls]
#     df1 = pd.DataFrame({"Type": "INCM", "Acc": bar_data_front, "Forgetting Classes": x_labels})
#     df2 = pd.DataFrame({"Type": "NRM", "Acc": bar_data_back, "Forgetting Classes": x_labels})
#     df = pd.concat([df1, df2], axis=0)
#
#     plt.clf()
#     ax = sn.barplot(df, x="Forgetting Classes", y="Acc", hue="Type",
#                     palette="Set1")
#     ax.bar_label(ax.containers[0], fontsize=10, fmt="%.2f")
#     ax.bar_label(ax.containers[1], fontsize=10, fmt="%.2f")
#     y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     ax.set_yticks(y_ticks)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     ax.xaxis.label.set_size(14)
#     ax.yaxis.label.set_size(14)
#     legend = ax.legend(fontsize=12)
#     legend.set_title('')
#     ax.figure.set_size_inches(size)
#
#     if title is not None:
#         ax.set_title(title, fontdict={'size': 24, 'weight': 'bold'})
#
#     if save_path is not None:
#         ax.figure.savefig(save_path)
#     else:
#         plt.show()


def show_bars(bar_data_front, bar_data_middle, bar_data_back, forget_cls, size=(7, 7), title=None, save_path=None):
    x_labels = [f"C{y}" for y in forget_cls]
    df1 = pd.DataFrame({"Type": "ULM", "Acc": bar_data_front, "Forgetting Classes": x_labels})
    df2 = pd.DataFrame({"Type": "ITM", "Acc": bar_data_middle, "Forgetting Classes": x_labels})
    df3 = pd.DataFrame({"Type": "DUE", "Acc": bar_data_back, "Forgetting Classes": x_labels})
    df = pd.concat([df1, df2, df3], axis=0)

    plt.clf()
    ax = sn.barplot(df, x="Forgetting Classes", y="Acc", hue="Type",
                    palette="Set1")
    ax.bar_label(ax.containers[0], fontsize=10, fmt="%.2f")
    ax.bar_label(ax.containers[1], fontsize=10, fmt="%.2f")
    ax.bar_label(ax.containers[2], fontsize=10, fmt="%.2f")
    y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(y_ticks)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    legend = ax.legend(fontsize=16)
    legend.set_title('')
    ax.figure.set_size_inches(size)

    if title is not None:
        ax.set_title(title, fontdict={'size': 30, 'weight': 'bold'})

    if save_path is not None:
        ax.figure.savefig(save_path)
    else:
        plt.show()


def show_conf_mt(y_true, y_pred, forget_cls, size=(7, 7), title=None, save_path=None):
    y_true = [y if y in forget_cls else -1 for y in y_true]
    y_pred = [y if y in forget_cls else -1 for y in y_pred]
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    tick_labels = ["Other"] + [f"C{y}" for y in forget_cls]

    plt.clf()

    ax = sn.heatmap(cm, xticklabels=tick_labels, yticklabels=tick_labels,
                    annot=True, fmt='.2f', cbar=False, annot_kws={"fontsize": 14})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax.figure.set_size_inches(size)

    if title is not None:
        ax.set_title(title, fontdict={'size': 30, 'weight': 'bold'})

    if save_path is not None:
        ax.figure.savefig(save_path)
    else:
        plt.show()


def show_tsne(embeddings, labels, forget_cls, size=(5, 5), title=None, save_path=None):
    # Apply t-SNE using MulticoreTSNE for speedup
    tsne_data = TSNE(n_components=2).fit_transform(embeddings).T
    # styles = ["Forgotten" if y in forget_cls else "Others" for y in labels]
    labels = [f"C{y}" if y in forget_cls else "Others" for y in labels]
    tsne_df = pd.DataFrame({"x": tsne_data[0], "y": tsne_data[1], "Class": labels})

    # Plotting the result of tsne
    plt.clf()

    custom_order = sorted(list(set(labels)))

    ax = sn.scatterplot(data=tsne_df, x='x', y='y',
                   hue='Class', palette="muted", hue_order=custom_order)
    # hide x-axis
    ax.get_xaxis().set_visible(False)
    # hide y-axis
    ax.get_yaxis().set_visible(False)
    ax.legend().set_title('')
    ax.figure.set_size_inches(size)
    plt.legend(fontsize=10)

    if title is not None:
        ax.set_title(title, fontdict={'size': 24, 'weight': 'bold'})

    if save_path is not None:
        ax.figure.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    try:
        pargs = parse_args()
        execute(pargs)
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
