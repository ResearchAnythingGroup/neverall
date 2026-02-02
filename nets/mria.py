import os
from functools import partial

from torch import nn
import torch
import torch.nn.functional as F
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasetloader import get_dataset_loader, MixupDataset, NormalizeDataset
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper
from configs import settings
from train_test import model_test, model_forward, model_train
import torchvision.transforms as transforms

import logging
import utils
import arg_parser


def laplace_smooth(probs, beta=0.02, axis=-1):
    probs += beta
    return probs / np.sum(probs, axis=axis, keepdims=True)


def label_smooth(labels, num_classes, gamma=0.0):
    gm = gamma / num_classes
    label_diag = np.diag(np.ones(num_classes) - gamma)
    return (label_diag + gm)[labels]


def auto_mixup(images, labels=None, alpha=0.75):
    nb_images = len(images)
    beta_dist = torch.distributions.beta.Beta(alpha, alpha)

    lbd = beta_dist.sample(sample_shape=(nb_images,))
    idx = lbd < 0.5
    lbd[idx] = 1 - lbd[idx]
    perm_idx = torch.randperm(nb_images)
    lbd_img = lbd.reshape((-1, 1, 1, 1))
    mixed_img = lbd_img * images + (1 - lbd_img) * images[perm_idx]
    if labels is not None:
        lbd_label = lbd.reshape((-1, 1))
        mixed_labels = lbd_label * labels + (1 - lbd_label) * labels[perm_idx]

    return mixed_img if labels is None else (mixed_img, mixed_labels)


def mixup_label(label1, label2, alpha=0.75, device='cuda'):
    nb_labels = len(label1)
    beta_dist = torch.distributions.beta.Beta(alpha, alpha)
    lbd = beta_dist.sample(sample_shape=(nb_labels, 1)).to(device)
    mixed_labels = lbd * label1 + (1 - lbd) * label2
    return mixed_labels


def mixup(images1, labels1, images2, labels2, alpha=0.75, device='cuda'):
    nb_images = len(images1)
    beta_dist = torch.distributions.beta.Beta(alpha, alpha)
    lbd = beta_dist.sample(sample_shape=(nb_images,)).to(device)
    # idx = lbd < 0.5
    # lbd[idx] = 1 - lbd[idx]
    rnd_idx = torch.randint(low=0, high=len(images2), size=(nb_images,))
    lbd_img = lbd.reshape((-1,1,1,1))
    mixed_img = lbd_img * images1 + (1 - lbd_img) * images2[rnd_idx]
    lbd_label = lbd.reshape((-1, 1))
    mixied_label = lbd_label * labels1 + (1 - lbd_label) * labels2[rnd_idx]

    return mixed_img, mixied_label


def mix_up_dataloader(
    first_data,
    first_probs,
    second_data,
    second_probs,
    batch_size,
    alpha=1.0,
    transforms=None,
    shuffle=True,
    first_max=True
):
    mixed_dataset = MixupDataset(
        data_pair=(first_data, second_data),
        label_pair=(first_probs, second_probs),
        mixup_alpha=alpha,
        transforms=transforms,
        first_max=first_max
    )
    return DataLoader(mixed_dataset, batch_size, drop_last=False, shuffle=shuffle)


def model_distill(model_teacher, model_student, epoch, data_loader,
                  transforms, criterion, optimizer, need_mixup, device):
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1} Distillation") as pbar:
        model_teacher.eval()
        model_student.train()
        for i, (image, target) in enumerate(data_loader):
            # predictions on data augmentation
            img_aug = transforms(image).to(device)
            # image_mixup = auto_mixup(image, None, alpha=0.75, device=device)
            pred_t = model_teacher(img_aug)
            pred_t = F.softmax(pred_t, dim=1)

            pred_infer = model_student(img_aug)
            if need_mixup:
                pred_st = F.softmax(pred_infer, dim=1)
                pred_t = mixup_label(pred_st, pred_t, alpha=0.75, device=device)

            optimizer.zero_grad()
            loss = criterion(pred_infer, pred_t)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)


def get_conf_data_loader(data, num_classes, conf_thresh, probs1, probs2, beta=0.02):
    joint_probs = torch.as_tensor(laplace_smooth(probs1, beta) * laplace_smooth(probs2, beta))
    nb_samples = len(probs1)
    k = max(1, round(conf_thresh * nb_samples / num_classes))
    _, conf_topk = torch.topk(joint_probs, k=k, dim=0)
    conf_topk = conf_topk.numpy().flatten()
    conf_labels = np.tile(np.arange(num_classes), k)
    conf_data = data[conf_topk]
    # agree_labels = np.argmax(conf_probs, axis=-1)
    conf_agree_probs = label_smooth(conf_labels, num_classes, gamma=args.ls_gamma)
    conf_dataset = NormalizeDataset(conf_data, conf_agree_probs)
    return DataLoader(conf_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True)

def mria_train(args):
    num_classes = settings.num_classes_dict[args.dataset]
    # kwargs = parse_kwargs(args.kwargs)
    case = settings.get_case(args.forget_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(settings.root_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_path = os.path.join(log_path, "core_execution.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    learning_rate = args.learning_rate
    lr_student = args.lr_student
    weight_decay = args.weight_decay
    optimizer_type = args.optimizer
    num_epochs = args.num_epochs
    uni_name = args.uni_name
    update_teacher = args.no_t_update

    ul_model_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="ul", unique_name=uni_name,
    )

    model_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="restore", unique_name=uni_name,
    )

    distill_only = args.align_epochs < 1
    if distill_only:
        student_suffix = "distill"
    else:
        student_suffix = "student" if update_teacher else "student_only"

    st_model = args.st_model if args.st_model else args.model
    model_student_path = settings.get_ckpt_path(
        args.dataset, case, st_model, model_suffix=student_suffix, unique_name=uni_name,
    )
    model_student = load_custom_model(st_model, num_classes)
    model_student = ClassifierWrapper(model_student, num_classes)
    optimizer, lr_scheduler = create_optimizer_scheduler(
        optimizer_type,
        model_student.parameters(),
        num_epochs,
        lr_student,
        weight_decay,
    )

    backbone = load_custom_model(args.model, num_classes, load_pretrained=False)
    model_teacher = ClassifierWrapper(backbone, num_classes)
    checkpoint = torch.load(ul_model_path)
    model_teacher.load_state_dict(checkpoint, strict=False)
    t_optimizer, t_lr_scheduler = create_optimizer_scheduler(
        optimizer_type,
        model_teacher.parameters(),
        num_epochs,
        learning_rate,
        weight_decay,
    )

    model_teacher.to(device=device)
    model_student.to(device=device)
    # switch to train mode

    loss_fn = nn.CrossEntropyLoss()

    train_data, train_labels, train_loader = get_dataset_loader(
        args.dataset,
        ["test", "forget"],
        [None, case],
        batch_size=args.batch_size,
        shuffle=False,
    )

    _, _, test_loader = get_dataset_loader(
        args.dataset,
        ["test"],
        [None],
        batch_size=args.batch_size,
        shuffle=False,
    )

    _, _, forget_loader = get_dataset_loader(
        args.dataset,
        ["forget"],
        [case],
        batch_size=args.batch_size,
        shuffle=False,
    )

    _, _, data_loader = get_dataset_loader(
        args.dataset,
        ["test", "forget"],
        [None, case],
        batch_size=args.batch_size,
        shuffle=True,
    )

    auto_mix = partial(auto_mixup, labels=None, alpha=0.2)

    model_test(forget_loader, model_teacher, device)
    model_test(test_loader, model_teacher, device)

    # lr_scheduler = None
    # ul_lr_scheduler = None
    for ep in tqdm(range(num_epochs), desc="MRIA"):
        # Distillation Stage
        print(f"MRIA Epoch {ep}: Distillation Stage")
        for epoch in range(args.distill_epochs):
            need_mixup = (not distill_only) and (ep > 0)
            model_distill(model_teacher, model_student, epoch, data_loader,
                          auto_mix, loss_fn, optimizer, need_mixup, device)
            model_test(forget_loader, model_student, device)
        if lr_scheduler:
            lr_scheduler.step(ep)

        # Alignment Stage
        top_conf = args.top_conf
        iters = 5

        print(f"MRIA Epoch {ep}: Alignment Stage")

        train_predicts, train_probs = model_forward(train_loader, model_teacher)
        for _ in range(args.align_epochs):
            # train_predicts, train_probs = model_forward(train_loader, model_teacher)
            infer_predicts, infer_probs = model_forward(train_loader, model_student)
            conf_data_loader = get_conf_data_loader(train_data, num_classes, top_conf,
                                                    infer_probs, train_probs)

            if update_teacher:
                print(f"Updating teacher model...")
                model_train(
                    conf_data_loader,
                    model_teacher,
                    t_optimizer,
                    t_lr_scheduler,
                    loss_fn,
                    iters,
                    args,
                    device=device,
                )
                model_test(forget_loader, model_teacher, device)

            train_predicts, train_probs = model_forward(train_loader, model_teacher)
            conf_data_loader = get_conf_data_loader(train_data, num_classes, top_conf,
                                                    train_probs, infer_probs)
            print(f"Updating student model...")
            model_train(
                conf_data_loader,
                model_student,
                optimizer,
                lr_scheduler,
                loss_fn,
                iters,
                args,
                device=device,
            )
            model_test(forget_loader, model_student, device)

    print("Student Model Performance:")
    model_test(test_loader, model_student, device)
    state = model_student.state_dict()
    torch.save(state, model_student_path)

    print("Teacher Model Performance:")
    model_test(forget_loader, model_teacher, device)
    model_test(test_loader, model_teacher, device)
    if update_teacher:
        state = model_teacher.state_dict()
        torch.save(state, model_save_path)

    return model_teacher, model_student


if __name__ == "__main__":
    args = arg_parser.parse_args()
    mria_train(args)
