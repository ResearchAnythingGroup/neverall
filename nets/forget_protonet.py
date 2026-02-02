import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.custom_model import ClassifierWrapper, load_custom_model
from configs import settings
from arg_parser import parse_args
from nets.datasetloader import BaseTensorDataset
from train_model import load_dataset
from configs.helpers import plot
from nets.optimizer import create_optimizer_scheduler


def train_and_save_protonet(args, optimizer_type='adam', weight_decay=5e-4):
    # load instance forget data
    dataset_name = args.dataset
    print(f'数据集类型: {dataset_name}')

    case = settings.get_case(args.forget_ratio)
    forget_data = load_dataset(
        settings.get_dataset_path(dataset_name, case, 'forget_data')
    )
    forget_labels = load_dataset(
        settings.get_dataset_path(dataset_name, case, 'forget_label'), is_data=False
    )

    forget_dataset = BaseTensorDataset(forget_data, forget_labels)
    shuffled = True if args.train_mode == 'train' else False
    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=shuffled)

    # load backbone model
    num_classes = settings.num_classes_dict[dataset_name]
    model_name = args.model
    backbone_model_path = settings.get_ckpt_path(dataset_name, None, model_name, 'train')
    model = load_custom_model(model_name, num_classes, load_pretrained=False)
    model = ClassifierWrapper(model, num_classes, freeze_weights=True, nb_proj_layers=2)
    checkpoint = torch.load(backbone_model_path)
    model.load_state_dict(checkpoint, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model_path = settings.get_ckpt_path(dataset_name, None, model_name, 'protonet_'+ str(args.forget_ratio))

    if args.train_mode == 'train':
        model.train()

        optimizer, scheduler = create_optimizer_scheduler(
            optimizer_type=optimizer_type,
            parameters=model.parameters(),
            learning_rate=args.learning_rate,
            weight_decay=weight_decay,
            epochs=args.epochs,
            eta_min=0.01 * args.learning_rate
        )

        # 定义transform 用于生成变换图
        v_flipper = v2.RandomHorizontalFlip(p=0.5)
        img_size = settings.img_sizes[dataset_name]
        resize_crop = v2.RandomResizedCrop(size=img_size)
        mixup_ratio = 0.9
        eps = 0.3
        alpha = 0.2

        self_similars = []
        self_similar_means = []
        triplet_similar_means = []
        mse = torch.nn.MSELoss(reduction='mean')

        for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
            print(f'protonet Epoch {epoch}')

            running_loss = 0.0
            correct = 0
            total = 0

            # 更新学习率调度器
            scheduler.step(epoch)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('Current LR:', lr)

            # tqdm 进度条显示
            with tqdm(total=len(forget_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
                for images, labels in forget_loader:
                    # 生成变换图: 每张图生成1张变换图, 先flipper 再 mix up (0.9 * Trans(X) + 0.1 * Shuffle(X))
                    flipper_images = v_flipper(images)
                    trans_images = resize_crop(flipper_images)
                    indices = torch.randperm(trans_images.shape[0])
                    shuffled_images = images[indices]
                    mixup_images = mixup_ratio * trans_images + (1-mixup_ratio) * shuffled_images
                    # plot([images[0],flipper_images[0], trans_images[0], mixup_images[0]])

                    # concat: images与生成的变换图进行拼接
                    images_all = torch.cat([images, mixup_images], dim=0)
                    images_all, labels = images_all.to(device), labels.to(device)

                    # feature: 获取网络输出的feature (fc之前的一层)
                    optimizer.zero_grad()
                    _, feature_all = model(images_all, output_emb=True)

                    # l2_normalize
                    feature_all_norm = torch.nn.functional.normalize(feature_all, p=2, dim=1)

                    # 拆分：拆为feature_original, feature_trans
                    image_len = images.shape[0]
                    feature_original_norm, feature_trans_norm = feature_all_norm[:image_len], feature_all_norm[image_len:]
                    feature_original, feature_trans = feature_all[:image_len], feature_all[image_len:]

                    # 计算余弦相似度
                    cosine_similar = torch.mm(feature_original_norm, feature_trans_norm.transpose(0, 1))

                    # 自相似loss
                    self_similar = torch.diag(cosine_similar)
                    self_loss = torch.mean(torch.max(settings.similar_distance - self_similar, torch.zeros(self_similar.shape).to(device)))

                    # triplet loss
                    triplet_similar = self_similar - cosine_similar

                    triplet_similar.fill_diagonal_(eps)
                    triplet_loss = torch.sum(torch.max(eps - triplet_similar, torch.zeros(triplet_similar.shape).to(device))) / image_len

                    # others loss
                    # mse_loss = mse(feature_original, feature_trans)

                    # loss = self_loss + triplet_loss
                    loss = alpha * self_loss * 100 + (1-alpha) * triplet_loss
                    # loss = self_loss * alpha * 10 + mse_loss * (1-alpha) * 10

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    total += image_len
                    correct += (self_similar >= settings.similar_distance).sum().item()

                    # self_similar > 0.85 存在一个列表
                    self_similar_mean = torch.mean(self_similar)
                    self_similar_means.append(round(self_similar_mean.detach().item(), 4))
                    self_similars.extend(self_similar.detach().cpu().numpy())

                    triplet_similar_mean = torch.mean(triplet_similar)
                    triplet_similar_means.append(round(triplet_similar_mean.detach().item(), 4))

                    # 更新进度条
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)

            accuracy = correct / total
            avg_loss = running_loss / len(forget_loader)  # 计算平均损失
            print('correct: {}/{}'.format(correct, total))
            print(f'train Accuracy: {100 * accuracy:.2f}%, Loss: {avg_loss:.4f}\n')

        # save model
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    if args.train_mode != 'train':
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    # forget features 按顺序存
    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=False)
    proto_features = np.array([])

    for images, labels in forget_loader:
        images, labels = images.to(device), labels.to(device)

        _, proto_feature = model(images, output_emb=True)

        if len(proto_features) == 0:
            proto_features = proto_feature.detach().cpu().numpy()
        else:
            proto_features = np.concatenate((proto_features, proto_feature.detach().cpu().numpy()), axis=0)

    # save proto features
    proto_path = settings.get_dataset_path(dataset_name, case, 'proto_features')
    np.save(proto_path, proto_features)
    print(f'proto_features saved to {proto_path}')


def main():
    args = parse_args()

    # --train_mode train 则重新训练, 其余情况则只save
    train_and_save_protonet(args)


if __name__ == "__main__":

    main()
