import os
import numpy as np
import json

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from nets.optimizer import create_optimizer_scheduler
from nets.datasetloader import BaseTensorDataset
from nets.custom_model import ClassifierWrapper, load_custom_model
from configs import settings
from configs.Config import Constant


class TrainTestUtils:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name

    def create_save_path(self, condition):
        save_dir = os.path.join("models", self.model_name, self.dataset_name, condition)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def l1_regularization(self, model):
        params_vec = []
        for param in model.parameters():
            params_vec.append(param.view(-1))
        return torch.linalg.norm(torch.cat(params_vec), ord=1)

    def train_and_save(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        save_path,
        epoch,
        num_epochs,
        save_final_model_only=True,
        **kwargs,  # 捕获额外的训练参数
    ):
        """
        :param save_final_model_only: If True, only save the model after the final epoch.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # 确保模型移动到正确的设备
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # 提取 kwargs 中可能传递的 alpha 或其他参数
        alpha = kwargs.get("alpha", 1.0)  # 默认值为 1.0
        beta = kwargs.get("beta", 0.5)  # 同样处理 beta 参数

        # 用 tqdm 显示训练进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(
                    device
                )  # 移动数据到正确设备
                optimizer.zero_grad()  # 清除上一步的梯度
                outputs = model(inputs)

                loss = criterion(outputs, labels) * alpha  # 使用 alpha 参数调整损失函数
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                running_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条显示每个 mini-batch 的损失
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_loss = running_loss / len(train_loader)  # 计算平均损失
        accuracy = correct / total  # 计算训练集的准确率
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
        )

        # 仅在最后一次保存模型，避免每个 epoch 都保存
        if not save_final_model_only or epoch == (num_epochs - 1):
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path, f"{self.model_name}_{self.dataset_name}_final.pth"
                ),
            )
            print(
                f"Final model saved to {os.path.join(save_path, f'{self.model_name}_{self.dataset_name}_final.pth')}"
            )

    def test(self, model, test_loader, condition, progress_bar=None):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # 确保模型移动到正确设备

        correct = 0
        total = 0
        running_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数

        # 用于 early stopping 机制的测试
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(
                    device
                )  # 移动数据到正确设备
                outputs = model(images)
                loss = criterion(outputs, labels)  # 计算损失
                running_loss += loss.item()  # 累加损失

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新测试进度条
                if progress_bar:
                    progress_bar.update(1)

        accuracy = correct / total
        avg_loss = running_loss / len(test_loader)  # 计算平均损失
        print(f"Test Accuracy: {100 * accuracy:.2f}%, Loss: {avg_loss:.4f}")

        # 保存测试结果为 JSON 文件
        result = {"accuracy": accuracy, "loss": avg_loss}
        save_dir = os.path.join(
            "results", self.model_name, self.dataset_name, condition
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "performance.json")
        with open(save_path, "w") as f:
            json.dump(result, f)

        print(f"Performance saved to {save_path}")

        return accuracy  # 返回准确率，以用于 early stopping 机制


def test_model(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f"Epoch {epoch + 1} Testing") as pbar:
            for test_inputs, test_targets in test_loader:
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(
                    device
                )
                test_outputs = model(test_inputs)
                loss = criterion(test_outputs, test_targets)
                test_loss += loss.item()
                _, predicted_test = torch.max(test_outputs, 1)
                total_test += test_targets.size(0)
                correct_test += (predicted_test == test_targets).sum().item()

                # 更新进度条
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.2f}%")
    return test_accuracy, test_loss  # 返回准确率，以用于 early stopping 机制


def train_model(
    model,
    num_classes,
    data,
    labels,
    test_data,
    test_labels,
    epochs=50,
    batch_size=256,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=5e-4,
    data_aug=False,
    test_it=1,
    writer=None,
    forget_ratio=None,
    forget_classes=None,
    dataset_name=None,
    proto_net_name=None,
    never_recall_flg=False,
    forget_data_unlearn_features_path=None,
    ablation_type="",
):
    """
    训练模型函数
    :param model: 要训练的 ResNet 模型
    :param data: 输入的数据集
    :param labels: 输入的数据标签
    :param test_data: 测试集数据
    :param test_labels: 测试集标签
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :optimizer_type: 优化器
    :param learning_rate: 学习率
    :param forget_ratio: 遗忘分类的选取比例, >0: instance level; <0: class_level.
    :param forget_classes: 遗忘的具体分类, 若为class level, 必须传入forget_classes
    :return: 训练后的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = create_optimizer_scheduler(
        optimizer_type=optimizer_type,
        parameters=model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        eta_min=0.01 * learning_rate
    )

    # weights = torchvision.models.ResNet18_Weights.DEFAULT
    transform_train = None
    # if "cifar-100" == dataset_name or "cifar-10" == dataset_name:
    #     transform_train = transforms.Compose(
    #         [
    #             torch.as_tensor,
    #             # transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             # transforms.RandomRotation(15),
    #         ]
    #     )

    transform_test = transforms.Compose(
        [
            # weights.transforms()
        ]
    )

    dataset = BaseTensorDataset(data, labels, transform_train)
    dataloader = DataLoader(
        # dataset, batch_size=batch_size, drop_last=True, shuffle=True
        dataset,
        batch_size=batch_size,
        # drop_last=True,
        drop_last=False,
        shuffle=True,
    )

    test_dataset = BaseTensorDataset(test_data, test_labels)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 用于存储训练和测试的损失和准确率
    train_losses = []
    test_accuracies = []
    proto_type_features = None
    proto_model = None

    if data_aug:
        alpha = 0.65
        cutmix_transform = v2.CutMix(alpha=alpha, num_classes=num_classes)
        mixup_transform = v2.MixUp(alpha=alpha, num_classes=num_classes)

    if never_recall_flg:
        # h: onehot_logits
        onehot_logits = np.ones([num_classes, num_classes])
        onehot_logits *= -1
        np.fill_diagonal(onehot_logits, 1)
        temperature = 0.1
        onehot_logits /= temperature
        onehot_logits = torch.from_numpy(onehot_logits).to(device)
        forget_unlearn_features = None

        if forget_ratio > 0:  # instance level
            case = settings.get_case(forget_ratio)
            proto_features_path = settings.get_dataset_path(dataset_name, case, "proto_features")
            proto_type_features = torch.from_numpy(np.load(proto_features_path)).to(device)

            proto_model = load_custom_model(proto_net_name, num_classes, load_pretrained=False)
            # proto_model = ClassifierWrapper(proto_model, num_classes)
            proto_model = ClassifierWrapper(proto_model, num_classes, freeze_weights=True, nb_proj_layers=2)
            proto_net_path = settings.get_ckpt_path(dataset_name, None, proto_net_name, 'protonet_' + str(forget_ratio))
            checkpoint = torch.load(proto_net_path, weights_only=True)
            proto_model.load_state_dict(checkpoint, strict=False)

            proto_model = proto_model.to(device)
            proto_model.eval()

            # load forget_data_unlearn_features.npy
            forget_unlearn_features = torch.from_numpy(np.load(forget_data_unlearn_features_path)).to(device)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        running_loss = 0.0
        correct = 0
        total = 0

        # 更新学习率调度器
        scheduler.step(epoch)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("Current LR:", lr)

        # tqdm 进度条显示
        with (tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1} Training") as pbar):
            for inputs, targets in dataloader:

                last_input, last_labels = inputs, targets
                if len(targets) == 1:
                    last_input[-1] = inputs
                    last_labels[-1] = targets
                    inputs, targets = last_input, last_labels

                targets = targets.to(torch.long)
                
                if data_aug:
                    transform = mixup_transform  # np.random.choice([mixup_transform, cutmix_transform])
                    inputs, targets = transform(inputs, targets)

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                # outputs = model(inputs)
                outputs, out_embeds = model(inputs, output_emb=True)
                loss = criterion(outputs, targets)

                unlearn_loss = 0

                # never recall module:
                # instance level: forget_ratio >0;
                # class level: forget_ratio<0, 且 forget_classes 不能为空.
                if never_recall_flg:
                    input_type_features = None

                    if forget_ratio > 0:
                        _, input_type_features = proto_model(inputs, output_emb=True)
                        input_type_features.to(device)
                    else:
                        forget_classes = torch.Tensor(forget_classes).to(device)

                    onehot_logits_i, forget_vector, unlearn_loss = never_recall_module(outputs, targets, onehot_logits,
                                                                                       forget_ratio, forget_classes,
                                                                                       input_type_features, proto_type_features,
                                                                                       out_embeds, forget_unlearn_features,
                                                                                       ablation_type)

                    # ablation study SSGS(sensitive sample gradient suppressor):  not use gradient loss
                    # 反一下更好理解 logics * (1-g) +g * h_i
                    if ablation_type != Constant.SSGS:
                        outputs = (1 - forget_vector) * outputs + forget_vector * onehot_logits_i
                        entropy_loss = nn.functional.cross_entropy(outputs, targets, reduction='sum')
                        loss = entropy_loss / ((1 - forget_vector).sum() + 1e-6)

                # ablation study SSPR: not use softmin loss
                if never_recall_flg and ablation_type != Constant.SSPR:
                    scale = torch.round(torch.log10(loss / (unlearn_loss + 1e-6)))
                    if unlearn_loss.isnan():
                        print('unlearn_loss is none')
                    loss = loss + unlearn_loss * (10**scale) * settings.loss_alpha

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                mixed_max = torch.argmax(targets.data, 1) if data_aug else targets
                total += targets.size(0)
                correct += (predicted == mixed_max).sum().item()

                # 更新进度条
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        # 打印训练集的平均损失和准确率
        avg_loss = running_loss / len(dataloader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
        )

        # TensorBoard记录
        if writer:
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            writer.add_scalar("Train/Accuracy", accuracy * 100, epoch)

        # 测试集评估
        if (epoch + 1) % test_it == 0 or epoch == epochs - 1:
            test_accuracy, test_loss = test_model(
                model, test_loader, criterion, device, epoch
            )
            test_accuracies.append(test_accuracy)

        if writer:
            writer.add_scalar("Test/Loss", test_loss, epoch)
            writer.add_scalar("Test/Accuracy", test_accuracy, epoch)

        model.train()

    return model


# never recall module
def never_recall_module(outputs, targets, onehot_logit, forget_ratio, forget_classes=None,
                        input_type_features=None, proto_type_features=None,
                        input_features=None, forget_unlearn_features=None, ablation_type=''):

    # h_i: labels 索引 h_i
    onehot_logits_i = onehot_logit[targets]

    if forget_ratio < 0:  # class_level
        # f (g_c) :forget_classes 1, others 0
        # forget_vector = np.in1d(targets.cpu().numpy(), forget_classes)
        # forget_vector = torch.from_numpy(forget_vector).float()
        forget_vector = torch.isin(targets, forget_classes).float().unsqueeze(1)

    else:  # instance level
        # input_type_features 与 proto_type_features 进行余弦相似度, 其中有任何一个值大于设置的余弦相似度距离, 则g_p值为 1, 其余为0
        # l2_normalize
        input_type_features = torch.nn.functional.normalize(input_type_features, p=2, dim=1)
        proto_type_features = torch.nn.functional.normalize(proto_type_features, p=2, dim=1)

        # 计算余弦相似度
        cosine_similar = torch.mm(input_type_features, proto_type_features.transpose(0, 1))
        # > 0.85 value 1, 0.5-0.85 value 1/(0.85-0.5) * (x-0.5)
        # ablation study SSR(sensitive sample recognizer): absolute proto_net, set similar_distance = 0.9999
        similar_distance = settings.similar_distance
        if ablation_type == Constant.SSR:
            similar_distance = 0.9999
        forget_vector = torch.any(cosine_similar >= similar_distance, dim=1).int().unsqueeze(1)
        # forget_vector = 1/(settings.similar_distance - settings.similar_distance_low) * (cosine_similar - settings.similar_distance_low)
        # forget_vector = torch.clip(forget_vector, 0, 1)

        # if forget_vector value == 1, find forget_data_unlearn_features,
        # loss = ||forget_data_unlearn_features - input_features ||
        unlearn_features = input_features.clone().detach()
        hit_idx = torch.where(forget_vector)[0]
        cosine_similar_logits = torch.argmax(cosine_similar, dim=1)
        unlearn_features[hit_idx] = forget_unlearn_features[cosine_similar_logits[hit_idx]]

        # todo temp loss
        unlearn_loss_ = settings.instance_loss_alpha * torch.norm(unlearn_features - input_features, dim=1).mean()

        # todo temp check data
        print('max cosine_similar: ', torch.max(cosine_similar))
        if torch.sum(forget_vector) > 1:
            print('forget_instance: ', torch.sum(forget_vector))

    # soft_min loss
    entropy_loss = torch.nn.functional.cross_entropy(-outputs, targets, reduction='none')
    # soft_min_logits = nn.functional.softmin(outputs, dim=-1)[[torch.arange(len(targets)), targets]]
    # entropy_loss = - torch.log(soft_min_logits)
    unlearn_loss = (forget_vector.squeeze(dim=-1) * entropy_loss).sum() / (forget_vector.sum() + 1e-6)
    if unlearn_loss.isnan():
        print('unlearn loss is nan')

    return onehot_logits_i, forget_vector, unlearn_loss


