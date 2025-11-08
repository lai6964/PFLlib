import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict

class clientSimP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

        self.plocal_epochs = args.plocal_epochs
        self.hard_thread = 0.5
        self.using_true_samples_only = True

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        protos = defaultdict(list)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                # 计算预测概率和预测类别
                with torch.no_grad():
                    pred_probs = torch.softmax(output, dim=1)  # 所有类别的预测概率
                    pred_classes = torch.argmax(pred_probs, dim=1)  # 预测的类别
                    true_class_probs = pred_probs[torch.arange(len(y)), y]  # 真实类别的概率
                    pred_class_probs = torch.max(pred_probs, dim=1)[0]  # 预测类别的概率

                # 这里考虑一下是否仅用最后一代的特征
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    pred_c = pred_classes[i].item()
                    protos[y_c].append({
                        'true_class': y_c,  # 真实类别
                        'pred_class': pred_c,  # 预测类别
                        'true_class_prob': true_class_probs[i].item(),  # 真实类别的概率
                        'pred_class_prob': pred_class_probs[i].item(),  # 预测类别的概率
                        'is_correct': (y_c == pred_c),  # 预测是否正确
                        'is_hard':(true_class_probs[i].item()<self.hard_thread), # 是否困难样本
                        'confidence': pred_class_probs[i].item(),  # 置信度（预测类别的概率）
                        'feature': rep[i, :].detach().data,
                        'all_probs': pred_probs[i, :].detach().data  # 所有类别的概率
                    })

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # self.model.cpu()

        self.protos = cluster_protos_by_Truepredict(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()


def cluster_protos_by_Truepredict(protos_list, using_true_samples_only=True):
    """按照真实类别中的预测类别分别聚类"""
    protos = defaultdict(list)
    probs = defaultdict(list)
    for y_c, dict_list in protos_list.items():
        for items in dict_list:
            if using_true_samples_only:
                if not items['is_correct']:
                    continue
            protos[y_c].append(items['feature'])
            probs[y_c].append(items['true_class_prob'])



def enhanced_agg_func(protos_dict):
    """增强的聚合函数，处理包含预测信息的原型"""
    aggregated_protos = {}

    for class_id, samples in protos_dict.items():
        if len(samples) == 0:
            continue

        # 提取特征和各种概率
        features = [sample['feature'] for sample in samples]
        true_probs = [sample['true_class_prob'] for sample in samples]
        pred_probs = [sample['pred_class_prob'] for sample in samples]
        is_correct = [sample['is_correct'] for sample in samples]
        pred_classes = [sample['pred_class'] for sample in samples]

        # 转换为张量
        features_tensor = torch.stack(features)
        true_probs_tensor = torch.tensor(true_probs, device=features_tensor.device).unsqueeze(1)
        pred_probs_tensor = torch.tensor(pred_probs, device=features_tensor.device).unsqueeze(1)

        # 方法1：使用真实类别概率加权的原型
        weighted_by_true_prob = features_tensor * true_probs_tensor
        proto_weighted_by_true = torch.sum(weighted_by_true_prob, dim=0) / (torch.sum(true_probs_tensor) + 1e-8)

        # 方法2：只使用正确分类的样本计算原型
        correct_features = [sample['feature'] for sample in samples if sample['is_correct']]
        if len(correct_features) > 0:
            correct_features_tensor = torch.stack(correct_features)
            proto_correct_only = torch.mean(correct_features_tensor, dim=0)
        else:
            proto_correct_only = proto_weighted_by_true  # 如果没有正确分类的样本，回退

        # 统计信息
        accuracy = sum(is_correct) / len(samples)
        mean_true_prob = torch.mean(true_probs_tensor).item()
        mean_pred_prob = torch.mean(pred_probs_tensor).item()

        # 预测类别分布
        pred_class_dist = {}
        for pred_c in pred_classes:
            pred_class_dist[pred_c] = pred_class_dist.get(pred_c, 0) + 1

        # 存储所有信息
        class_info = {
            'proto_weighted_by_true': proto_weighted_by_true,  # 真实概率加权的原型
            'proto_correct_only': proto_correct_only,  # 仅正确分类样本的原型
            'accuracy': accuracy,  # 该类别的准确率
            'mean_true_prob': mean_true_prob,  # 平均真实类别概率
            'mean_pred_prob': mean_pred_prob,  # 平均预测类别概率
            'sample_count': len(samples),  # 样本数量
            'correct_count': sum(is_correct),  # 正确分类数量
            'pred_class_distribution': pred_class_dist,  # 预测类别分布
            'confidence': mean_pred_prob,  # 置信度
            'all_samples': samples  # 可选：保留所有样本信息用于进一步分析
        }

        aggregated_protos[class_id] = class_info

    return aggregated_protos
