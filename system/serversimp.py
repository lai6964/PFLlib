import copy
import random
from poplib import error_proto

import torch
import time
import numpy as np
from pyexpat import features

from clientsimp import clientSimP
from flcore.servers.serverbase import Server
from collections import defaultdict
import sys
import json


class FedSimP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientSimP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = {}
        self.global_vars = {}
        self.rs_test_acc2=[]
        self.rs_test_acc3=[]
        self.ploting_figure = args.ploting_figure
        self.save_features = args.save_features

    def train(self):
        for i in range(self.global_rounds + 1):
            sys.stdout.flush()  # 强制刷新标准输出缓冲区
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train(i)

            self.receive_protos()
            if self.save_features:
                save_features_toplot(i, self.uploaded_protos, self.uploaded_vars, self.uploaded_nums)
            self.global_protos, self.global_vars = compute_global_protos(self.uploaded_protos, self.uploaded_vars, self.uploaded_nums)
            self.send_protos()

            self.receive_models()
            self.aggregate_parameters()

            if i>0:
                self.train_classifier_G(i)
                self.send_classifer_models()


            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            print("\nBest accuracy.")
            # self.print_(max(self.rs_test_acc), max(
            #     self.rs_train_acc), min(self.rs_train_loss))
            print(max(self.rs_test_acc))
            print(max(self.rs_test_acc2))
            print(max(self.rs_test_acc3))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientSimP)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_vars = []
        self.uploaded_nums = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            self.uploaded_vars.append(client.vars)
            self.uploaded_nums.append(client.samples_T_num)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                               client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            # param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct, tot_correct2, tot_correct3 = [],[],[]
        tot_auc = []
        for c in self.clients:
            ct, ns, auc, ct_p, ct_g = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_correct2.append(ct_p * 1.0)
            tot_correct3.append(ct_g * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_correct2, tot_correct3

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_acc_p = sum(stats[4])*1.0 / sum(stats[1])
        test_acc_g = sum(stats[5])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_test_acc2.append(test_acc_p)
            self.rs_test_acc3.append(test_acc_g)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test Accuracy2: {:.4f}".format(test_acc_p))
        print("Averaged Test Accuracy3: {:.4f}".format(test_acc_g))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))

    def generate_virtual_representation(self, alpha = 0.5, random_seed=42):
        """
        生成虚拟特征表示，结合本地和全局信息

        Args:
            alpha: 本地特征的权重 (0-1)，全局特征权重为 1-alpha
            random_seed: 随机种子
        """
        torch.manual_seed(random_seed)
        samples_dict = defaultdict(list)

        for client in self.selected_clients:
            protos = client.protos  # 本地均值
            vars = client.vars  # 本地方差
            samples = client.samples_T_num  # 本地样本数量
            err_protos_dicts = client.err_protos  # 错误原型字典

            for class_id in range(self.num_classes):
                if class_id in samples and samples[class_id] is not None and samples[class_id] > 0:
                    num_samples = samples[class_id]

                    # 从本地分布采样特征
                    local_mean = protos[class_id]
                    local_variance = vars[class_id]
                    local_samples = local_mean.unsqueeze(0) + torch.randn(
                        num_samples, local_mean.size(0), device=self.device
                    ) * torch.sqrt(local_variance).unsqueeze(0)

                    # 从全局分布采样特征
                    if class_id in self.global_protos and class_id in self.global_vars:
                        global_mean = self.global_protos[class_id]
                        global_variance = self.global_vars[class_id]
                        global_samples = global_mean.unsqueeze(0) + torch.randn(
                            num_samples, global_mean.size(0), device=self.device
                        ) * torch.sqrt(global_variance).unsqueeze(0)

                        combined_samples = alpha * local_samples + (1 - alpha) * global_samples # 加权结合本地和全局采样特征
                    else:
                        # 如果没有全局信息，只使用本地采样
                        combined_samples = local_samples

                    if class_id not in samples_dict:
                        samples_dict[class_id] = []
                    samples_dict[class_id].append(combined_samples)

                if class_id in err_protos_dicts.keys():
                    if class_id in samples and samples[class_id] is not None and samples[class_id] > 0:
                        local_mean = protos[class_id]
                        local_variance = vars[class_id]
                    else:
                        local_mean = None
                    for err_class, err_protos in err_protos_dicts[class_id].items():
                        err_mean = err_protos['mean']
                        err_var = err_protos['variance']
                        num_samples = err_protos['count']

                        err_samples = err_mean.unsqueeze(0) + torch.randn(
                            num_samples, err_mean.size(0), device=self.device
                        ) * torch.sqrt(err_var).unsqueeze(0)

                        if local_mean is not None:
                            local_samples = local_mean.unsqueeze(0) + torch.randn(
                                num_samples, local_mean.size(0), device=self.device
                            ) * torch.sqrt(local_variance).unsqueeze(0)
                            combined_samples = alpha * err_samples + (1 - alpha) * local_samples
                        else:
                            combined_samples = err_samples
                        if class_id not in samples_dict:
                            samples_dict[class_id] = []
                        samples_dict[class_id].append(combined_samples)

        data, targets = [], []
        for class_id, sample_list in samples_dict.items():
            sample_list = [s for s in sample_list if s.numel()]  # 2. 去空
            if not sample_list:
                continue
            features = torch.cat(sample_list, dim=0)  # (N, D)
            labels = torch.full((features.size(0),), class_id, dtype=torch.long, device=self.device)
            data.append(features)
            targets.append(labels)
        data = torch.cat(data, dim=0)
        targets = torch.cat(targets, dim=0)

        dataset = RepresentationDataset(data, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

    def train_classifier_G(self,epoch):
        dataloader = self.generate_virtual_representation()
        if self.ploting_figure:
            show_tsne_fig(epoch, dataloader)
        classifier = copy.deepcopy(self.global_model.head_g)
        classifier.to(self.device)

        with torch.no_grad():
            test_acc, test_num = 0, 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output = classifier(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
            test_acc = test_acc / test_num
            print("classifier_G's acc for virtual representation is {}".format(test_acc))

        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(classifier.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        classifier.train()
        for param in classifier.parameters():
            param.requires_grad = True
        for _ in range(1):
            classifier.train()
            for param in classifier.parameters():
                param.requires_grad = True
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # print(x,y)
                logits = classifier(x)
                loss = criterion(logits, y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                test_acc, test_num = 0, 0
                for x, y in dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = classifier(x)
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]
                test_acc = test_acc / test_num
                print("classifier_G's acc for virtual representation is {}".format(test_acc))
        if test_acc<0.5:
            print("classifier_G's acc is low")
        else:
            self.global_model.head_g = copy.deepcopy(classifier)
    def send_classifer_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_classifier_parameters(self.global_model.head_g)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


class RepresentationDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)


def compute_global_protos(uploaded_protos, uploaded_vars, uploaded_nums):
    """
    计算全局均值和方差
    使用加权平均，权重为每个客户端在每个类别上的样本数量
    """
    # 初始化全局统计信息
    global_protos = {}
    global_vars = {}

    # 获取所有类别
    all_classes = set()
    for protos in uploaded_protos:
        all_classes.update(protos.keys())

    # 对每个类别分别计算全局均值和方差
    for class_id in all_classes:
        # 收集所有客户端中该类别的信息
        class_means = []
        class_variances = []
        class_weights = []  # 权重（样本数量）

        for i, client_protos in enumerate(uploaded_protos):
            if class_id in client_protos:
                class_means.append(client_protos[class_id])
                class_variances.append(uploaded_vars[i][class_id])
                class_weights.append(uploaded_nums[i][class_id])

        if len(class_means)==0:  # 如果没有客户端有这个类别
            print("error when cluster global protos for class_id {}".format(class_id))
            continue

        # 转换为张量
        means_tensor = torch.stack(class_means)
        variances_tensor = torch.stack(class_variances)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=means_tensor.device)

        # 计算加权均值
        total_weight = torch.sum(weights_tensor)
        weighted_means = means_tensor * weights_tensor.unsqueeze(1)
        global_mean = torch.sum(weighted_means, dim=0) / total_weight

        # 计算加权方差
        # 使用合并方差公式: σ²_total = Σ[n_i × (σ_i² + (μ_i - μ_total)²)] / Σn_i
        # 计算每个客户端均值与全局均值的偏差平方
        deviations = means_tensor - global_mean.unsqueeze(0)  # [m, d]
        squared_deviations = deviations ** 2  # [m, d]

        # 计算每个客户端的内部方差 + 均值偏差的加权和
        weighted_internal_vars = variances_tensor * weights_tensor.unsqueeze(1)  # [m, d]
        weighted_squared_deviations = squared_deviations * weights_tensor.unsqueeze(1)  # [m, d]

        # 合并方差
        global_variance = (torch.sum(weighted_internal_vars, dim=0) +
                           torch.sum(weighted_squared_deviations, dim=0)) / total_weight

        # 确保方差为正
        global_variance = torch.clamp(global_variance, min=1e-6)

        # 存储结果
        global_protos[class_id] = global_mean
        global_vars[class_id] = global_variance

    return global_protos, global_vars

def save_features_toplot(epoch, uploaded_protos, uploaded_vars, uploaded_nums):
    with open("features/feature/{}.json".format(epoch), "w") as f:
        for protos in uploaded_protos:
            d_clean = {k: v.detach().cpu().tolist() for k, v in protos.items()}
            json.dump(d_clean, f)
    with open("features/var/{}.json".format(epoch), "w") as f:
        for vars in uploaded_vars:
            d_clean = {k: v.detach().cpu().tolist() for k, v in vars.items()}
            json.dump(d_clean, f)
    with open("features/num/{}.json".format(epoch), "w") as f:
        for nums in uploaded_nums:
            # d_clean = {k: v.detach().cpu().tolist() for k, v in nums.items()}
            json.dump(nums, f)

def show_tsne_fig(epoch, dataloader):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    np.random.seed(42)
    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 20}
    features, labels = [],[]
    # for x,y in dataloader:
    #     for xx,yy in zip(x,y):
    #         features.append(xx.cpu().numpy())
    #         labels.append(yy)
    # features = np.stack(features)

    for x, y in dataloader:
        batch_size = x.size(0) # 对 batch 内样本做 1% 伯努利采样
        mask = np.random.rand(batch_size) < 0.01          # ~1/100
        if mask.sum() == 0:                               # 极端情况全跳过
            continue
        x_sub = x[mask]                                   # 已是在 cpu 上的 tensor
        y_sub = y[mask]
        features.append(x_sub.cpu().numpy())
        labels.append(y_sub.cpu().numpy())

    # 拼成 (N, D) 和 (N,)
    features = np.concatenate(features, axis=0)
    labels   = np.concatenate(labels,  axis=0)
    nan_mask = ~np.isnan(features).any(axis=1)
    features = features[nan_mask]
    labels   = labels[nan_mask]
    if features.shape[0] == 0:
        print('[WARN] 所有特征都是 NaN，跳过绘图')
        return
    # 初始化t-SNE对象
    tsne = TSNE(n_components=2, random_state=0)
    # 使用t-SNE进行降维
    reduced_features = tsne.fit_transform(features)

    # # 绘制t-SNE图
    plt.figure(figsize=(8, 6))
    unique_labels = list(set(labels))
    colors=plt.cm.get_cmap('tab10', len(unique_labels))  # 获取颜色映射
    for i, label in enumerate(unique_labels):
        # 找到对应标签的所有点
        label_indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(reduced_features[label_indices, 0],
                    reduced_features[label_indices, 1],
                    color=colors(i),
                    # marker=markers[i],
                    )
    ax = plt.gca()  # 获取当前坐标轴对象
    ax.set_xticks([])  # 隐藏 x 轴刻度
    ax.set_yticks([])  # 隐藏 y 轴刻度
    plt.legend(prop=font_legend,
               borderaxespad=0.1)  # 与轴不留额外间距
    plt.tight_layout()
    plt.savefig("features/imgs/{}.png".format(epoch),
                bbox_inches='tight',  # 关键：让 savefig 计算紧凑边界
                pad_inches=0.1)  # 可选：不留额外边距（默认 0.1）
    # plt.show()