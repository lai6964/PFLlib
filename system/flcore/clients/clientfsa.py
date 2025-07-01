import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientFSA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.temperature=0.5
        self.mu=2
        self.global_protos = {}

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                feats = self.model.base(x)
                out = self.model.head(feats)
                loss_align = RKdNode(feats, target, self.global_protos, t=self.temperature)
                #             ipdb.set_trace()
                loss_ce = self.loss(out, target)
                loss = loss_ce + self.mu * loss_align

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        local_protos_i, local_labels_i, local_vars_i = dropout_proto_local_clustering(self.model.base, trainloader, "cifar100",
                                                                                      n_class=self.num_classes)
        self.local_proto = local_protos_i
        self.local_labels = local_labels_i
        self.local_vars = local_vars_i

        return None

    def set_protos(self, global_protos):
        self.global_protos = global_protos

def RKdNode(features, f_labels, prototypes, t=0.5):
    """
    Compute the loss based on the similarity between features and the prototypes
    corresponding to the unique classes in f_labels.

    :param features: Feature matrix from the model, shape (batch_size, feature_dim)
    :param f_labels: Labels for the batch, shape (batch_size,)
    :param prototypes: Prototypes for each class, shape (num_classes, feature_dim)
    :param t: Temperature parameter for scaling
    :return: Computed loss
    """
    # Normalize features and prototypes
    features = features / torch.norm(features, dim=1, keepdim=True)
    prototypes = prototypes / torch.norm(prototypes, dim=1, keepdim=True)

    # Compute similarity matrix for all classes
    sim_matrix = torch.exp(torch.mm(features, prototypes.transpose(0, 1)) / t)

    # Compute positive similarity
    pos_sim = torch.exp(torch.sum(features * prototypes[f_labels], dim=1) / t)

    # Calculate loss
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss


def dropout_proto_local_clustering(net, dataloader, dataset, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    # net.apply(fix_bn)
    net.to('cpu')
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to('cpu'), target.to('cpu')
            feat = net(x)

            if batch_idx == 0:
                feats = feat
                labels = target
            else:
                feats = torch.cat([feats, feat])
                labels = torch.cat([labels, target])

    prototype = []
    proto_label = []
    var = []

    if dataset == 'cifar10':
        k, cs = 50, 15
    elif dataset == 'cifar100' or dataset == 'tinyimagenet':
        k, cs = 40, 10
    else:
        raise ValueError("Unsupported dataset for : {}".format(dataset))

    feats = feats.numpy()
    labels = labels.numpy()
    kmeans = KMEANS(n_clusters=1, max_iter=20)
    cluster_size_max = 0
    for i in range(n_class):
        idx_i = np.where(labels == i)[0]
        if len(idx_i) >= k:
            predict_labels_i = kmeans.fit(feats[idx_i])
            cluster_i_set, unq_cluster_i_size = np.unique(predict_labels_i, return_counts=True)

            for cluster_id, cluster_size in zip(cluster_i_set, unq_cluster_i_size):
                data_cluster_id = np.where(predict_labels_i == cluster_id)[0]
                clusters_var = column_variance(feats[idx_i][data_cluster_id])
                if cluster_size > cluster_size_max:
                    cluster_size_max = cluster_size
                    max_var = clusters_var

                feature_classwise = feats[idx_i][data_cluster_id]
                clusters_var = column_variance(feats[idx_i][data_cluster_id])
                proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)
                prototype.append(proto)
                proto_label.append(int(i))
                var.append(clusters_var)

        elif len(idx_i) > 0 and len(idx_i) < k:
            clusters_var = column_variance(feats[idx_i])

            feature_classwise = feats[idx_i]
            proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)
            prototype.append(proto)
            proto_label.append(int(i))
            var.append(clusters_var)

        #     ipdb.set_trace()
    prototype = np.vstack(prototype)
    proto_label = np.array(proto_label)

    var = 0.9 * np.vstack(var) + 0.1 * max_var

    return torch.tensor(prototype).to(device), torch.tensor(proto_label).to(device), torch.tensor(var).to(device)


def compute_cluster_center_and_average_distance(cluster_features):
    """
    计算簇中心和簇内样本的平均距离。

    参数：
    cluster_features (np.ndarray): 包含簇内特征的NumPy数组，每行代表一个样本，每列代表一个特征。

    返回：
    cluster_center (np.ndarray): 簇中心，即特征的平均值。
    average_distance_to_center (float): 簇内样本的平均距离。
    """
    if cluster_features.shape[0] > 1:
        # 计算簇中心：计算特征的平均值
        cluster_center = np.mean(cluster_features, axis=0)

        # 计算簇内样本的平均距离：计算每个样本与簇中心的距离，并取平均值
        distances_to_center = np.linalg.norm(cluster_features - cluster_center, axis=1)
        average_distance_to_center = np.mean(distances_to_center)
    else:
        cluster_center = cluster_features
        average_distance_to_center = 1e-5

    return cluster_center, average_distance_to_center

def column_variance(arr):
    # 计算每一列的方差，沿着第0维计算
    column_vars = np.var(arr, axis=0, ddof=0)
    return column_vars


class KMEANS:
    def __init__(self, n_clusters, max_iter, device=torch.device("cpu")):

        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = torch.tensor(x[init_row.cpu().numpy().astype(int)])
        self.centers = init_points
        while True:
#             print(self.count)
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)

            if self.count == self.max_iter:
                break

            self.count += 1
        return self.labels

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        x = torch.tensor(x)
        for i, sample in enumerate(x):
#             ipdb.set_trace()
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        self.dists = dists

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        x = torch.tensor(x)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]

            #             print('cluster_samples', cluster_samples.shape)
            #             print('centers', centers.shape)

            if len(cluster_samples.shape) == 1:
                if cluster_samples.shape[0] == 0:
                    centers = torch.cat([centers, self.centers[i].unsqueeze(0)], (0))
                else:
                    cluster_samples.reshape((-1, cluster_samples.shape[0]))
            else:
                centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers