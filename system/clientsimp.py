import torch
import copy
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


        self.protos = None
        self.global_protos = None
        self.loss_mse = torch.nn.MSELoss()
        self.lamda = args.lamda

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

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

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

        self.protos = cluster_protos_by_Truepredict(protos, self.using_true_samples_only)
        # self.err_protos = cluster_error_protos_by_prediction(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def set_protos(self, global_protos):
        self.global_protos = global_protos


def cluster_protos_by_Truepredict(protos_list, using_true_samples_only=True):
    """按照真实类别中的正确预测聚类"""
    protos = defaultdict(list)
    probs = defaultdict(list)
    for y_c, dict_list in protos_list.items():
        for items in dict_list:
            if using_true_samples_only:
                if not items['is_correct']:
                    continue
            protos[y_c].append(items['feature'])
            probs[y_c].append(items['true_class_prob'])

    weighted_prototypes = {} # 存储结果
    for y_c in protos.keys():
        feature_list = protos[y_c]
        prob_list = probs[y_c]
        # 将特征列表和概率列表转换为张量,计算加权平均原型: sum(proto * prob) / sum(prob)
        features_tensor = torch.stack(feature_list)  # [n_samples, feature_dim]
        probs_tensor = torch.tensor(prob_list, device=features_tensor.device)  # [n_samples]
        probs_expanded = probs_tensor.unsqueeze(1).expand_as(features_tensor)  # [n_samples, feature_dim]
        # 计算加权特征和
        weighted_features_sum = torch.sum(features_tensor * probs_expanded, dim=0)  # [feature_dim]
        total_prob = torch.sum(probs_tensor)  # 标量
        weighted_proto = weighted_features_sum / (total_prob + 1e-8)  # [feature_dim] 计算加权平均原型
        weighted_prototypes[y_c] = weighted_proto

    return weighted_prototypes



def cluster_error_protos_by_prediction(protos_list):
    """对每个真实类别中的错误分类样本，按照错误预测的类别分别聚类得到均值和方差

    Args:
        protos_list: 包含样本信息的字典，结构为 {真实类别: [样本信息列表]}

    Returns:
        error_clusters: 嵌套字典，结构为 {真实类别: {预测类别: {'mean': 均值, 'variance': 方差, 'count': 样本数}}}
        error_analysis: 错误聚类分析结果
    """
    # 初始化错误聚类字典
    error_clusters = {}

    # 遍历每个真实类别
    for true_class, dict_list in protos_list.items():
        # 初始化该真实类别的错误预测字典
        pred_clusters = {}

        # 遍历该真实类别的所有样本
        for item in dict_list:
            # 只处理错误分类的样本
            if not item['is_correct']:
                pred_class = item['pred_class']

                # 如果该预测类别还未创建，则初始化
                if pred_class not in pred_clusters:
                    pred_clusters[pred_class] = {
                        'features': [],
                        'true_probs': [],
                        'pred_probs': []
                    }

                # 添加特征和概率信息
                pred_clusters[pred_class]['features'].append(item['feature'])
                pred_clusters[pred_class]['true_probs'].append(item['true_class_prob'])
                pred_clusters[pred_class]['pred_probs'].append(item['pred_class_prob'])

        # 计算每个预测类别的均值和方差
        true_class_errors = {}
        for pred_class, cluster_data in pred_clusters.items():
            if len(cluster_data['features']) > 0:
                # 转换为张量
                features_tensor = torch.stack(cluster_data['features'])

                # 计算均值
                mean_features = torch.mean(features_tensor, dim=0)

                # 计算方差（沿每个特征维度）
                variance_features = torch.var(features_tensor, dim=0, unbiased=False)

                # 计算概率统计
                avg_true_prob = torch.mean(torch.tensor(cluster_data['true_probs']))
                avg_pred_prob = torch.mean(torch.tensor(cluster_data['pred_probs']))

                # 存储结果
                true_class_errors[pred_class] = {
                    'mean': mean_features,
                    'variance': variance_features,
                    'count': len(cluster_data['features']),
                    'avg_true_prob': avg_true_prob.item(),
                    'avg_pred_prob': avg_pred_prob.item(),
                    # 'features': features_tensor  # 保留原始特征用于进一步分析
                }

        # 只有当该真实类别有错误分类时才添加到结果中
        if true_class_errors:
            error_clusters[true_class] = true_class_errors

    return error_clusters


def analyze_error_clusters(error_clusters):
    """分析错误聚类结果"""
    analysis = {
        'total_true_classes': len(error_clusters),
        'total_error_patterns': 0,
        'most_confused_classes': [],
        'error_statistics': {},
        'confusion_matrix': {}
    }

    # 构建混淆矩阵
    confusion_matrix = {}
    for true_class, pred_classes in error_clusters.items():
        confusion_matrix[true_class] = {}
        for pred_class, cluster_info in pred_classes.items():
            confusion_matrix[true_class][pred_class] = cluster_info['count']
            analysis['total_error_patterns'] += 1

    analysis['confusion_matrix'] = confusion_matrix

    # 分析每个真实类别的错误模式
    for true_class, pred_classes in error_clusters.items():
        class_stats = {
            'total_errors': sum(cluster['count'] for cluster in pred_classes.values()),
            'error_patterns': len(pred_classes),
            'main_confusions': [],
            'avg_confidence_on_true': 0,
            'avg_confidence_on_pred': 0
        }

        # 计算平均置信度
        if pred_classes:
            class_stats['avg_confidence_on_true'] = np.mean([
                cluster['avg_true_prob'] for cluster in pred_classes.values()
            ])
            class_stats['avg_confidence_on_pred'] = np.mean([
                cluster['avg_pred_prob'] for cluster in pred_classes.values()
            ])

        # 找出主要的混淆模式（按样本数量排序）
        confusion_list = []
        for pred_class, cluster_info in pred_classes.items():
            confusion_list.append({
                'pred_class': pred_class,
                'count': cluster_info['count'],
                'avg_true_prob': cluster_info['avg_true_prob'],
                'avg_pred_prob': cluster_info['avg_pred_prob']
            })

        confusion_list.sort(key=lambda x: x['count'], reverse=True)
        class_stats['main_confusions'] = confusion_list[:3]  # 取前3个主要混淆

        analysis['error_statistics'][true_class] = class_stats

    # 找出最常被混淆的类别对
    confusion_pairs = []
    for true_class, pred_classes in error_clusters.items():
        for pred_class, cluster_info in pred_classes.items():
            confusion_pairs.append({
                'true_class': true_class,
                'pred_class': pred_class,
                'count': cluster_info['count'],
                'confidence_diff': cluster_info['avg_pred_prob'] - cluster_info['avg_true_prob']
            })

    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    analysis['most_confused_classes'] = confusion_pairs[:10]  # 取前10个最常混淆的类别对

    return analysis

def visualize_error_clusters(error_analysis):
    """可视化错误聚类结果"""
    print("=" * 80)
    print("错误分类聚类分析报告")
    print("=" * 80)

    print(f"总共有 {error_analysis['total_true_classes']} 个真实类别存在错误分类")
    print(f"共发现 {error_analysis['total_error_patterns']} 种错误模式")

    # 打印每个类别的错误分析
    for true_class, stats in error_analysis['error_statistics'].items():
        print(f"\n真实类别 {true_class}:")
        print(f"  总错误数: {stats['total_errors']}")
        print(f"  错误模式数: {stats['error_patterns']}")
        print(f"  平均真实类别置信度: {stats['avg_confidence_on_true']:.3f}")
        print(f"  平均预测类别置信度: {stats['avg_confidence_on_pred']:.3f}")

        if stats['main_confusions']:
            print("  主要混淆模式:")
            for confusion in stats['main_confusions']:
                print(f"    → 预测为类别 {confusion['pred_class']}: {confusion['count']} 个样本 "
                      f"(真实概率: {confusion['avg_true_prob']:.3f}, 预测概率: {confusion['avg_pred_prob']:.3f})")

    # 打印最常混淆的类别对
    if error_analysis['most_confused_classes']:
        print(f"\n最常混淆的类别对 (前10):")
        for i, pair in enumerate(error_analysis['most_confused_classes']):
            confidence_diff = f"{pair['confidence_diff']:+.3f}"
            print(f"  {i + 1}. {pair['true_class']} → {pair['pred_class']}: {pair['count']} 次 "
                  f"(置信度差异: {confidence_diff})")

def calculate_cluster_quality_metrics(error_clusters):
    """计算错误聚类的质量指标"""
    quality_metrics = {
        'total_clusters': 0,
        'avg_samples_per_cluster': 0,
        'cluster_size_std': 0,
        'high_variance_clusters': [],
        'low_variance_clusters': [],
        'cluster_quality_scores': {}
    }

    all_cluster_sizes = []
    all_variances = []

    for true_class, pred_classes in error_clusters.items():
        for pred_class, cluster_info in pred_classes.items():
            quality_metrics['total_clusters'] += 1
            cluster_size = cluster_info['count']
            all_cluster_sizes.append(cluster_size)

            # 计算平均方差（所有特征维度的平均值）
            avg_variance = torch.mean(cluster_info['variance']).item()
            all_variances.append(avg_variance)

            # 识别高方差和低方差聚类
            cluster_id = f"{true_class}→{pred_class}"
            if avg_variance > 0.1:  # 阈值可根据实际情况调整
                quality_metrics['high_variance_clusters'].append(cluster_id)
            elif avg_variance < 0.01:
                quality_metrics['low_variance_clusters'].append(cluster_id)

            # 计算聚类质量评分（综合考虑样本数量和方差）
            # 样本数量越多、方差越小，质量越高
            size_score = min(cluster_size / 10, 1.0)  # 样本数量评分，最多10个样本得满分
            variance_score = max(0, 1 - avg_variance * 10)  # 方差评分，方差越小得分越高

            quality_score = size_score * 0.6 + variance_score * 0.4
            quality_metrics['cluster_quality_scores'][cluster_id] = {
                'score': quality_score,
                'size': cluster_size,
                'variance': avg_variance,
                'size_score': size_score,
                'variance_score': variance_score
            }

    # 计算总体统计
    if all_cluster_sizes:
        quality_metrics['avg_samples_per_cluster'] = np.mean(all_cluster_sizes)
        quality_metrics['cluster_size_std'] = np.std(all_cluster_sizes)

    return quality_metrics

def complete_error_cluster_analysis(protos_list):
    """完整的错误聚类分析流程"""
    print("开始错误分类聚类分析...")

    # 1. 聚类错误样本
    error_clusters = cluster_error_protos_by_prediction(protos_list)

    error_analysis = calculate_cluster_quality_metrics(error_clusters)

    # 2. 可视化结果
    visualize_error_clusters(error_clusters, error_analysis)

    # 3. 计算质量指标
    quality_metrics = calculate_cluster_quality_metrics(error_clusters)

    print(f"\n聚类质量指标:")
    print(f"  总聚类数: {quality_metrics['total_clusters']}")
    print(
        f"  平均每聚类样本数: {quality_metrics['avg_samples_per_cluster']:.2f} ± {quality_metrics['cluster_size_std']:.2f}")
    print(f"  高方差聚类: {len(quality_metrics['high_variance_clusters'])}")
    print(f"  低方差聚类: {len(quality_metrics['low_variance_clusters'])}")