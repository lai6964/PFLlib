import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from main import *
from utils.data_utils import read_client_data

from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_datalodaer(dataset, id):
    train_data = read_client_data(dataset, id, is_train=True)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

font2 = {'family': 'Times New Roman',
         'size': 17}

font_axis = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 15}

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 20}


def tmp(ids,labels, reduced_features):
    fig, ax = plt.subplots(figsize=(8, 6))

    # ---------- 1. 准备映射 ----------
    unique_ids = sorted(set(ids))
    unique_labels = sorted(set(labels))
    cmap = cm.get_cmap('tab20', len(unique_ids))
    client2color = {c: cmap(i) for i, c in enumerate(unique_ids)}
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
    label2marker = {l: markers[i % len(markers)] for i, l in enumerate(labels)}

    # ---------- 2. 画散点 ----------
    for c in unique_ids:
        for l in unique_labels:
            mask = (np.array(ids) == c) & (np.array(labels) == l)
            ax.scatter(reduced_features[mask, 0],
                       reduced_features[mask, 1],
                       color=client2color[c],
                       marker=label2marker[l],
                       s=60)
            # **不加 label**

    # ---------- 3. 生成两个独立图例 ----------
    # 3-a 颜色图例
    color_handles = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=client2color[c], markersize=8)
                     for c in unique_ids]
    leg1 = ax.legend(color_handles, [f'client_{c}' for c in unique_ids],
                     title='Client', bbox_to_anchor=(1.02, 1), loc='upper left')

    # 3-b 形状图例
    shape_handles = [Line2D([0], [0], marker=label2marker[l], color='k',
                            markersize=8, linestyle='None')
                     for l in unique_labels]
    ax.legend(shape_handles, [str(l) for l in unique_labels],
              title='Class', bbox_to_anchor=(1.02, 0.5), loc='upper left')

    # 把第一个图例再添加回去，否则会丢失
    ax.add_artist(leg1)

    plt.tight_layout()
    plt.show()

def show_clinet(ids,labels, reduced_features):

    plt.figure(figsize=(10, 6))
    unique_clients = list(set(ids))
    colors = plt.cm.get_cmap('tab20', len(unique_clients))  # 获取颜色映射
    for c in unique_clients:
        mask = np.array(ids) == c
        plt.scatter(reduced_features[mask, 0],
                    reduced_features[mask, 1],
                    label='C'+str(c),
                    color=colors(c),
                    # marker=markers[l],
                    )
    ax = plt.gca()  # 获取当前坐标轴对象
    ax.set_xticks([])  # 隐藏 x 轴刻度
    ax.set_yticks([])  # 隐藏 y 轴刻度
    plt.legend(prop=font_legend,
               ncol=2,  # 或需要的列数
               bbox_to_anchor=(1.02, 1),  # 锚点：轴外右侧顶部
               loc='upper left',  # 图例的“左上角”对齐到锚点
               borderaxespad=0)  # 与轴不留额外间距
    plt.tight_layout()
    plt.savefig(f"tmp.pdf",
                bbox_inches='tight',  # 关键：让 savefig 计算紧凑边界
                pad_inches=0)  # 可选：不留额外边距（默认 0.1）
    plt.show()

def show_label(ids,labels, reduced_features):

    # # 绘制t-SNE图
    plt.figure(figsize=(8, 6))
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # 获取颜色映射
    markers = ['o', 's', '^', 'v', '<']#, '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    for i, label in enumerate(unique_labels):
        # 找到对应标签的所有点
        label_indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(reduced_features[label_indices, 0],
                    reduced_features[label_indices, 1],
                    label=labelname[label],
                    color=colors(i),
                    # marker=markers[i],
                    )
    ax = plt.gca()  # 获取当前坐标轴对象
    ax.set_xticks([])  # 隐藏 x 轴刻度
    ax.set_yticks([])  # 隐藏 y 轴刻度
    plt.legend(prop=font_legend,
               ncol=1,  # 或需要的列数
               # bbox_to_anchor=(1.02, 1),  # 锚点：轴外右侧顶部
               # loc='upper left',  # 图例的“左上角”对齐到锚点
               borderaxespad=0.1)  # 与轴不留额外间距
    plt.tight_layout()
    plt.savefig(f"tmp.pdf",
                bbox_inches='tight',  # 关键：让 savefig 计算紧凑边界
                pad_inches=0)  # 可选：不留额外边距（默认 0.1）
    plt.show()

def show_img(ids,labels,reduced_features):

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
    plt.savefig(f"tmp.pdf",
                bbox_inches='tight',  # 关键：让 savefig 计算紧凑边界
                pad_inches=0)  # 可选：不留额外边距（默认 0.1）
    plt.show()

if __name__ == '__main__':

    labelname = ['apple', 'aquarium', 'baby', 'bear', 'beaver']

    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = FedAvgCNN(in_features=3, num_classes=100, dim=10816).to(device)
    # head = copy.deepcopy(model.fc)
    # headL = copy.deepcopy(model.fc)
    # model.fc = nn.Identity()
    # decoder = Decoder(512, device=device)
    # model = BaseMineSplit(model, head, headL, decoder).to(device)
    label_select = [0,1,2,3,4]#[20,10]
    client_select = [i for i in range(20)]
    dataset = 'Cifar100'
    algos = ['FedDFCC','PerAvg','FedProto']
    algo = algos[2]
    # protos = {'0':[],'1':[],'2':[],'3':[],'4':[]}
    protos = defaultdict(list)
    client_id = defaultdict(list)
    for id in client_select:
        dataloader = get_datalodaer(dataset,id)
        # tmp_model = copy.deepcopy(model)
        # tmp_model.load_state_dict(torch.load("models/Cifar100_01/FedDFCC_client_{}.pt".format(id), weights_only=False))
        tmp_model = torch.load("models/Cifar100_head/{}_client_{}.pt".format(algo, id), weights_only=False)
        # tmp_model = torch.load("models/Cifar100/{}_client_{}.pt".format(algo, id), weights_only=False)
        # tmp_model = torch.load("models/Cifar100/{}_server.pt".format(algo), weights_only=False)
        with torch.no_grad():
            for x, y in dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                if algo=='PerAvg':
                    rep = tmp_model(x)
                else:
                    rep = tmp_model.base(x)
                    # if algo == 'FedDFCC':
                    d = rep.size(1)
                    rep = rep[:, :d // 2]
                        # rep = rep[:, d // 2:]
                # embedding_inv = rep[:, :d // 2]
                # embedding_spe = rep[:, d // 2:]
                # out_g = tmp_model.head(embedding_inv)
                # out_p = tmp_model.headL(embedding_spe)

                for yy,pp in zip(y,rep):
                    label = yy.item()
                    if label in label_select:
                        protos[label].append(pp)
                        client_id[label].append(id)
        for key in protos.keys():
            print("{}:{}".format(key,len(protos[key])))

    np.random.seed(42)
    features, labels, ids = [],[],[]
    for key, proto in protos.items():
        for p in proto:
            labels.append(key)
            features.append(p.cpu().numpy())
        for id in client_id[key]:
            ids.append(id)

    print(len(labels),"==",len(features),"==",len(ids))
    features = np.stack(features)
    # 初始化t-SNE对象
    tsne = TSNE(n_components=2, random_state=0)
    # 使用t-SNE进行降维
    reduced_features = tsne.fit_transform(features)

    # tmp(ids,labels,reduced_features)
    show_img(ids,labels,reduced_features)
    # show_clinet(ids,labels,reduced_features)
