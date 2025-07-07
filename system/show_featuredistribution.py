import copy

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


if __name__ == '__main__':

    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = FedAvgCNN(in_features=3, num_classes=100, dim=10816).to(device)
    # head = copy.deepcopy(model.fc)
    # headL = copy.deepcopy(model.fc)
    # model.fc = nn.Identity()
    # decoder = Decoder(512, device=device)
    # model = BaseMineSplit(model, head, headL, decoder).to(device)
    label_select = [0,1,2,3,4]
    client_select = [i for i in range(20)]
    dataset = 'Cifar100'
    algo = 'FedDFCC'
    # protos = {'0':[],'1':[],'2':[],'3':[],'4':[]}
    protos = defaultdict(list)
    for id in client_select:
        dataloader = get_datalodaer(dataset,id)
        # tmp_model = copy.deepcopy(model)
        # tmp_model.load_state_dict(torch.load("models/Cifar100_01/FedDFCC_client_{}.pt".format(id), weights_only=False))
        tmp_model = torch.load("models/Cifar100/{}_client_{}.pt".format(algo, id), weights_only=False)
        # tmp_model = torch.load("models/Cifar100/{}_server.pt".format(algo, id), weights_only=False)
        with torch.no_grad():
            for x, y in dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                rep = tmp_model.base(x)
                d = rep.size(1)
                embedding_inv = rep[:, :d // 2]
                embedding_spe = rep[:, d // 2:]
                out_g = tmp_model.head(embedding_inv)
                out_p = tmp_model.headL(embedding_spe)

                for yy,pp in zip(y,embedding_spe):
                    label = yy.item()
                    if label in label_select:
                        protos[label].append(pp)
        for key in protos.keys():
            print("{}:{}".format(key,len(protos[key])))

    np.random.seed(42)
    features, labels = [],[]
    for key, proto in protos.items():
        for p in proto:
            labels.append(key)
            features.append(p.cpu().numpy())

    print(len(labels),"==",len(features))
    features = np.stack(features)
    # 初始化t-SNE对象
    tsne = TSNE(n_components=2, random_state=0)
    # 使用t-SNE进行降维
    reduced_features = tsne.fit_transform(features)

    # 绘制t-SNE图
    plt.figure(figsize=(8, 6))
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # 获取颜色映射
    for i, label in enumerate(unique_labels):
        # 找到对应标签的所有点
        label_indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(reduced_features[label_indices, 0], reduced_features[label_indices, 1], label=label,
                    color=colors(i))

    plt.title('t-SNE visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()