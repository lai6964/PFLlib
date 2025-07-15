import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
from utils.finch import FINCH
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import torch.nn.functional as F
import os

from openpyxl.styles.builtins import output


def get_protos_meanvar(protos_in):
    protos = clone_defaultdict(protos_in)
    protos_means, protos_vars = {}, {}
    for key, protos_list in protos.items():
        if len(protos_list) > 1:
            protos_tensor = torch.stack(protos_list)
            protos_means[key] = protos_tensor.mean(dim=0)
            protos_vars[key] = protos_tensor.var(dim=0)
        else:
            protos_means[key] = protos_list[0]
            protos_vars[key] = torch.zeros_like(protos_means[key])
    return protos_means, protos_vars

def get_local_protos(model, dataloader, device):
    agg_protos_label = {}
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            features, _ = model(images)
            for i in range(len(labels)):
                key = labels[i].item()
                if key in agg_protos_label:
                    agg_protos_label[key].append(features[i, :].detach())
                else:
                    agg_protos_label[key] = [features[i, :].detach()]
    return agg_protos_label

def get_special_protos(protos_in):
    agg_protos_label = clone_defaultdict(protos_in)
    # 捷径特征只取后半段聚类，前半段要置0，否则影响聚类中心
    protos_meas, protos_vars = {}, {}
    for key, proto_list in agg_protos_label.items():
        d = proto_list[0].size(0)
        for proto in proto_list:
            proto[:d // 2] = 0

        if len(proto_list) > 1:
            proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
            proto_list = np.array(proto_list)

            c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                        ensure_early_exit=False, verbose=False)

            m, n = c.shape
            class_cluster_list = []
            for index in range(m):
                class_cluster_list.append(c[index, -1])

            class_cluster_array = np.array(class_cluster_list)
            uniqure_cluster = np.unique(class_cluster_array).tolist()
            agg_selected_proto = []
            agg_selected_proto_var = []

            for _, cluster_index in enumerate(uniqure_cluster):
                selected_array = np.where(class_cluster_array == cluster_index)
                selected_proto_list = proto_list[selected_array]
                proto = np.mean(selected_proto_list, axis=0, keepdims=True)
                proto_var = np.var(selected_proto_list, axis=0, keepdims=True)

                agg_selected_proto.append(torch.tensor(proto))
                agg_selected_proto_var.append(torch.tensor(proto_var))
            protos_meas[key] = agg_selected_proto
            protos_vars[key] = agg_selected_proto_var
        else:
            protos_meas[key] = [proto_list[0].data]
            protos_vars[key] = [torch.zeros_like(proto_list[0].data)]
    return protos_meas, protos_vars

class clientDFGC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda
        self.margin = 0.5
        self.using_tripletloss = args.using_tripletloss
        self.using_reconstructloss = args.using_reconstructloss
        self.using_specialloss = args.using_specialloss
        self.using_normal = args.using_normal
        self.using_glocla = args.using_glocla
        self.firsttime = True
        self.protos_num=self.extract_protonum()

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()


        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.headL.parameters():
            param.requires_grad = True
        if not self.using_glocla:
            for param in self.model.head.parameters():
                param.requires_grad = True


        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            rep = self.model.base(x)
            d = rep.size(1)
            output = self.model.headL(rep[:,d//2:])
            loss = self.loss(output, y)
            if not self.using_glocla:
                output = self.model.head(rep)
                loss += self.loss(output, y)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.headL.parameters():
            param.requires_grad = False
        if not self.using_glocla:
            for param in self.model.head.parameters():
                param.requires_grad = False

        if self.using_glocla:# and self.firsttime:
            for param in self.model.head.parameters():
                param.requires_grad = False
            # self.firsttime = False

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        protos_inv = defaultdict(list)
        protos_spe = defaultdict(list)
        for epoch in range(max_local_epochs):
            total_invloss, total_speloss, total_triloss, total_proloss = 0,0,0,0
            for images, labels in trainloader:
                if type(images) == type([]):
                    images[0] = images[0].to(self.device)
                else:
                    images = images.to(self.device)
                labels = labels.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                embedding = self.model.base(images)
                d = embedding.size(1)
                embedding_inv = embedding[:, :d // 2]
                embedding_spe = embedding[:, d // 2:]
                output_inv = self.model.head(embedding)
                lossCE_inv = self.loss(output_inv, labels)
                total_invloss+=lossCE_inv.item()
                loss = lossCE_inv
                if self.using_specialloss:
                    output_spe = self.model.headL(embedding_spe)
                    lossCE_spe = self.loss(output_spe, labels)
                    loss += lossCE_spe
                    total_speloss+=lossCE_spe.item()

                if self.using_reconstructloss:
                    reconstructed_image = self.model.decoder(embedding)
                    loss_recon = self.loss_mse(images, reconstructed_image)
                    loss += loss_recon * 0.1

                if self.using_normal:
                    embedding = F.normalize(embedding, dim=1)
                    embedding_inv = F.normalize(embedding_inv, dim=1)
                    embedding_spe = F.normalize(embedding_spe, dim=1)

                if self.global_protos is not None:
                    proto_g = copy.deepcopy(embedding_inv.detach())
                    for i, label in enumerate(labels):
                        key = label.item()
                        if type(self.global_protos[key]) != type([]):
                            proto_g[i, :] = self.global_protos[key].data[:d//2]
                    loss_proto = self.loss_mse(embedding_inv, proto_g.detach()) * self.lamda
                    loss += loss_proto
                    total_proloss+=loss_proto.item()


                    if self.using_tripletloss:
                        proto_g = copy.deepcopy(embedding_spe.detach())
                        proto_l = copy.deepcopy(embedding_spe.detach())
                        for i, label in enumerate(labels):
                            key = label.item()
                            if key in self.global_protos.keys():
                                proto_g[i, :] = self.global_protos[key].data[d//2:]
                                proto_l[i, :] = self.special_protos[key].data
                        distance_positive = torch.nn.functional.pairwise_distance(embedding_spe, proto_l)
                        distance_negative = torch.nn.functional.pairwise_distance(embedding_spe, proto_g)
                        # loss_triplet = torch.nn.functional.relu(distance_positive - distance_negative + self.margin).mean()
                        loss_triplet = torch.mean(torch.clamp(distance_positive - distance_negative + self.margin, min=0.0)) * 0.1
                        loss += loss_triplet
                        total_triloss+=loss_triplet.item()

                if epoch==max_local_epochs-1:
                    for i, yy in enumerate(labels):
                        y_c = yy.item()
                        protos[y_c].append(embedding[i, :].detach().data)
                        protos_inv[y_c].append(embedding[i, :].detach())
                        protos_spe[y_c].append(embedding_spe[i, :].detach())


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # print("epoch:{}\tinv:{:.6f}\tspe:{:.6f}\ttri:{:.6f}\tpro:{:.6f}".format(epoch,total_invloss/self.train_samples,total_speloss/self.train_samples,total_triloss/self.train_samples,total_proloss/self.train_samples))
        # print("id",self.id)
        # print("len(protos):",len(protos))

        # self.model.cpu()
        # rep = self.model.base(x)
        # print(torch.sum(rep!=0).item() / rep.numel())

        # protos = self.collect_protos()
        self.protos = agg_func(protos)
        # print("len(self.protos):",len(self.protos))
        self.local_protos, self.local_protos_vars = get_protos_meanvar(protos_inv)
        # self.special_protos, self.special_protos_vars = get_special_protos(protos_spe)
        self.special_protos, self.special_protos_vars = get_protos_meanvar(protos_spe)
        # print("len(self.protos):",len(self.protos))

        # tmplist = []
        # for images, labels in trainloader:
        #     for label in labels:
        #         y_c = label.item()
        #         if y_c not in tmplist:
        #             tmplist.append(y_c)
        # print("len(tmplist):",len(tmplist))
        # for key in tmplist:
        #     if key not in self.protos.keys():
        #         print("{}=={}?".format(len(tmplist), len(self.protos.keys())))
        #         print("aaa not key {} in this client {}".format(key, self.id))
        #         if key in protos.keys():
        #             print(protos[key])
        #         else:
        #             print("not in protos too")
        #         self.protos = agg_func(protos)
        #         raise ValueError("aaa not key {} in this client {}".format(key, self.id))


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        # self.test_acc, self.test_num, self.auc, self.test_acc2 = self.test_metrics2(self.model)

    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                if self.using_normal:
                    rep = F.normalize(rep, dim=1)
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)
        return protos

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0

        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)
                    if self.using_normal:
                        rep = F.normalize(rep, dim=1)
                    d = rep.size(1)
                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in self.global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r[:d//2], pro[:d//2])

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                d = rep.size(1)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                d = rep.size(1)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new[:d//2], rep[:d//2]) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def test_metrics2(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_acc2 = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                d = rep.size(1)
                embedding_inv = rep[:, :d // 2]
                embedding_spe = rep[:, d // 2:]
                out_g = self.model.head(rep)
                out_p = self.model.headL(embedding_spe)
                output = torch.nn.functional.softmax(out_g.detach()) + torch.nn.functional.softmax(out_p.detach())

                test_acc += (torch.sum(torch.argmax(out_g, dim=1) == y)).item()
                test_acc2 += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(torch.nn.functional.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        self.save_local_model()

        return test_acc, test_num, auc, test_acc2
    #
    # def train_metrics(self):
    #     trainloader = self.load_train_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()
    #
    #     train_num = 0
    #     losses = 0
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             rep = self.model.base(x)
    #             out_g = self.model.head(rep)
    #             out_p = self.model.headL(rep.detach())
    #             output = out_g.detach() + out_p
    #             loss = self.loss(output, y)
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]
    #
    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')
    #
    #     return losses, train_num

    def set_classifier_parameters(self, classifier_model):
        for new_param, old_param in zip(classifier_model.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()

    def set_parameters(self, model):
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def extract_protonum(self):
        py = torch.zeros(self.num_classes)
        for x, y in self.load_train_data():
            for yy in y:
                py[yy.item()] += 1
        return py
    def save_local_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_client_{}.pt".format(self.id))
        torch.save(self.model, model_path)

    def train_one_step(self):
        trainloader = self.load_train_data(self.batch_size)
        iter_loader = iter(trainloader)
        # self.model.to(self.device)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        (x, y) = next(iter_loader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        rep = self.model.base(x)
        d = rep.size(1)
        embedding_inv = rep[:, :d // 2]
        embedding_spe = rep[:, d // 2:]
        out_g = self.model.head(rep)
        out_p = self.model.headL(embedding_spe)

        loss = self.loss(out_g, y)+self.loss(out_p, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos_in):
    """
    Returns the average of the weights.
    """
    protos = clone_defaultdict(protos_in)
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def clone_defaultdict(protos):
    cloned_protos = {}#defaultdict(type(protos.default_factory))
    for key, value in protos.items():
        if isinstance(value, torch.Tensor):
            cloned_protos[key] = value.clone()
        elif isinstance(value, list):
            cloned_protos[key] = [v.clone() if isinstance(v, torch.Tensor) else v for v in value]
        else:
            raise ValueError(f"Unsupported type for cloning: {type(value)}")
    return cloned_protos