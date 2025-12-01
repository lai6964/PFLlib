import copy
import random
from poplib import error_proto

import torch
import time
import numpy as np
from pyexpat import features

from clientclip2fl import clientCLIP2FL
from flcore.servers.serverbase import Server
from collections import defaultdict
import sys
import json
import clip
from torchvision import datasets
import torch.nn as nn

class FedCLIP2FL(Server):
    def __init__(self, args, times, label_name):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCLIP2FL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.num_of_feature = 512
        self.contrast_alpha = 0.001
        self.lr_net = 0.01
        self.match_epoch = 100
        self.dis_metric = 'ours'
        self.batch_size_local_training = 32
        self.crt_epoch = 300
        self.ins_temp = 0.1
        self.feature_syn = torch.randn(size=(args.num_classes * self.num_of_feature, 512), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)], dtype=torch.long,
                                      requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

        self.label_name = label_name
        self.feature_net = copy.deepcopy(self.global_model.head)
        self.contras_criterion = SupConLoss_text(args.device, self.ins_temp, args.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)
        self.optimizer_feature = torch.optim.SGD([self.feature_syn, ], lr=0.1)  # optimizer_img for synthetic data


    def train(self):
        clip_model, _ = clip.load('ViT-B/16', device=self.device)
        clip_model.eval()
        text_inputs = clip.tokenize([f"a photo of a {c}" for c in self.label_name]).to(self.device)  # torch.size([10, 77])
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
        text_features = text_features.float()
        text_features /= text_features.norm(dim=-1, keepdim=True)  # torch.size([10, 512])
        new_text_features = text_features[0].repeat(100, 1)
        for i in range(1, self.num_classes):
            new_text_features = torch.cat([new_text_features, text_features[i].repeat(100, 1)], 0)

        for i in range(self.global_rounds + 1):
            sys.stdout.flush()  # 强制刷新标准输出缓冲区
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train(i)


            self.receive_models()
            self.aggregate_parameters()
            self.update_feature_syn(new_text_features)
            self.feature_re_train()

            # # aggregating local models with FedAvg
            # fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
            # global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params), list_clients_gradient,
            #                                 new_text_features)
            # # re-trained classifier
            # syn_params, ft_params = global_model.feature_re_train(copy.deepcopy(fedavg_params),
            #                                                       args.batch_size_local_training)
            # # global eval
            # one_re_train_acc = global_model.global_eval(ft_params, data_global_test, args.batch_size_test)
            # re_trained_acc.append(one_re_train_acc)
            # global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))


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
            self.set_new_clients(clientCLIP2FL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.list_clients_gradient = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                truth_gradient = client.compute_gradient(self.global_model)
                self.list_clients_gradient.append(copy.deepcopy(truth_gradient))
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def update_feature_syn(self, new_text_features):
        for param, target_param in zip(self.global_model.head.parameters(), self.feature_net.parameters()):
            target_param.data = param.data.clone()
        self.feature_net.train()
        net_global_parameters = list(self.feature_net.parameters())
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        for gradient_one in self.list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        gw_real_avg = {class_index: [] for class_index in range(self.num_classes)}
        # aggregate the real feature gradients
        for i in range(self.num_classes):
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]

            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp
        # update the federated features.
        for ep in range(self.match_epoch):
            loss_feature = torch.tensor(0.0).to(self.device)
            for c in range(self.num_classes):
                if len(gw_real_avg[c]) != 0:
                    feature_syn = self.feature_syn[c * self.num_of_feature:(c + 1) * self.num_of_feature].reshape(
                        (self.num_of_feature, 512))
                    lab_syn = torch.ones((self.num_of_feature,), device=self.device, dtype=torch.long) * c
                    # print("test lab_syn: ", lab_syn, lab_syn.shape)
                    output_syn = self.feature_net(feature_syn)
                    loss_syn = self.criterion(output_syn, lab_syn)
                    # compute the federated feature gradients of class c
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    loss_feature += match_loss(gw_syn, gw_real_avg[c], self.dis_metric, self.device)
            contrast_loss = self.contras_criterion(self.feature_syn, self.label_syn, new_text_features)
            # Eq. 8
            loss_feature += self.contrast_alpha * contrast_loss
            self.optimizer_feature.zero_grad()
            loss_feature.backward()
            self.optimizer_feature.step()

    def feature_re_train(self):
        feature_syn_train_ft = copy.deepcopy(self.feature_syn.detach())
        label_syn_train_ft = copy.deepcopy(self.label_syn.detach())
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        ft_model = nn.Linear(512, self.num_classes).to(self.device)
        optimizer_ft_net = torch.optim.SGD(ft_model.parameters(), lr=self.lr_net)  # optimizer_img for synthetic data
        ft_model.train()
        for epoch in range(self.crt_epoch):
            trainloader_ft = torch.utils.data.dataloader.DataLoader(dataset=dst_train_syn_ft,
                                        batch_size=self.batch_size_local_training,
                                        shuffle=True)
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = self.criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
        ft_model.eval()

        for param, target_param in zip(ft_model.parameters(), self.global_model.head.parameters()):
            target_param.data = param.data.clone()
        return None


# Dataset.Gradient_matching_loss.py
def match_loss_cpu(gw_syn, gw_real, args):
    # dis = torch.tensor(0.0).to(args.device)
    dis = torch.tensor(0.0)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                    torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis


def match_loss(gw_syn, gw_real, dis_metric, device):
    # dis = torch.tensor(0.0).to(args.device)
    dis = torch.tensor(0.0).to(device)

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                    torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        # return 0

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

# Dataset/dataset.py
class TensorDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

# losses.py
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# clear those instances that have no positive instances to avoid training error
class SupConLoss_clear(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss

# clear those instances that have no positive instances to avoid training error
class SupConLoss_text(nn.Module):
    def __init__(self, device="0", temperature=0.07, num_classes=10):
        super(SupConLoss_text, self).__init__()
        self.device = device
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, features, labels, text_features):
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)  # features, labels: [1000, 512], [1000, 1]
        # mask = F.one_hot(labels, labels.T).float().to(self.device)  # mask [1000,1000]
        mask = torch.eq(labels, labels.T).float().to(self.device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, text_features.T),
            self.temperature)  # anchor_dot_contrast: [1000, 10]

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # logits_max: [1000, 1]
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )  # logits_mask: [1000, 1000]

        mask = mask * logits_mask  # mask: [1000, 1000]
        single_samples = (mask.sum(1) == 0).float()  # single_samples: [1000]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss

class SupConLoss_text_cpu(nn.Module):
    def __init__(self, device="0", temperature=0.07, num_classes=10):
        super(SupConLoss_text_cpu, self).__init__()
        self.device = device
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, features, labels, text_features):
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)  # features, labels: [1000, 512], [1000, 1]
        # mask = F.one_hot(labels, labels.T).float().to(self.device)  # mask [1000,1000]
        mask = torch.eq(labels, labels.T).float()

        anchor_dot_contrast = torch.div(
            torch.matmul(features, text_features.T),
            self.temperature)  # anchor_dot_contrast: [1000, 10]

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # logits_max: [1000, 1]
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1),
            0
        )  # logits_mask: [1000, 1000]

        mask = mask * logits_mask  # mask: [1000, 1000]
        single_samples = (mask.sum(1) == 0).float()  # single_samples: [1000]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss