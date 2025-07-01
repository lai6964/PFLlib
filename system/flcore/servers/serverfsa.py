import time
from flcore.clients.clientfsa import clientFSA
from flcore.servers.serverbase import Server
from threading import Thread
import sys
import torch
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class FedFSA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFSA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.global_protos = get_clip_textembeddings(args.dataset)

    def send_protos(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.set_protos(self.global_protos)

    def train(self):
        self.send_protos()
        for i in range(self.global_rounds+1):
            sys.stdout.flush()  # 强制刷新标准输出缓冲区
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()


            self.compute_local_plv_()
            self.retrain_cls_relation_(i)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            print("\nBest accuracy.")
            # self.print_(max(self.rs_test_acc), max(
            #     self.rs_train_acc), min(self.rs_train_loss))
            print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFSA)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def compute_local_plv_(self):
        local_proto_list = []
        local_proto_label_list = []
        local_vars_list = []

        for client in self.selected_clients:
            local_proto_list.append(client.local_proto)
            local_proto_label_list.append(client.local_labels)
            local_vars_list.append(client.local_vars)
        self.local_protos = torch.cat(local_proto_list)
        self.proto_labels = torch.cat(local_proto_label_list)
        self.local_vars = torch.cat(local_vars_list)

    def retrain_cls_relation_(self, epoch, re_k=2, re_bs=32, dist_weight=1.0, angle_weight=1.0, re_mu=1.0, re_phase='p2'):
        prototypes = self.local_protos
        proto_labels = self.proto_labels
        local_vars = self.local_vars


        if epoch < 5:
            init_lr = 1e-1
        else:
            init_lr = 1e-2

        model = self.global_model.head
        model.to(self.device)
        lr_decay = 40
        decay_rate = 0.1

        prototypes = prototypes.to(self.device)
        proto_labels = proto_labels.to(self.device)

        random_vectors = generate_random_vectors(local_vars, k=re_k).to(self.device)
        local_protos = torch.cat([prototypes] * re_k).to(self.device) + random_vectors
        local_labels = torch.cat([proto_labels] * re_k).to(self.device)

        local_protos = torch.cat([prototypes, local_protos])
        local_labels = torch.cat([proto_labels, local_labels])
        # print(proto_labels.shape)
        #     ipdb.set_trace()
        #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        relation_criterion = Relation_Loss(dist_weight, angle_weight)
        if re_phase != 'p5':
            idx_list = np.array(np.arange(len(proto_labels)))
        else:
            idx_list = np.array(np.arange(len(local_labels)))
        batch_size = re_bs

        for epoch in range(100):
            optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
            random.shuffle(idx_list)

            epoch_loss_collector = []
            epoch_relationloss_collector = []
            if re_phase != 'p5':
                for i in range((len(proto_labels) + batch_size - 1) // batch_size):  # 向上取整计算需要多少批次
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, len(proto_labels))  # 防止索引超出范围
                    if start_index < end_index:  # 确保批次不为空
                        x = prototypes[idx_list[start_index:end_index]]
                        target = proto_labels[idx_list[start_index:end_index]]
                        batch_vars = local_vars[idx_list[start_index:end_index]]

                        if re_phase == 'p1':
                            optimizer.zero_grad()
                            x.requires_grad = True
                            target.requires_grad = False
                            target = target.long()
                            # 只需要分类
                            # feats, out = model(x)
                            out = model.head(x)
                            celoss = criterion(out, target)
                            relationloss = torch.tensor(0).cuda()

                        elif re_phase == 'p2':
                            optimizer.zero_grad()
                            target = target.long()
                            #                         if round > 1:
                            #                             ipdb.set_trace()
                            # 增广+分类
                            random_vectors = generate_random_vectors(batch_vars, k=re_k).cuda()
                            x = torch.cat([x] * re_k)
                            target = torch.cat([target] * re_k)
                            x.requires_grad = True
                            target.requires_grad = False
                            # feats, out = model(x + random_vectors)
                            # print(x.shape, random_vectors.shape)
                            out = model(x + random_vectors)

                            celoss = criterion(out, target)
                            relationloss = torch.tensor(0).cuda()

                        elif re_phase == 'p3':
                            optimizer.zero_grad()
                            target = target.long()
                            # 增广+分类+原始
                            random_vectors = generate_random_vectors(batch_vars, k=re_k).cuda()
                            x_ = torch.cat([x] * re_k)
                            x_ = x_ + random_vectors
                            x_all = torch.cat([x, x_], dim=0)
                            target_all = torch.cat([target] * (re_k + 1))
                            x_all.requires_grad = True
                            target_all.requires_grad = False
                            # feats, out = model(x_all)
                            out = model(x_all)
                            celoss = criterion(out, target_all)
                            relationloss = torch.tensor(0).cuda()
                        # elif re_phase == 'p4':
                        #     optimizer.zero_grad()
                        #     target = target.long()
                        #     # 增广+分类+原始+关系
                        #     random_vectors = generate_random_vectors(batch_vars).cuda()
                        #     x_ = x + random_vectors
                        #     x_all = torch.cat([x, x_], dim=0)
                        #     target_all = torch.cat([target, target])
                        #     x_all.requires_grad = True
                        #     target_all.requires_grad = False
                        #     # feats, out = model(x_all)
                        #     out = model(x_all)
                        #     celoss = criterion(out, target_all)
                        #     relationloss = relation_criterion(feats, self.global_protos[target_all])
                        loss = celoss + re_mu * relationloss
                        epoch_loss_collector.append(celoss.data)
                        epoch_relationloss_collector.append(relationloss.data)

                        loss.backward()
                        optimizer.step()
            # elif re_phase == 'p5':
            #     for i in range((len(local_labels) + batch_size - 1) // batch_size):  # 向上取整计算需要多少批次
            #         start_index = i * batch_size
            #         end_index = min((i + 1) * batch_size, len(local_protos))  # 防止索引超出范围
            #         if end_index - start_index > 1:  # 确保批次不为空
            #             x = local_protos[idx_list[start_index:end_index]]
            #             target = local_labels[idx_list[start_index:end_index]]
            #
            #             optimizer.zero_grad()
            #             x.requires_grad = True
            #             target.requires_grad = False
            #             target = target.long()
            #             # feats, out = model(x)
            #             out = model(x)
            #             celoss = criterion(out, target)
            #             relationloss = relation_criterion(feats, self.global_protos[target])
            #             #                     if (torch.isnan(celoss).any()) or (torch.isnan(relationloss).any()):
            #         #                         ipdb.set_trace()
            #         # #                     import ipdb; ipdb.set_trace()
            #         loss = celoss + re_mu * relationloss
            #
            #         epoch_loss_collector.append(celoss.data)
            #         epoch_relationloss_collector.append(relationloss.data)
            #
            #         loss.backward()
            #         optimizer.step()
            # print(epoch, sum(epoch_loss_collector) / len(epoch_loss_collector), sum(epoch_relationloss_collector) / len(epoch_relationloss_collector))
        self.global_model.head = model

def generate_random_vectors(var_vectors, k=1):
    random_vectors = []
    for _ in range(k):
        for vars_i in var_vectors:
            std_vector = torch.sqrt(vars_i)  # 计算标准差
            mean_vector = torch.zeros(std_vector.size(0)).cuda()  # 定义均值向量

            # 根据均值和标准差生成随机向量
            random_vector = torch.normal(mean=mean_vector, std=std_vector)
            random_vectors.append(random_vector)

    # 将列表转换为矩阵
    #     ipdb.set_trace()
    random_matrix = torch.stack(random_vectors)

    return random_matrix

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class KDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        # kd = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
        #               F.softmax(out_t / self.T, dim=1),
        #               reduction='none').mean(dim=0)
        kd_loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return kd_loss

def js_divergence(p, q):
    kl_loss = KDLoss(T=0.5).cuda()
    #     ipdb.set_trace()
    half = torch.div(p + q, 2)
    s1 = kl_loss(p, half)
    s2 = kl_loss(q, half)
    #     ipdb.set_trace()
    return torch.div(s1 + s2, 2)

class RKdAngle(nn.Module):
    def forward(self, student, teacher, args):
        # N x C
        # N x N x C
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1)) + 1e-6
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
#         import ipdb; ipdb.set_trace()
        if args.mode_angle == 'L2':
            loss = F.mse_loss(s_angle, t_angle, reduction='mean')
        elif args.mode_angle == 'L1':
            loss = F.l1_loss(s_angle, t_angle, reduction='mean')
        elif args.mode_angle == 'smooth_l1':
            loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
#         if (torch.isnan(loss).any()):
#             ipdb.set_trace()
        return loss

class RkdDistance(nn.Module):
    def forward(self, student, teacher, args):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
#         ipdb.set_trace()
        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        if args.mode_dis == 'KL':
            KL_loss = KDLoss(T=0.5).to(args.device)
            loss = KL_loss(d, t_d)
        elif args.mode_dis == 'L2':
            loss = F.mse_loss(d, t_d, reduction='mean')
        elif args.mode_dis == 'L1':
            loss = F.l1_loss(d, t_d, reduction='mean')
        elif args.mode_dis == 'smooth_l1':
            loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        elif args.mode_dis == 'JS':
            loss = js_divergence(d, t_d)
#         if (torch.isnan(loss).any()):
#             ipdb.set_trace()
        return loss

class Relation_Loss(nn.Module):
    def __init__(self, dist_weight=1.0, angle_weight=1.0):
        super(Relation_Loss, self).__init__()
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dist_weight = dist_weight
        self.angle_weight = angle_weight

    def forward(self, student, teacher, args):
        dis_loss = self.dist_criterion(student, teacher, args)
        angle_loss = self.angle_criterion(student, teacher, args)
        relational_loss = self.dist_weight * dis_loss + self.angle_weight * angle_loss
        #         import ipdb; ipdb.set_trace()
        return relational_loss

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    # 每四次epoch调整一下lr，将lr减半
    lr = init_lr * (decay_rate ** (epoch // lr_decay))  # *是乘法，**是乘方，/是浮点除法，//是整数除法，%是取余数

    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 返回改变了学习率的optimizer
    return optimizer

def get_clip_textembeddings(dataset_name):
    file_path = 'flcore/clipdata/prototypes_{}.pth'.format(dataset_name.lower())
    loaded_tensor = torch.load(file_path)
    return loaded_tensor