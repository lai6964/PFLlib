import copy
import random
import time
import numpy as np
from flcore.clients.clientdfgc import clientDFGC
from flcore.servers.serverbase import Server
from threading import Thread
from collections import defaultdict
import sys
import torch


class FedDFGC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDFGC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]
        self.classifier = copy.deepcopy(args.model.head)
        self.virtual_representation_perclass = 200
        self.rs_test_acc2=[]
        self.rs_test_acc3=[]
        self.using_aggregate = args.using_aggregate
        self.using_glocla = args.using_glocla
        self.using_normal = args.using_normal

    def train(self):
        for i in range(self.global_rounds + 1):
            sys.stdout.flush()  # 强制刷新标准输出缓冲区
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if (i % self.eval_gap == 0) and (i>0):
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
                # self.evaluate_one_step()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            if self.using_aggregate:
                self.receive_models()
                self.aggregate_parameters()
                self.send_models()

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            # self.global_protos = proto_aggregation_withratio(self.uploaded_protos, self.uploaded_protos_num)
            self.send_protos()

            if self.using_glocla:
                self.train_classifier_G()
                self.send_classifer_models()


            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            print("\nBest accuracy.")
            # self.print_(max(self.rs_test_acc), max(
            #     self.rs_train_acc), min(self.rs_train_loss))
            if len(self.rs_test_acc):
                print(max(self.rs_test_acc))
                print(max(self.rs_test_acc2))
                print(max(self.rs_test_acc3))
            self.save_global_model()
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

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
        self.uploaded_protos_num = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            self.uploaded_protos_num.append(client.protos_num)

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_correct2, tot_correct3 = [],[]
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            ct2, ns2, auc2, ct3 = c.test_metrics2()
            tot_correct.append(ct * 1.0)
            tot_correct2.append(ct2 * 1.0)
            tot_correct3.append(ct3 * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns2)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_correct2, tot_correct3

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_acc2 = sum(stats[4])*1.0 / sum(stats[1])
        test_acc3 = sum(stats[5])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_test_acc2.append(test_acc2)
            self.rs_test_acc3.append(test_acc3)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test Accuracy2: {:.4f}".format(test_acc2))
        print("Averaged Test Accuracy3: {:.4f}".format(test_acc3))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))

    def generate_virtual_representation(self):
        inv_protos, spe_protos = {}, {}
        inv_vars, spe_vars = {}, {}
        for client in self.selected_clients:
            protos = client.local_protos
            protos_vars = client.local_protos_vars
            for key in protos.keys():
                if key in inv_protos.keys():
                    inv_protos[key].append(protos[key])
                    inv_vars[key].append(protos_vars[key])
                else:
                    inv_protos[key] = [protos[key]]
                    inv_vars[key] = [protos_vars[key]]

            # protos = client.special_protos
            # for key, proto in protos.items():
            #     if key in spe_protos.keys():
            #         spe_protos[key].extend(proto)
            #     else:
            #         spe_protos[key] = proto

        data, targets = [], []
        for key in inv_protos.keys():
            inv_proto = inv_protos[key]
            inv_var = inv_vars[key]
            # spe_proto = spe_protos[key]
            # d = len(inv_proto)
            for i in range(self.virtual_representation_perclass):
                if random.random()>0.5:
                    inv_lambda = get_random_lambda(len(inv_proto))
                    # spe_lambda = get_random_lambda(len(spe_proto))

                    tmp_inv = sum(w*t.cpu() for w,t in zip(inv_lambda, inv_proto))
                    tmp_inv = 0.5*self.global_protos[key][:len(tmp_inv)].cpu()+0.5+tmp_inv
                    # tmp_spe = sum(w*t.cpu() for w,t in zip(spe_lambda, spe_proto))
                    # tmpdata = tmp_inv#+tmp_spe
                else:
                    index = random.randint(0, len(inv_proto) - 1)
                    # tmp_inv = np.random.normal(inv_proto[index], inv_var[index])
                    tmp_inv = torch.normal(inv_proto[index], inv_var[index])

                data.append(tmp_inv.cpu())
                targets.append(key)
        # data = torch.cat(data)
        targets = torch.Tensor(targets)
        # print(data.shape, targets.shape)
        dataset = RepresentationDataset(data, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

    def train_classifier_G(self):
        dataloader = self.generate_virtual_representation()
        criterion = torch.nn.CrossEntropyLoss()
        self.classifier = copy.deepcopy(self.global_model.head)
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.learning_rate)
        for param in self.classifier.parameters():
            param.requires_grad = True
        # optimizer = optim.Adam(self.classifier.parameters(), lr=self.args.local_lr, weight_decay=1e-5)
        self.classifier.to(self.device)
        self.classifier.train()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            # print(x,y)
            logits = self.classifier(x)
            loss = criterion(logits, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.global_model.head = copy.deepcopy(self.classifier)

    def send_classifer_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_classifier_parameters(self.classifier)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def evaluate_one_step(self):
        acc2list, acc3list, auclist, numlist = [],[],[],[]
        for c in self.clients:
            acc2list.append(c.test_acc)
            acc3list.append(c.test_acc2)
            auclist.append(c.auc)
            numlist.append(c.num)
        test_acc2 = sum(acc2list) * 1.0 / sum(numlist)
        test_acc3 = sum(acc3list) * 1.0 / sum(numlist)

        self.rs_test_acc2.append(test_acc2)
        self.rs_test_acc3.append(test_acc3)

        print("Averaged Test Accuracy2: {:.4f}".format(test_acc2))
        print("Averaged Test Accuracy3: {:.4f}".format(test_acc3))


def proto_aggregation(local_protos_list):
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos[label].append(local_protos[label])

    for [label, proto_list] in agg_protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos[label] = proto / len(proto_list)
        else:
            agg_protos[label] = proto_list[0].data

    return agg_protos

def proto_aggregation_withratio(local_protos_list, local_protos_num_list):
    nums_list = []
    for idx in range(len(local_protos_num_list[0])):
        num = 0
        for protos_num in local_protos_num_list:
            num += protos_num[idx]
        nums_list.append(num)

    agg_protos = defaultdict(list)
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        protos_num = local_protos_num_list[idx]
        for label in local_protos.keys():
            proto = local_protos[label] * protos_num[label] / nums_list[label]
            agg_protos[label].append(proto)

    for [label, proto_list] in agg_protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos[label] = proto / len(proto_list)
        else:
            agg_protos[label] = proto_list[0].data

    return agg_protos


def get_random_lambda(num):
    alpha = np.ones(num)  # Dirichlet 分布的参数，所有元素为 1
    random_numbers = np.random.dirichlet(alpha)
    return random_numbers

class RepresentationDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)