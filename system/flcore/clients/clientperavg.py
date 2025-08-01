import numpy as np
import torch
import time
import copy
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client
import os

class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate

        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data(self.batch_size*2)
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):  # local update
            for X, Y in trainloader:
                temp_model = copy.deepcopy(list(self.model.parameters()))

                # step 1
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # step 2
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer.step(beta=self.beta)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_one_step(self):
        trainloader = self.load_train_data(self.batch_size)
        iter_loader = iter(trainloader)
        # self.model.to(self.device)
        self.model.train()

        (x, y) = next(iter_loader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        output = self.model(x)
        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.model.cpu()
        self.save_local_model()


    def train_metrics(self, model=None):
        trainloader = self.load_train_data(self.batch_size*2)
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        for X, Y in trainloader:
            # step 1
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][:self.batch_size].to(self.device)
                x[1] = X[1][:self.batch_size]
            else:
                x = X[:self.batch_size].to(self.device)
            y = Y[:self.batch_size].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            # step 2
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][self.batch_size:].to(self.device)
                x[1] = X[1][self.batch_size:]
            else:
                x = X[self.batch_size:].to(self.device)
            y = Y[self.batch_size:].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            loss1 = self.loss(output, y)

            train_num += y.shape[0]
            losses += loss1.item() * y.shape[0]

        return losses, train_num

    def train_one_epoch(self):
        trainloader = self.load_train_data(self.batch_size)
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            output = self.model(x)
            loss = self.loss(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.save_local_model()

    def save_local_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_client_{}.pt".format(self.id))
        torch.save(self.model, model_path)