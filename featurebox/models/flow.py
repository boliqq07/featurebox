import torch
from torch.nn import Module

from featurebox.featurizers.generator import MGEDataLoader


class BaseLearning:
    def __init__(self, model: Module, train_loader: MGEDataLoader, test_loader: MGEDataLoader = None, device="cpu",
                 opt=None, clf=False, loss_method=None):

        self.train_loader = train_loader
        self.test_loader = test_loader

        device = torch.device(device)
        self.train_loader.to_cuda(device)
        if self.test_loader is not None:
            self.test_loader.to_cuda(device)
        self.device = device
        self.model = model
        self.model.to(device)
        self.opt = opt
        self.clf = clf
        self.loss_method = loss_method
        self.train_batch_number = len(self.train_loader.loader)
        self.test_batch_number = len(self.test_loader.loader) if self.test_loader is not None else 0

    def method(self, opt=None, clf=False, loss_method=None, learning_rate=1e-3):
        """ "sum" is recommended, for batch training"""
        opt = opt if opt else self.opt
        if opt is None:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            # L2 regularization
        else:
            self.opt = opt(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        if loss_method is None:
            if clf is True:
                self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            elif clf == "multi_label":
                self.loss_fn = torch.nn.L1Loss(reduction='sum')
                # 主要是用来判定实际的输出与期望的输出的接近程度 MAE: 1/N |y_pred-y|
            else:
                self.loss_fn = torch.nn.MSELoss(reduction='sum')

        else:
            self.loss_fn = loss_method(reduction="sum")

    def run_train(self, epochs=30,  threshold=230):
        #############defination###########################

        if self.opt is None:
            self.method()

        self.model.to(self.device)
        self.model.train()

        train_loader = self.train_loader

        ##############tarin###########################
        for epochi in range(epochs):
            
            train_loader.reset()
            loss = 0
            for m, (batch_x, batch_y) in enumerate(train_loader):
                self.opt.zero_grad()
                y_pred = self.model(*batch_x)
                lossi = self.loss_fn(y_pred, batch_y)
                loss += lossi.item()
                lossi.backward()
                self.opt.step()
            if loss <=threshold:
                break
            print("Train {} epochi loss:".format(epochi), loss)  #

    def run_test(self, force=True):
        ##############test###########################
        self.model.eval()

        if self.test_loader is None:
            if force:
                print("There is no test data loader. try with train data")
                test_loader = self.train_loader
            else:
                raise TypeError("There is no test loader!")
        else:
            test_loader = self.test_loader

        loss = 0.0
        for m, (batch_x, batch_y) in enumerate(test_loader):

            y_pred = self.model(*batch_x)
            lossi = self.loss_fn(y_pred, batch_y)
            loss += lossi.item()

        print("Test loss", loss)  #
