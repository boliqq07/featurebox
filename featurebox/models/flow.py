import torch
from torch.nn import Module
from torch.optim.lr_scheduler import MultiStepLR

from featurebox.featurizers.generator import MGEDataLoader
from featurebox.utils.general import AverageMeter


class BaseLearning:
    def __init__(self, model: Module, train_loader: MGEDataLoader, test_loader: MGEDataLoader, device="cpu",
                 optimizer=None, clf=False, loss_method=None, learning_rate=1e-3, milestones=None,
                 loss_threshold=230):

        self.train_loader = train_loader
        self.test_loader = test_loader

        device = torch.device(device)
        self.train_loader.to_cuda(device)
        if self.test_loader is not None:
            self.test_loader.to_cuda(device)
        self.device = device
        self.model = model
        self.model.to(device)
        self.clf = clf
        self.loss_method = loss_method
        self.milestones =milestones
        self.train_batch_number = len(self.train_loader.loader)
        self.test_batch_number = len(self.test_loader.loader) if self.test_loader is not None else 0

        if self.optimizer is None or self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        elif self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=0.01
                                             )
            # L2 regularization
        else:
            self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        if self.loss_method is None:
            if self.clf is True:
                self.loss_method = torch.nn.CrossEntropyLoss()
            elif self.clf == "multi_label":
                self.loss_method = torch.nn.L1Loss()
                # 主要是用来判定实际的输出与期望的输出的接近程度 MAE: 1/N |y_pred-y|
            else:
                self.loss_method = torch.nn.MSELoss()

        else:
            self.loss_method = loss_method()
        if self.milestones is None:
            self.milestones = [30, 50, 80]
        self.scheduler = MultiStepLR(self.optimizer, gamma=0.1, milestones=self.milestones)

        self.loss_threshold = loss_threshold

    def run(self,start_epoch=0, epoch=50, ):

        self.model.to(self.device)
        self.model.train()
        train_loader = self.train_loader
        for epochi in range(start_epoch,epoch):

            train_loader.reset()

            self._train(epochi)

            loss = self._validate()

            self.scheduler.step()

            if loss <= self.loss_threshold:
                break

    def _train(self, epochi):

        if self.clf is False:
            looses = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()

        self.train_loader.reset()
        loss = 0
        for m, (batch_x, batch_y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            y_pred = self.model(*batch_x)
            lossi = self.loss_method(y_pred, batch_y)

            loss += lossi.cpu().item()

            if self.clf is False:

                looses.update(loss.cpu(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))

            lossi.backward()
            self.optimizer.step()

        print("Train {} epochi loss:".format(epochi), loss)  #

    # def run_train(self, start_epoch=0, epochs=30, threshold=230):
    #     #############defination###########################
    #

    #
    #     self.model.to(self.device)
    #     self.model.train()
    #
    #     train_loader = self.train_loader
    #
    #     ##############tarin###########################
    #     for epochi in range(epochs):
    #
    #         train_loader.reset()
    #         loss = 0
    #         for m, (batch_x, batch_y) in enumerate(train_loader):
    #             self.opt.zero_grad()
    #             y_pred = self.model(*batch_x)
    #             lossi = self.loss_fn(y_pred, batch_y)
    #             loss += lossi.item()
    #             lossi.backward()
    #             self.opt.step()
    #         if loss <= threshold:
    #             break
    #         print("Train {} epochi loss:".format(epochi), loss)  #
    #
    # def run_test(self, force=True):
    #     ##############test###########################
    #     self.model.eval()
    #
    #     if self.test_loader is None:
    #         if force:
    #             print("There is no test data loader. try with train data")
    #             test_loader = self.train_loader
    #         else:
    #             raise TypeError("There is no test loader!")
    #     else:
    #         test_loader = self.test_loader
    #
    #     loss = 0.0
    #     for m, (batch_x, batch_y) in enumerate(test_loader):
    #         y_pred = self.model(*batch_x)
    #         lossi = self.loss_fn(y_pred, batch_y)
    #         loss += lossi.item()
    #
    #     print("Test loss", loss)  #
