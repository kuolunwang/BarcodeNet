#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
import copy
import wandb

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

# from dataset.dataset import RetinopathyDataset
from model import BarcodeNet

class Trainer():
    def __init__(self,args):
        self.args = args

        # wandb
        self.run = wandb.init(project="BarcodeNet", entity="kuolunwang")
        config = wandb.config

        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.epochs = args.epochs

        # data
        # trainset = RetinopathyDataset("train")
        # evaset = RetinopathyDataset("evaluate")
        
        # file name 
        self.file_name ="BarcodeNet_{0}_{1}_{2}".format(args.batch_size, args.epochs, args.learning_rate)

        # crate folder
        self.weight_path = os.path.join(args.save_folder,"weight")
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
            
        # dataloader
        self.trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4)
        self.evaloader = DataLoader(dataset=evaset, batch_size=args.batch_size, 
        shuffle=False, num_workers=4)

        # model 
        self.model = BarcodeNet(2, pretrained=args.pretrained)

        # show model parameter
        print(self.model)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 5e-4)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # using cuda
        if args.cuda: 
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            print("using cuda")
            
        # load model if need 
        if(args.load_model != None):
            artifact = self.run.use_artifact(args.load_model, type='model')
            artifact_dir = artifact.download()
            files = os.listdir(artifact_dir)
            for f in files:
                if ".pkl" in f:
                    file_name = f
            self.model.load_state_dict(torch.load(os.path.join(artifact_dir, file_name)))

        self.evaluate(self.testloader) if args.test else self.training()

    # training
    def training(self):
        
        best_model_weight = None
        best_accuracy = 0.0
        self.loss_record = []
        wandb.watch(self.model)
        for e in range(self.args.epochs):
            train_loss = 0.0
            tbar = tqdm(self.trainloader)
            self.model.train()
            for i, (data, label) in enumerate(tbar):
                data, label = Variable(data),Variable(label)

                # using cuda
                if self.args.cuda:
                    data, label = data.cuda(), label.cuda()

                prediction = self.model(data)
                loss = self.criterion(prediction, label.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                tbar.set_description('Train loss: {0:.6f}'.format(train_loss / (i + 1)))

            self.loss_record.append(train_loss / (i + 1))
            tr_ac = self.evaluate(self.trainloader)
            te_ac = self.evaluate(self.testloader)
            wandb.log({"loss":train_loss / (i + 1)})
            wandb.log({"Test accuracy": te_ac})
            wandb.log({"Train accuracy": tr_ac})
            self.record(e+1, train_loss / (i + 1), tr_ac, te_ac)

            if(te_ac > best_accuracy):
                best_accuracy = te_ac
                best_model_weight = copy.deepcopy(self.model.state_dict())

        self.plot_confusion_matrix(5)
        # save the best model 
        torch.save(best_model_weight, os.path.join(self.weight_path, self.file_name + '.pkl'))
        
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(self.weight_path, self.file_name + '.pkl'))
        self.run.log_artifact(artifact)
        self.run.join()

        # plot learning curve
        self.plt_lr_cur()

    # evaluation
    def evaluate(self, d):

        correct = 0.0
        total = 0.0
        tbar = tqdm(d)
        self.model.eval()
        for i, (data, label) in enumerate(tbar):
            data, label = Variable(data),Variable(label)
                            
            # using cuda
            if self.args.cuda:
                data, label = data.cuda(), label.cuda()
                
            with torch.no_grad():
                prediction = self.model(data)
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += np.sum(np.squeeze(pred.eq(label.data.view_as(pred))).cpu().numpy())
                total += data.size(0)

            text = "Train accuracy" if d == self.trainloader else "Test accuracy"
            tbar.set_description('{0}: {1:2.2f}% '.format(text, 100. * correct / total))

        return 100.0 * correct / total

    # plot learning curve 
    def plt_lr_cur(self):

        plt.figure(2)
        plt.title("learning curve", fontsize = 18)
        ep = [x for x in range(1, self.args.epochs + 1)]
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(ep, self.loss_record)
        plt.savefig(os.path.join(self.picture_path, self.file_name + '_learning_curve.png'))
        # plt.show()

    # record information
    def record(self, epochs, loss, tr_ac, te_ac):

        file_path = os.path.join(self.record_path, self.file_name + '.txt')
        with open(file_path, "a") as f:
            f.writelines("Epochs : {0}, train loss : {1:.6f}, train accurancy : {2:.2f}, test accurancy : {3:.2f}\n".format(epochs, loss, tr_ac, te_ac))

    def plot_confusion_matrix(self, num_class):

        matrix = np.zeros((num_class,num_class))
        self.model.eval()
        for i, (data, label) in enumerate(self.testloader):
            data, label = Variable(data),Variable(label)
                            
            # using cuda
            if self.args.cuda:
                data, label = data.cuda(), label.cuda()
                
            with torch.no_grad():
                prediction = self.model(data)
                pred = prediction.data.max(1, keepdim=True)[1]

                ground_truth = pred.cpu().numpy().flatten()
                actual = label.cpu().numpy().astype('int')

                for j in range(len(ground_truth)):
                    matrix[actual[j]][ground_truth[j]] += 1
                
        for i in range(num_class):
            matrix[i,:] /=  sum(matrix[i,:])

        plt.figure(1)
        plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Normalized confusion matrix")
        plt.colorbar()

        thresh = np.max(matrix) / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

        tick_marks = np.arange(num_class)
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(os.path.join(self.picture_path, self.file_name + '_confusion_matrix.png'))