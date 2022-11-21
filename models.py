import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightly
import torch.nn.functional as F
import torchvision
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import numpy as np

knn_k = 200
knn_t = 0.1
classes = 10
nb_features = 100

class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback
    At the end of every training epoch we create a feature bank by inferencing
    the backbone on the dataloader passed to the module. 
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the 
    feature_bank features from the train data.
    We can access the highest accuracy during a kNN prediction using the 
    max_accuracy attribute.
    """

    def __init__(self, dataloader_kNN, epochs):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.epochs = epochs

    def training_epoch_end(self, outputs):
        # print losses
        losses = [i['loss'].item() for i in outputs]
        loss_avg = sum(losses)/len(losses)
        print(f'Epoch {self.current_epoch+1}/{self.epochs}: train loss = {loss_avg:.2f}')

        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                if torch.cuda.is_available():
                    img = img.cuda()
                    target = target.cuda()
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(
            self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(
            self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            return feature, targets

    def validation_epoch_end(self, outputs):
        if outputs:
            print(zip(*outputs))
            feature, targets = outputs[0]
            print("feature shape", feature.shape)
            print("targets shape", targets.shape)

            pred_labels_knn = knn_predict(
                feature, self.feature_bank, self.targets_bank, classes, knn_k, knn_t)
            num = feature.size(0)
            top1_knn = (pred_labels_knn[:, 0] == targets).float().sum().item()

            pred_labels_pfn = pfn_predict(feature.cpu(), self.feature_bank.cpu(), self.targets_bank.cpu())

            top1_pfn = (pred_labels_pfn == targets.cpu()).float().sum().item()
            """
            total_num = 0
            total_top1_knn = 0.
            total_top1_pfn = 0.
            for (num, top1_knn, top1_pfn) in outputs:
                total_num += num
                total_top1_knn += top1_knn
                total_top1_pfn += top1_pfn
            """
            total_top1_knn = top1_knn
            total_top1_pfn = top1_pfn
            total_num = num
            acc_knn = float(total_top1_knn / total_num)
            acc_pfn = float(total_top1_pfn / total_num)
            if max(acc_knn, acc_pfn) > self.max_accuracy:
                self.max_accuracy = max(acc_knn, acc_pfn)
            print(f'Epoch {self.current_epoch+1}/{self.epochs}: KNN acc = {100*acc_knn:.2f}')
            print(f'Epoch {self.current_epoch+1}/{self.epochs}: PFN acc = {100*acc_pfn:.2f}')
            self.log('kNN_accuracy', acc_knn * 100.0, prog_bar=True)
            self.log('PFN_accuracy', acc_pfn * 100.0, prog_bar=True)


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        #resnet = lightly.models.ResNetGenerator('resnet-18', num_classes=nb_features)
        
        #self.backbone = torchvision.models.resnet18(num_classes = nb_features)
        self.backbone = LeNetModel()
        
        # create a simclr model based on ResNet
        print(self.backbone)
        self.resnet_simclr = \
            lightly.models.SimCLR(self.backbone, num_ftrs=nb_features, out_dim = nb_features)  # add a 2-layer projection head
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(x0, x1)
        # self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_simclr.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.epochs)
        return [optim], [scheduler]



# code from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
#
# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR

def knn_predict(feature, feature_bank, feature_labels, classes: int, knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features based on a feature bank

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: 

    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

    # we do a reweighting of the similarities 
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels

def pfn_predict(feature, feature_bank, feature_labels):

    print("feature shape: ", feature.shape)
    print("feature_bank shape: ", feature_bank.shape)
    print("feature_labels shape: ", feature_labels.shape)
    
    feature_bank = feature_bank.T

    #pick 1000 random features from the feature bank
    random_indices = np.random.choice(feature_bank.shape[0], 1000, replace=False)
    feature_bank = feature_bank[random_indices, :]
    feature_labels = feature_labels[random_indices]



    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=32)
    classifier.fit(feature_bank, feature_labels)
    pred_labels, pred_probs = classifier.predict(feature, return_winning_probability=True)

    return torch.as_tensor(pred_labels)



class LeNetModel(torch.nn.Module):
    def __init__(self, final_dim = 100 ):
        super(LeNetModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, final_dim)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(final_dim, final_dim)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y