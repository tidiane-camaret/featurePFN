
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torch


# define the LightningModule
class LitTransformer(pl.LightningModule):
    def __init__(self,):
        super().__init__()


        tabpnf_classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32, no_preprocess_mode=True)
        self.tabpfn_transformer = tabpnf_classifier.model[2]


        self.encoder = nn.Linear(in_features=784,out_features=100)



    def training_step(self, batch, batch_idx):

        x, y = batch
        x = x.view(x.size(0), -1)
        x = self.encoder(x)

        y =  y.to(torch.float)
        y_hat = self.tabpfn_transformer((x,y))

        loss = nn.functional.cross_entropy(y_hat,y)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

littransformer = LitTransformer()

trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=littransformer, train_dataloaders=train_loader)