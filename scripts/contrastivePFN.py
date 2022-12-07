from featurePFN.models import LeNetModel
from pytorch_lightning import LightningModule, Trainer
import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
from featurePFN.models import LeNetPFN
from pytorch_lightning.callbacks.progress import TQDMProgressBar


transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


ds_class = datasets.MNIST
ds_name = ds_class.__name__

ds = ds_class("data/"+ ds_name, train=True,  transform=transform)
target_classes = [0,1]
ds = torch.utils.data.Subset(ds, [i for i, t in enumerate(ds.targets) if t in target_classes])

train_loader = torch.utils.data.DataLoader(
    ds,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)



lenetpfn_model = LeNetPFN()

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)

# Train the model ⚡
trainer.fit(lenetpfn_model, train_loader)