#model definition

import torch
from torch import nn
import torchvision
import copy

import lightly
from lightly.data import LightlyDataset
from lightly.data import MoCoCollateFunction, SimCLRCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from sklearn.preprocessing import normalize

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier


input_size = 32
resnet_features = 40 #100
moco_hidden_features = 30 #512
moco_projection_features = 20 #128

class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(resnet_features, moco_hidden_features, moco_projection_features)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key



resnet = torchvision.models.resnet18(num_classes = resnet_features)

backbone = resnet
model = MoCo(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

collate_fn = SimCLRCollateFunction(input_size=input_size,
                                   gaussian_blur=0.,
                                    )


cifar10_train = torchvision.datasets.CIFAR10("data/cifar10", download=True)

dataset_train = LightlyDataset.from_torch_dataset(cifar10_train)

# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")



dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)



test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(
    #    mean=lightly.data.collate.imagenet_normalize['mean'],
    #    std=lightly.data.collate.imagenet_normalize['std'],
    #)
])

cifar10_test = torchvision.datasets.CIFAR10("data/cifar10", download=True)

dataset_test = LightlyDataset.from_torch_dataset(cifar10_test ,transform=test_transforms)

# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")



dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)


criterion = NTXentLoss(memory_bank_size=4096)
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)





print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for (x_query, x_key), _, _ in dataloader_train:
        update_momentum(model.backbone, model.backbone_momentum, m=0.99)
        update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        query = model(x_query)
        key = model.forward_momentum(x_key)
        loss = criterion(query, key)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader_train)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")


pretrained_resnet_backbone = model.backbone

state_dict = {
    'resnet18_parameters': pretrained_resnet_backbone.state_dict()
}
torch.save(state_dict, 'models/pretrained_resnet_backbone_'+str(resnet_features)+'.pth')
"""

backbone_new = resnet

ckpt = torch.load('models/pretrained_resnet_backbone_'+str(resnet_features)+'.pth')
backbone_new.load_state_dict(ckpt['resnet18_parameters'])

model.backbone = backbone_new
"""

def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    labels = []
    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device)
            emb = model.backbone(img).flatten(start_dim=1).cpu()
            embeddings.append(emb)
            labels.extend(label)


    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, labels

model.eval()
embeddings, labels = generate_embeddings(model, dataloader_test)
labels = [l.item() for l in labels]



tabpfn_trainsize = 1000

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, 
                                                    train_size=tabpfn_trainsize, 
                                                    test_size = int(tabpfn_trainsize*0.2),
                                                    random_state=42, 
                                                    stratify=labels)



# N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
# When N_ensemble_configurations > #features * #classes, no further averaging is applied.

classifier = TabPFNClassifier(device='cuda:1', N_ensemble_configurations=32)

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))