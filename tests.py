import sys
import torch
import torchvision
import pytorch_lightning as pl
import numpy as np
import lightly
import argparse
import models
import PIL
import matplotlib.pyplot as plt


input_size = 32

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="simclr, simsiam, twins, moco")
parser.add_argument("--epochs", type=int, default="800", help="number of training epochs (800 epochs take around 10h on a single V100)")
parser.add_argument("--batch_size", type=int, default="4", help="batch size")
parser.add_argument("--num_runs", type=int, default="1", help="number of runs")
parser.add_argument("--color_strength", type=float, default="0.5", help="color distortion strength")
parser.add_argument("--augs", type=str, default="default", help="augmentation combinations: default, color, a, ab, abc, abcd, abcde")
parser.add_argument("--data_folder", type=str, default="./CIFAR10/", help="cifar-10 dataset directory")
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--verbose', type=int, default=0, help='verbosity level (0, 1, 2)')

args = parser.parse_args()

collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    min_scale=0.08,
    gaussian_blur=0.0,
    random_gray_scale=0.0,
    cj_prob=1.0,  # force to do color distortion
    cj_strength=args.color_strength,
    hf_prob=0.0,
)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #torchvision.transforms.Normalize(
    #    mean=lightly.data.collate.imagenet_normalize['mean'],
    #    std=lightly.data.collate.imagenet_normalize['std'],
    #)
])

ds_class = torchvision.datasets.MNIST
ds_name = ds_class.__name__

ds_test = ds_class("data/"+ ds_name, train=False, download=True)
ds_train_kNN = ds_class("data/"+ ds_name, train=False, download=True)
ds_train_ssl = ds_class(root="data/"+ ds_name, train=True, download=True)

dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(ds_train_ssl, transform=transforms)
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(ds_train_kNN ,transform=test_transforms)
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(ds_test ,transform=test_transforms)

dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train_ssl,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=args.num_workers
)

dataloader_train_kNN = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)

resnet = torchvision.models.resnet18(num_classes = 100)

print(lightly.data.collate.imagenet_normalize['mean'])



# print first element of the dataloader
for (x1, x2), y, z in dataloader_train_ssl:
    # display all pairs of the first batch
    fig, axs = plt.subplots(x1.shape[0], 2)
    for i in range(x1.shape[0]):
        axs[i, 0].imshow(x1[i].permute(1, 2, 0))
        axs[i, 1].imshow(x2[i].permute(1, 2, 0))
    plt.show()

    res = resnet(x1)
    print(res.shape)
    break


for x, y, z in dataloader_train_kNN:
    print(x.shape)
    # display all pairs of the first batch
    fig, axs = plt.subplots(x.shape[0], )
    for i in range(x.shape[0]):
        axs[i].imshow(x[i].permute(1, 2, 0))
    plt.show()

    res = resnet(x)
    print(res.shape)
    break


