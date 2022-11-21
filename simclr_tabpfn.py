import sys
import torch
import torchvision
import pytorch_lightning as pl
import numpy as np
import lightly
import argparse
import models

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="simclr, simsiam, twins, moco")
parser.add_argument("--epochs", type=int, default="800", help="number of training epochs (800 epochs take around 10h on a single V100)")
parser.add_argument("--batch_size", type=int, default="512", help="batch size")
parser.add_argument("--num_runs", type=int, default="1", help="number of runs")
parser.add_argument("--color_strength", type=float, default="0.5", help="color distortion strength")
parser.add_argument("--augs", type=str, default="default", help="augmentation combinations: default, color, a, ab, abc, abcd, abcde")
parser.add_argument("--data_folder", type=str, default="./CIFAR10/", help="cifar-10 dataset directory")
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--verbose', type=int, default=0, help='verbosity level (0, 1, 2)')
parser.add_argument('--dataset', type=str, default="mnist", help='dataset name')

args = parser.parse_args()





if args.dataset == 'cifar10':

    input_size = 32 

    ds_class = torchvision.datasets.CIFAR10
    ds_name = ds_class.__name__

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])
    ds_test = ds_class("data/"+ ds_name, train=False, download=True)
    ds_train_kNN = ds_class("data/"+ ds_name, train=False, download=True)
    ds_train_ssl = ds_class(root="data/"+ ds_name, train=True, download=True)

    dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(ds_train_ssl)
    dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(ds_train_kNN ,transform=test_transforms)
    dataset_test = lightly.data.LightlyDataset.from_torch_dataset(ds_test ,transform=test_transforms)


elif args.dataset == 'mnist':

    input_size = 28

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),

    ])

    ds_class = torchvision.datasets.MNIST
    ds_name = ds_class.__name__

    ds_test = ds_class("data/"+ ds_name, train=False, download=True)
    ds_train_kNN = ds_class("data/"+ ds_name, train=False, download=True)
    ds_train_ssl = ds_class(root="data/"+ ds_name, train=True, download=True)

    dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(ds_train_ssl, transform=transforms)
    dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(ds_train_kNN ,transform=test_transforms)
    dataset_test = lightly.data.LightlyDataset.from_torch_dataset(ds_test ,transform=test_transforms)

collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    min_scale=0.08,
    gaussian_blur=0.0,
    random_gray_scale=0.0,
    cj_prob=1.0,  # force to do color distortion
    cj_strength=args.color_strength,
    hf_prob=0.0,
)

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
    #collate_fn=collate_fn,
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

Model = models.SimCLRModel

# loop through configurations and train models
gpu_memory_usage = []
runs = []
for seed in range(args.num_runs):
    pl.seed_everything(seed, workers=True)
    model = Model(dataloader_train_kNN, args.epochs)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices = [0],
        gpus=int(torch.cuda.is_available()),
        #progress_bar_refresh_rate=args.fresh_rate,
        check_val_every_n_epoch=1,
        #val_check_interval = 1,
        deterministic=True
    )
    trainer.fit(
        model,
        train_dataloaders=dataloader_train_ssl,
        val_dataloaders=dataloader_test
    )
    gpu_memory_usage.append(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    runs.append(model.max_accuracy)

    # delete model and trainer + free up cuda memory
    del model
    del trainer
    torch.cuda.empty_cache()

result = np.asarray(runs)
mean = result.mean()
std = result.std()
gpu_usage = np.asarray(gpu_memory_usage).mean()
model = args.model + '_epoch_' + str(args.epochs) + '_batch_' + \
    str(args.batch_size) + '_augs_' + args.augs
if args.augs == 'color':
    model = model + '_strength_' + str(args.color_strength)

print(f'{model}: {100*mean:.2f} +- {100*std:.2f}%, GPU used: {gpu_usage / (1024.0**3):.1f} GByte', flush=True)
