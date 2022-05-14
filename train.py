import os
import sys
import pathlib
import argparse

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

"""
TODOS
1. Write Data Pipeline for cifar10, imagenet
2. Implement at a very least the ResNet50 architecture
3.
"""

def load_imagenet_dataset(args):
    imgnt_path = args.imagenet_path

    # normalize
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    # consider the following: https://github.com/pytorch/vision/issues/39
    train_preprocess = transforms.Compose([transforms.RandomChoice([transforms.Resize(256),transforms.Resize(480)]),
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,])

    val_preprocess = transforms.Compose([transforms.Compose([transforms.Resize(256),
                                                             transforms.CenterCrop(224),
                                                             transforms.ToTensor(),
                                                             normalize,])])



    # this is the torch.data.utils.Dataset
    imgnt_train_data = torchvision.datasets.Imagenet(imgnt_path,split='train',transform=train_preprocess)
    imgnt_val_data = torchvision.dataset.Imagenet(imgnt_path,split='val',transform=val_preprocess)

    # wrap them in DataLoaders
    imgnt_train_data_loader = DataLoader(imgnt_train_data, batch_size=args.batch_size, shuffle=True)
    imgnt_val_data_loader = DataLoader(imgnt_val_data, batch_size=args.batch_size,shuffle=True)

    # TODO: look at https://pytorch.org/vision/stable/transforms.html and ResNet paper for how to setup this section
    return imgnt_train_data,imgnt_val_data


def load_cifar10_dataset(args):
    pass


def get_args():
    parser = argparse.ArgumentParser(description='script to train ResNets on Imagenet')
    parser.add_argument('--num_epochs',type=int,help='number of epochs to run')
    parser.add_argument('--dataset',type=str,help='dataset name , cifar10 or Imagenet')
    parser.add_argument('--imgnt_path',type=str,default='/data/petabyte/IMAGENET/Imagenet2012_for_torchvision',help='path to imagenet dataset')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--arch',type=str,help='choice of ResNet50, ResNet18, ResNet34, Resnet101, ResNet152')
    args = parser.parser_args()
    return args

def get_dataset(args)
    if args.dataset.lower() == 'cifar10':
        return load_cifar10_dataset(args)
    elif args.dataset.lower() == 'imagenet':
        # do imagenet loading
        return load_imagenet_dataset(args)

def main():
    args = get_args()

    # make sure to include the

    # write data pipeline for cifar
