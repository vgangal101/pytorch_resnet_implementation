import os
import sys
import pathlib
import argparse

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from metrics_track import AverageMeter, ProgressMeter, accuracy, Summary
from model_arch2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from torch.optim.lr_scheduler import MultiStepLR
from torch.multiprocessing as mp
#from torchvision_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import matplotlib.pyplot as plt
import logging
import datetime


def plot_graphs(args,track_train_acc,track_val_top1_acc,track_val_top5_acc,track_val_loss,track_train_loss):

    accuracy = track_train_acc
    val_accuracy = track_val_top1_acc
    top5_acc = track_val_top5_acc
    plt.figure()
    plt.title("Epoch vs Accuracy")
    plt.plot(accuracy,label='training accuracy')
    plt.plot(val_accuracy,label='val_accuracy')
    plt.plot(top5_acc,label='top5 accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    viz_file = 'accuracy_graph_' + args.dataset.lower() + '_' + args.model.lower() + '_bs' + str(args.batch_size) + '_epochs' + str(args.num_epochs) + '.png'
    plt.savefig(viz_file)
    plt.show()

    plt.figure()
    plt.title("Epoch vs Loss")
    plt.plot(track_train_loss,label='training loss')
    plt.plot(track_val_loss,label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')

    viz_file2 = 'loss_graph_' + args.dataset.lower() + '_' + args.model.lower() + '_bs' + str(args.batch_size) + '_epochs' + str(args.num_epochs) + '.png'
    plt.savefig(viz_file2)
    plt.show()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.1.3'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def main_worker():
    pass


def main():
