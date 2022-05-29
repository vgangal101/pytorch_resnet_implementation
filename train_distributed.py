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

    parser = argparse.ArgumentParser(description='script to train ResNets on Imagenet & cifar10(sanity check)')
    parser.add_argument('--num_epochs',type=int,default=90,help='number of epochs to run')
    parser.add_argument('--dataset',type=str,help='dataset name , cifar10 or Imagenet')
    parser.add_argument('--imgnt_path',type=str,default='/data/petabyte/IMAGENET/Imagenet2012_for_torchvision',help='path to imagenet dataset')
    parser.add_argument('--model',type=str,help='model architecture')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--lr',type=float,default=0.1,help='learning rate')
    parser.add_argument('--weight_decay',type=float,default=1e-4,help='weight decay to use')
    parser.add_argument('--momentum',type=float,default=0.9,help='momentum to use')

    parser.add_argument('--distributed',type=bool,default=True,help='whether to run distributed')
    parser.add_argument('--nodes', default=1, type=int)
    #parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int,help='number of gpus per node')
    parser.add_argument('--node-rank',type=int,default='0')

    args = parser.parse_args()



def main_worker():

    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    return



def main():
    args = get_args()

    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.nodes * args.gpus
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    else:
        # Simply call main_worker function
        gpu = torch.device('cuda:0')
        main_worker(gpu, 1, args)
