import os
import sys
import pathlib
import argparse

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import transforms

from metrics_track import AverageMeter, ProgressMeter, accuracy, Summary
from model_arch_final import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from torch.optim.lr_scheduler import MultiStepLR
import torch.multiprocessing as mp
#from torchvision_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import matplotlib.pyplot as plt
import logging
import datetime




def get_args():
    parser = argparse.ArgumentParser(description='script to train ResNets on Imagenet & cifar10(sanity check)')
    parser.add_argument('--num_epochs',type=int,default=90,help='number of epochs to run')
    parser.add_argument('--dataset',type=str,help='dataset name , cifar10 or Imagenet')
    parser.add_argument('--imgnt_path',type=str,default='/data/petabyte/IMAGENET/Imagenet2012/',help='path to imagenet dataset')
    parser.add_argument('--model',type=str,help='model architecture')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--lr',type=float,default=0.1,help='learning rate')
    parser.add_argument('--weight_decay',type=float,default=1e-4,help='weight decay to use')
    parser.add_argument('--momentum',type=float,default=0.9,help='momentum to use')
    parser.add_argument('--num_nodes',type=int,default=1,help='number of nodes to use')
    parser.add_argument('--num_gpus',type=int,default=torch.cuda.device_count(),help='number of gpus on a node')
    parser.add_argument('--dist_url',default='tcp://127.0.0.1:50000', type=str, help='url used in distributed training')
    parser.add_argument('--dist_backend',default='nccl',type=str,help='distributed backend')
    #parser.add_argument('--num_nodes',default=1,type=int,help='number of compute nodes')
    #parser.add_argument('--num_gpus',default=torch.cuda.device_count(),help='number of gpus available')


    args = parser.parse_args()
    return args


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


def process_vals(x):
    if isinstance(x,torch.Tensor):
        return x.cpu().detach().item()
    else:
        return x


def train(model,train_dataset,optimizer,scheduler,loss_function):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    for i,(images, target) in enumerate(train_dataset):

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output,target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg

def validate(model,val_dataset,loss_function):

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_dataset),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_dataset):

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss = loss_function(outputs,labels)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(),images.size(0))
            top1.update(acc1[0],images.size(0))
            top5.update(acc5[0],images.size(0))

    return losses.avg, top1.avg, top5.avg




def load_imagenet_dataset(rank, args):
    #imgnt_path = args.imagenet_path

    traindir = os.path.join(args.imgnt_path, 'train')
    valdir = os.path.join(args.imgnt_path, 'val')

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
    #imgnt_train_data = torchvision.datasets.Imagenet(imgnt_path,split='train',transform=train_preprocess)
    #imgnt_val_data = torchvision.dataset.Imagenet(imgnt_path,split='val',transform=val_preprocess)
    train_dataset = datasets.ImageFolder(traindir, train_preprocess)
    val_dataset = datasets.ImageFolder(valdir, val_preprocess)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    bs = int(args.batch_size/(args.num_gpus * args.num_nodes))
    print('bs=',bs)


    # wrap them in DataLoaders
    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False,num_workers=5,pin_memory=True,sampler=train_sampler)
    val_data_loader = DataLoader(val_dataset, batch_size=bs,shuffle=False,num_workers=5,pin_memory=True)

    # TODO: look at https://pytorch.org/vision/stable/transforms.html and ResNet paper for how to setup this section
    return train_data_loader, val_data_loader, train_sampler



def load_cifar10_dataset(rank,args):
    # checkout https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    cifar_train_data = torchvision.datasets.CIFAR10(f'./cifar10_dataset',transform=transform_train,download=True)
    cifar_val_data = torchvision.datasets.CIFAR10(f'./cifar10_dataset',transform=transform_val,download=True)


    train_sampler = torch.utils.data.distributed.DistributedSampler(cifar_train_data)

    bs = int(args.batch_size/(args.num_gpus * args.num_nodes))

    # wrap them in DataLoaders
    train_data_loader = DataLoader(cifar_train_data, batch_size=bs, shuffle=False,num_workers=5,pin_memory=True,sampler=train_sampler)
    val_data_loader = DataLoader(cifar_val_data, batch_size=bs,shuffle=False,num_workers=5,pin_memory=True)

    cifar_train_dl = DataLoader(cifar_train_data,batch_size=args.batch_size,shuffle=True)
    cifar_val_dl = DataLoader(cifar_val_data,batch_size=args.batch_size,shuffle=True)

    return cifar_train_dl, cifar_val_dl, train_sampler





def get_dataset(rank,args):
    if args.dataset.lower() == 'cifar10':
        return load_cifar10_dataset(rank,args)
    elif args.dataset.lower() == 'imagenet':
        return load_imagenet_dataset(rank,args)
    else:
        raise ValueError("Invalid dataset dataset=",args.dataset)




def train_model(rank,world_size,args):

    print('rank is=',rank)
    print(f'Using GPU: {rank} for training')
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size= world_size, rank=rank)

    if rank == 0:
        filename = f'train_log--model={args.model},dataset={args.dataset},date={datetime.datetime.now()}'
        logging.basicConfig(level=logging.INFO,filename=filename)

    num_classes = None

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'imagenet':
        num_classes = 1000

    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'ResNet34':
        model = ResNet34(num_classes=num_classes)
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes)
    elif args.model == 'ResNet101':
        model = ResNet101(num_classes=num_classes)
    elif args.model == 'ResNet152':
        model == ResNet152(num_classes=num_classes)
    else:
        raise ValueError('Did not receive a valid model type, recieved=',args.model)

    # All the necessary distributed code

    # set the gpu device number
    torch.cuda.set_device(rank)

    # move the model to the gpu
    model.cuda(rank)

    # the batch size to use , total batch size div by workers
    batch_size = int(args.batch_size / world_size)

    # wrap model with DistributedDataParallel and use only current rank
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_function = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer,milestones=[30,60,80],gamma=0.1)

    train_dataset, val_dataset, train_sampler = get_dataset(rank,args)

    track_train_loss = []
    track_train_acc = []
    track_val_loss = []
    track_val_top1_acc = []
    track_val_top5_acc = []

    num_epochs = args.num_epochs

    # write training loop
    print('starting training')
    for epoch in range(num_epochs):

        #train_sampler.set_epoch(epoch)

        train_loss, train_acc = train(model,train_dataset,optimizer,scheduler,loss_function)
        val_loss, val_top1_acc, val_top5_acc = validate(model,val_dataset,loss_function)

        track_train_loss.append(process_vals(train_loss))
        track_train_acc.append(process_vals(train_acc))
        track_val_loss.append(process_vals(val_loss))
        track_val_top1_acc.append(process_vals(val_top1_acc))
        track_val_top5_acc.append(process_vals(val_top5_acc))

        scheduler.step()
        print(f'epoch={epoch} train_loss={train_loss} train_acc={train_acc} val_loss={val_loss} val_top1_acc={val_top1_acc} val_top5_acc={val_top5_acc}')

        if rank == 0:
            logging.info(f'epoch={epoch} train_loss={train_loss} train_acc={train_acc} val_loss={val_loss} val_top1_acc={val_top1_acc} val_top5_acc={val_top5_acc}')

    print("training complete")

    if rank == 0:
        # display graphs
        plot_graphs(args,track_train_acc,track_val_top1_acc,track_val_top5_acc,track_val_loss,track_train_loss)

        # store checkpoint
        save_file = args.model + '_' + args.dataset + '.pth'
        torch.save(model,save_file)


def main():

    args = get_args()
    print('num nodes present=',args.num_nodes)
    print('number of gpus on node=',args.num_gpus)

    mp.spawn(train_model, nprocs= args.num_nodes * args.num_gpus, args=(args.num_nodes * args.num_gpus, args))


if __name__ == '__main__':
    main()
