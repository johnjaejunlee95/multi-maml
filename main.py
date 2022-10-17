import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import learn2learn as l2l
import torch.optim as optim
import argparse

from torchvision.models import resnet50, ResNet50_Weights
from resnet import Bottleneck, ResNet, ResNet50, ResNet152
from tqdm import tqdm
from triplet import TripletLoss
from embedding_network import EmbeddingNet
from torchviz import make_dot
from Miniimagenet import MiniImagenet as Mini
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from copy import deepcopy
from PIL import Image
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from learn2learn.data.utils import partition_task, InfiniteIterator, OnDeviceDataset
# from MiniImagenet import miniImageNetGenerator as MiniImagenet

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        
def main():
    
    torch.manual_seed(283)
    torch.cuda.manual_seed_all(283)
    np.random.seed(823)
    
    device = torch.device("cuda")
    # Embeddingnet = EmbeddingNet(args.drop_out, args.imgsz, args.imgc).to(device)
    Embeddingnet = ResNet152(2).to(device)
    tmp = filter(lambda x: x.requires_grad, Embeddingnet.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    
    print(num)
    
    train_set = Mini("/data01/jjlee_hdd/data", mode="train")
    val_set = Mini("/data01/jjlee_hdd/data", mode="validation")
    test_set = Mini("/data01/jjlee_hdd/data", mode="test")

    train_dataset = l2l.data.MetaDataset(train_set)
    val_dataset = l2l.data.MetaDataset(val_set)
    test_dataset = l2l.data.MetaDataset(test_set)

    train_transforms = [
        NWays(train_dataset, n=args.n_way),
        # KShots(train_dataset, k=(args.k_qry + args.k_spt)*2),
        KShots(train_dataset, k=args.k_spt*2),
        LoadData(train_dataset),
        # RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset)
    ]
    negative_train_transforms = [
        NWays(train_dataset, n=args.n_way),
        KShots(train_dataset, k=args.k_spt*2),
        LoadData(train_dataset),
        # RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset)
    ]
    val_transforms = [
        NWays(val_dataset, args.n_way),
        KShots(val_dataset, args.k_qry + args.k_spt),
        LoadData(val_dataset),
        # RemapLabels(val_dataset),
        ConsecutiveLabels(val_dataset)
    ]
    test_transforms = [
        NWays(test_dataset, args.n_way),
        KShots(test_dataset, args.k_spt * 2),
        LoadData(test_dataset),
        # RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset)
    ]
    
    anchor_train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=args.epochs*args.task_num)
    anchor_train_loader = DataLoader(anchor_train_tasks, batch_size = args.task_num, pin_memory=True, shuffle=True)

    #batch_size = args.task_num,
    negative_train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=negative_train_transforms, num_tasks=args.epochs*args.task_num)
    negative_train_loader = DataLoader(negative_train_tasks, batch_size = args.task_num, pin_memory=True, shuffle=True)
    
    # val_tasks = l2l.data.TaskDataset(val_dataset, task_transforms = val_transforms, num_tasks=600)
    # val_loader = DataLoader(val_tasks, pin_memory=True, shuffle = True)
    
    
    optimizer = optim.Adam(Embeddingnet.parameters(), lr=0.001)
    Embeddingnet.apply(init_weights)
    # criterion = torch.jit.script(TripletLoss())
    # criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.KLDivLoss(reduction= "batchmean", log_target=True ))
    criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance())
    
    
    Embeddingnet.train()
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        
        running_loss = []
        x, y = next(iter(anchor_train_loader))
        x_n, y_n = next(iter(negative_train_loader))
        for i in tqdm(range(args.task_num), desc="Batch Training"):  
            
            # x_spt_a, y_spt_a, x_qry_a, y_qry_a  = [], [], [], []
            # x_spt_p, y_spt_p, x_qry_p, y_qry_p = [], [], [], []
            # x_spt_n, y_spt_n, x_qry_n, y_qry_n = [], [], [], []
            
            (x_spt, y_spt), (x_qry, y_qry) = partition_task(x[i], y[i], shots=args.k_spt)
            x_spt, y_spt, x_qry, y_qry = (x_spt.to(device), 
                                            y_spt.to(device),
                                            x_qry.to(device), 
                                            y_qry.to(device))
                    
            
            (x_spt_n, y_spt_n), (x_qry_n, y_qry_n) = partition_task(x_n[i], y_n[i], shots=args.k_spt)
            x_spt_n, y_spt_n, x_qry_n, y_qry_n = (x_spt_n.to(device), 
                                                y_spt_n.to(device),
                                                x_qry_n.to(device), 
                                                y_qry_n.to(device))
            
            optimizer.zero_grad()
            anchor_out = Embeddingnet(x_spt)
            positive_out = Embeddingnet(x_qry)
            negative_out = Embeddingnet(x_spt_n)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, args.epochs, np.mean(running_loss)))
        torch.save(Embeddingnet, "../embedding_model.pt")
        # torch.save(Embeddingnet.state_dict(), "./embedding_model_state_dict.pt" )
    # torch.save(Embeddingnet, "../embedding_model2.pt")
    # torch.save(Embeddingnet.state_dict(), "./embedding_model_state_dict2.pt" )

    # with torch.no_grad():
    #     for img, _, _, label = 
            
            
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, help='epoch number', default=100)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--num_filters', type=int, help='size of filters of convblock', default=32)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--drop_out', type=list, help='drop out rate of ResNet Model', default= [0.3, 0.2, 0.2, 0.2])

    args = argparser.parse_args()

    main()