import torch
import os
import numpy as np
import learn2learn as l2l
import torch.optim as optim
import argparse
# import faiss 
import torch.nn as nn
import time 
# import clustering

from torch.utils.data import Dataset, DataLoader, random_split
# from kmeans_pytorch import kmeans, kmeans_predict
from fast_pytorch_kmeans import KMeans
from torch.nn import functional as F
from triplet import TripletLoss
from resnet import Bottleneck, ResNet, ResNet50, ResNet101, ResNet152
from embedding_network import EmbeddingNet
from Miniimagenet import MiniImagenet as Mini
from copy import deepcopy
from PIL import Image
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from learn2learn.data.utils import partition_task, InfiniteIterator, OnDeviceDataset
# from MiniImagenet import miniImageNetGenerator as MiniImagenet


torch.manual_seed(283)
torch.cuda.manual_seed_all(283)
np.random.seed(823)

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

class Classifier(nn.Module):
    def __init__(self, model, class_num):
        super(Classifier, self).__init__()
        self.model = model
        self.class_num = class_num
        self.fc = torch.nn.Linear(self.model.size(1), self.class_num)
    
    def forward(self, x):
        x = self.fc(x)
        return x

class New_datasets(Dataset): 
  def __init__(self, data, label):
      
      self.data = data
      self.label = label

  def __len__(self): 
    return len(self.data)

  def __getitem__(self, idx): 
    x = self.data[idx]
    y = self.label[idx]
    return x, y

def main():
    
    device = torch.device("cuda")
    num_cluster = args.num_cluster
    # classifier = EmbeddingNet(args.drop_out, args.imgsz, args.imgc, num_cluster).to(device)
    # Embeddingnet = ResNet152(64).to(device)
    classifier = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device)
    model_weight = torch.load("/data01/jjlee_hdd/backbone_model/dino_vitbase16_pretrain.pth")
    classifier.load_state_dict(model_weight)
    tmp = filter(lambda x: x.requires_grad, classifier.parameters())
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
        KShots(train_dataset, k=args.k_spt*2),
        LoadData(train_dataset),
        # RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset)
    ]
    negative_train_transforms = [
        NWays(train_dataset, args.n_way),
        KShots(train_dataset, args.k_spt * 2),
        LoadData(train_dataset),
        # RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset)
    ]
    
    
    anchor_train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=args.task_epochs)
    anchor_train_loader = DataLoader(anchor_train_tasks, batch_size = args.task_num, pin_memory=True, shuffle=True)

    # negative_train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=negative_train_transforms, num_tasks=args.epochs*args.task_num)
    # negative_train_loader = DataLoader(negative_train_tasks, batch_size = args.task_num, pin_memory=True, shuffle=True)
    
    
    # val_train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=args.val_task_num)
    # val_train_loader = DataLoader(val_train_tasks, pin_memory=True, shuffle=True)
    
    # neg_val_train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=negative_train_transforms, num_tasks=args.val_task_num)
    # neg_val_train_loader = DataLoader(neg_val_train_tasks, pin_memory=True, shuffle=True)

    # val_tasks = l2l.data.TaskDataset(val_dataset, task_transforms = val_transforms, num_tasks=600)
    # val_loader = DataLoader(val_tasks, pin_memory=True, shuffle = True)
    
    train_result = []
    train_data = []
    train_label = [] 
    
    for epoch in range(args.task_epochs):
        
        x, y = next(iter(anchor_train_loader))
        
        (x_spt, y_spt), (x_qry, y_qry) = partition_task(x[0], y[0].to(torch.int64), shots=args.k_spt)
        x_spt, y_spt, x_qry, y_qry = (x_spt.to(device), 
                                        y_spt.to(device),
                                        x_qry.to(device), 
                                        y_qry.to(device))
            
        anchor_out = classifier(x_spt)
        anchor_out_mean = torch.mean(anchor_out, dim=0)
        # print(anchor_out_mean.size())
        train_result.append(np.array(anchor_out_mean.detach().cpu().numpy()))
        train_data.append(np.array(x_spt.detach().cpu().numpy()))
        # train_label.append(np.array(y_spt[0].detach().cpu().numpy()))
        print("Epoch: {0}/{1}".format(epoch+1, args.task_epochs))
    
    train_result = torch.from_numpy(np.array(train_result))
    train_data = torch.from_numpy(np.array(train_data))
    # train_label = torch.from_numpy(np.array(train_label))
    # cluster_ids_x, cluster_centers = kmeans(X=train_result, num_clusters=num_cluster, distance='euclidean', device=device, tol=1e-15)
    kmeans = KMeans(n_clusters=num_cluster, mode='euclidean', verbose=1)
    clustered_label = kmeans.fit_predict(train_result)
    

    overall_datasets = New_datasets(train_data, clustered_label)
    
    
    data_size= len(overall_datasets)
    training_size = int(data_size*0.8)
    test_size = int(data_size*0.2)
    
    training_datasets, test_datasets = random_split(overall_datasets, [training_size, test_size])
    
    overall_dataloader = DataLoader(overall_datasets, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(training_datasets, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=True)
    
    # classifier = Classifier(train_result, 5).to(device)
    optimizer = optim.NAdam(classifier.parameters(), lr=1e-5)
    classifier.apply(init_weights)
    loss = torch.nn.CrossEntropyLoss()
    
    
    classifier.train()
    for epoch in range(args.training_epochs):
        running_loss, correct, num = 0, 0, 0
        i = 0
        for batch, (xx, yy) in enumerate(train_dataloader):
            
            # xx, yy = data
            xx, yy = xx[0].to(device), yy[0].to(device)
            logits = torch.mean(classifier(xx), dim=0)
            loss_ = loss(logits, yy)
            
            optimizer.zero_grad()
            loss_.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10)
            optimizer.step()
            
            pred = F.softmax(logits, dim=0).argmax(dim=0)
            correct += torch.eq(pred, yy).sum().item()
            running_loss += loss_
            i += 1

        acc = correct / i
        loss_ = running_loss/ i
        print("Epoch: {0}/{1} - loss: {2:.4f}, acc: {3:.4f}".format(epoch+1, args.training_epochs, loss_, acc))
        # torch.save(classifier, "/data01/jjlee_hdd/backbone_model/embedding_ResNet.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'loss': loss_,
            },"/data01/jjlee_hdd/backbone_model/embedding_ResNet.pt")
        
        
    # checkpoint = torch.load("/data01/jjlee_hdd/backbone_model/embedding_ResNet.pt")
    # classifier.load_state_dict(checkpoint['model_state_dict'])
        
    classifier.eval()
    test_loss, test_correct, num = 0, 0, 0
    for batch, (xx, yy) in enumerate(test_dataloader):
        num += 1
        with torch.no_grad():
            xx, yy = xx[0].to(device), yy[0].to(device)
            logits_q = torch.mean(classifier(xx), dim=0)
            print(logits_q.size())
            test_loss += loss(logits_q, yy)
            # print(loss(logits_q, yy))
            # print(logits_q)
            pred_q = F.softmax(logits_q, dim=0).argmax(dim=0)
            test_correct += torch.eq(pred_q, yy).sum().item()
            # print(test_correct)
            # correct += (logits_q.argmax(0) == yy).type(torch.float).sum().item()
            
    print((test_loss/num), (test_correct/num) )
        
        
        
    
    
    
    
    
    
    
    
    
    
    # optimizer = optim.Adam(Embeddingnet.parameters(), lr=1e-3)
    # Embeddingnet.apply(init_weights)
    # # criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance())
    # # min_loss= 100
    # train_result = []
    # train_label = []
    # loss = torch.nn.CrossEntropyLoss()
    
    # for epoch in range(args.epochs):
        
        # x, y = next(iter(anchor_train_loader))
        # x_n, y_n = next(iter(negative_train_loader))
        
        # for i in (range(args.task_num)):  
        #     negative_loss_ = 0
        #     (x_spt, y_spt), (x_qry, y_qry) = partition_task(x[i], y[i].to(torch.int64), shots=args.k_spt)
        #     x_spt, y_spt, x_qry, y_qry = (x_spt.to(device), 
        #                                     y_spt.to(device),
        #                                     x_qry.to(device), 
        #                                     y_qry.to(device))
            
        #     (x_spt_n, y_spt_n), (x_qry_n, y_qry_n) = partition_task(x_n[i], y_n[i].to(torch.int64), shots=args.k_spt)
        #     x_spt_n, y_spt_n, x_qry_n, y_qry_n = (x_spt_n.to(device), 
        #                                         y_spt_n.to(device),
        #                                         x_qry_n.to(device), 
        #                                         y_qry_n.to(device))

                        
            # anchor_out = classifier(Embeddingnet(x_spt),5)
            # positive_out = classifier(Embeddingnet(x_qry), 5)
            # negative_out = classifier(Embeddingnet(x_spt_n),5)
                        
            # anchor_out_mean = torch.mean(anchor_out, dim=0)
            # train_result.append(np.array(anchor_out_mean.detach().cpu().numpy()))
            # positive_out_mean = torch.mean(positive_out, dim=0)
            # negative_out_mean = torch.mean(negative_out, dim=0)
            
            # criterion = TripletLoss(margin=2)

            # loss_ = criterion(anchor_out_mean, positive_out_mean, negative_out_mean)
        
            # running_loss += loss_
            
        # loss_q = running_loss / args.task_num
        # optimizer.zero_grad()
        # loss_q.backward()
        # optimizer.step()
        
        # print("Epoch: {0}/{1}".format(epoch+1, args.epochs, ))
        
        
        # if (epoch+1) % 50 == 0 or epoch == 0:
        #     Embeddingnet.eval()
        #     loss_val = 0
        #     for k in range(args.val_task_num):
        #         negative_val_loss_ = 0 
        #         x_val, y_val = next(iter(val_train_loader))
        #         x_val_n, y_val_n = next(iter(neg_val_train_loader))
                
        #         (x_spt_val, y_spt_val), (x_qry_val, y_qry_val) = partition_task(x_val[0], y_val[0].to(torch.int64), shots=args.k_spt)
        #         x_spt_val, y_spt_val, x_qry_val, y_qry_val = x_spt_val.to(device), y_spt_val.to(device), x_qry_val.to(device), y_qry_val.to(device)
                
        #         (x_spt_val_n, y_spt_val_n), (__, ___) = partition_task(x_val_n[0], y_val_n[0].to(torch.int64), shots=args.k_spt)
        #         x_spt_val_n, y_spt_val_n= x_spt_val_n.to(device), y_spt_val_n.to(device)
                
        #         # class_diff_val = len(list(set(y_spt_val.detach().cpu().numpy()) - set(y_spt_val_n.detach().cpu().numpy())))
                
        #         with torch.no_grad():
        #             anchor_val = classifier(Embeddingnet(x_spt_val), 64)
        #             positive_val = classifier(Embeddingnet(x_qry_val), 64)
        #             negative_val = classifier(Embeddingnet(x_spt_val_n), 64)
                    
        #             anchor_val_mean = torch.mean(anchor_val, dim=0)
        #             positive_val_mean = torch.mean(positive_val, dim=0)
        #             negative_val_mean = torch.mean(negative_val, dim=0)
                    
                   
        #             criterion2 = TripletLoss(margin=2)
        #             loss__ = criterion2(anchor_val_mean, positive_val_mean, negative_val_mean)
        #             loss_val += loss__

        #     loss_val = loss_val / args.val_task_num
        #     if min_loss > loss_val:
        #         min_loss = loss_val
        #         torch.save(Embeddingnet, "/data01/jjlee_hdd/backbone_model/embedding_ResNet.pt")
        #         print("Epoch: {}/{} - Loss: {:.4f} --> saved!!\n".format(epoch+1, args.epochs, min_loss))
        #     else:
        #         min_loss = min_loss
        #         print("Epoch: {}/{} - Loss: {:.4f} --> not saved!!\n".format(epoch+1, args.epochs, loss_val))
            
            
            
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--task_epochs', type=int, help='task epoch number', default=5000)
    argparser.add_argument('--training_epochs', type=int, help='training epoch number', default=5)
    argparser.add_argument('--num_cluster', type=int, help='num of clustering', default=5)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    # argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    # argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--num_filters', type=int, help='size of filters of convblock', default=32)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    # argparser.add_argument('--val_task_num', type=int, help='validation batch size, namely val_num', default=100)
    # argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--drop_out', type=list, help='drop out rate of ResNet Model', default= [0.3, 0.2, 0.2, 0.2])

    args = argparser.parse_args()

    main()