import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, drop_out, imgsz, imgc):
        super(EmbeddingNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(num_features=64, eps=2e-05),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(num_features=64, eps=2e-05),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(num_features=64, eps=2e-05))
        self.conv1_r = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(num_features=64))
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                                   nn.Dropout2d(p=drop_out[0]))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(128, eps=2e-05),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(128, eps=2e-05),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(128, eps=2e-05))
        self.conv2_r = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(128, eps=2e-05))
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                                   nn.Dropout2d(p=drop_out[1]))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(256, eps=2e-05),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(256, eps=2e-05),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(256, eps=2e-05))
        self.conv3_r = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(256, eps=2e-05))
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                                   nn.Dropout2d(p=drop_out[2]))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(512, eps=2e-05),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(512, eps=2e-05),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(512, eps=2e-05))
        self.conv4_r = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(512, eps=2e-05))
        self.pool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                                   nn.Dropout2d(p=drop_out[3]),
                                   nn.AvgPool2d(kernel_size=6))
        self.fc = nn.Linear(512, 2)


    def forward(self, x):
        # x = x.view(-1, 84, 84, 3)
        # x = x.permute(0, 3, 1, 2).contiguous()  # shape = (25, 3, 84, 84)
        out1 = self.pool1(F.relu(self.conv1(x) + self.conv1_r(x)))
        out2 = self.pool2(F.relu(self.conv2(out1) + self.conv2_r(out1)))
        out3 = self.pool3(F.relu(self.conv3(out2) + self.conv3_r(out2)))
        out4 = self.pool4(F.relu(self.conv4(out3) + self.conv4_r(out3)))
        h_t = out4.view(x.shape[0], -1)
        fc = self.fc(h_t)

        return fc