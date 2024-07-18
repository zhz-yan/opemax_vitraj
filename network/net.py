import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv2D(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.2):
        super(Conv2D, self).__init__()
        self.in_size = in_size
        self.hidden_out = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, 16, 5, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU()
        )
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_out, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, out_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x