import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class FC(nn.Module):
    def __init__(self, in_ch, med_ch, out_ch):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, med_ch),
            nn.ReLU(inplace=True),
            nn.Linear(med_ch, out_ch)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv(1, 20)
        self.conv2 = Conv(20, 50)
        self.fc = FC(4*4*50, 500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 4*4*50)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
