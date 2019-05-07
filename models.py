import torch.nn as nn
import torch.nn.functional as F
from config import CAP_LEN, CHARACTERS

__all__ = ["Model1", "Model2", "Model3", "Model4"]

def get_convoluted_len(n, number_of_conv):
    for _ in range(number_of_conv):
        n = (n - 4) / 2
    return int(n)

class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.img_height = 60
        self.img_width = 160
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.fc1 = nn.Linear(self.img_height*self.img_width*64, 1024)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout1d = nn.Dropout(p=0.25)
        self.dropout2d = nn.Dropout2d(p=0.25)
        self.fc3 = nn.Linear(512, CAP_LEN * len(CHARACTERS))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2d(x)
        x = F.relu(self.conv3(x))
        x = self.dropout2d(x)

        x = x.view(-1, self.img_height*self.img_width*64)
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1d(x)
        x = self.fc3(x)
        return x


class Model3(nn.Module):    
    def __init__(self):
        super(Model3, self).__init__()
        self.img_height = 60
        self.img_width = 160

        self.convoluted_height = get_convoluted_len(self.img_height, 2)
        self.convoluted_width = get_convoluted_len(self.img_width, 2)
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(self.convoluted_height*self.convoluted_width*64, 512)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(512, 128)
        self.dropout1d = nn.Dropout(p=0.25)
        self.dropout2d = nn.Dropout2d(p=0.25)
        self.fc3 = nn.Linear(128, CAP_LEN * len(CHARACTERS))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2d(x)

        x = x.view(-1, self.convoluted_height*self.convoluted_width*64)
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1d(x)
        x = self.fc3(x)
        return x


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.img_height = 32
        self.img_width = 80

        self.convoluted_height = get_convoluted_len(self.img_height, 2)
        self.convoluted_width = get_convoluted_len(self.img_width, 2)
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(self.convoluted_width *
                             self.convoluted_height*50, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, CAP_LEN * len(CHARACTERS))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.convoluted_width*self.convoluted_height*50)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.fc3(x)
        return x


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.img_height = 32
        self.img_width = 80

        self.convoluted_height = get_convoluted_len(self.img_height, 2)
        self.convoluted_width = get_convoluted_len(self.img_width, 2)
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(self.convoluted_width *
                             self.convoluted_height*50, 500)
        self.fc2 = nn.Linear(500, CAP_LEN * len(CHARACTERS))

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.convoluted_width*self.convoluted_height*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
