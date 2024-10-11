import torch
from torch import nn

class FirstModel(nn.Module):

    def __init__(self):
        super(FirstModel, self).__init__()

        self.conv1 = nn.Conv3d(11, 32, (1, 3, 3), padding=(0, 1, 1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d((1,2,2))
        self.conv2 = nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d((1,2,2))
        self.flatten1 = nn.Flatten(start_dim=2)

        self.lstm = nn.LSTM(910656, 50, batch_first = True)

        self.lin1 = nn.Linear(50, 64 * 40 * 40)
        self.relu3 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(64, 64, (3, 3))
        self.up1 = nn.Upsample(scale_factor=(2,2))
        self.relu4 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(64, 32, (3, 3))
        self.up2 = nn.Upsample(scale_factor=(2,2))
        self.relu5 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(32, 11, (3, 3))
        self.up3 = nn.Upsample(scale_factor=(3,3))
        self.relu6 = nn.ReLU()
        self.conv6 = nn.ConvTranspose2d(11, 11, (3, 3))
        self.up4 = nn.Upsample(scale_factor=(1,2))
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.swapaxes(x, 1, 2)
        x = self.flatten1(x)
        outputs, (x, c) = self.lstm(x)
        x = torch.squeeze(x,0)
        x = self.lin1(x)
        x = self.relu3(x)
        x = torch.reshape(x, (x.shape[0], 64, 40, 40))
        x = self.conv3(x)
        x = self.up1(x)

        x = self.relu4(x)
        x = self.conv4(x)
        x = self.up2(x)
        x = self.relu5(x)
        x = self.conv5(x)
        x = self.up3(x)
        x = self.relu6(x)
        x = self.conv6(x)
        x = self.up4(x)
        x = self.sig(x)
        x = x[:,:,:372,:614]

        return x
