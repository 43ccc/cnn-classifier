import torch.nn as nn

# Squeeze and Excitation Layer
class SEL(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.lin1 = nn.Linear(channels, channels//reduction)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(channels//reduction, channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = y.permute(0, 2, 3, 1)
        y = self.relu(self.lin1(y))
        y = self.sig(self.lin2(y))
        y = y.permute(0, 3, 2, 1)

        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels*2)
        self.actv = nn.GLU(dim=1)

    def forward(self, x):
        return self.actv(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size, stride=1, padding=1),
            ConvBlock(in_channels, in_channels, kernel_size, stride=1, padding=1)
        )
        self.sel = SEL(channels=in_channels)

    def forward(self, x):
        return self.sel(self.model(x)) + x

class Classifier(nn.Module):
    def __init__(self, features_in=3, features_out=10):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels=features_in, out_channels=62, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=62, out_channels=126, kernel_size=3, stride=2, padding=1),
            ResBlock(in_channels=126, kernel_size=3),
            ResBlock(in_channels=126, kernel_size=3),
            ResBlock(in_channels=126, kernel_size=3),
            ResBlock(in_channels=126, kernel_size=3),
            ResBlock(in_channels=126, kernel_size=3),
            ResBlock(in_channels=126, kernel_size=3),
            ResBlock(in_channels=126, kernel_size=3),
            ConvBlock(in_channels=126, out_channels=255, kernel_size=3, stride=2, padding=1), 
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), 
            nn.Linear(255, features_out)
        )

    def forward(self, x):
        return self.model(x)
    
    def calc_loss(self, inputs, targets):
        loss_function = nn.CrossEntropyLoss()

        pred = self(inputs)
        loss = loss_function(pred, targets)

        num_correct = sum(pred.argmax(dim=1) == targets)

        return loss, num_correct