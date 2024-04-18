import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_layers = self.make_conv_layers(64, 17)
        self.conv_final = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.subtract = Subtract()

    def make_conv_layers(self, channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv_layers(out)
        out = self.conv_final(out)
        out = self.subtract(x, out)
        return out

class Subtract(nn.Module):
    def __init__(self):
        super(Subtract, self).__init__()

    def forward(self, x, y):
        return x - y
if __name__ == "__main__":
    # Example usage:
    input = torch.randn(64, 3, 40, 40) # Assuming input size is (batch_size, channels, height, width)
    model = DnCNN()
    output = model(input)
    print(output.shape)