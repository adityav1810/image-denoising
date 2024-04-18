import torch 
from torch import nn
import torch.nn.functional as F
class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding='same')
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding='same')

        
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, dilation=3, padding='same')
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, dilation=4, padding='same')

        self.concatconv1 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        self.conv4_1 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv4_3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        self.conv5_1 = nn.Conv2d(64, 4, kernel_size=3, padding='same')
        self.conv5_2 = nn.Conv2d(4, 64, kernel_size=3, padding='same')

    def forward(self, x):
        branch1 = torch.relu(self.conv1_2(torch.relu(self.conv1_1(x))))
        # print("Branch 1 : ", branch1.shape)
        branch2 = torch.relu(self.conv2_2(torch.relu(self.conv2_1(x))))
        # print("Branch 2 : ", branch2.shape)
        add1 = torch.cat([branch1, branch2],dim = 1)
        # print("Concat : ",add1.shape)
        add1 = torch.relu(self.concatconv1(add1))
        # print("Concat Conv: ",add1.shape)
        add1 = torch.add(add1,x)
        # print("Add1: ",add1.shape)
        z = torch.relu(self.conv3_1(add1))
        z = torch.relu(self.conv3_2(z))
        # print("Z: ",z.shape)

        add2 = torch.relu(torch.add(z,add1))
        # print("add2: ",add2.shape)

        z = torch.relu(self.conv4_1(add2))
        z = torch.relu(self.conv4_2(z))
        add_3 = torch.relu(torch.add(self.conv4_3(z),add2))
        # print("add3: ",add_3.shape)

        z = F.adaptive_avg_pool2d(add_3, (1, 1))
        # print("Global Average Pooling : ", z.shape)
        # Expand dimensions
        
        # print("Squeeze dims ", z.shape)
        z = torch.relu(self.conv5_1(z))
        # print("conv5_1 :",z.shape)
        z = torch.sigmoid(self.conv5_2(z))
        # print("conv5_2 :",z.shape)

        out = z* add_3
        # print("Multiply: ",out.shape)
        

        return out
class RIDNET(nn.Module):
    def __init__(self):
        super(RIDNET, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.eam1 = EAM()
        self.eam2 = EAM()
        self.eam3 = EAM()
        self.eam4 = EAM()
        self.conv_final = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.eam1(x)
        x = self.eam2(x)
        x = self.eam3(x)
        x = self.eam4(x)
        x = self.conv_final(x)
        return x
if __name__ == "__main__":
    # Example usage:
    input = torch.randn(64, 3, 40, 40) # Assuming input size is (batch_size, channels, height, width)
    model = RIDNET()
    output = model(input)
    print(output.shape)
