import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch.nn.functional as F
# load dependent functions from other files 
from ridnet import RIDNET
from dncnn import DnCNN
from metrics import get_batch_psnr_ssim
from evaluate import evaluate_model 
from train import train_model


#Get dat filepaths
train_files=['dataset/patches/train/'+filename for filename in os.listdir('dataset/patches/train')]
test_files=['dataset/patches/test/'+filename for filename in os.listdir('dataset/patches/test')]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 30
# Create a dataloader class for pytorch to ingest data into model
class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_decoded = Image.open(file_path).convert('RGB')
        image = transforms.ToTensor()(image_decoded)
        # add random noise in patches
        noise_level = np.random.choice([20,30,40])
        noise = torch.randn(image.size()) * noise_level / 255.
        noisy_image = torch.clamp(image + noise, 0., 1.)
        return noisy_image, image

# Create dataset instances
train_dataset = CustomDataset(train_files)
test_dataset = CustomDataset(test_files)

# Create DataLoader instances for batching
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = RIDNET().to(device)
trained_model ,epoch_losses, epoch_psnrs,epoch_ssim, learning_rates = train_model(model,
                                                                                  train_data_loader,
                                                                                  test_data_loader,
                                                                                  device = device,
                                                                                  num_epochs = NUM_EPOCHS ,save_path='ridnet-mseloss.pth')

