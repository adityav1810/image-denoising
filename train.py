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


def train_model(model, train_loader,test_data_loader, num_epochs=30, lr=1e-3,device = 'cpu', save_path='best_model_ridnet.pth'):
    
    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="ridnet-imgdenoise",

    # track hyperparameters and run metadata
    config={
    "architecture": "dncnn",
    "epochs": num_epochs,
    }
    )
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    best_loss = float('inf')
    epoch_losses = []
    epoch_psnrs = []
    epoch_ssim = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            psnr_res, ssim_res = get_batch_psnr_ssim(targets,outputs)
            epoch_loss += loss.item()
            total_psnr += psnr_res
            total_ssim += ssim_res
            
        
        # Compute average epoch loss and PSNR
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)
        epoch_lr = scheduler.get_last_lr()[0]
        wandb.log({"Learning Rate": epoch_lr, "Training Loss": avg_epoch_loss, "Training PSNR": avg_psnr, "Training SSIM": avg_ssim})
        # Record epoch, loss, and PSNR
        # print(f"Epoch: {epoch+1}/{num_epochs},Learning Rate : {epoch_lr:.6f},Loss: {avg_epoch_loss:.6f},PSNR: {avg_psnr:.3f},SSIM: {avg_ssim:.3f}")
        epoch_losses.append(avg_epoch_loss)
        epoch_psnrs.append(avg_psnr)
        epoch_ssim.append(avg_ssim)
        learning_rates.append(epoch_lr)

        if (epoch + 1 )% 2 == 0 : 
            val_psnr , val_ssim = evaluate_model(model,test_data_loader)
            wandb.log({"Validation PSNR": val_psnr, "Validation SSIM": val_ssim})
            
        
        # Check if current model is the best so far
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at {save_path}")
        
        # Adjust learning rate
        scheduler.step()
    return model, epoch_losses, epoch_psnrs,epoch_ssim , learning_rates

