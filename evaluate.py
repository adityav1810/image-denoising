import torch
from metrics import get_batch_psnr_ssim

def evaluate_model(model, val_loader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()  # Set the model to evaluation mode
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            psnr_value, ssim_value = get_batch_psnr_ssim(targets,outputs)
            total_psnr += psnr_value
            total_ssim += ssim_value
            num_batches += 1
    
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    return avg_psnr,avg_ssim