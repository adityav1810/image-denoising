import torch
import numpy as np
import cv2
import torch
import torch.nn.functional as F




def get_batch_psnr_ssim(ground_truth_image_tensor,predicted_image_tensor,verbose = 0):
    batch_size = ground_truth_image_tensor.shape[0]
    avg_psnr = 0.0
    avg_ssim = 0.0
    ground_truth_image_tensor= ground_truth_image_tensor.detach().cpu().numpy()
    predicted_image_tensor= predicted_image_tensor.detach().cpu().numpy()

    for i in range(batch_size):
        psnr_value = psnr(ground_truth_image_tensor[i].transpose(1, 2, 0),predicted_image_tensor[i].transpose(1, 2, 0))
        ssim_value = ssim(ground_truth_image_tensor[i].transpose(1, 2, 0), predicted_image_tensor[i].transpose(1, 2, 0))
        
        avg_ssim +=ssim_value
        avg_psnr +=psnr_value
        
        if verbose: 
            print("PSNR:", psnr_value)
            print(" SSIM:", ssim_value)
    return avg_psnr/batch_size , avg_ssim/batch_size
    
def ssim(clean_image, noisy_image, window_size=11, sigma=1.5, K1=0.01, K2=0.03, L=255):
    '''
    Calculates SSIM of 2 images, takes in a numpy array
    '''
    # Convert images to float32
    clean_image = clean_image.astype(np.float32)
    noisy_image = noisy_image.astype(np.float32)

    # Define spatial Gaussian filter
    spatial_filter = cv2.getGaussianKernel(window_size, sigma)
    spatial_filter = np.outer(spatial_filter, spatial_filter.T)

    # Compute local means and variances
    mu_clean = cv2.filter2D(clean_image, -1, spatial_filter)
    mu_noisy = cv2.filter2D(noisy_image, -1, spatial_filter)
    mu_clean_sq = mu_clean ** 2
    mu_noisy_sq = mu_noisy ** 2
    mu_clean_noisy = mu_clean * mu_noisy
    sigma_clean_sq = cv2.filter2D(clean_image ** 2, -1, spatial_filter) - mu_clean_sq
    sigma_noisy_sq = cv2.filter2D(noisy_image ** 2, -1, spatial_filter) - mu_noisy_sq
    sigma_clean_noisy = cv2.filter2D(clean_image * noisy_image, -1, spatial_filter) - mu_clean_noisy

    # Compute SSIM components
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    num = (2 * mu_clean_noisy + c1) * (2 * sigma_clean_noisy + c2)
    den = (mu_clean_sq + mu_noisy_sq + c1) * (sigma_clean_sq + sigma_noisy_sq + c2)

    # Compute SSIM map
    ssim_map = num / den

    # Compute mean SSIM value
    ssim_value = np.mean(ssim_map)

    return ssim_value
def psnr(clean_image, noisy_image):
    '''
        Calculates PSNR of 2 images, takes in a numpy array
    '''
    mse = np.mean((clean_image - noisy_image) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr.item()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example run 
    random_tensor1 = torch.randn(64, 3, 40, 40).to(device)
    random_tensor2 = torch.randn(64, 3, 40, 40).to(device)
    
    # Verify the shape of the tensor
    print("Shape of the random tensor:", random_tensor1.shape)
    print("Shape of the random tensor:", random_tensor2.shape)
    psnr_res, ssim_res = get_batch_psnr_ssim(random_tensor1,random_tensor1)
    print("PSNR : ",psnr_res)
    print("SSIM : ",ssim_res)


