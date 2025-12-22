import cv2
import numpy as np
import os

def check_reconstruction():
    input_path = "/home/anar/own_projects/antigravity_prs/distribution_functions/GBR/input.png"
    output_path = "/home/anar/own_projects/antigravity_prs/distribution_functions/GBR/output.png"
    
    if not os.path.exists(output_path):
        print("Output image not found!")
        return
        
    img1 = cv2.imread(input_path).astype(np.float32)
    img2 = cv2.imread(output_path).astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    if psnr > 40:
        print("Success: Reconstruction is sharp and mathematically consistent!")
    else:
        print("Warning: Reconstruction is still inconsistent or blurred.")

if __name__ == "__main__":
    check_reconstruction()
