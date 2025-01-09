
import argparse
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import matplotlib.pyplot as plt

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='Directory of the test dataset')
    parser.add_argument('--set_names', default=['Set68'], nargs='+', help='List of dataset names')
    parser.add_argument('--sigma', default=15, type=int, help='Noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma15'), type=str, help='Directory of the model')
    parser.add_argument('--model_name', default='model_180.pth', type=str, help='Name of the model file')
    parser.add_argument('--result_dir', default='results', type=str, help='Directory to save the results')
    parser.add_argument('--quantum_data', default='q1_train_images.npy', type=str, help='Path to the quantum preprocessed data')
    parser.add_argument('--save_result', default=1, type=int, choices=[0, 1], help='Whether to save the denoised images: 1 to save, 0 to not save')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

# 保存结果
def save_result(result, path):
    path = path if '.' in path else path + '.png'
    imsave(path, result)


def save_comparison_plot(original, noisy, denoised, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("original")
    axes[0].axis('off')

    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title("noise")
    axes[1].axis('off')

    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title("denoised")
    axes[2].axis('off')


    plt.subplots_adjust(wspace=0.00)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=5, kernel_size=3):
        super(DnCNN, self).__init__()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, 1, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        y = x[:, :1, :, :]
        return y - self.dncnn(x)


if __name__ == '__main__':
    args = parse_args()


    log("Starting image denoising evaluation script")

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log("Using CUDA for inference")
    else:
        log("CUDA not available, using CPU")


    log("Loading model...")
    model = DnCNN()
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        log(f"Model file {model_path} does not exist. Please check the model path.")
        exit(1)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        log("Model loaded successfully.")
    except Exception as e:
        log(f"Error loading model: {e}")
        exit(1)
    model.to(device)
    model.eval()


    log("Loading quantum preprocessed data...")
    try:
        quantum_data = np.load(args.quantum_data)
        log(f"Loaded quantum preprocessed data with shape: {quantum_data.shape}")
    except Exception as e:
        log(f"Error loading quantum data: {e}")
        exit(1)


    os.makedirs(args.result_dir, exist_ok=True)
    log(f"Results will be saved to directory: {args.result_dir}")


    for set_cur in args.set_names:
        log(f"Processing test set: {set_cur}")
        test_dir = os.path.join(args.set_dir, set_cur)
        if not os.path.isdir(test_dir):
            log(f"Test set directory {test_dir} does not exist. Skipping.")
            continue

        save_dir = os.path.join(args.result_dir, f"{set_cur}_sigma{args.sigma}")
        os.makedirs(save_dir, exist_ok=True)
        log(f"Denoised results will be saved to: {save_dir}")

        psnrs, ssims, times = [], [], []

        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".bmp"))]
        num_images = len(image_files)
        if num_images == 0:
            log(f"No image files found in test set {set_cur}.")
            continue

        for idx, im_name in enumerate(image_files):
            im_path = os.path.join(test_dir, im_name)
            try:

                start_time = time.time()


                x = np.array(imread(im_path), dtype=np.float32) / 255.0
                if x.ndim == 3:
                    x = np.mean(x, axis=2)  # Convert to grayscale


                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)
                y = np.clip(y, 0, 1).astype(np.float32)


                y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()
                y_tensor = y_tensor.to(device)


                quantum_tensor = torch.from_numpy(quantum_data[idx].transpose((2, 0, 1))).unsqueeze(0).float().to(device)
                quantum_upsampled = torch.nn.functional.interpolate(
                    quantum_tensor, size=y_tensor.shape[2:], mode='bilinear', align_corners=False
                )


                input_tensor = torch.cat((y_tensor, quantum_upsampled), dim=1)


                with torch.no_grad():
                    output_tensor = model(input_tensor).cpu().squeeze().numpy()


                elapsed_time = time.time() - start_time
                times.append(elapsed_time)


                psnr = compare_psnr(x, output_tensor, data_range=1.0)
                ssim = compare_ssim(x, output_tensor, data_range=1.0)
                psnrs.append(psnr)
                ssims.append(ssim)


                log(f"Image: {im_name}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Processing Time: {elapsed_time:.4f} seconds")


                if args.save_result:
                    output_image = np.clip(output_tensor * 255, 0, 255).astype(np.uint8)
                    noisy_image = np.clip(y * 255, 0, 255).astype(np.uint8)
                    original_image = np.clip(x * 255, 0, 255).astype(np.uint8)


                    original_save_path = os.path.join(save_dir, f"{os.path.splitext(im_name)[0]}_original.png")
                    save_result(original_image, original_save_path)


                    denoised_save_path = os.path.join(save_dir, f"{os.path.splitext(im_name)[0]}_denoised.png")
                    save_result(output_image, denoised_save_path)


                    comparison_save_path = os.path.join(save_dir, f"{os.path.splitext(im_name)[0]}_comparison.png")
                    save_comparison_plot(
                        original_image,
                        noisy_image,
                        output_image,
                        comparison_save_path
                    )

            except Exception as e:
                log(f"Error processing image {im_name}: {e}")
                continue

        # Calculate average metrics
        if len(psnrs) == 0:
            log(f"No valid images in test set {set_cur}.")
            continue

        psnr_avg, ssim_avg = np.mean(psnrs), np.mean(ssims)
        time_avg, time_min, time_max = np.mean(times), np.min(times), np.max(times)

        log(f"Dataset: {set_cur}, Average PSNR: {psnr_avg:.2f} dB, Average SSIM: {ssim_avg:.4f}")
        log(f"Processing Time Statistics - Average: {time_avg:.4f} seconds, Min: {time_min:.4f} seconds, Max: {time_max:.4f} seconds\n")

        # Save all metrics to results.txt (if needed)
        if args.save_result:
            results_path = os.path.join(save_dir, 'results.txt')
            try:
                with open(results_path, 'w') as f:
                    f.write("Image\tPSNR(dB)\tSSIM\tTime(s)\n")
                    for im, psnr, ssim, t in zip(image_files, psnrs, ssims, times):
                        f.write(f"{im}\t{psnr:.2f}\t{ssim:.4f}\t{t:.4f}\n")
                    # Write average metrics
                    f.write(f"Average\t{psnr_avg:.2f}\t{ssim_avg:.4f}\t{time_avg:.4f}\n")
                log(f"Saved metrics to {results_path}")
            except Exception as e:
                log(f"Error saving metrics to {results_path}: {e}")

            # Generate and save processing time chart
            try:
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(times)), times, color='skyblue')
                plt.xlabel('Image Index')
                plt.ylabel('Processing Time (seconds)')
                plt.title(f'Processing Time Distribution for {set_cur}')
                plt.tight_layout()
                processing_times_path = os.path.join(save_dir, 'processing_times.png')
                plt.savefig(processing_times_path)
                plt.close()
                log(f"Saved processing time chart to {processing_times_path}")
            except Exception as e:
                log(f"Error generating processing time chart: {e}")
