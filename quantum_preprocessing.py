import os
import numpy as np
import torch
import pennylane as qml
from data_generator import datagenerator
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse

SAVE_PATH = "/home/xc/QDNCNN/"
PREPROCESS = True

def parse_args():
    parser = argparse.ArgumentParser(description="Add noise and preprocess images using quantum circuits.")
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    return parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def init_worker():

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

def process_patch(img):

    dev = qml.device("lightning.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(phi):
        qml.RY(np.pi * phi[0], wires=0)
        qml.RY(np.pi * phi[1], wires=1)
        qml.RY(np.pi * phi[2], wires=2)
        qml.RY(np.pi * phi[3], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        qml.RZ(-1 * phi[1], wires=1)
        qml.RZ(-1 * phi[3], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        return [qml.expval(qml.PauliZ(j)) for j in range(4)]

    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    out = np.zeros((14, 14, 4))

    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            try:
                phi = [
                    img[j, k, 0],
                    img[j, k + 1, 0],
                    img[j + 1, k, 0],
                    img[j + 1, k + 1, 0]
                ]
                q_results = circuit(phi)
                for c in range(4):
                    out[j // 2, k // 2, c] = q_results[c]
            except Exception as e:
                print(f"Error processing patch at ({j}, {k}): {e}")
                out[j // 2, k // 2, c] = 0
    return out

def add_noise(clean_patches, sigma):

    clean_patches = clean_patches.astype(np.float32) / 255.0
    noise = np.random.randn(*clean_patches.shape).astype(np.float32) * (sigma / 255.0)
    noisy_patches = clean_patches + noise
    noisy_patches = np.clip(noisy_patches, 0.0, 1.0)
    noisy_patches = (noisy_patches * 255.0).astype(np.uint8)
    return noisy_patches

def visualize_noisy_images(noisy_patches, num_images=4):

    indices = np.random.choice(len(noisy_patches), num_images, replace=False)
    selected_images = noisy_patches[indices]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[i].imshow(selected_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Noisy Image {indices[i] + 1}')
    plt.suptitle('Sample Noisy Image Patches', fontsize=16)
    plt.show()

def visualize_quantum_preprocessed_images(q1_train_images, num_images=4):

    indices = np.random.choice(len(q1_train_images), num_images, replace=False)
    selected_images = q1_train_images[indices]

    fig, axes = plt.subplots(4, num_images, figsize=(15, 15))
    for i in range(num_images):
        for c in range(4):
            axes[c, i].imshow(selected_images[i, :, :, c], cmap='gray')
            axes[c, i].axis('off')
            axes[c, i].set_title(f'Q{c + 1} Image {indices[i] + 1}')
    plt.suptitle('Sample Quantum Preprocessed Image Patches', fontsize=16)
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    sigma = args.sigma

    if PREPROCESS:
        print("Loading clean image patches...")
        clean_patches = datagenerator(data_dir='data/Train400', verbose=True)
        print(f"Loaded a total of {len(clean_patches)} clean image patches.")

        print(f"Adding Gaussian noise to clean image patches with noise level: {sigma}...")
        noisy_patches = add_noise(clean_patches, sigma=sigma)
        print("Gaussian noise added.")

        print("Visualizing 4 noisy image patches...")
        visualize_noisy_images(noisy_patches, num_images=4)

        print("Starting quantum preprocessing of training image patches:")
        start_time = time.time()

        num_workers = mp.cpu_count()
        print(f"Using {num_workers} processes for quantum preprocessing.")

        with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
            results = list(
                tqdm(pool.imap(process_patch, noisy_patches), total=len(noisy_patches), desc="Quantum Preprocessing Progress", unit="image")
            )

        q1_train_images = np.array(results)
        elapsed_time = time.time() - start_time
        print(f"\nQuantum preprocessing completed. Total time: {elapsed_time:.2f} seconds.")

        print("Saving noisy image patches and quantum preprocessed images...")
        np.save(os.path.join(SAVE_PATH, "noisy_patches.npy"), noisy_patches)
        np.save(os.path.join(SAVE_PATH, "q1_train_images.npy"), q1_train_images)
        print("Saving completed.")

        print("Visualizing 4 quantum preprocessed image patches...")
        visualize_quantum_preprocessed_images(q1_train_images, num_images=4)

    else:
        print("Loading preprocessed quantum images...")
        q1_train_images = np.load(os.path.join(SAVE_PATH, "q1_train_images.npy"))
        print("Preprocessed quantum images loaded.")

    print("Shape of q1_train_images:", q1_train_images.shape)
