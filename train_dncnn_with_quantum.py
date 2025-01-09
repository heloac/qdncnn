import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
import matplotlib.pyplot as plt

SAVE_PATH = "/home/xc/QDNCNN/"
PREPROCESS = False

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch DnCNN with Quantum Preprocessing')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--preprocessed_data', default=os.path.join(SAVE_PATH, "q1_train_images.npy"), type=str, help='path of preprocessed train data')
parser.add_argument('--noisy_data', default=os.path.join(SAVE_PATH, "noisy_patches.npy"), type=str, help='path of noisy train data')
parser.add_argument('--clean_data', default='data/Train400', type=str, help='path of clean train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

save_dir = os.path.join('models', args.model + '_' + 'sigma' + str(sigma))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=5, use_bnorm=True, kernel_size=3):

        super(DnCNN, self).__init__()
        padding = 1
        layers = []
        self.kernel_size = kernel_size

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=self.kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=self.kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=self.kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x[:, :1, :, :]
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class SumSquaredError(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(SumSquaredError, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, reduction='sum') / 2

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(r"model_(\d+).pth", file_)
            if result:
                epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

class QuantumDenoisingDataset(Dataset):
    def __init__(self, clean_patches, noisy_patches, quantum_data, sigma):

        super(QuantumDenoisingDataset, self).__init__()
        self.clean_patches = clean_patches
        self.noisy_patches = noisy_patches
        self.quantum_data = quantum_data
        self.sigma = sigma
        assert len(self.clean_patches) == len(self.noisy_patches) == len(self.quantum_data), "Number of clean patches, noisy patches, and quantum preprocessed data are inconsistent"

    def __len__(self):
        return len(self.clean_patches)

    def __getitem__(self, idx):
        clean_image = self.clean_patches[idx]
        clean_image = clean_image.astype('float32') / 255.0
        clean_image = torch.from_numpy(clean_image.transpose((2, 0, 1)))

        noisy_image = self.noisy_patches[idx]
        noisy_image = noisy_image.astype('float32') / 255.0
        noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1)))

        quantum_image = self.quantum_data[idx]
        quantum_image = quantum_image.astype('float32')
        quantum_image = torch.from_numpy(quantum_image.transpose((2, 0, 1)))

        quantum_image_upsampled = torch.nn.functional.interpolate(quantum_image.unsqueeze(0), size=noisy_image.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

        input_tensor = torch.cat((noisy_image, quantum_image_upsampled), dim=0)

        return input_tensor, clean_image

if __name__ == '__main__':
    if PREPROCESS:
        import subprocess
        subprocess.run(["python", "quantum_preprocessing.py"])
        print("Quantum preprocessing completed.")

    print("Loading clean image patches...")
    clean_patches = dg.datagenerator(data_dir=args.clean_data, verbose=True)
    print(f"Loaded a total of {len(clean_patches)} clean image patches.")

    print("Loading noisy image patches...")
    noisy_patches = np.load(args.noisy_data)
    print(f"Loaded a total of {len(noisy_patches)} noisy image patches.")

    print("Loading quantum preprocessed images...")
    quantum_data = np.load(args.preprocessed_data)
    print(f"Shape of quantum preprocessed data: {quantum_data.shape}")

    dataset = QuantumDenoisingDataset(clean_patches, noisy_patches, quantum_data, sigma=args.sigma)
    DLoader = DataLoader(dataset=dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)

    print('===> Building model')
    model = DnCNN()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print(f'Resuming from epoch {initial_epoch:03d}')
        model.load_state_dict(torch.load(os.path.join(save_dir, f'model_{initial_epoch:03d}.pth')))
    model.train()

    criterion = SumSquaredError()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)

    for epoch in range(initial_epoch, n_epoch):
        scheduler.step(epoch)
        epoch_loss = 0
        start_time = time.time()

        for n_count, (batch_input, batch_clean) in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                batch_input = batch_input.cuda()
                batch_clean = batch_clean.cuda()

            output = model(batch_input)
            loss = criterion(output, batch_clean)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if n_count % 10 == 0:
                print(f'{epoch + 1:4d} {n_count:4d} / {len(DLoader):4d} loss = {loss.item() / batch_size:.4f}')

        elapsed_time = time.time() - start_time

        log(f'epoch = {epoch + 1:4d} , loss = {epoch_loss / len(DLoader):.4f} , time = {elapsed_time:.2f} s')
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / len(DLoader), elapsed_time)), fmt='%2.4f')

        torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch + 1:03d}.pth'))
