import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from moled_dataset import ampout_moled_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_dataset = ampout_moled_dataset(root='Your Target Data Path',name='target')
target_loader = DataLoader(target_dataset, batch_size=64, num_workers=1)
source_dataset = ampout_moled_dataset(root='Your Source Data Path',name='source')
source_loader = DataLoader(source_dataset, batch_size=64, num_workers=1)

distance_matrix1 = np.zeros((256, 256))
distance_matrix2 = np.zeros((256, 256))

source_data = []
target_data = []
for images in tqdm(source_loader):
    amp1 = images['amp1'].to(device)
    amp2 = images['amp2'].to(device)
    source_data.append((amp1, amp2))

for images in tqdm(target_loader):
    amp1 = images['amp1'].to(device)
    amp2 = images['amp2'].to(device)
    target_data.append((amp1, amp2))

for i in tqdm(range(256)):
    for j in tqdm(range(256)):
        source_amp1 = []
        target_amp1 = []
        source_amp2 = []
        target_amp2 = []
        for amp1, amp2 in source_data:
            pixel_value = amp1[:, i, j]
            source_amp1.append(pixel_value)
            pixel_value = amp2[:, i, j]
            source_amp2.append(pixel_value)
        source_amp1 = torch.cat(source_amp1, axis=0)
        source_amp2 = torch.cat(source_amp2, axis=0)

        for amp1, amp2 in target_data:
            pixel_value = amp1[:, i, j]
            target_amp1.append(pixel_value)
            pixel_value = amp2[:, i, j]
            target_amp2.append(pixel_value)
        target_amp1 = torch.cat(target_amp1, axis=0)
        target_amp2 = torch.cat(target_amp2, axis=0)

        source_amp1 = source_amp1.view(-1).cpu().numpy()
        target_amp1 = target_amp1.view(-1).cpu().numpy()
        source_amp2 = source_amp2.view(-1).cpu().numpy()
        target_amp2 = target_amp2.view(-1).cpu().numpy()

        wasserstein_dist1 = wasserstein_distance(source_amp1, target_amp1)
        distance_matrix1[i, j] = wasserstein_dist1
        wasserstein_dist2 = wasserstein_distance(source_amp2, target_amp2)
        distance_matrix2[i, j] = wasserstein_dist2

np.save('wasserstein_dist1.npy', distance_matrix1)
np.save('wasserstein_dist2.npy', distance_matrix2)

