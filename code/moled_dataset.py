import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import random

def get_full_filelist(base_dir='.', target_ext='') -> list:
    fname_list = []  
    for fname in os.listdir(base_dir):
        path = os.path.join(base_dir, fname)  
        if os.path.isfile(path):  
            _, fname_ext = os.path.splitext(fname) 
            if fname_ext == target_ext or target_ext == '':  
                fname_list.append(path)  
        elif os.path.isdir(path):  
            temp_list = get_full_filelist(path, target_ext) 
            fname_list = fname_list + temp_list 
        else:
            pass
    return fname_list 

class moled_dataset(Dataset):
    def __init__(self, data_path, name):
        self.name = name
        if self.name == 'source':
            self.file = sorted(glob.glob(os.path.join(data_path,"*.Charles")))
            self.file = self.file[0:8000]
            self.len = len(self.file)
            print('Source num:',self.len)
        if self.name == 'target':
            self.file = get_full_filelist(data_path)
            self.len = len(self.file)
            print('Target num:',self.len)

        distance_matrix1 = np.load('wasserstein_dist1.npy')
        self.stand_distance_matrix1 = (distance_matrix1-np.mean(distance_matrix1))/(np.std(distance_matrix1))
        distance_matrix2 = np.load('wasserstein_dist2.npy')
        self.stand_distance_matrix2 = (distance_matrix2-np.mean(distance_matrix2))/(np.std(distance_matrix2))

    def __getitem__(self, index):
        if self.name == 'source':
            charles = np.fromfile(self.file[index], dtype=np.float32)
            reshape_charles = charles.reshape(256,256,-1)
            real1, imag1 = reshape_charles[:,:,0], reshape_charles[:,:,1]
            real2, imag2 = reshape_charles[:,:,2], reshape_charles[:,:,3]
            mqmoled_data = reshape_charles[:,:,0:4]

        elif self.name == 'target':
            charles = np.fromfile(self.file[index], dtype=np.float32)
            reshape_charles = charles.reshape(256,256,-1)
            real1, imag1 = reshape_charles[:,:,0], reshape_charles[:,:,1]
            real2, imag2 = reshape_charles[:,:,2], reshape_charles[:,:,3]
            mqmoled_data = reshape_charles[:,:,0:4]

        i = random.randint(0, 255)
        j = random.randint(0, 255)
        r = random.choices([-1, 1])[0]
        A_spectrum = np.zeros((1,256,256), dtype=complex)
        A_spectrum[:, i, j] = 1
        A_basis = np.fft.ifft2(A_spectrum)
        Uij = A_basis[0]/ np.linalg.norm(A_basis, axis=(-2, -1))
        # if np.linalg.norm(A_basis, axis=(-2, -1)) * 30 <= 8:
        #     Uij = A_basis[0]*8 / np.linalg.norm(A_basis, axis=(-2, -1))
        # else:
        #     print("-------------->8-----------------")
        #     Uij = A_basis[0]*30

        complex1 = real1 + 1j * imag1
        complex2 = real2 + 1j * imag2
        new_complex1 = complex1 + r * Uij * self.stand_distance_matrix1[i,j]
        new_complex2 = complex2 + r * Uij * self.stand_distance_matrix2[i,j]

        new_real1 = np.real(new_complex1)
        new_imag1 = np.imag(new_complex1)
        new_real2 = np.real(new_complex2)
        new_imag2 = np.imag(new_complex2)

        disturb_mqmoled_data = np.stack([new_real1, new_imag1, new_real2, new_imag2], axis=-1)
        disturb_mqmoled_data_tensor = torch.from_numpy(disturb_mqmoled_data).type(torch.FloatTensor)
        disturb_mqmoled_data_tensor = disturb_mqmoled_data_tensor.permute(2,0,1)

        mqmoled_data_tensor = torch.from_numpy(mqmoled_data).type(torch.FloatTensor)
        mqmoled_data_tensor = mqmoled_data_tensor.permute(2,0,1)

        if self.name == 'source':
            t2 = reshape_charles[:,:,5:6]
            t2_tensor = torch.from_numpy(t2).type(torch.FloatTensor)
            t2_tensor = t2_tensor.permute(2,0,1)

            adc = reshape_charles[:,:,6:7]
            adc_tensor = torch.from_numpy(adc).type(torch.FloatTensor)
            adc_tensor = adc_tensor.permute(2,0,1)

            return mqmoled_data_tensor, disturb_mqmoled_data_tensor, t2_tensor, adc_tensor

        elif self.name == 'target':
            return mqmoled_data_tensor, disturb_mqmoled_data_tensor

    def __len__(self):
        return len(self.file)
    

class ampout_moled_dataset(Dataset):
    def __init__(self, root, name, transform=None):
        self.name = name
        if self.name == 'source':
            self.file = sorted(glob.glob(os.path.join(root,"*.Charles")))
            self.file = self.file[0:8000]
            self.len = len(self.file)
            print('Source num:',self.len)
        elif self.name == 'target':
            self.file = get_full_filelist(root)
            self.len = len(self.file)
            print('Target num:',self.len)
        self.transform = transform

    def __getitem__(self, index):
        if self.name == 'source':
            real_charles = np.fromfile(self.file[index], dtype=np.float32)
            sim_reshape_charles = real_charles.reshape(256,256,-1)
            sim_real1, sim_imag1, sim_real2, sim_imag2 = sim_reshape_charles[:,:,0], sim_reshape_charles[:,:,1], sim_reshape_charles[:,:,2], sim_reshape_charles[:,:,3]
            sim_complex1 = sim_real1 + 1j * sim_imag1
            sim_complex2 = sim_real2 + 1j * sim_imag2
            fft_sim_complex1 = np.fft.fft2(sim_complex1)
            fft_sim_complex2 = np.fft.fft2(sim_complex2)
            amp_sim_complex1, _ = np.abs(fft_sim_complex1), np.angle(fft_sim_complex1)
            amp_sim_complex2, _ = np.abs(fft_sim_complex2), np.angle(fft_sim_complex2)

        elif self.name == 'target':
            real_charles = np.fromfile(self.file[index], dtype=np.float32)
            sim_reshape_charles = real_charles.reshape(256,256,-1)
            sim_real1, sim_imag1, sim_real2, sim_imag2 = sim_reshape_charles[:,:,0], sim_reshape_charles[:,:,1], sim_reshape_charles[:,:,2], sim_reshape_charles[:,:,3]
            sim_complex1 = sim_real1 + 1j * sim_imag1
            sim_complex2 = sim_real2 + 1j * sim_imag2
            fft_sim_complex1 = np.fft.fft2(sim_complex1)
            fft_sim_complex2 = np.fft.fft2(sim_complex2)
            amp_sim_complex1, _ = np.abs(fft_sim_complex1), np.angle(fft_sim_complex1)
            amp_sim_complex2, _ = np.abs(fft_sim_complex2), np.angle(fft_sim_complex2)
            
        amp_sim_complex1 = torch.from_numpy(amp_sim_complex1).type(torch.FloatTensor)
        amp_sim_complex2 = torch.from_numpy(amp_sim_complex2).type(torch.FloatTensor)

        sample = {'amp1': amp_sim_complex1, 'amp2': amp_sim_complex2} 
        return sample

    def __len__(self):
        return len(self.file)