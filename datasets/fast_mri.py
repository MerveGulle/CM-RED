from torch.utils.data import Dataset
from pathlib import Path
import scipy.io as sio


class KneeMVU_MatDataset(Dataset):
    def __init__(self, mat_dir):
        self.mat_dir = Path.cwd() / mat_dir
        self.file_list = sorted(list(self.mat_dir.rglob("*.mat")))

        if len(self.file_list) == 0:
            raise ValueError(f"No .mat files found in {mat_dir}")
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        mat_path = self.file_list[idx]
        data = sio.loadmat(mat_path)

        # Image domian crop (PE, RO)
        kspace = data['kspace']         # (15, 320, 320), complex64 
        coils = data['coils']        # (15, 320, 320), complex64 
         
        file_name = (str(mat_path).split('/')[-1]).split('.')[0]
        if isinstance(file_name, tuple):
            file_name = file_name[0]  

        return kspace, coils, str(file_name)  