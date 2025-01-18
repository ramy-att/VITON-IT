from data.base_data_loader import BaseDataLoader
from torch.utils.data import DataLoader
from PIL import Image
import os

class PairedDataset:
    def __init__(self, root, transform=None):
        self.cloth_dir = os.path.join(root, 'cloth')
        self.image_dir = os.path.join(root, 'image')
        self.files = sorted(os.listdir(self.cloth_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cloth_path = os.path.join(self.cloth_dir, self.files[idx])
        image_path = os.path.join(self.image_dir, self.files[idx])
        cloth = Image.open(cloth_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            cloth = self.transform(cloth)
            image = self.transform(image)
        return {'cloth': cloth, 'image': image}

class CreateDataLoader(BaseDataLoader):
    def initialize(self, opt):
        self.dataset = PairedDataset(opt.dataroot, transform=None)  # Add your transforms here
        self.dataloader = DataLoader(self.dataset, batch_size=opt.batchSize, shuffle=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
