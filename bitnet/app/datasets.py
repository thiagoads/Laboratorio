import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features: list, 
                        targets: list, 
                        transform = None, 
                        target_transform = None,
                        dtype = torch.float):
        super().__init__()
        self.features = features
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype

    def __getitem__(self, idx):
        feature, target = self.features[idx], self.targets[idx]
        
        feature = torch.tensor(feature, 
                               dtype=self.dtype)
        
        target = torch.tensor(target)
        
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            target = self.target_transform(target)
        return feature, target
    
    def __len__(self):
        return len(self.targets)