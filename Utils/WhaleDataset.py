import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class WhaleDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = self.data['image'].tolist()
        self.labels = self.data['individual_id'].tolist()
        self.classes = self.data['individual_id'].unique()
        self.encode = {k: i for i,k in enumerate(self.classes)}

    def __len__(self):
        return len(self.data) 

    def __getitem__(self,idx):
        image_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(image_name).convert('RGB')
        label = self.encode[self.labels[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label