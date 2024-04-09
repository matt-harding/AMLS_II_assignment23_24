import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

class WhaleDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file) # Pandas dataframe with training label data. Assumes csv file has columns image, species and individual_id
        self.image_dir = image_dir # currently ./Datasets/train_images
        self.image_names = self.data['image'].tolist()
        self.labels = self.data['individual_id'].tolist()
        self.classes = self.data['individual_id'].unique()
        self.encode = {k: i for i, k in enumerate(self.classes)} # dictionary that maps each unique individual ID to a numerical index for one-hot encoding

    '''
    Step 1: Converts PIL image to a PyTorch tensor, scaling ther pixel values to the range [0,1]
    Step 2: Normalizes the tensor by subtracting the mean and dividing by the standard deviation. The provided mean and standard deviation values are typical for ImageNet data,
    '''
    @property
    def transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.image_names[idx]) # constructs the full file path of the image
        image = Image.open(image_name).convert('RGB') # opens the image file using PIL's Image.open method and converts it to the RGB color mode
        label = self.encode[self.labels[idx]] # retrieved one-hot encoding for label value
        if self.transform:
            image = self.transform(image) # applied transformation on image
        return image, label