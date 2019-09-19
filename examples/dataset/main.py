import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data_augmentation import Rescale, RandomCrop

# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.landmarks_frame.iloc[idx, 1:]
        label = np.array([label])
        label = label.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

def CreateData():
    os.makedirs('data', exist_ok=True)
    w = open('data/label.csv', 'w')
    w.write('image_name,label_0,label_1')
    for i in range(10):
        label_0 = np.random.randint(10)
        label_1 = np.random.randint(10)
        w.write('\n' + str(i) + '.jpg,' + str(label_0) + ',' + str(label_1))
        img = np.random.randint(256, size=(300, 300, 3), dtype=np.uint8)
        io.imsave('data/' + str(i) + '.jpg', img)
    w.close()

def main():
    CreateData()

    transformed_dataset = FaceLandmarksDataset(csv_file='data/label.csv',
                                               root_dir='data/',
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())

if __name__ == '__main__':
    main()
