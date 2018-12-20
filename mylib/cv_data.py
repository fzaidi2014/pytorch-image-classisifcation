import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import *
class ImageDataset(Dataset):

    def __init__(self, csv_path,
                 transforms=None,
                 labels=False):
       
        self.labels = None
        self.transforms = None

        self.df = pd.read_csv(csv_path)
        
        self.ids = np.asarray(self.df.iloc[:, 0])
        
        self.images = np.asarray(self.df.iloc[:, 1])
        
        if labels:
            self.labels = np.asarray(self.df.iloc[:, 1])
        
        self.data_len = len(self.df.index)
        if transforms is not None:
            self.transforms = transforms
            
        #print(self.data_len)

    def __getitem__(self, index):
        
        image_name = self.images[index]
        id_ = self.ids[index]
        img_ = Image.open(image_name)
        
        if self.transforms is not None:
            img_ = self.transforms(img_)[:3,:,:]
        
        label = 0
        if self.labels is not None:
            label = self.labels[index]
        

        return (id_,img_,label)

    def __len__(self):
        return self.data_len