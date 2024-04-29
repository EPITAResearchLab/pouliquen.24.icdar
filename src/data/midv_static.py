
from os.path import join as pjoin, splitext
from os import listdir
from PIL import Image
import numpy as np
import torchvision.transforms as T

class MIDVStatic:
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    def __init__(self, input_dir, input_size=224, applytransform=True) -> None:
        

        self.input_size = input_size
        self.transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(
                mean=self.IMAGENET_NORMALIZE["mean"],
                std=self.IMAGENET_NORMALIZE["std"],
            ),
        ])
        self.applytransform = applytransform

        self.videos = [[self.get_frames(pjoin(input_dir, d_type, v_id)) for v_id in listdir(pjoin(input_dir, d_type))] for d_type in listdir(input_dir)] #pjoin(input_dir, d_type, v_id): 
        self.videos = sum(self.videos, [])

    
    def get_frames(self, path):
        return [pjoin(path, x) for x in sorted(listdir(path), key=lambda x:int(splitext(x)[0][:-2]))]

    
    def getVideos(self, idx):
        return self.videos[idx]
    
    def __getitem__(self, idx):
        for i, im_p in enumerate(self.videos[idx]):
            if i % 2: # 10 fps and midv holo is at 5fps
                continue
            im = Image.open(im_p)
            if self.applytransform:
                im_t = self.transform(im)
            else:
                im_t = np.asarray(im)
            yield im_t
    
    def isFraud(self, idx):
        return True

    def __len__(self) -> int:
        return len(self.videos)
    

class MIDVStaticFull:
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    def __init__(self, input_dir, input_size=224, applytransform=True) -> None:
        self.input_size = input_size
        self.transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(
                mean=self.IMAGENET_NORMALIZE["mean"],
                std=self.IMAGENET_NORMALIZE["std"],
            ),
        ])
        self.applytransform = applytransform

        self.videos = [[self.get_frames(pjoin(input_dir, d_type, v_id)) for v_id in listdir(pjoin(input_dir, d_type))] for d_type in listdir(input_dir)] #pjoin(input_dir, d_type, v_id): 
        self.videos = sum(self.videos, [])

    
    def get_frames(self, path):
        return [pjoin(path, x) for x in sorted(listdir(path), key=lambda x:int(splitext(x)[0]))]

    
    def getVideos(self, idx):
        return self.videos[idx]
    
    def __getitem__(self, idx):
        for i, im_p in enumerate(self.videos[idx]):
            if i % 2: # 10 fps and midv holo is at 5fps
                continue
            im = Image.open(im_p)
            if self.applytransform:
                im_t = self.transform(im)
            else:
                im_t = np.asarray(im)
            yield im_t
    
    def isFraud(self, idx):
        return True

    def __len__(self) -> int:
        return len(self.videos)