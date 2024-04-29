from os.path import join as pjoin, normpath, basename, dirname
from PIL import Image
import numpy as np
import torchvision.transforms as T

class MIDVHoloDataPhotoRSplit:
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    def __init__(self, input_dir, split_dir="", train=True, input_size=224, applytransform=True) -> None:
        self.labels_dict = {"fraud/photo_replacement":{}}
        self.shorttopath = {"photo_replacement":"fraud/photo_replacement"}
        self.fraud_names = [k for k in self.labels_dict if k != "origins"]
        self.input_dir = normpath(input_dir)
        self.videos = []
        self.labels_vid = []
        self.getFilesSplit(pjoin(self.input_dir, "fraud/photo_replacement"), split_dir, train)
        
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
    
    def getFilesSplit(self, input_dir, split_dir="", train=True):
        images = []
        labels = []
        general_type = basename(input_dir)
        if len(split_dir):
            with open(pjoin(split_dir, self.shorttopath[general_type], f"{'train' if train else 'test'}.txt")) as f:
                video_names = f.read().split("\n")
        else:
            with open(pjoin(input_dir, f"{general_type}.lst")) as f:
                video_names = f.read().split("\n")[:-1]
        for vn in video_names:
            name = general_type if general_type == "origins" else "fraud/"+general_type
            l = f"{name}/{dirname(vn)}"
            with open(pjoin(input_dir, vn)) as f:
                tmp_lst = [pjoin(l, v) for v in f.read().split("\n") if v != ""]
                images += tmp_lst
                labels += [l] * len(tmp_lst)
                self.labels_dict[name][l] = tmp_lst
                self.videos.append(tmp_lst)
        assert len(images) == len(labels), "images must be the same size as labels"
        return images, labels
    
    def getFraudNames(self):
        return ["photo_replacement"]
    
    def getVideos(self, idx):
        return self.videos[idx]
    
    def __getitem__(self, idx):
        for im_p in self.videos[idx]:
            im = Image.open(pjoin(self.input_dir, im_p))
            if self.applytransform:
                im_t = self.transform(im)
            else:
                im_t = np.asarray(im)
            yield im_t
    
    def isFraud(self, idx):
        return True

    def __len__(self) -> int:
        return len(self.videos)
    

class MIDVHoloDataPhotoRFullSplit:
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    def __init__(self, input_dir, split_dir="", train=True, input_size=224, applytransform=True) -> None:
        self.labels_dict = {"fraud/photo_replacement":{}}
        self.shorttopath = {"photo_replacement":"fraud/photo_replacement"}
        self.fraud_names = [k for k in self.labels_dict if k != "origins"]
        self.input_dir = normpath(input_dir)
        self.videos = []
        self.labels_vid = []
        self.getFilesSplit(pjoin(self.input_dir, "fraud/photo_replacement"), split_dir, train)
        
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
    
    def getFilesSplit(self, input_dir, split_dir="", train=True):
        images = []
        labels = []
        general_type = basename(input_dir)
        if len(split_dir):
            with open(pjoin(split_dir, self.shorttopath[general_type], f"{'train' if train else 'test'}.txt")) as f:
                video_names = f.read().split("\n")
        else:
            with open(pjoin(input_dir, f"{general_type}.lst")) as f:
                video_names = f.read().split("\n")[:-1]
        for vn in video_names:
            name = general_type if general_type == "origins" else "fraud/"+general_type
            l = f"{name}/{dirname(vn)}"
            with open(pjoin(input_dir, vn)) as f:
                tmp_lst = [pjoin(l, v) for v in f.read().split("\n") if v != ""]
                images += tmp_lst
                labels += [l] * len(tmp_lst)
                self.labels_dict[name][l] = tmp_lst
                self.videos.append(tmp_lst)
        assert len(images) == len(labels), "images must be the same size as labels"
        return images, labels
    
    def getFraudNames(self):
        return ["photo_replacement"]
    
    def getVideos(self, idx):
        return self.videos[idx]
    
    def __getitem__(self, idx):
        for im_p in self.videos[idx]:
            im = Image.open(pjoin(self.input_dir, im_p))
            if self.applytransform:
                im_t = self.transform(im)
            else:
                im_t = np.asarray(im)
            yield im_t
    
    def isFraud(self, idx):
        return True

    def __len__(self) -> int:
        return len(self.videos)