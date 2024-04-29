from os.path import join as pjoin, normpath, basename, dirname
from PIL import Image
import numpy as np
import torchvision.transforms as T

class MIDVHoloDataSplit:
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    def __init__(self, input_dir, split_dir="", split_file="train.txt", input_size=224, applytransform=True) -> None:
        # self.input_dir = input_dir
        # self.transform = transform
        self.labels_dict = {"fraud/copy_without_holo":{}, "fraud/photo_holo_copy":{}, "fraud/pseudo_holo_copy":{}, "origins":{}}
        self.shorttopath = {"copy_without_holo":"fraud/copy_without_holo", "photo_holo_copy":"fraud/photo_holo_copy", "pseudo_holo_copy":"fraud/pseudo_holo_copy", "origins":"origins"}
        self.fraud_names = [k for k in self.labels_dict if k != "origins"]
        self.files = []
        self.labels = []
        self.input_dir = normpath(input_dir)
        self.videos = []
        self.labels_vid = []
        for l in self.labels_dict:
            files_tmp, labels_tmp = self.getFilesSplit(pjoin(self.input_dir, l), split_dir, split_file)
            self.files += files_tmp
            self.labels += labels_tmp

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

        # self.videos = sum(list(self.labels_dict.values()), [])
        # print(self.videos)
    
    def getFilesSplit(self, input_dir, split_dir="", split_file=""):
        images = []
        labels = []
        general_type = basename(input_dir)
        if len(split_dir):
            with open(pjoin(split_dir, self.shorttopath[general_type], split_file)) as f:
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
                self.labels_vid.append(l)
        assert len(images) == len(labels), "images must be the same size as labels"
        return images, labels
    
    def getFraudNames(self):
        return [k for k in self.labels_dict.keys() if k.startswith("fraud")]
    
    def getVideos(self, idx):
        return self.videos[idx]
    
    def __getitem__(self, idx):
        for im_p in self.videos[idx]:
            im = Image.open(pjoin(self.input_dir, im_p))
            if self.applytransform:
                im_t = self.transform(im)
            else:
                im_t = np.asarray(im)
            # print(type(im_t))
            yield im_t
    
    def isFraud(self, idx):
        return "fraud" in self.labels_vid[idx]

    def __len__(self) -> int:
        return len(self.videos)
    
class MIDVHoloDataFullSplit:
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    def __init__(self, input_dir, split_dir="", split_file="", input_size=224, applytransform=True) -> None:
        # self.input_dir = input_dir
        # self.transform = transform
        self.labels_dict = {"fraud/copy_without_holo":{}, "fraud/photo_holo_copy":{}, "fraud/pseudo_holo_copy":{}, "origins":{}}
        self.shorttopath = {"copy_without_holo":"fraud/copy_without_holo", "photo_holo_copy":"fraud/photo_holo_copy", "pseudo_holo_copy":"fraud/pseudo_holo_copy", "origins":"origins"}
        self.fraud_names = [k for k in self.labels_dict if k != "origins"]
        self.files = []
        self.labels = []
        self.input_dir = normpath(input_dir)
        self.videos = []
        self.labels_vid = []
        for l in self.labels_dict:
            files_tmp, labels_tmp = self.getFilesSplit(pjoin(self.input_dir, l), split_dir, split_file)
            self.files += files_tmp
            self.labels += labels_tmp

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

        # self.videos = sum(list(self.labels_dict.values()), [])
        # print(self.videos)
    
    def getFilesSplit(self, input_dir, split_dir="", split_file=""):
        images = []
        labels = []
        general_type = basename(input_dir)
        if len(split_dir):
            with open(pjoin(split_dir, self.shorttopath[general_type], split_file)) as f:
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
                self.labels_vid.append(l)
        assert len(images) == len(labels), "images must be the same size as labels"
        return images, labels
    
    def getFraudNames(self):
        return [k for k in self.labels_dict.keys() if k.startswith("fraud")]
    
    def getVideos(self, idx):
        return self.videos[idx]
    
    def __getitem__(self, idx):
        for im_p in self.videos[idx]:
            im = Image.open(pjoin(self.input_dir, im_p))
            if self.applytransform:
                im_t = self.transform(im)
            else:
                im_t = np.asarray(im)
            # print(type(im_t))
            yield im_t
    
    def isFraud(self, idx):
        return "fraud" in self.labels_vid[idx]

    def __len__(self) -> int:
        return len(self.videos)
    