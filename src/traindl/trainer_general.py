import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import torch.nn as nn
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
from PIL import Image
from os.path import join as pjoin 
import os
import random

class MIDVHoloDataset:
    IMAGES_TRANSFORM = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
    def __init__(self, input_dir, transform, split_dir="", split_file="train.txt", only_label=None, flip_rot=True) -> None:
        # self.input_dir = input_dir
        self.transform = transform
        self.labels_dict = {"fraud/copy_without_holo":{}, "fraud/photo_holo_copy":{}, "fraud/pseudo_holo_copy":{}, "origins":{}}
        self.shorttopath = {"copy_without_holo":"fraud/copy_without_holo", "photo_holo_copy":"fraud/photo_holo_copy", "pseudo_holo_copy":"fraud/pseudo_holo_copy", "origins":"origins"}
        self.fraud_names = [k for k in self.labels_dict if k != "origins"]
        self.files = []
        self.labels = []
        self.input_dir = os.path.normpath(input_dir)
        self.only_label = only_label
        for l in self.labels_dict:
            files_tmp, labels_tmp = self.getFilesSplit(pjoin(self.input_dir, l), split_dir, split_file)
            self.files += files_tmp
            self.labels += labels_tmp
        self.lenght = self.__len__()
        self.flip_rot = flip_rot
        if self.flip_rot:
            print("random flip and rotation")
    
    def randomFlipRotation(self, imgs):
        op = random.choice(self.IMAGES_TRANSFORM)
        return [img.transpose(op) for img in imgs]
    
    def getFilesSplit(self, input_dir, split_dir, split_file=""):
        images = []
        labels = []
        general_type = os.path.basename(input_dir)
        if len(split_dir):
            with open(pjoin(split_dir, self.shorttopath[general_type], split_file)) as f: #f"train.txt"
                video_names = f.read().split("\n")
        else:
            with open(pjoin(input_dir, f"{general_type}.lst")) as f:
                video_names = f.read().split("\n")[:-1]
        for vn in video_names:
            name = general_type if general_type == "origins" else "fraud/"+general_type
            if self.only_label is not None:
                # will only takes origins (only_label True) or frauds (only_label False)
                if (general_type == "origins") != self.only_label:
                    continue
            l = f"{name}/{os.path.dirname(vn)}"
            with open(pjoin(input_dir, vn)) as f:
                tmp_lst = [v for v in f.read().split("\n") if v != ""]
                images += tmp_lst
                labels += [l] * len(tmp_lst)
                self.labels_dict[name][l] = tmp_lst
        assert len(images) == len(labels), "images must be the same size as labels"
        return images, labels

    def __getitem__(self, idx: int):
        f = self.files[idx]
        l = self.labels[idx]
        if "origins" in l:
            im = Image.open(pjoin(self.input_dir, l, f))
            tmp_l = self.labels[idx+1 if idx+1 < self.lenght else idx-1]
            if tmp_l == l:
                im_n = Image.open(pjoin(self.input_dir, tmp_l, self.files[idx+1 if idx+1 < self.lenght else idx-1]))
            else:
                im_n = Image.open(pjoin(self.input_dir, self.labels[idx-1], self.files[idx-1]))
                    
            if self.flip_rot and random.random() < 0.5:
                im, im_n = self.randomFlipRotation((im, im_n))

            return [self.transform(im), self.transform(im), self.transform(im_n)], l
        else:
            im = Image.open(pjoin(self.input_dir, l, f))
            fraud = "/".join(l.split("/")[:2])
            img_path_tmp = random.choice(self.labels_dict[fraud][l])
            im_p = Image.open(pjoin(self.input_dir, l, img_path_tmp))
            possible_frauds = [k for k in self.fraud_names if k != fraud]

            fraud_n = random.choice(possible_frauds)
            k_n = fraud_n + "/"+"/".join(l.split("/")[2:])
            im_n = random.choice(self.labels_dict[fraud_n][k_n])
            im_n = Image.open(pjoin(self.input_dir, k_n, im_n))

            if self.flip_rot and random.random() < 0.5:
                im, im_p, im_n = self.randomFlipRotation((im, im_p, im_n))

            return [self.transform(im), self.transform(im_p), self.transform(im_n)], l
    
    def __len__(self) -> int:
        return len(self.files)

class MIDVHoloDataModule(pl.LightningDataModule):
    def __init__(self, input_dir: str, split_dir, transform=None, batch_size: int = 32, num_workers=15, only_label=None, flip_rot=True):
        super().__init__()
        self.data_dir = input_dir
        self.batch_size = batch_size
        print(transform)
        self.transform=transform
        self.split_dir = split_dir
        self.num_workers = num_workers
        print("only label", only_label)
        self.only_label = only_label
        self.flip_rot = flip_rot

    def setup(self, stage: str):
        self.midvholo_train = MIDVHoloDataset(self.data_dir, self.transform, self.split_dir, "trainval/train_train.txt", self.only_label, self.flip_rot)
        self.midvholo_val = MIDVHoloDataset(self.data_dir, self.transform, self.split_dir, "trainval/train_val.txt", self.only_label, self.flip_rot)

    def train_dataloader(self):
        return DataLoader(self.midvholo_train, shuffle=True,batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.midvholo_val, batch_size=self.batch_size, num_workers=self.num_workers)

class ModelTriplet(pl.LightningModule):
    def __init__(self, model, lr=0.1):
        super().__init__()
        self.backbone = model
        self.lr = lr
        self.criterion = nn.TripletMarginLoss()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        return f

    def training_step(self, batch, batch_idx):
        ancor_img, positive_img, negative_img = batch[0]

        anchor_out = self.forward(ancor_img)
        positive_out = self.forward(positive_img)
        negative_out = self.forward(negative_img)

        loss = self.criterion(anchor_out, positive_out, negative_out)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
        return loss
    
    def validation_step(self, batch, batch_idx):
        ancor_img, positive_img, negative_img = batch[0]

        anchor_out = self.forward(ancor_img)
        positive_out = self.forward(positive_img)
        negative_out = self.forward(negative_img)

        loss = self.criterion(anchor_out, positive_out, negative_out)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.lr)
        return ({"optimizer": optim})

class Trainer:
    def __init__(self, epochs_max, model, seed=0, lr=0.1, accelerator="cuda", checkpoint_callback=None, run_name="") -> None:
        torch.set_float32_matmul_precision('high')
        pl.seed_everything(seed, workers=True)
        self.seed = seed
        self.model = model
        self.epochs = epochs_max
        self.accelerator = accelerator
        self.checkpoint_callback = checkpoint_callback
        self.run_name = run_name
        self.decision = None
        self.val_fullvid = None
        self.lr = lr

    def train(self, datamodule, task_name):
        model = ModelTriplet(self.model, self.lr)
        tags = {"task_name": task_name}
        checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_loss", filename='{epoch}-{val_loss:.2f}')
        mllogger = MLFlowLogger(log_model=True, run_name=f"{task_name}_{self.run_name}", tags=tags)
        run_id = mllogger.run_id
        mllogger.experiment.log_param(run_id, "lr", model.lr)
        mllogger.experiment.log_param(run_id, "transform", datamodule.transform)
        mllogger.experiment.log_param(run_id, "batch_size", datamodule.batch_size)
        loggers = [mllogger]
        # callbacks=([self.checkpoint_callback] if self.checkpoint_callback is not None else None)
        pl.seed_everything(self.seed, workers=True)
        trainer = pl.Trainer(max_epochs=self.epochs, accelerator=self.accelerator, logger=loggers, callbacks=[checkpoint], deterministic=True)
        trainer.fit(model, datamodule)

        # saving best model for latter use
        best_model_path = trainer.checkpoint_callback.best_model_path
        # print(best_model_path)
        model.load_state_dict(torch.load(best_model_path)["state_dict"])
        torch.save(model.backbone.state_dict(), os.path.join(os.path.dirname(best_model_path), f"backbone_{os.path.basename(best_model_path)}"))
        mllogger.experiment.log_metric(run_id, "best_val_loss", checkpoint.best_model_score)
        mllogger.experiment.log_param(run_id, "best_model_name", f"backbone_{os.path.basename(best_model_path)}")
        mllogger.experiment.log_param(run_id, "epochs_max", self.epochs)