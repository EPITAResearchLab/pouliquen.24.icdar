import numpy as np
import torchvision.transforms as T
import torch

class ClassificationModelGeneral:
    i = -1
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    def __init__(self, model, accelerator="cuda", input_size=224) -> None:
        self.model = model
        self.model.eval()

        self.device = torch.device(accelerator)
        self.model = self.model.to(self.device)

        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.Normalize(mean=self.IMAGENET_NORMALIZE["mean"], std=self.IMAGENET_NORMALIZE["std"]),
            T.ToTensor()
        ])
        self.reset()
        
    def reset(self):
        self.i = -1
        self.preds = np.array([])
        
    def apply(self, img_t):
        self.i += 1

        with torch.no_grad():
            pred = self.model(img_t.unsqueeze(0).to(self.device)).cpu().softmax(1).max(dim=1).indices.numpy()

            self.preds = np.append(self.preds, pred)
        if self.preds.size > 2:
            return self.preds.mean()
        return 0