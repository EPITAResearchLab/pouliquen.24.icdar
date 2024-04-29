import numpy as np
from sklearn.metrics import pairwise_distances
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import torch

class SSLGeneral:
    i = -1
    mask_holo_coarse = None
    embeddings = np.array([])
    diffs = np.array([])
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    def __init__(self, model, accelerator="cuda", input_size=224, method=0) -> None:
        self.model = model
        self.model.eval()

        self.model = self.model
        self.device = torch.device(accelerator)
        self.model.to(self.device)
        if method == 0:
            print("np1")
            self.method = self.np1
        elif method == 1:
            print("np2")
            self.method = self.nscore
        elif method == 2:
            print("np3")
            self.n = method
            self.method = self.npn
        else:
            print("mean sim")
            self.method = self.mean_sim

        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_NORMALIZE["mean"], std=self.IMAGENET_NORMALIZE["std"]),
        ])

    def nscore(self):
        n = 3
        if self.embeddings.shape[0] < n:
            return None
        base_emb = self.embeddings[-1].reshape(1, -1)
        res = []
        for i in range(-2,-n-1, -1):
            embedding2 = self.embeddings[i].reshape(1, -1)
            res+=[cosine_similarity(base_emb, embedding2)[0, 0]]*-i
        return np.mean(res)

    def np1(self):
        return cosine_similarity(self.embeddings[-2].reshape(1, -1), self.embeddings[-1].reshape(1, -1))
    
    def npn(self):
        if self.embeddings.shape[0] <= self.n:
            return None
        # print(self.n, self.embeddings.shape[0])
        return cosine_similarity(self.embeddings[-self.n-1].reshape(1, -1), self.embeddings[-1].reshape(1, -1))
    
    def mean_sim(self):
        if self.embeddings.shape[0] <= 2:
            return None
        similarity = pairwise_distances(self.embeddings, metric="cosine")
        np.fill_diagonal(similarity, 0)
        return similarity.mean()
        
    def reset(self):
        self.i = -1
        self.h_percent = 0
        self.embeddings = np.array([])
        self.diffs = np.array([])
    
    def apply(self, img_t):
        self.i += 1
        # img_t = self.transform(img)

        with torch.no_grad():
            embedding = self.model(img_t.unsqueeze(0).to(self.device)).flatten(start_dim=1).cpu().numpy()

            if self.embeddings.size == 0:
                self.embeddings = embedding
            else:
                self.embeddings = np.concatenate((self.embeddings, embedding))
        if self.method == self.mean_sim:
            return self.method() if self.embeddings.shape[0] > 2 else None
        
        if self.embeddings.shape[0] > 1:
            # diff = cosine_similarity(self.embeddings[-2].reshape(1, -1), self.embeddings[-1].reshape(1, -1))
            diff = self.method()
            if diff is not None:
                self.diffs = np.append(self.diffs, diff)
    
        if self.diffs.size > 1:
            # return cosine_similarity(self.embeddings[-1].reshape(1, -1), self.embeddings[-2].reshape(1, -1))
            # print(self.diffs)
            return 1-np.median(self.diffs)
        return None
    
    def get_vid_embeddings(self, ims_t, batch_n=8):
        imgs_b = []
        embeddings = None
        with torch.no_grad():
            for im_t in ims_t:
                imgs_b.append(im_t)
                if len(imgs_b) == batch_n:
                    res = self.model(torch.stack(imgs_b).cuda()).flatten(start_dim=1).cpu()
                    if embeddings is None:
                        embeddings = res
                    else:
                        embeddings = torch.cat((embeddings, res))
                    imgs_b = []

            if len(imgs_b) != 0:
                res = self.model(torch.stack(imgs_b).cuda()).flatten(start_dim=1).cpu()
                if embeddings is None:
                    embeddings = res
                else:
                    embeddings = torch.cat((embeddings, res))
        return embeddings.numpy()