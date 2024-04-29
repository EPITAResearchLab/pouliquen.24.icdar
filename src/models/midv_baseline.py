import cv2
import numpy as np

class MIDVBaseline:
    C = None
    h = 0
    i = -1
    mask_holo_coarse = None

    def __init__(self, hight_threshold, s_t, T, n_min=4) -> None:
        self.th_white = hight_threshold
        self.n_min = n_min
        self.reset()

        self.s_t = s_t
        self.T = T
        
    def reset(self):
        self.C = None
        self.h = 0
        self.i = -1
        self.h_percent = 0

    def normalize_gray_world(self, img):
        img_array = img.copy()
        img_array = img_array.astype(np.float32)

        # mask of highlight
        mask = (cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) <= self.th_white)
        
        b_mean, g_mean, r_mean = np.mean(img_array[:,:,0][mask]), np.mean(img_array[:,:,1][mask]), np.mean(img_array[:,:,2][mask])
        
        global_mean = np.mean([r_mean, g_mean, b_mean])
        
        img_array[:,:,0] = img_array[:,:,0] * global_mean / b_mean
        img_array[:,:,1] = img_array[:,:,1] * global_mean / g_mean
        img_array[:,:,2] = img_array[:,:,2] * global_mean / r_mean

        img_array = np.clip(img_array, 0, 255)

        img_array[~mask] = 0
        
        return img_array.astype(np.uint8), mask
    
    def apply(self, img):
        self.i += 1
        if self.C is None:
            self.C = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float64)
            self.smax = np.zeros((img.shape[0], img.shape[1]))
            self.n = np.zeros((img.shape[0], img.shape[1]))

        img_n, mask = self.normalize_gray_world(img)

        h, s, _ = cv2.split(cv2.cvtColor(img_n, cv2.COLOR_BGR2HSV))
        h = h.astype(np.float64)
        s = img_n.max(axis=2) - img_n.min(axis=2)
        # mask[v < self.t_v] = False
        self.smax = np.max(np.stack([self.smax, s]), axis=0)
        self.smax[~mask] = 0

        self.C[:, :, 0][mask] += s[mask]*np.cos(np.deg2rad(h[mask]*2))
        self.C[:, :, 1][mask] += s[mask]*np.sin(np.deg2rad(h[mask]*2))
        self.n = self.n + mask

        C_f = self.C.copy()

        n_min = self.i//2
        C_f[:, :, 0][self.n > n_min] /= self.n[self.n > n_min]
        C_f[:, :, 1][self.n > n_min] /= self.n[self.n > n_min]

        c_map = np.linalg.norm(C_f, axis=-1)
        smax_tmp = self.smax.copy()
        smax_tmp[self.n<=n_min] = 0
        
        s_th = smax_tmp > self.s_t
        I_no = np.ones(s_th.shape) * 255
        I_no[s_th] = (c_map[s_th] / smax_tmp[s_th]) * 255
        I_no_bin = (I_no > self.T)

        self.h_percent = I_no_bin.sum()/I_no_bin.size
        
        return 1-self.h_percent