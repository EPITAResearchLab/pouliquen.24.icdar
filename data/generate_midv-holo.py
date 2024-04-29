import os
import cv2
from joblib import Parallel, delayed
import numpy as np
from os.path import join as pjoin, basename, dirname
import tqdm
import json
import shutil

n_jobs = 10

base_path = "data/midv-holo/"
markup_p = pjoin(base_path, "markup")
images_p = pjoin(base_path, "images")

rectified_name = "rectified"
ovd_name = "crop_ovds"

print(images_p, pjoin(images_p, "origins.lst"))
os.makedirs(dirname(pjoin(images_p, "origins/origins.lst").replace("/images/", f"/{rectified_name}/")), exist_ok=True)
shutil.copy(pjoin(images_p, "origins/origins.lst"), pjoin(images_p, "origins/origins.lst").replace("/images/", f"/{rectified_name}/"))
os.makedirs(dirname(pjoin(images_p, "origins/origins.lst").replace("/images/", f"/{ovd_name}/")), exist_ok=True)
shutil.copy(pjoin(images_p, "origins/origins.lst"), pjoin(images_p, "origins/origins.lst").replace("/images/", f"/{ovd_name}/"))
with open(pjoin(images_p, "origins/origins.lst")) as f:
    files = f.read().splitlines(False)

images_d = {}
for f in files:
    dirname_p = dirname(f)
    with open(pjoin(images_p, "origins", f)) as f:
        images_d[basename(dirname_p)] = [pjoin(images_p, "origins", dirname_p, x.replace("\n", "")) for x in f.readlines()]

images_d_fraud = {}
for fraud in os.listdir(pjoin(images_p, "fraud")):
    print(fraud, pjoin(images_p, f"fraud/{fraud}/{fraud}.lst"))
    os.makedirs(dirname(pjoin(images_p, f"fraud/{fraud}/{fraud}.lst").replace("/images/", f"/{rectified_name}/")), exist_ok=True)
    shutil.copy(pjoin(images_p, f"fraud/{fraud}/{fraud}.lst"), pjoin(images_p, f"fraud/{fraud}/{fraud}.lst").replace("/images/", f"/{rectified_name}/"))
    os.makedirs(dirname(pjoin(images_p, f"fraud/{fraud}/{fraud}.lst").replace("/images/", f"/{ovd_name}/")), exist_ok=True)
    shutil.copy(pjoin(images_p, f"fraud/{fraud}/{fraud}.lst"), pjoin(images_p, f"fraud/{fraud}/{fraud}.lst").replace("/images/", f"/{ovd_name}/"))

    with open(pjoin(images_p, f"fraud/{fraud}/{fraud}.lst")) as f:
        files_no_holo = f.read().splitlines(False)#[i[:-1] for i in f.readlines()] #
    
    if fraud not in images_d_fraud:
        images_d_fraud[fraud] = {}
    for f in files_no_holo:
        dirname_p = dirname(f)
        with open(pjoin(images_p, f"fraud/{fraud}", f)) as f:
            images_d_fraud[fraud][basename(dirname_p)] = [pjoin(images_p, f"fraud/{fraud}", dirname_p, x.replace("\n", "")) for x in f.readlines()]

def getLabel(p):
    with open(p.replace("/images/", "/markup/")+".json") as f:
        markup = json.load(f)
    return list(markup["document"]["templates"].values())[0]["template_quad"]

def save_roi(imgs_p, width, height, roi):
    dirname_p = dirname(imgs_p[0].replace("/images/", "/rectified/"))
    dirname2_p = dirname(imgs_p[0].replace("/images/", "/crop_ovds/"))
    os.makedirs(dirname_p, exist_ok=True)
    os.makedirs(dirname2_p, exist_ok=True)
    shutil.copyfile(dirname(imgs_p[0])+"/list.lst", dirname_p+"/list.lst")
    shutil.copyfile(dirname(imgs_p[0])+"/list.lst", dirname2_p+"/list.lst")
    for im_p in imgs_p:
        im = cv2.imread(im_p)
        quad = getLabel(im_p)
        T = cv2.getPerspectiveTransform(np.array(quad, dtype=np.float32), np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32))
        im_w = cv2.warpPerspective(im, T, (width, height))

        p = im_p.replace("/images/", "/rectified/")
        cv2.imwrite(p, im_w)

        im_roi = im_w[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

        p_roi = im_p.replace("/images/", "/crop_ovds/")

        cv2.imwrite(p_roi, im_roi)

width, height = 1123, 709
roi = [[141, 224], [498, 557]]
# origins
Parallel(n_jobs=n_jobs)(delayed(save_roi)(img_p, width, height, roi) for img_p in tqdm.tqdm(images_d.values()))

# frauds
for fraud in images_d_fraud:
    Parallel(n_jobs=n_jobs)(delayed(save_roi)(img_p, width, height, roi) for img_p in tqdm.tqdm(images_d_fraud[fraud].values()))