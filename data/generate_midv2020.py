import cv2
import os
from os.path import join as pjoin
import json
import numpy as np
import tqdm
from joblib import Parallel, delayed

n_jobs = 10
width, height = 1123, 709

roi = np.int64([[141, 224], [498, 557]])

def getQuadDoc(dict_regions, test_str="doc_quad"):
    res = next((x for x in dict_regions if x["region_attributes"]["field_name"] == test_str), None)
    return res    

def write_annotations(annotation, width, height, roi):
    region = getQuadDoc(annotation["regions"])["shape_attributes"]
    all_x = region["all_points_x"]
    all_y = region["all_points_y"]
    im_name = annotation["filename"]
    doc_n = os.path.splitext(anno_p)[0]

    im = cv2.imread(pjoin("images", d_type, doc_n, im_name))

    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts1 = np.float32([[x, y] for x, y in zip(all_x, all_y)])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(im, matrix, (width, height))
    os.makedirs(pjoin("rectified", d_type, doc_n), exist_ok=True)
    cv2.imwrite(pjoin("rectified", d_type, doc_n, im_name), result)

    os.makedirs(pjoin("crop_ovd", d_type, doc_n), exist_ok=True)
    cv2.imwrite(pjoin("crop_ovd", d_type, doc_n, im_name), result[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]])

for d_type in os.listdir("annotations"):
    p = pjoin("annotations", d_type)
    print(d_type)
    for anno_p in tqdm.tqdm(os.listdir(p)):
        base_path = pjoin(p, anno_p)
        with open(base_path) as f:
            annotations = json.load(f)
        Parallel(n_jobs=n_jobs)(delayed(write_annotations)(annotations["_via_img_metadata"][img_k], width, height, roi) for img_k in annotations["_via_image_id_list"])
