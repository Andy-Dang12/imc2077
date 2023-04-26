# from sys import path; path.append('dkm')
import warnings; warnings.filterwarnings('ignore')
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from dkm.dkm import DKMv3_outdoor
from dkm.dkm.utils import tensor_to_pil


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dkm_model = DKMv3_outdoor(device=device)
w_h_muts_dkm = [[680 * 510, 1]]

im1_path = 'DKM/assets/sacre_coeur_A.jpg'
im2_path = 'DKM/assets/sacre_coeur_B.jpg'
save_path = 'demo/dkmv3_warp_sacre_coeur.jpg'

def inference():
    H, W = 864, 1152
    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = dkm_model.match(im1_path, im2_path, device=device)
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    dkm_model.sample(warp, certainty)
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
    x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
    x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    white_im = torch.ones((H,2*W),device=device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    tensor_to_pil(vis_im, unnormalize=False).save(save_path)
    return

def inference_dkm(image_0_BGR:cv2.Mat, image_1_BGR:cv2.Mat,
                  mkpts0_loftr_all:    np.ndarray=...,
                  mkpts0_superglue_all:np.ndarray=... ) -> tuple[np.ndarray, np.ndarray]:
    # ===========================
    #            DKM
    # ===========================
    img0PIL = Image.fromarray(cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2RGB))
    img1PIL = Image.fromarray(cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2RGB))

    mkpts0_dkm_all = []
    mkpts1_dkm_all = []

    for w_h_mut, param in w_h_muts_dkm:
        ratio = (image_0_BGR.shape[0] + image_1_BGR.shape[0]) / (image_0_BGR.shape[1] + image_1_BGR.shape[1]) * param
        dkm_model.w_resized = int(np.sqrt(w_h_mut / ratio))
        dkm_model.h_resized = int(ratio * dkm_model.w_resized)

        dense_matches, dense_certainty = dkm_model.match(img0PIL, img1PIL, device=device)
        dense_certainty = dense_certainty.pow(0.6)

        sparse_matches, sparse_certainty = dkm_model.sample(
            dense_matches, dense_certainty,
            # max(min(500, (len(mkpts0_loftr_all) + len(mkpts0_superglue_all)) // int(4 * len(w_h_muts_dkm))), 100),
            max(min(500, (3448 + 1335) // int(4 * len(w_h_muts_dkm))), 100)
        )

        mkpts0_dkm = sparse_matches[:, :2]
        mkpts1_dkm = sparse_matches[:, 2:]
        h, w, c = image_0_BGR.shape
        mkpts0_dkm[:, 0] = ((mkpts0_dkm[:, 0] + 1) / 2) * w
        mkpts0_dkm[:, 1] = ((mkpts0_dkm[:, 1] + 1) / 2) * h
        h, w, c = image_1_BGR.shape
        mkpts1_dkm[:, 0] = ((mkpts1_dkm[:, 0] + 1) / 2) * w
        mkpts1_dkm[:, 1] = ((mkpts1_dkm[:, 1] + 1) / 2) * h

        mkpts0_dkm_all.append(mkpts0_dkm)
        mkpts1_dkm_all.append(mkpts1_dkm)

    mkpts0_dkm_all = np.concatenate(mkpts0_dkm_all, axis=0)
    mkpts1_dkm_all = np.concatenate(mkpts1_dkm_all, axis=0)

    return mkpts0_dkm_all, mkpts1_dkm_all

if __name__ == '__main__':
    img0 = cv2.imread('superGlue/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg')
    img1 = cv2.imread('superGlue/assets/phototourism_sample_images/london_bridge_49190386_5209386933.jpg')
    from time import time
    s = time()
    mkpts0, mkpts1 = inference_dkm(img0, img1)
    ss = time()
    print('mkpts0 =\n', mkpts0.shape)   #(500, 2)
    print('mkpts1 = \n', mkpts1.shape)  #(500, 2)
    print(ss - s)                       #49.428354024887085
