import torch
import kornia
import cv2
import numpy as np
import kornia as K
from kornia.feature import LoFTR


# ===========================
#          LoFTR
# ===========================
def resize_img_loftr(img, max_len, enlarge_scale, variant_scale, device):
    if max_len == -1:
        scale = 1
    else:
        scale = max(max_len, max(img.shape[0], img.shape[1]) * enlarge_scale) / max(img.shape[0], img.shape[1])
    w = int(round(img.shape[1] * scale) / 8) * 8
    h = int(round(img.shape[0] * scale) / 8) * 8

    isResized = False
    if w >= h:
        if int(h * variant_scale) <= w:
            isResized = True
            h = int(h * variant_scale / 8) * 8
    else:
        if int(w * variant_scale) <= h:
            isResized = True
            w = int(w * variant_scale / 8) * 8
    img_resize = cv2.resize(img, (w, h))
    img_resize = K.image_to_tensor(img_resize, False).float() / 255.

    return img_resize.to(device), (w / img.shape[1], h / img.shape[0]), isResized


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
matcher_loftr = LoFTR(pretrained=None)
matcher_loftr.load_state_dict(torch.load('loFTR/loftr_outdoor.ckpt')['state_dict'])
# matcher_loftr = LoFTR()
matcher_loftr = matcher_loftr.eval().to(device)
scales_lens_loftr = ((1.1, 1000, 1.0), (1, 1200, 1.3), (0.9, 1400, 1.6))


def inference_loftr(image_0_BGR:cv2.Mat, image_1_BGR:cv2.Mat,
                    debug:bool=False) -> tuple[np.ndarray, np.ndarray]:
    mkpts0_loftr_all = []
    mkpts1_loftr_all = []
    image_0_GRAY = cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2GRAY)
    image_1_GRAY = cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2GRAY)

    for variant_scale, max_len, enlarge_scale in scales_lens_loftr:

        image_0_resize, scale_0, isResized_0 = resize_img_loftr(image_0_GRAY, max_len, enlarge_scale, variant_scale, device)
        image_1_resize, scale_1, isResized_1 = resize_img_loftr(image_1_GRAY, max_len, enlarge_scale, variant_scale, device)

        if isResized_0 == False or isResized_1 == False: continue

        input_dict = {"image0": image_0_resize,
                      "image1": image_1_resize}
        correspondences = matcher_loftr(input_dict)
        confidence = correspondences['confidence'].cpu().numpy()

        if len(confidence) < 1: continue

        confidence_quantile = np.quantile(confidence, 0.6)
        idx = np.where(confidence >= confidence_quantile)

        mkpts0_loftr = correspondences['keypoints0'].cpu().numpy()[idx]
        mkpts1_loftr = correspondences['keypoints1'].cpu().numpy()[idx]

        if debug:
            print("loftr scale_0", scale_0)
            print("loftr scale_1", scale_1)

        mkpts0_loftr = mkpts0_loftr / scale_0
        mkpts1_loftr = mkpts1_loftr / scale_1

        mkpts0_loftr_all.append(mkpts0_loftr)
        mkpts1_loftr_all.append(mkpts1_loftr)

    mkpts0_loftr_all = np.concatenate(mkpts0_loftr_all, axis=0)
    mkpts1_loftr_all = np.concatenate(mkpts1_loftr_all, axis=0)
    return mkpts0_loftr_all, mkpts1_loftr_all


if __name__ == '__main__':
    img0 = cv2.imread('superGlue/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg')
    img1 = cv2.imread('superGlue/assets/phototourism_sample_images/london_bridge_49190386_5209386933.jpg')
    from time import time
    s = time()
    mkpts0, mkpts1 = inference_loftr(img0, img1, True)
    ss = time()
    print('mkpts0 =\n', mkpts0.shape)   #(3448, 2)
    print('mkpts1 = \n', mkpts1.shape)  #(3448, 2)
    print(ss - s)                       #1016.5355575084686
