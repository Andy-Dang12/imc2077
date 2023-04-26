import torch
import cv2
import gc
import numpy as np
from superGlue.models.matching import Matching
from superGlue.models.utils import frame2tensor

gc.enable()
def resize_img_superglue(img:cv2.Mat, max_len:int, enlarge_scale:float, variant_scale:float
                         ) -> tuple[cv2.Mat, tuple[float, float], bool]:
    if max_len == -1:
        scale = 1
    else:
        scale = max(max_len, max(img.shape[0], img.shape[1]) * enlarge_scale) / max(img.shape[0], img.shape[1])
    w = int(round(img.shape[1] * scale))
    h = int(round(img.shape[0] * scale))

    isResized = False
    if w >= h:
        if int(h * variant_scale) <= w:
            isResized = True
            h = int(h * variant_scale)
    else:
        if int(w * variant_scale) <= h:
            isResized = True
            w = int(w * variant_scale)
    img_resize = cv2.resize(img, (w, h))
    return img_resize, (w / img.shape[1], h / img.shape[0]), isResized


# ===========================
#          SuperGlue
# ===========================
resize = [-1, ]
resize_float = True
config = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 10,
        "match_threshold": 0.2,
    }
}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
matching_superglue = Matching(config).eval().to(device)
scales_lens_superglue = ((1.2, 1200, 1.0), (1.2, 1600, 1.6), (0.8, 2000, 2), (1, 2800, 3))


def inference_superglue(image_0_BGR:cv2.Mat, image_1_BGR:cv2.Mat,
                        debug:bool=False) -> tuple[np.ndarray, np.ndarray]:
    mkpts0_superglue_all = []
    mkpts1_superglue_all = []
    image_0_GRAY = cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2GRAY)
    image_1_GRAY = cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2GRAY)

    for variant_scale, max_len, enlarge_scale in scales_lens_superglue:
        image_0, scale_0, isResized_0 = resize_img_superglue(image_0_GRAY, max_len, enlarge_scale, variant_scale)
        image_1, scale_1, isResized_1 = resize_img_superglue(image_1_GRAY, max_len, enlarge_scale, variant_scale)

        if not isResized_0 or not isResized_1: break

        image_0 = frame2tensor(image_0, device)
        image_1 = frame2tensor(image_1, device)

        pred = matching_superglue({"image0": image_0, "image1": image_1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches, conf = pred["matches0"], pred["matching_scores0"]

        valid = matches > -1
        mkpts0_superglue = kpts0[valid]
        mkpts1_superglue = kpts1[matches[valid]]

        if debug:
            print("superglue scale_0", scale_0)
            print("superglue scale_1", scale_1)

        mkpts0_superglue /= scale_0
        mkpts1_superglue /= scale_1

        mkpts0_superglue_all.append(mkpts0_superglue)
        mkpts1_superglue_all.append(mkpts1_superglue)

    if len(mkpts0_superglue_all) > 0:
        mkpts0_superglue_all = np.concatenate(mkpts0_superglue_all, axis=0)
        mkpts1_superglue_all = np.concatenate(mkpts1_superglue_all, axis=0)

    return mkpts0_superglue_all, mkpts1_superglue_all


if __name__ == '__main__':
    img0 = cv2.imread('superGlue/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg')
    img1 = cv2.imread('superGlue/assets/phototourism_sample_images/london_bridge_49190386_5209386933.jpg')
    from time import time
    s = time()
    mkpts0, mkpts1 = inference_superglue(img0, img1, True)
    ss = time()
    print('mkpts0 =\n', mkpts0.shape)   #(1335, 2)
    print('mkpts1 = \n', mkpts1.shape)  #(1335, 2)
    print(ss - s)
