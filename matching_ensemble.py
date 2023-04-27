import cv2
import torch
import numpy as np
from PIL import Image
from matching_super_glue import inference_superglue
from matching_LoFRT import inference_loftr
from matching_dkm import inference_dkm

torch.set_grad_enabled(False)

def ensemble(image_0_BGR:cv2.Mat, image_1_BGR:cv2.Mat, debug:bool=False):
    mkpts0_superglue_all, mkpts1_superglue_all = inference_superglue(image_0_BGR, image_1_BGR, debug)
    mkpts0_loftr_all, mkpts1_loftr_all         = inference_loftr(image_0_BGR, image_1_BGR, debug)
    mkpts0_dkm_all, mkpts1_dkm_all             = inference_dkm(image_0_BGR, image_1_BGR)

    mkpts0 = []; mkpts1 = []; F_dict = {}

    if len(mkpts0_loftr_all) > 0:
        mkpts0.append(mkpts0_loftr_all)
        mkpts1.append(mkpts1_loftr_all)

    if len(mkpts0_superglue_all) > 0:
        mkpts0.append(mkpts0_superglue_all)
        mkpts1.append(mkpts1_superglue_all)

    mkpts0.append(mkpts0_dkm_all)
    mkpts1.append(mkpts1_dkm_all)

    mkpts0 = np.concatenate(mkpts0, axis=0)
    mkpts1 = np.concatenate(mkpts1, axis=0)

    return mkpts0, mkpts1


def get_test_samples(src:str):
    import csv
    _test_samples = []
    with open(f'{src}/test.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if not i:
                continue
            _test_samples += [row]
    return _test_samples

def flatten_matrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])

def run(debug:bool=False):
    src = 'image-matching-challenge-2022'
    test_samples = get_test_samples(src)
    F_dict = {}
    for i, row in enumerate(test_samples):
        sample_id, batch_id, image_0_id, image_1_id = row

        image_0_BGR = cv2.imread(f'{src}/test_images/{batch_id}/{image_0_id}.png')
        image_1_BGR = cv2.imread(f'{src}/test_images/{batch_id}/{image_1_id}.png')
        mkpts0, mkpts1 = ensemble(image_0_BGR, image_1_BGR, debug)

        if len(mkpts0) > 8:
            F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
            F_dict[sample_id] = F
        else:
            F_dict[sample_id] = np.zeros((3, 3))

    print('write_test_sample')
    with open('submission.csv', 'w') as f:
        f.write('sample_id,fundamental_matrix\n')
        for sample_id, F in F_dict.items():
            f.write(f'{sample_id},{flatten_matrix(F)}\n')

    return

if __name__ == '__main__':
    from time import time
    s = time()
    run(True)
    ss = time()
    print('runtime = ', ss-s)
