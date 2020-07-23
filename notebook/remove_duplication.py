import os
import sys
import numpy
import cv2


def color_feature(img):
    if img.dtype == 'uint8':
        img = img.astype(float) / 255.0
    else:
        img = img.astype(float)
    r_img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    r2 = cv2.resize(img**2, (8, 8), interpolation=cv2.INTER_AREA)
    r2 -= r_img**2
    r2 = np.sqrt(np.maximum(r2, 0))
    r_img = np.concatenate([r_img, 1.0 - r_img, r2, 1.0 - r2], axis=2).ravel()
    r_img = r_img / np.sqrt(np.sum(r_img**2))
    return r_img


def remove_duplication(fn_list):
    '''
    Args:
        fn_list: list of image filenames.

    Returns:
        filtered_list: list of filenames duplication removed.
    '''
    all_clr = [color_feature(cv2.imread(fn)) for fn in fn_list]

    N = len(fn_list)

    fidx = np.random.permutation(np.arange(N))
    # hogs = np.array(all_hog)[fidx]
    clrs = np.array(all_clr)
    kept = []
    remaining = fidx.copy()

    for ii in range(N):
        if len(remaining) == 0:
            break
        pidx = remaining[0]
        kept.append(pidx)
        if len(remaining) == 1:
            break
        
        p_clr = clrs[pidx]
        sims_clr = np.sum(clrs[remaining] * np.reshape(p_clr, (1, -1)), axis=1)**100
        
    #     p_hog = hogs[pidx]
    #     sims_hog = np.sum(hogs[remaining] * np.reshape(p_hog, (1, -1)), axis=1) / 36
        
        ridx = np.where((sims_clr) < 0.5)[0]
        remaining = remaining[ridx]

    return np.array(fn_list)[kept]
