import numpy as np
from dataflow.dataflow.imgaug import Transform, ImageAugmentor, PhotometricAugmentor
from dataflow.dataflow.imgaug import ResizeTransform
import cv2


class AffineTransform(Transform):
    def __init__(self, roi, out_wh, mean_rgbgr):
        super(AffineTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)

        a = (self.out_hw[1]) / (self.roi[2] - self.roi[0])
        b = (self.out_hw[0]) / (self.roi[3] - self.roi[1])
        c = -a * (self.roi[0] - 0.0)
        d = -b * (self.roi[1] - 0.0)
        self.mapping = np.array([[a, 0, c],
                                 [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(img, self.mapping,
                              (self.out_hw[1], self.out_hw[0]),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=self.mean_rgbgr)
        return crop

    def apply_coords(self, coords):
        a, b, c, d = self.mapping.ravel()[[0, 4, 2, 5]]
        coords[:, 0] = coords[:, 0] * a + c
        coords[:, 1] = coords[:, 1] * b + d
        return coords


class ShiftScaleAugmentor(ImageAugmentor):
    """ 
    shift scale the bounding box.
    crop image accordinly, centering the augmented bb.
    """

    def __init__(self,
                 scale_exp,
                 aspect_exp,
                 out_hw,
                 mean_rgbgr=np.array([127, 127, 127])):
        """
        1. Create a box, [x0, y0, x1, y1] = [0, 0, ww, hh]
        2. Randomly scale, change aspect ratio, translate the box
        3. Crop or pad image according to the box
        4. Rescale the output to [out_size, out_size]

        Args:
            scale_exp: scale base, input will be scaled to [scale_exp^-1, scale_exp^1]
            aspect_exp: aspect base, same as scale_exp
            out_size: size of the output
            wmin, hmin, wmax, hmax: range to sample shape.
            max_aspect_ratio (float): the upper bound of ``max(w,h)/min(w,h)``.
        """
        self._init(locals())

    def _log_rand(self, rrange, r):
        '''
        rrange: tuple of [min, max], should be > 0
        r: random float, range of [0, 1)
        '''
        logrr = np.log(rrange)
        w = logrr[1] - logrr[0]
        return np.exp(r * w + logrr[0])

    def get_transform(self, img):

        h, w = img.shape[:2]
        cx = w / 2.0
        cy = h / 2.0
        sz = np.sqrt(float(h*w))

        # augmentation params
        rval = self.rng.uniform(size=[4])
        # no_aug = rval[0]
        rtx, rty = rval[:2] # [0, 1)
        ss = np.power(self.scale_exp, rval[2] * 2.0 - 1.0)
        asp = np.power(self.aspect_exp, rval[3] * 2.0 - 1.0)

        sx = ss * np.sqrt(asp)
        sy = ss / np.sqrt(asp)
        # initial crop box: (cx-t2, cy-t2, cx+t2, cy+t2)
        # cx, cy = image centre, t2 = half of target size

        ww = sz / sx
        hh = sz / sy

        x0 = rtx * (w - ww)
        y0 = rty * (h - hh)
        # tx = rtx * (w - ww)
        # ty = rty * (h - hh)

        # augmentation
        # cx += tx
        # cy += ty

        roi = [x0, y0, x0+ww, y0+hh]
        # roi = [cx - ww/2.0, cy - hh/2.0, cx + ww/2.0, cy + hh/2.0]
        return AffineTransform(roi, self.out_hw, self.mean_rgbgr)


class ResizeAugmentor(ImageAugmentor):
    def __init__(self, size, interp=cv2.INTER_LINEAR):
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        return ResizeTransform(h, w, self.size, self.size, self.interp) 


class ColorJitterAugmentor(PhotometricAugmentor):
    ''' Random color jittering '''
    def __init__(self, \
                 mean_rgbgr=[127.0, 127.0, 127.0], \
                 rand_l=0.45 * 255, \
                 rand_c=0.75, \
                 rand_h=0.15 * 255):
        super(ColorJitterAugmentor, self).__init__()
        if not isinstance(mean_rgbgr, np.ndarray):
            mean_rgbgr = np.array(mean_rgbgr)
        min_rgbgr = -mean_rgbgr
        max_rgbgr = min_rgbgr + 255.0
        self._init(locals())

    def _get_augment_params(self, _):
        return self.rng.uniform(-1.0, 1.0, [8])

    def _augment(self, img, rval):
        old_dtype = img.dtype
        img = img.astype(np.float32)
        rflag = rval[5:] * 0.5 + 0.5
        rflag = (rflag > 0.25).astype(float)
        rval[0] *= (self.rand_l * rflag[0])
        rval[1] = np.power(1.0 + self.rand_c, rval[3] * rflag[1])
        rval[2:4] *= (self.rand_h * rflag[2])
        rval[4] = -(rval[2] + rval[3])

        for i in range(3):
            add_val = (rval[0] + rval[i+2] - self.mean_rgbgr[i]) * rval[1] + self.mean_rgbgr[i]
            img[:, :, i] = img[:, :, i] * rval[1] + add_val
            # img[:, :, i] = np.maximum(0.0, np.minimum(255.0,
            #     (img[:, :, i] + add_val) * rval[1] + self.mean_rgbgr[i]))
        img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


def box_to_point8(boxes, offset_rb=0):
    """
    Args:
        boxes: nx4
    Returns:
        (nx4)x2
    """
    b = np.reshape(boxes, [-1, 4])
    # b[:, 2:] -= offset_rb
    b = b[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points, offset_rb=0):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    b = np.concatenate((minxy, maxxy), axis=1)
    # b[:, 2:] += offset_rb
    if b.shape[0] == 1:
        b = np.ravel(b)
    return b
