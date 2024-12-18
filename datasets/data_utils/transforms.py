import random
import cv2

import numpy as np

import torch

from torchvision import transforms

# -------------------- Useful Function --------------------
MAX_VALUES_BY_DTYPE = {
    np.dtype('uint8'): 255,
    np.dtype('float32'): 1.0,
}

NPDTYPE_TO_OPENCV_DTYPE = {
    np.dtype('uint8'): cv2.CV_8U,
    np.dtype('float32'): cv2.CV_32F,
}

def to_float(x, max_value=None):
    if max_value is None:
        try: 
            max_value = MAX_VALUES_BY_DTYPE[x.dtype]
        except KeyError:
            raise RuntimeError

    return x.astype(np.float32) / max_value

def from_float(x, dtype, max_value=None):
    if max_value is None:
        try: 
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError

    return (x * max_value).astype(dtype)

def resize_fn(img, out_h, out_w):
    return transforms.ToTensor()(
        transforms.Resize((out_h, out_w), transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)
        )
    )

# -------------------- Overall --------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)

        return x

# -------------------- Numpy + CV2 --------------------
class ToFloat:
    def __call__(self, x):
        return to_float(x)

class Resize:
    def __init__(self, out_shape):
        self.out_shape = out_shape

    def __call__(self, x):
        h = int(self.out_shape[0])
        w = int(self.out_shape[1])

        return cv2.resize( x, (w, h), interpolation=cv2.INTER_CUBIC )

class BasePad:
    def __init__(self, base=64):
        self.base = base

    def __call__(self, x):
        h, w, c = x.shape

        pad_h = 0 if h % self.base == 0 else self.base - (h % self.base)
        pad_w = 0 if w % self.base == 0 else self.base - (w % self.base)

        pad_t = pad_h // 2
        pad_b = pad_h - pad_t

        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        return cv2.copyMakeBorder(x, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT), [pad_t, pad_b, pad_l, pad_r]

class BaseCrop:
    def __init__(self, base=64):
        self.base = base

    def __call__(self, x):
        h, w, c = x.shape

        crop_h = h % self.base
        crop_w = w % self.base

        return x[crop_h // 2: h - crop_h + crop_h // 2, crop_w // 2: w - crop_w + crop_w // 2, ...].copy()

class CenterCrop:
    def __init__(self, out_shape):
        self.out_shape = out_shape

    def __call__(self, x):
        out_h = int(self.out_shape)
        out_w = int(self.out_shape)

        h, w, c = x.shape

        h0 = int( round( (h - out_h) * 0.5 ) )
        w0 = int( round( (w - out_w) * 0.5 ) )

        return x[h0: h0 + out_h, w0: w0 + out_w, ...].copy()

class CenterCropForERF:
    def __init__(self, out_shape):
        self.out_shape = out_shape

    def __call__(self, x):
        out_h = int(self.out_shape)
        out_w = int(self.out_shape)

        c, h, w = x.shape

        h0 = int( round( (h - out_h) * 0.5 ) )
        w0 = int( round( (w - out_w) * 0.5 ) )

        return x[..., h0: h0 + out_h, w0: w0 + out_w]

class RandomCrop:
    def __init__(self, out_shape):
        self.out_shape = out_shape

    def __call__(self, x):
        out_h = int(self.out_shape)
        out_w = int(self.out_shape)

        h, w, c = x.shape

        h0 = random.randint(0, h - out_h)
        w0 = random.randint(0, w - out_w)
        
        return x[h0: h0 + out_h, w0: w0 + out_w, ...].copy()

class PairedRandomCrop:
    def __init__(self, out_shape):
        self.out_shape = out_shape

    def __call__(self, x, y):
        out_h = int(self.out_shape)
        out_w = int(self.out_shape)

        h, w, c = x.shape

        h0 = random.randint(0, h - out_h)
        w0 = random.randint(0, w - out_w)

        patch_x = x[h0: h0 + out_h, w0: w0 + out_w, ...].copy()
        patch_y = y[h0: h0 + out_h, w0: w0 + out_w, ...].copy()

        return patch_x, patch_y

# -------------------- Torch --------------------
class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(np.moveaxis(x, -1, 0))

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        for t, m, s in zip(x, self.mean, self.std):
            t.sub_(m).div_(s)

        return x