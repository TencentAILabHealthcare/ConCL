import cv2
import inspect
import numpy as np
from PIL import Image, ImageFilter

import torch
from torchvision import transforms as _transforms
import torchvision.transforms.functional as F
import random

from openselfsup.utils import build_from_cfg
from .transform_utils import _get_size, _update_transf_and_ratio, _with_trans_info

from ..registry import PIPELINES_WITH_INFO

# borrow from https://github.com/kakaobrain/scrl/blob/master/augment/transforms.py
# Spatially Consistent Representation Learning: https://arxiv.org/abs/2103.06122

# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur', 'CenterCrop', 'Resize', 'RandomResizedCrop',
                        'RandomHorizontalFlip', 'RandomOrder', 'RandomApply']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES_WITH_INFO.register_module(m[1])


@PIPELINES_WITH_INFO.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES_WITH_INFO) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES_WITH_INFO.register_module
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES_WITH_INFO.register_module
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 -img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES_WITH_INFO.register_module
class CenterCrop(_transforms.CenterCrop):
    def with_trans_info(self, img, transf, ratio):
        w, h = img.size
        oh, ow = _get_size(self.size)
        i = int(round((w - ow) * 0.5))
        j = int(round((h - oh) * 0.5))
        transf_local = [i, j, oh, ow]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, None)
        return F.center_crop(img, self.size), transf, ratio


@PIPELINES_WITH_INFO.register_module
class Resize(_transforms.Resize):
    def with_trans_info(self, img, transf, ratio):
        w, h = img.size  # PIL.Image
        resized_img = F.resize(img, self.size, self.interpolation)
        # get the size directly from resized image rather than using _get_size()
        # since only smaller edge of the image will be matched in this class.
        ow, oh = resized_img.size
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, None, ratio_local)
        return resized_img, transf, ratio
    
    
@PIPELINES_WITH_INFO.register_module
class RandomResizedCrop(_transforms.RandomResizedCrop):
    def with_trans_info(self, img, transf, ratio):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        
        oh, ow = _get_size(self.size)
        transf_local = [i, j, h, w]
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, ratio_local)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return img, transf, ratio


@PIPELINES_WITH_INFO.register_module
class RandomHorizontalFlip(_transforms.RandomHorizontalFlip):
    def with_trans_info(self, img, transf, ratio):
        if torch.rand(1) < self.p:
            transf.append(True)
            return F.hflip(img), transf, ratio
        transf.append(False)
        return img, transf, ratio


@PIPELINES_WITH_INFO.register_module
class RandomOrder(_transforms.RandomOrder):
    def with_trans_info(self, img, transf, ratio):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            t = _with_trans_info(self.transforms[i])
            img, transf, ratio = t(img, transf, ratio)
        return img, transf, ratio


@PIPELINES_WITH_INFO.register_module
class RandomApply(_transforms.RandomApply):
    def with_trans_info(self, img, transf, ratio):
        if self.p < random.random():
            return img, transf, ratio
        for t in self.transforms:
            t = _with_trans_info(t)
            img, transf, ratio = t(img, transf, ratio)
        return img, transf, ratio
