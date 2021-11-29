
from typing import NamedTuple, List, Tuple
from functools import wraps

import torch
import torch.nn.functional as F
import random
from torchvision.transforms import (Resize, CenterCrop, RandomHorizontalFlip,
                                    ColorJitter, RandomGrayscale, ToTensor)
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import ImageOps

# borrow from https://github.com/kakaobrain/scrl/blob/master/augment/transforms.py
# Spatially Consistent Representation Learning: https://arxiv.org/abs/2103.06122


class ImageWithTransInfo(NamedTuple):
    """to improve readability"""
    image: torch.Tensor  # image
    transf: List         # cropping coord. in the original image + flipped or not
    ratio: List          # resizing ratio w.r.t. the original image
    size: List           # size (width, height) of the original image


def free_pass_trans_info(func):
    """Wrapper to make the function bypass the second argument(transf)."""
    @wraps(func)
    def decorator(img, transf, ratio):
        return func(img), transf, ratio
    return decorator


def _with_trans_info(transform):
    """use with_trans_info function if possible, or wrap original __call__."""
    if hasattr(transform, 'with_trans_info'):
        transform = transform.with_trans_info
    else:
        transform = free_pass_trans_info(transform)
    return transform


def _get_size(size):
    if isinstance(size, int):
        oh, ow = size, size
    else:
        oh, ow = size
    return oh, ow


def _update_transf_and_ratio(transf_global, ratio_global,
                             transf_local=None, ratio_local=None):
    if transf_local:
        i_global, j_global, *_ = transf_global
        i_local, j_local, h_local, w_local = transf_local
        i = int(round(i_local / ratio_global[0] + i_global))
        j = int(round(j_local / ratio_global[1] + j_global))
        h = int(round(h_local / ratio_global[0]))
        w = int(round(w_local / ratio_global[1]))
        transf_global = [i, j, h, w]

    if ratio_local:
        ratio_global = [g * l for g, l in zip(ratio_global, ratio_local)]

    return transf_global, ratio_global


class Compose(object):
    def __init__(self, transforms, with_trans_info=False, seed=None):
        self.transforms = transforms
        self.with_trans_info = with_trans_info
        self.seed = seed
        
    @property
    def with_trans_info(self):
        return self._with_trans_info
    
    @with_trans_info.setter
    def with_trans_info(self, value):
        self._with_trans_info = value

    def __call__(self, *args, **kwargs):
        if self.with_trans_info:
            return self._call_with_trans_info(*args, **kwargs)
        return self._call_default(*args, **kwargs)

    def _call_default(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def _call_with_trans_info(self, img):
        w, h = img.size
        transf = [0, 0, h, w]
        ratio = [1., 1.]

        for t in self.transforms:
            t = _with_trans_info(t)
            try:
                if self.seed:
                    random.seed(self.seed)
                    torch.manual_seed(self.seed)
                img, transf, ratio = t(img, transf, ratio)
            except Exception as e:
                raise Exception(f'{e}: from {t.__self__}')

        return ImageWithTransInfo(img, transf, ratio, (h, w))


def decompose_collated_batch(collated_batch_transf):
    batch_transf = []
    # collated_batch_transf: List
    # len(collated_batch_transf) = 2 (two views)
    # View1: list of length 5, each list contains a tensor of size batch_size.
    # 5 components: h_start, w_start, h, w, flipped?
    for view_transf in collated_batch_transf:
        # step1: concatnate all view-specific transformations (5 x batch_size, 1)
        # step2: reshape them to (5,batch_size)
        transf = torch.cat(view_transf).reshape(len(view_transf), len(view_transf[0]))
        # step3: transpose them to (batch_size, 5)
        transf = torch.transpose(transf, 1, 0)
        batch_transf.append(transf)

    # transformations of two views:
    #       [ tensor of (batch_size, 5), tensor of (batch_size, 5)]
    return batch_transf

def decompose_collated_batch_boxes(collated_batch_boxes):
    batch_boxes = []
    for view_bboxes in collated_batch_boxes:
        bboxes = view_bboxes.view(-1, 4)
        bixs = torch.arange(view_bboxes.size(0), dtype=torch.float32).repeat_interleave(view_bboxes.size(1))
        bixs = bixs.view(-1,1).cuda()
        bboxes = torch.cat([bixs, bboxes], dim=-1)
        batch_boxes.append(bboxes)

    return batch_boxes
