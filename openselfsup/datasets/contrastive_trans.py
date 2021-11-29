from numpy import isin
import torch
from PIL import Image
from .registry import DATASETS, PIPELINES, PIPELINES_WITH_INFO
from .base import BaseDataset
from .utils import to_numpy
from torch.utils.data import Dataset
from openselfsup.utils import print_log, build_from_cfg
from openselfsup.datasets.pipelines.transform_utils import Compose
from .builder import build_datasource
import torchvision.transforms.functional as F

@DATASETS.register_module
class ContrastiveDatasetTrans(Dataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, resized_size=(224,224), prefetch=False, with_trans_info=True):
        data_source['return_label'] = False
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES_WITH_INFO) for p in pipeline]
        self.pipeline = Compose(pipeline, with_trans_info=with_trans_info)
        self.prefetch = prefetch
        self.resized_size = resized_size
        
        img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor_and_normalized = Compose(
            [build_from_cfg(p, PIPELINES) for p in 
                [dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)]]
        )

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        view1 = self.pipeline(img)
        view2 = self.pipeline(img)
        
        if isinstance(view1, Image.Image):
            if self.prefetch:
                img1 = torch.from_numpy(to_numpy(view1))
                img2 = torch.from_numpy(to_numpy(view2))
            img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
            return dict(img=img_cat)
        else:
            # with transformation information.
            img1, transf1, ratio1, size1 = view1.image, view1.transf, view1.ratio, view1.size
            img2, transf2, ratio2, size2 = view2.image, view2.transf, view2.ratio, view2.size

            int_l = min(transf1[1], transf2[1])
            int_r = max(transf1[1] + transf1[3], transf2[1] + transf2[3])
            int_t = min(transf1[0], transf2[0])
            int_b = max(transf1[0] + transf1[2], transf2[0] + transf2[2])

            i = int_t
            j = int_l
            w = int_r - int_l
            h = int_b - int_t

            img_reference = F.resized_crop(img, i, j, h, w, self.resized_size, Image.BICUBIC)

            if self.prefetch:
                img1 = torch.from_numpy(to_numpy(img1))
                img2 = torch.from_numpy(to_numpy(img2))
                img_reference = torch.from_numpy(to_numpy(img_reference))
            
            else:
                img_reference = self.to_tensor_and_normalized(img_reference)

            img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0), img_reference.unsqueeze(0)), dim=0)

            view_1_crop_range_and_flip = [
                (transf1[0] - i)/h, # top
                (transf1[1] - j)/w, # left
                (transf1[2]+transf1[0] - i)/h, # bottom
                (transf1[3]+transf1[1] - j)/w, # right
                float(transf1[4])
            ]

            view_2_crop_range_and_flip = [
                (transf2[0] - i)/h,
                (transf2[1] - j)/w,
                (transf2[2]+transf2[0] - i)/h,
                (transf2[3]+transf2[1] - j)/w,
                float(transf2[4])
            ]

            return dict(img=img_cat, 
                        transf=[view_1_crop_range_and_flip, view_2_crop_range_and_flip])      




    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
