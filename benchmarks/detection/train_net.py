#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.data.datasets import register_coco_instances

import argparse
import sys

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """

    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # if "coco" in dataset_name or "pannuke" in dataset_name or "GlaS" in dataset_name:
        #     return COCOEvaluator(dataset_name, cfg, True, output_folder)
        # else:
        #     assert "voc" in dataset_name
        #     return PascalVOCDetectionEvaluator(dataset_name)
        if "voc" in dataset_name:
            return PascalVOCDetectionEvaluator(dataset_name)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    
    ##### register datasets #####
    # GlaS
    for fold in ['train', 'test']:
        register_coco_instances(
            f"GlaS_{fold}",
            {},
            f"datasets/GlaS/annotations/{fold}.json",
            f"datasets/GlaS/{fold}"
        )
    # CRAG
    for fold in ['train', 'test']:
        register_coco_instances(
            f"CRAG_{fold}",
            {},
            f"datasets/CRAG/annotations/{fold}.json",
            f"datasets/CRAG/{fold}"
        )
    ##############################
    
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(
            model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    if len(cfg.MODEL.WEIGHTS) > 0:
        trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    ####
    epilog = None
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
        Examples:

        Run on single machine:
            $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

        Change some config options:
            $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

        Run on multiple machines:
            (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument("--dev", action="store_true", help='debug_mode')
    parser.add_argument(
        "opts",
        help="""
                Modify config options at the end of the command. For Yacs configs, use
                space-separated "PATH.KEY VALUE" pairs.
                For python-based LazyConfig, use "path.key=value".""".strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    print("Command Line Args:", args)
    if args.dev:
        args.opts[3]='1'
        args.opts.extend(
            [
             'INPUT.MAX_SIZE_TRAIN', '400',
             'INPUT.MIN_SIZE_TRAIN', (224,),
             'INPUT.MAX_SIZE_TEST', '400',
             'INPUT.MIN_SIZE_TEST', 224,
            ]
        )

    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
