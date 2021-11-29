from math import cos, pi
from mmcv.runner import Hook
from mmcv.parallel import is_module_wrapper

from .registry import HOOKS


@HOOKS.register_module
class ConCLHook(Hook):
    def __init__(self, warm_up_epoch=10, **kwargs):
        self.warm_up_epoch = warm_up_epoch

    def before_train_epoch(self, runner):
        if runner._epoch >= self.warm_up_epoch:
            runner.model.module.local_start = True

