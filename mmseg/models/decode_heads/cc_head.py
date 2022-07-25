# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from ..builder import HEADS
from .fcn_head import FCNHead

try:
    from mmcv.ops import CrissCrossAttention
except ModuleNotFoundError:
    CrissCrossAttention = None


@HEADS.register_module()
class CCHead(FCNHead):
    """CCNet: Criss-Cross Attention for Semantic Segmentation.

    This head is the implementation of `CCNet
    <https://arxiv.org/abs/1811.11721>`_.

    Args:
        recurrence (int): Number of recurrence of Criss Cross Attention
            module. Default: 2.
    """

    def __init__(self, recurrence=2, **kwargs):
        if CrissCrossAttention is None:
            raise RuntimeError('Please install mmcv-full for '
                               'CrissCrossAttention ops')
        super(CCHead, self).__init__(num_convs=2, **kwargs)
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(self.channels)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        for _ in range(self.recurrence):
            output = self.cca(output)
        output = self.convs[1](output)
        print('Output shape after cca and selfconv', output.shape)
        mean = 0
        var = .1
        sigma = var ** 0.5

        #noise = np.random.normal(mean, sigma, output[0][1].shape)
        for index1 in range(output.shape[0]):
            for index2 in range(output.shape[1]):
                noise = np.random.normal(mean, sigma, output[index1][index2].shape)
                output[index1][index2] = output[index1][index2] + noise
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        #print('Output shape after cca,selfconv and cls_seg', output.shape)
        
        return output
