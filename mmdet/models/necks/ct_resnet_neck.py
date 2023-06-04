# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig


@MODELS.register_module()
class CTResNetNeck(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.
    大概流程是,conv+up_conv -> conv+up_conv -> conv+up_conv
    Args:
         in_channels (int): 输入维度.
         num_deconv_filters (tuple[int]): 每个stage卷积核的输出维度.
         num_deconv_kernels (tuple[int]): 每个stage卷积核大小.
         use_dcn (bool): 是否应用DCNv2.
         init_cfg (:obj:`ConfigDict` or dict or list[dict] or
             list[:obj:`ConfigDict`], optional): 初始化配置字典.
    """

    def __init__(self,
                 in_channels: int,
                 num_deconv_filters: Tuple[int, ...],
                 num_deconv_kernels: Tuple[int, ...],
                 use_dcn: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channels = in_channels
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters,
                                                     num_deconv_kernels)

    def _make_deconv_layer(
            self, num_deconv_filters: Tuple[int, ...],
            num_deconv_kernels: Tuple[int, ...]) -> nn.Sequential:
        """使用反卷积对backbone的输出进行上采样."""
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channels = num_deconv_filters[i]
            conv_module = ConvModule(
                self.in_channels,
                feat_channels,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=dict(type='BN'))
            layers.append(conv_module)
            upsample_module = ConvModule(
                feat_channels,
                feat_channels,
                num_deconv_kernels[i],
                stride=2,
                padding=1,
                conv_cfg=dict(type='deconv'),
                norm_cfg=dict(type='BN'))
            layers.append(upsample_module)
            self.in_channels = feat_channels

        return nn.Sequential(*layers)

    def init_weights(self) -> None:
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def forward(self, x: Sequence[torch.Tensor]) -> Tuple[torch.Tensor]:
        """model forward."""
        assert isinstance(x, (list, tuple))
        outs = self.deconv_layers(x[-1])
        return outs,
