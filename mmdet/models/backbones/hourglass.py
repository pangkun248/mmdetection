# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from ..layers import ResLayer
from .resnet import BasicBlock


class HourglassModule(BaseModule):
    """HourglassNet的Hourglass模块.

    递归生成模块并使用 res18/34中的BasicBlock 作为基本单元.

    Args:
        depth (int): 当前 HourglassModule 的深度.
        stage_channels (list[int]): 当前和后续 HourglassModule 中子模块的特征通道.
        stage_blocks (list[int]): 当前和后续 HourglassModule 中堆叠的子模块数量.
        norm_cfg (dict): 构造和配置norm层的字典.
        upsample_cfg (dict, optional): 插值层的配置字典.默认: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): 初始化配置字典.默认: None
    """

    def __init__(self,
                 depth: int,
                 stage_channels: List[int],
                 stage_blocks: List[int],
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 upsample_cfg: ConfigType = dict(mode='nearest'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)

        self.depth = depth

        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = ResLayer(
            BasicBlock, cur_channel, cur_channel, cur_block, norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            cur_channel,
            next_channel,
            cur_block,
            stride=2,
            norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassModule(depth - 1, stage_channels[1:],
                                        stage_blocks[1:])
        else:
            self.low2 = ResLayer(
                BasicBlock,
                next_channel,
                next_channel,
                next_block,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            next_channel,
            cur_channel,
            cur_block,
            norm_cfg=norm_cfg,
            downsample_first=False)

        self.up2 = F.interpolate
        self.upsample_cfg = upsample_cfg

    def forward(self, x: torch.Tensor) -> nn.Module:
        """Forward function."""
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        # Fixing `scale factor` (e.g. 2) is common for upsampling, but
        # in some cases the spatial size is mismatched and error will arise.
        if 'scale_factor' in self.upsample_cfg:
            up2 = self.up2(low3, **self.upsample_cfg)
        else:
            shape = up1.shape[2:]
            up2 = self.up2(low3, size=shape, **self.upsample_cfg)
        return up1 + up2


@MODELS.register_module()
class HourglassNet(BaseModule):
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`_ .

    Args:
        downsample_times (int): HourglassModule 中的下采样次数.
        num_stacks (int): 堆叠的 HourglassModule 模块数量,
            1:Hourglass-52, 2:Hourglass-104.
        stage_channels (Sequence[int]): HourglassModule 中每个子模块的特征通道.
        stage_blocks (Sequence[int]): HourglassModule 中堆叠的子模块数量.
        feat_channel (int): HourglassModule 后卷积的特征通道.
        norm_cfg (dict): 构造和配置norm层的字典.
        init_cfg (dict or ConfigDict, optional): 权重初始化配置字典.默认: None

    Example:
        >>> from mmdet.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 downsample_times: int = 5,
                 num_stacks: int = 2,
                 stage_channels: Sequence = (256, 256, 384, 384, 384, 512),
                 stage_blocks: Sequence = (2, 2, 2, 2, 2, 4),
                 feat_channel: int = 256,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 init_cfg: OptMultiConfig = None) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        cur_channel = stage_channels[0]

        self.stem = nn.Sequential(
            ConvModule(
                3, cur_channel // 2, 7, padding=3, stride=2,
                norm_cfg=norm_cfg),
            ResLayer(
                BasicBlock,
                cur_channel // 2,
                cur_channel,
                1,
                stride=2,
                norm_cfg=norm_cfg))

        self.hourglass_modules = nn.ModuleList([
            HourglassModule(downsample_times, stage_channels, stage_blocks)
            for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock,
            cur_channel,
            cur_channel,
            num_stacks - 1,
            norm_cfg=norm_cfg)

        self.conv1x1s = nn.ModuleList([
            ConvModule(
                cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.out_convs = nn.ModuleList([
            ConvModule(
                cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(num_stacks)
        ])

        self.remap_convs = nn.ModuleList([
            ConvModule(
                feat_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self) -> None:
        """Init module weights."""
        # Training Centripetal Model needs to reset parameters for Conv2d
        super().init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward function."""
        inter_feat = self.stem(x)
        out_feats = []

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]

            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)

            if ind < self.num_stacks - 1:
                inter_feat = self.conv1x1s[ind](
                    inter_feat) + self.remap_convs[ind](
                        out_feat)
                inter_feat = self.inters[ind](self.relu(inter_feat))
        # num_stacks=2时,[[bs, 256, h/4, w/4], [bs, 256, h/4, w/4]]
        return out_feats
