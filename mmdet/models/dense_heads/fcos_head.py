# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from ..utils import multi_apply
from .anchor_free_head import AnchorFreeHead

INF = 1e8


@MODELS.register_module()
class FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    FCOS head 不使用anchor. 它在每一个特征点上预测box,同时centerness 用于抑制低质量的预测.
    这里 norm_on_bbox、centerness_on_reg、dcn_on_last_conv 是官方代码 中使用的训练技巧,
    这将带来高达 4.9 的显著mAP增益. 详情参考 https://github.com/tianzhi0549/FCOS.

    Args:
        num_classes (int): 不包括背景类别的类别数.
        in_channels (int): 输入特征图中的通道数.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): 各层级特征图上的下采样倍数. 
            默认: (4, 8, 16, 32, 64),对应FPN中start_level=0.
        regress_ranges (Sequence[Tuple[int, int]]): 各层级特征点的回归范围.
        center_sampling (bool): 是否使用中心采样.
        center_sample_radius (float): 中心采样半径. Default: 1.5, 单位:stride
        norm_on_bbox (bool):  是否使用下采样倍数对回归目标进行归一化.以缩小不同层级间的差距.
        centerness_on_reg (bool): 如果为true, 将 centerness 放置在回归分支.
            这种简单的修改可为FCOS-r101带来0.5 mAP增益
            参考 https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
        conv_bias (bool or str): 如果值为 `auto`,将由 norm_cfg 决定. 如果 `norm_cfg` 为 None,
            则 conv 的偏差将设置为 True, 否则False.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness
            loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 center_sampling: bool = False,
                 center_sample_radius: float = 1.5,
                 norm_on_bbox: bool = False,
                 centerness_on_reg: bool = False,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = MODELS.build(loss_centerness)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): [[bs, c, h, w], ] * nl.

        Returns:
            tuple:
                cls_scores (list[Tensor]): 所有层级的cls_scores,
                    [bs, nc, h, w] * nl.
                bbox_preds (list[Tensor]): 所有层级的box_reg,
                    [bs, 4, h, w] * nl.
                centernesses (list[Tensor]): 所有层级的centernesses,
                    [bs, 1, h, w] * nl.
        """
        return multi_apply(self.forward_single, x, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): 特定层级的输入特征.
            scale (:obj:`mmcv.cnn.Scale`): 用于调整 reg 大小的可学习比例模块.
            stride (int): 当前特征图的对应下采样倍数, 仅当 self.norm_on_bbox 为 True
                时用于norm化 bbox 预测.

        Returns:
            tuple: scores for each class, bbox predictions and centerness
            predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # 对不同层级输出的reg,进行不同的scale缩放,以避免在启用FP16时溢出
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # 使用 PyTorch 1.10 运行时,梯度计算所需的 bbox_pred 会被
            # F.relu(bbox_pred)修改. 所以使用bbox_pred.clamp(min=0)来替换此操作
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """计算head部分的loss.FCOS在每个特征点上生成的先验点个数为num_prior->np.
        在后续代码注释中为方便起见,令np=1带入计算各变量shape.

        Args:
            cls_scores (list[Tensor]): 所有层级的cls_scores,
                [[bs, np * nc, h, w],] * nl
            bbox_preds (list[Tensor]): 所有层级的box_reg,
                [[bs, np * 4, h, w],] * nl
            centernesses (list[Tensor]): 所有层级的box_obj,
                [[bs, np * 1, h, w],] * nl
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # [[h * w, 2], ] * nl, with_stride为True时为 2 -> 4, 后续以2为例
        # 以及这些prior中心处于grid右下角偏移stride_w/2, stride_h/2的位置
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # [[bs, ], ] * nl, [[bs, 4], ] * nl
        # 在cls_target的生成过程中,仅在gt box内部的先验点才为正样本,
        # 那些超出gt box范围或者reg target 超出对应层级回归范围的先验点皆为负样本
        labels, bbox_targets = self.get_targets(all_level_points,
                                                batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        # [[bs,nc,h,w],] * nl -> [[bs,h,w,nc],] * nl -> [[bs*h*w,nc],] * nl
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        # [[bs, 4, h, w],] * nl -> [[bs, h, w, 4],] * nl -> [[bs*h*w, 4],] * nl
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        # [[bs, 1, h, w],] * nl -> [[bs, h, w, 1],] * nl -> [[bs*h*w,],] * nl
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)  # [nl*bs*h*w, nc]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)  # [nl*bs*h*w, 4]
        flatten_centerness = torch.cat(flatten_centerness)  # [nl*bs*h*w,]
        flatten_labels = torch.cat(labels)  # [nl*bs*h*w,]
        flatten_bbox_targets = torch.cat(bbox_targets)  # [nl*bs*h*w, 4]
        # 执行repeat操作以和上面几个变量在维度上对齐
        # [[h * w, 2], ] * nl -> [[bs * h * w, 2], ] * nl -> [nl * bs * h * w, 2]
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # 前景: [0, num_classes), 背景: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        # 既是centerness_target, 又是reg_weight
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        # 1.最终是以IOU loss来计算回归损失的,在计算reg loss的平均因子时,
        # 往往一般除以正或(正+负)样本数量但这里取了个gt box内部范围内所有
        # 热力值(中心为1向四周递减至0)之和作为平均因子.也算是一种正样本数量总和?
        # 只是不同位置上的正样本所占权重不同,越向gt box中心权重占比越大
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_targets(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): 所有层级上的prior, [[h*w, 2],] * nl.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): [[bs * h * w, ], ] * nl.
            - concat_lvl_bbox_targets (list[Tensor]): [[bs * h * w, 4], ] * nl.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # 广播regress_ranges以与先验点对齐 [2,] -> [1, 2] -> [(h*w, 2),] * num_levels
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # cat 所有层级上的先验点以及回归范围
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # 各层级上的先验点数量
        num_points = [center.size(0) for center in points]

        # 获取batch幅图像的cls_target和reg_target,前两个变量是随图像变化的,后三个则是固定的.
        # [[num_level * h * w,] * bs] [[num_level * h * w, 4] * bs]
        labels_list, bbox_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)
        # 将labels_list分割, [tuple(h * w, ) * num_level, ] * bs.下同
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # 将不同图像上的同层级cls_target/reg_target合并到一起
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor]:
        """计算单张图像上的reg和cls目标.

        Args:
            gt_instances (:obj:`InstanceData`): It usually includes
            ``bboxes`` and ``labels``attributes.
            points (Tensor): [nl * h * w, 2] 所有层级上的prior中心点坐标
            regress_ranges (Tensor): [nl * h * w, 2] 所有层级上的prior回归范围
            num_points_per_lvl (list[Tensor]): [h * w, ] * nl 各层级上的prior数量
        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): cls target.[nl * h * w,]
                concat_lvl_bbox_targets (list[Tensor]): reg target.
                [nl * h * w, 4], 返回reg的原因是,如果开启norm_on_box
                则需要对reg值进行放缩,而放缩的原因是不同层级上的reg值不在一个范围
        """
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:  # 当图像上无目标时,默认cls/reg的target全为背景类/零,
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))
        # gt_bboxes对应的面积
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different, 可能expand仅仅是引用,而非真的复制.
        # areas = areas[None].expand(num_points, num_gts)
        # [num_gts,] -> [1, num_gts] -> [num_level * h * w, num_gts]
        areas = areas[None].repeat(num_points, 1)
        # [num_level*h*w, 2] -> [num_level*h*w, 1, 2] -> [num_level*h*w, num_gts, 2]
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        # 所有gt box的左右上下边界到所有prior的中心点的距离,距离可正可负.
        # 如果在最后一维度任一距离值<0, 则代表该prior处于该gt box外部
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        # [num_level * h * w, num_gts, 4] 所有prior中心到所有gt box边界的距离
        # 因为fcos的回归目标就是point坐标点到四个边界的距离,所以称为bbox_targets
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # 条件1: 在一个`center bbox`(以gt box中心为中心的方形区域)里面设置正样本
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)  # [num_level*h*w, num_gts, 4]
            stride = center_xs.new_zeros(center_xs.shape)  # [num_level*h*w, num_gts]

            # project the points on current lvl back to the `original` sizes
            # 在不同层级上的stride,其值也应不同.以方便后续计算出gt box在不同层级上的`center bbox`
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            # gt box 在各层级上的正样本范围不再是gt box 内部,而是其中心坐标向外扩张
            # center_sample_radius个对应层级的stride坐标单位形成的矩形区域
            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            # 限制正样本范围不得超出原始gt box区域
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)
            # 让(受限制的矩形区域)的prior坐标作为正样本区域
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # 条件1: prior中心处于gt box内部就算作正样本
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # 条件2: 计算出prior中心到gt box边界的最大距离.并获取最大距离在回归范围内的mask
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # 1.非正样本区域的area都设置为无限大
        # 2.和gt box边界的最大距离超出回归范围的prior的area也设置为无限大
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        # 单个prior与多个gt box 的最小area, 对应的gt 索引 二者shape -> [num_level*h*w,]
        min_area, min_area_inds = areas.min(dim=1)

        # 如果一个prior匹配多个gt box,则选择面积最小的一个作为该prior的target
        labels = gt_labels[min_area_inds]
        # 当某个prior与所有gt box的内部区域都不存在交集或者超出回归范围时.
        # 上面的流程会强制给予该prior某个gt box的label.下面一行代码就是为了防止该情况发生.
        labels[min_area == INF] = self.num_classes
        # bbox_targets也是利用labels的值来忽略计算背景类的box_target.
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
        """计算 centerness targets.它作为正样本权重的原因是,计算左右两边距离的
            最小值除以最大值,如果该结果越大代表两边距离越接近也就是说该prior越接近gt中心,
            自然该prior的权重就越大接近于1,反之就越小接近于0.上下边同理.

        Args:
            pos_bbox_targets (Tensor): 正样本的reg target -> [num_pos, 4]

        Returns:
            Tensor: Centerness target -> (num_pos,).
        """
        # 只计算正样本的 centerness targets, 否则可能会出现nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
