# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weight_reduce_loss, weighted_loss


@weighted_loss
def gaussian_focal_loss(pred: Tensor,
                        gaussian_target: Tensor,
                        alpha: float = 2.0,
                        gamma: float = 4.0,
                        pos_weight: float = 1.0,
                        neg_weight: float = 1.0) -> Tensor:
    """Focal Loss <https://arxiv.org/abs/1708.02002>`_的变体. 目标为高斯分布的浮点值.

    Args:
        pred (torch.Tensor): 网络输出值.
        gaussian_target (torch.Tensor): 网络输出值的拟合目标(高斯分布).
        alpha (float, optional): Focal Loss 中的平衡难易样本参数.
        gamma (float, optional): 调节负样本loss权重的gamma参数.
        pos_weight(float): Positive sample loss weight.
        neg_weight(float): Negative sample loss weight.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)  # 正样本权重,仅在gt box出现的地方为1
    neg_weights = (1 - gaussian_target).pow(gamma)  # 负(非正)样本权重,额外添加了一个指数参数
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_weight * pos_loss + neg_weight * neg_loss


def gaussian_focal_loss_with_pos_inds(
        pred: Tensor,
        gaussian_target: Tensor,
        pos_inds: Tensor,
        pos_labels: Tensor,
        alpha: float = 2.0,
        gamma: float = 4.0,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        reduction: str = 'mean',
        avg_factor: Optional[Union[int, float]] = None) -> Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Note: The index with a value of 1 in ``gaussian_target`` in the
    ``gaussian_focal_loss`` function is a positive sample, but in
    ``gaussian_focal_loss_with_pos_inds`` the positive sample is passed
    in through the ``pos_inds`` parameter.

    Args:
        pred (torch.Tensor): The prediction. The shape is (N, num_classes).
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution. The shape is (N, num_classes).
        pos_inds (torch.Tensor): The positive sample index.
            The shape is (M, ).
        pos_labels (torch.Tensor): The label corresponding to the positive
            sample index. The shape is (M, ).
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to 'mean`.
        avg_factor (int, float, optional): Average factor that is used to
            average the loss. Defaults to None.
    """
    eps = 1e-12
    neg_weights = (1 - gaussian_target).pow(gamma)

    pos_pred_pix = pred[pos_inds]
    pos_pred = pos_pred_pix.gather(1, pos_labels.unsqueeze(1))
    pos_loss = -(pos_pred + eps).log() * (1 - pos_pred).pow(alpha)
    pos_loss = weight_reduce_loss(pos_loss, None, reduction, avg_factor)

    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    neg_loss = weight_reduce_loss(neg_loss, None, reduction, avg_factor)

    return pos_weight * pos_loss + neg_weight * neg_loss


@MODELS.register_module()
class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss是Focal Loss的一种变体.

    详情参考`paper<https://arxiv.org/abs/1808.01244>`_
    代码修改自 `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    注意,GaussianFocalLoss 中的label target是高斯分布的热力图,而非 0/1 int型值.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """

    def __init__(self,
                 alpha: float = 2.0,
                 gamma: float = 4.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                pos_inds: Optional[Tensor] = None,
                pos_labels: Optional[Tensor] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[Union[int, float]] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        If you want to manually determine which positions are
        positive samples, you can set the pos_index and pos_label
        parameter. Currently, only the CenterNet update version uses
        the parameter.

        Args:
            pred (torch.Tensor): 网络输出值的拟合目标(高斯分布). [N, nc].
            target (torch.Tensor): 网络输出值的拟合目标(高斯分布). [N, nc].
            pos_inds (torch.Tensor): The positive sample index.
                Defaults to None.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index. Defaults to None.
            weight (torch.Tensor, optional): 每个输出值的loss权重.
            avg_factor (int, float, optional): 用于平均loss的平均因子(一般为正样本个数)
            reduction_override (str, optional): 用于覆盖Loss类初始化中的self.reduction.
                默认为None,表示不覆盖.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if pos_inds is not None:
            assert pos_labels is not None
            # Only used by centernet update version
            loss_reg = self.loss_weight * gaussian_focal_loss_with_pos_inds(
                pred,
                target,
                pos_inds,
                pos_labels,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            loss_reg = self.loss_weight * gaussian_focal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                reduction=reduction,
                avg_factor=avg_factor)
        return loss_reg
