# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from numpy import ndarray
from torch import Tensor

from mmdet.registry import TASK_UTILS
from ..assigners import AssignResult
from .base_sampler import BaseSampler


@TASK_UTILS.register_module()
class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): 样本总数
        pos_fraction (float): 正样本比例
        neg_pos_up (int): pos/neg的上限. 默认-1,即无上限.
        add_gt_as_proposals (bool): 是否将gt box添加进roi.
    """

    def __init__(self,
                 num: int,
                 pos_fraction: float,
                 neg_pos_ub: int = -1,
                 add_gt_as_proposals: bool = True,
                 **kwargs):
        from .sampling_result import ensure_rng
        super().__init__(
            num=num,
            pos_fraction=pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals)
        self.rng = ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery: Union[Tensor, ndarray, list],
                      num: int) -> Union[Tensor, ndarray]:
        """从样本中随机抽取指定数量样本.返回值类型与输入样本保持一致.

        Args:
            gallery (Tensor | ndarray | list): 样本池.
            num (int): 抽取样本数.

        Returns:
            Tensor or ndarray: 采样后的索引.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        # PyTorch的torch.randperm操作在GPU上偶尔会产生bug(生成非常大的数值),这是一个临时解决方案.
        # See: https://github.com/open-mmlab/mmdetection/pull/5014
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result: AssignResult, num_expected: int,
                    **kwargs) -> Union[Tensor, ndarray]:
        """Randomly sample some positive samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result: AssignResult, num_expected: int,
                    **kwargs) -> Union[Tensor, ndarray]:
        """Randomly sample some negative samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
