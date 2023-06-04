# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.utils import is_tuple_of
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import HorizontalBoxes

DeviceType = Union[str, torch.device]


@TASK_UTILS.register_module()
class AnchorGenerator:
    """用于 2D anchor-based 的检测网络的标准anchor生成器.

    Args:
        strides (list[int] | list[tuple[int, int]]): 按(w, h)顺序在多个特征级别中的anchor的下采样倍数.
        ratios (list[float]): 在单个层级上的anchor的高宽比列表.
        scales (list[int], Optional): 单个层级中的不同大小的基础anchor尺寸.
            它不能与 (`octave_base_scale` 和 `scales_per_octave`)同时设置.
        base_sizes (list[int], Optional): 多层级anchor的基本尺寸.
            如果没有给出,将用strides代替.(如果stride在w和h方向上不一致,则采用最小的stride.)
        scale_major (bool): 生成base anchors时是否对同一列中的anchor将具有相同的scale.
            为False时会导致同一行中的anchor将具有相同的scale.(MMDet V2.0开始默认为 True)
        octave_base_scale (int, Optional): anchor的基础尺寸,需要和scales_per_octave配合strides使用.
        scales_per_octave (int, Optional): 单个层级下anchor的不同尺寸个数.
            `octave_base_scale` and `scales_per_octave` 通常在retinanet中使用
            同时在这两个参数被设置时, 参数`scales` 必须为 None.
        centers (list[tuple[float, float]], Optional): 多个级别特征中anchor中心相对于grid左上角的偏移
            默认情况下,它设置为 None 并且不使用. 如果给出一个浮点元组列表,它们将用于anchor的中心偏移量.
        center_offset (float): 相对于grid左上角的anchor中心的偏移量.
            两者区别在于以下几点.
            centers                                             center_offset
            ∈[0,grid_size],单位是px,绝对值                        ∈[0,1],单位是anchor的基础尺寸(base_sizes),相对值
            与Strides长度呼应,因为它是每个层级上的偏移量              仅仅是一个float,代表所有层级上的偏移量
            与center_offset互斥,不能同时设置
        use_box_type (bool): Whether to warp anchors with the box type data
            structure. Defaults to False.

    Examples:
        >>> from mmdet.models.task_modules.
        ... prior_generators import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_priors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_priors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 ratios: List[float],
                 scales: Optional[List[int]] = None,
                 base_sizes: Optional[List[int]] = None,
                 scale_major: bool = True,
                 octave_base_scale: Optional[int] = None,
                 scales_per_octave: Optional[int] = None,
                 centers: Optional[List[Tuple[float, float]]] = None,
                 center_offset: float = 0.,
                 use_box_type: bool = False) -> None:
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, '当center_offset!=0时,center必须为None,' \
                                    f'实际centers={centers}.'
        if not (0 <= center_offset <= 1):
            raise ValueError(f'center_offset应在[0, 1]范围内,实际为{center_offset}.')
        if centers is not None:
            assert len(centers) == len(strides), \
                f'strides与centers的长度应一致,但实际为 {strides} 和 {centers}'

        # 计算anchor的基础尺寸
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            f'strides与base_sizes的长度应一致,但实际为{self.strides} 和 {self.base_sizes}'

        # 计算anchor的尺寸
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            '"scales" 和 "octave_base_scale 及 scales_per_octave" 不能同时设置'
        # self.scales要么由scales直接提供,要么由 scales_per_octave 和 octave_base_scale
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('"scales" 和 "octave_base_scale 及 scales_per_octave" 必须设置一个')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        # [[x1, y1, x2, y2] * num_scale * num_ratio, ] * nl
        self.base_anchors = self.gen_base_anchors()
        self.use_box_type = use_box_type

    @property
    def num_base_anchors(self) -> List[int]:
        """list[int]: 返回各个层级特征点上的基础anchor数量"""
        return self.num_base_priors

    @property
    def num_base_priors(self) -> List[int]:
        """list[int]: 返回各个层级特征点上的先验(anchor)数量"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self) -> int:
        """int: 基础anchor在生成时会需要特征图层数"""
        return len(self.strides)

    def gen_base_anchors(self) -> List[Tensor]:
        """生成基础anchor.

        Returns:
            list(torch.Tensor): 所有层级特征图上基于特征点的基础anchor,
                [[num_scale*num_ratio,4] * len(base_sizes)].
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size: Union[int, float],
                                      scales: Tensor,
                                      ratios: Tensor,
                                      center: Optional[Tuple[float]] = None) \
            -> Tensor:
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): anchor的基本尺寸.
            scales (torch.Tensor): anchor的不同大小尺寸系数,需要配合base_size使用.
            ratios (torch.Tensor): 单个层级上anchor的高宽比.
            center (tuple[float], optional): 单个级别特征中anchor中心相对于grid左上角的偏移,基于grid.

        Returns:
            torch.Tensor: 单个层级特征图上的anchor,[num_scale*num_ratio,4].
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # 使用浮点型anchor并且anchor的中心与特征点中心对齐
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs,
            x_center + 0.5 * ws, y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self,
                  x: Tensor,
                  y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor]:
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_priors(self,
                    featmap_sizes: List[Tuple],
                    dtype: torch.dtype = torch.float32,
                    device: DeviceType = 'cuda') -> List[Tensor]:
        """在多层级上生成anchor.

        Args:
            featmap_sizes (list[tuple]): 多层级上的特征图尺寸.[[h, w],] * num_level
            dtype (:obj:`torch.dtype`): anchor的数据类型.
            device (str | torch.device): 生成anchor的设备.

        Return:
            list[torch.Tensor]: 多层级上的anchor
            [[h * w * na, 4],] * num_level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: DeviceType = 'cuda') -> Tensor:
        """生成单层级特征图上的所有anchor.

        Note:
            此方法通常由方法“self.grid_priors”调用.

        Args:
            featmap_size (tuple[int, int]): 特征图的大小.
            level_idx (int): 特征图对应层别的索引.
            dtype (obj:`torch.dtype`): 生成数据的类型.默认``torch.float32``.
            device (str | torch.device): 在什么设备上生成数据.默认为 'cuda'.

        Returns:
            torch.Tensor: 单层级上的anchor, [h * w * na, 4].
        """

        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        # 获取特征图上所有grid坐标.直接使用torch.meshgrid的方式是否更简洁?
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)

        # 将bace_anchor -> [1,na,4] 以及grid坐标 -> [feat_h*feat_w,1,4].然后利用广播机制将二者
        # 相加得到单层级上的所有anchor坐标[h * w, na, 4],然后reshape为[h * w * na, 4]
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        if self.use_box_type:
            all_anchors = HorizontalBoxes(all_anchors)
        return all_anchors

    def sparse_priors(self,
                      prior_idxs: Tensor,
                      featmap_size: Tuple[int, int],
                      level_idx: int,
                      dtype: torch.dtype = torch.float32,
                      device: DeviceType = 'cuda') -> Tensor:
        """Generate sparse anchors according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int, int]): feature map size arrange as (h, w).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (str | torch.device): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 4), N should be equal to
                the length of ``prior_idxs``.
        """

        height, width = featmap_size
        num_base_anchors = self.num_base_anchors[level_idx]
        base_anchor_id = prior_idxs % num_base_anchors
        x = (prior_idxs //
             num_base_anchors) % width * self.strides[level_idx][0]
        y = (prior_idxs // width //
             num_base_anchors) % height * self.strides[level_idx][1]
        priors = torch.stack([x, y, x, y], 1).to(dtype).to(device) + \
            self.base_anchors[level_idx][base_anchor_id, :].to(device)

        return priors

    def valid_flags(self,
                    featmap_sizes: List[Tuple[int, int]],
                    pad_shape: Tuple,
                    device: DeviceType = 'cuda') -> List[Tensor]:
        """在多个层级的特征图上生成有效的anchor索引,它与anchor本身尺寸无关.
            在pipline中,Pad只负责将单一图像填充到指定尺寸(或者能被某一个整数整除),
            并没有将一个batch的图像统一到相同尺寸,这步操作是在...mmcv/parallel/collate.py
            中实现的,为了对齐最大尺寸.在Dataloader中的collate阶段会将较小图片进行padding,
            所以有些图片可能就会产生一些无意义像素.而这里就是为了将该无意义区域的anchor过滤掉的
        Args:
            featmap_sizes (list(tuple[int, int])): 多层级上的特征图尺寸.
                [[h, w],] * num_level
            pad_shape (tuple): pipline中Pad后的shape.
            device (str | torch.device): 在该设备上生成flags.

        Return:
            list(torch.Tensor): 多层特征图上有效anchor的mask
                [[h * w * na], ] * num_levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size: Tuple[int, int],
                                 valid_size: Tuple[int, int],
                                 num_base_anchors: int,
                                 device: DeviceType = 'cuda') -> Tensor:
        """在单层特征图中生成anchor的有效mask.

        Args:
            featmap_size (tuple[int]): 特征图的尺寸, (h, w).
            valid_size (tuple[int]): 特征图的有效尺寸.
            num_base_anchors (int): 单个特征点上基础anchor的数量.
            device (str | torch.device): mask将要生成的设备.默认为 'cuda'.

        Returns:
            torch.Tensor: 单层特征图上有效anchor的mask.[h * w * na, ].
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str


@TASK_UTILS.register_module()
class SSDAnchorGenerator(AnchorGenerator):
    """SSD的Anchor生成器.

    Args:
        strides (list[int]  | list[tuple[int, int]]): 多层级特征图的anchor步长.
        ratios (list[float]): 多层级上anchor的高宽比列表.
        min_sizes (list[float]): 多层级上的最小anchor尺寸.
        max_sizes (list[float]): 多层级上的最大anchor尺寸.
        basesize_ratio_range (tuple(float)): anchor的比例范围.
            未设置 min_sizes 和 max_sizes 时使用.
        input_size (int): 特征图的大小, 300 for SSD300, 512 for SSD512.
            在不设置 min_sizes 和 max_sizes 时使用.
        scale_major (bool): 生成base anchors时是否先乘尺度. 如果为True,
            同一行中的anchor将具有相同的比例. 在 SSD 中始终设置为 False.
        use_box_type (bool): Whether to warp anchors with the box type data
            structure. Defaults to False.
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 ratios: List[float],
                 min_sizes: Optional[List[float]] = None,
                 max_sizes: Optional[List[float]] = None,
                 basesize_ratio_range: Tuple[float] = (0.15, 0.9),
                 input_size: int = 300,
                 scale_major: bool = True,
                 use_box_type: bool = False) -> None:
        assert len(strides) == len(ratios)
        assert not (min_sizes is None) ^ (max_sizes is None)
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]

        if min_sizes is None and max_sizes is None:
            # 使用 hard code 生成SSD的anchors
            self.input_size = input_size
            assert is_tuple_of(basesize_ratio_range, float)
            self.basesize_ratio_range = basesize_ratio_range
            # 计算anchor长宽比和尺寸
            min_ratio, max_ratio = basesize_ratio_range
            min_ratio = int(min_ratio * 100)
            max_ratio = int(max_ratio * 100)
            # 这里使用np.floor是为了确保后续能添加self.num_levels - 1 个size
            step = int(np.floor(max_ratio - min_ratio) / (self.num_levels - 2))
            min_sizes = []  # [21, 45, 99, 153, 207, 261] ssd300默认配置
            max_sizes = []  # [45, 99, 153, 207, 261, 315] ssd300默认配置
            # range(15, 90, 18) * 3(ssd512时为5.12) -> [45, 99, 153, 207, 261]
            for ratio in range(int(min_ratio), int(max_ratio) + 1, step):
                min_sizes.append(int(self.input_size * ratio / 100))
                max_sizes.append(int(self.input_size * (ratio + step) / 100))
            if self.input_size == 300:
                if basesize_ratio_range[0] == 0.15:  # SSD300 COCO 因为COCO目标偏小
                    min_sizes.insert(0, int(self.input_size * 7 / 100))
                    max_sizes.insert(0, int(self.input_size * 15 / 100))
                elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                    min_sizes.insert(0, int(self.input_size * 10 / 100))
                    max_sizes.insert(0, int(self.input_size * 20 / 100))
                else:
                    raise ValueError(
                        '当input_size为300时, basesize_ratio_range[0]'
                        f'应为 0.15 或 0.2, 但实际为 {basesize_ratio_range[0]}.')
            elif self.input_size == 512:
                if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                    min_sizes.insert(0, int(self.input_size * 4 / 100))
                    max_sizes.insert(0, int(self.input_size * 10 / 100))
                elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                    min_sizes.insert(0, int(self.input_size * 7 / 100))
                    max_sizes.insert(0, int(self.input_size * 15 / 100))
                else:
                    raise ValueError(
                        '当未设置 min_sizes 和 max_sizes, 并且input_size为 512 时'
                        'basesize_ratio_range[0] 应该是0.1或0.15'
                        f'但实际为 {basesize_ratio_range[0]}.')
            else:
                raise ValueError(
                    '当不设置 min_sizes 和 max_sizes时,SSDAnchorGenerator'
                    f'仅支持 300 或 512 , 但实际为{self.input_size}.')

        assert len(min_sizes) == len(max_sizes) == len(strides)

        anchor_ratios = []
        anchor_scales = []
        for k in range(len(self.strides)):
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            anchor_ratio = [1.]
            for r in ratios[k]:
                anchor_ratio += [1 / r, r]  # 添加2 or 4 个ratio
            anchor_ratios.append(torch.Tensor(anchor_ratio))
            anchor_scales.append(torch.Tensor(scales))
        # anchor_scales, 代表每个层级上anchor的多种基础尺寸系数
        # [[1.0, 1.4639], [1.0, 1.4832], [1.0, 1.2432],
        # [1.0, 1.1632], [1.0, 1.1229], [1.0, 1.0986]]
        # ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        # anchor_ratios, 代表每个层级上anchor的多种比例
        # [[1.0, 0.5, 2.0], [1.0, 0.5, 2.0, 0.3333, 3.0],
        # [1.0, 0.5, 2.0, 0.3333, 3.0], [1.0, 0.5, 2.0, 0.3333, 3.0],
        # [1.0, 0.5, 2.0], [1.0, 0.5, 2.0]]
        # 注意!这些值仍需进行开方,作为高的系数与高相乘(取倒数作为宽的系数与宽相乘)
        # final_h = base_size*sqrt(ratio) ,final_w = base_size*1/sqrt(ratio)
        # 以使得最终面积与以base_size为边的正方形面积一致
        # final_h*final_w == base_size*base_size(忽略scale因素)
        self.base_sizes = min_sizes
        self.scales = anchor_scales
        self.ratios = anchor_ratios
        self.scale_major = scale_major
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()
        self.use_box_type = use_box_type

    def gen_base_anchors(self) -> List[Tensor]:
        """生成6个特征图上的以(stride/2,stride/2)为中心点的基础anchor.
        SSD的基础anchor生成策略中,并非是生成len(ratios)*len(scales)个anchor.
        而是针对ratio为1的不同scale(m个),scale为1的不同ratio(n个),m+n-1个anchor
        即理论上应生成 6, 10, 10, 10, 6, 6个anchor.实际生成4, 6, 6, 6, 4, 4个anchor
        Returns:
            list(torch.Tensor): 多层级特征图上的基本anchor.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            base_anchors = self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales[i],
                ratios=self.ratios[i],
                center=self.centers[i])
            indices = list(range(len(self.ratios[i])))
            indices.insert(1, len(indices))
            base_anchors = torch.index_select(base_anchors, 0,
                                              torch.LongTensor(indices))
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}input_size={self.input_size},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}basesize_ratio_range='
        repr_str += f'{self.basesize_ratio_range})'
        return repr_str


@TASK_UTILS.register_module()
class LegacyAnchorGenerator(AnchorGenerator):
    """Legacy anchor generator used in MMDetection V1.x.

    Note:
        Difference to the V2.0 anchor generator:

        1. The center offset of V1.x anchors are set to be 0.5 rather than 0.
        2. The width/height are minused by 1 when calculating the anchors' \
            centers and corners to meet the V1.x coordinate system.
        3. The anchors' corners are quantized.

    Args:
        strides (list[int] | list[tuple[int]]): Strides of anchors
            in multiple feature levels.
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int]): The basic sizes of anchors in multiple levels.
            If None is given, strides will be used to generate base_sizes.
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. It a list of float
            is given, this list will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0.5 in V2.0 but it should be 0.5
            in v1.x models.
        use_box_type (bool): Whether to warp anchors with the box type data
            structure. Defaults to False.

    Examples:
        >>> from mmdet.models.task_modules.
        ... prior_generators import LegacyAnchorGenerator
        >>> self = LegacyAnchorGenerator(
        >>>     [16], [1.], [1.], [9], center_offset=0.5)
        >>> all_anchors = self.grid_priors(((2, 2),), device='cpu')
        >>> print(all_anchors)
        [tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])]
    """

    def gen_single_level_base_anchors(self,
                                      base_size: Union[int, float],
                                      scales: Tensor,
                                      ratios: Tensor,
                                      center: Optional[Tuple[float]] = None) \
            -> Tensor:
        """Generate base anchors of a single level.

        Note:
            The width/height of anchors are minused by 1 when calculating \
                the centers and corners to meet the V1.x coordinate system.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between the height.
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature map.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * (w - 1)
            y_center = self.center_offset * (h - 1)
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * (ws - 1), y_center - 0.5 * (hs - 1),
            x_center + 0.5 * (ws - 1), y_center + 0.5 * (hs - 1)
        ]
        base_anchors = torch.stack(base_anchors, dim=-1).round()

        return base_anchors


@TASK_UTILS.register_module()
class LegacySSDAnchorGenerator(SSDAnchorGenerator, LegacyAnchorGenerator):
    """Legacy anchor generator used in MMDetection V1.x.

    The difference between `LegacySSDAnchorGenerator` and `SSDAnchorGenerator`
    can be found in `LegacyAnchorGenerator`.
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 ratios: List[float],
                 basesize_ratio_range: Tuple[float],
                 input_size: int = 300,
                 scale_major: bool = True,
                 use_box_type: bool = False) -> None:
        super(LegacySSDAnchorGenerator, self).__init__(
            strides=strides,
            ratios=ratios,
            basesize_ratio_range=basesize_ratio_range,
            input_size=input_size,
            scale_major=scale_major,
            use_box_type=use_box_type)
        self.centers = [((stride - 1) / 2., (stride - 1) / 2.)
                        for stride in strides]
        self.base_anchors = self.gen_base_anchors()


@TASK_UTILS.register_module()
class YOLOAnchorGenerator(AnchorGenerator):
    """Anchor generator for YOLO.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        base_sizes (list[list[tuple[int, int]]]): The basic sizes
            of anchors in multiple levels.
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 base_sizes: List[List[Tuple[int, int]]],
                 use_box_type: bool = False) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.base_sizes = []
        num_anchor_per_level = len(base_sizes[0])
        for base_sizes_per_level in base_sizes:
            assert num_anchor_per_level == len(base_sizes_per_level)
            self.base_sizes.append(
                [_pair(base_size) for base_size in base_sizes_per_level])
        self.base_anchors = self.gen_base_anchors()
        self.use_box_type = use_box_type

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.base_sizes)

    def gen_base_anchors(self) -> List[Tensor]:
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level,
                                                   center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_sizes_per_level: List[Tuple[int]],
                                      center: Optional[Tuple[float]] = None) \
            -> Tensor:
        """Generate base anchors of a single level.

        Args:
            base_sizes_per_level (list[tuple[int]]): Basic sizes of
                anchors.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size

            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchor = torch.Tensor([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

        return base_anchors
