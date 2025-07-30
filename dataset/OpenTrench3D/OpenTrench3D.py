import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
from openpoints.dataset.build import DATASETS

@DATASETS.register_module()
class OpenTrench3D(Dataset):
    def __init__(self, *, root, split, area='water',
                 num_classes=None, classes=None, **kwargs):
        """
        root: str           # 数据根目录
        split: str          # 'train' / 'val' / 'test'
        area: str           # 子目录名
        num_classes: int    # 类别数（可从配置里传入)
        classes: list       # 类标列表
        kwargs 包含 dataset.common.ignore_label 和 dataset.common.trench_class_id
        """
        # 构造数据路径并扫描文件
        area_path = os.path.join(root, area, split)
        if not os.path.isdir(area_path):
            raise ValueError(f"No directory: {area_path}")
        self.files = [
            os.path.join(area_path, f)
            for f in os.listdir(area_path) if f.endswith('.ply')
        ]

        # 类别及配色表
        self.num_classes = num_classes or 5
        self.classes     = classes if classes is not None else list(range(self.num_classes))
        # 从 kwargs 中读取忽略标签和 trench 类 id
        self.ignore_label     = kwargs.get('ignore_label', None)
        self.trench_class_id  = kwargs.get('trench_class_id', None)
        print(f"[DEBUG Dataset] ignore_label={self.ignore_label}, trench_class_id={self.trench_class_id}")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]

        # 1) 读坐标 & 颜色
        pcd  = o3d.io.read_point_cloud(ply_path)
        pts  = np.asarray(pcd.points, dtype=np.float32)   # [N,3]
        cols = np.asarray(pcd.colors, dtype=np.float32)   # [N,3]

        # 下采样到至多 max_pts 点，避免显存爆炸
        max_pts = 150000
        N = pts.shape[0]
        if N > max_pts:
            choice = np.random.choice(N, max_pts, replace=False)
            pts  = pts[choice]
            cols = cols[choice]

        # 2) 读取 class 标签（最后一列）
        with open(ply_path, 'r') as f:
            line = f.readline()
            while line and line.strip() != 'end_header':
                line = f.readline()
            lbl = np.loadtxt(f, dtype=np.int64, usecols=(-1,))
        if N > max_pts:
            lbl = lbl[choice]

        # 3) 过滤掉 ignore_label
        if self.ignore_label is not None:
            mask = lbl != self.ignore_label
            pts  = pts[mask]
            cols = cols[mask]
            lbl  = lbl[mask]

        # 4) 二分类重映射：Other→0, Trench→1
        if self.trench_class_id is not None:
            lbl = np.where(lbl == self.trench_class_id, 1, 0).astype(np.int64)

        # 5) 返回数据
        return {
            'pos': torch.from_numpy(pts),
            'x'  : torch.from_numpy(cols),
            'y'  : torch.from_numpy(lbl),  # 改为一维标签，不 unsqueeze
        }
