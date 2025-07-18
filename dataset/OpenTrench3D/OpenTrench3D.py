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
        num_classes: int    # 类别数（可从配置里传入）
        classes: list       # 类标列表
        **kwargs:           # 吃掉其它所有关键字参数
        """
        # 构造数据路径并扫描文件
        area_path = os.path.join(root, area, split)
        if not os.path.isdir(area_path):
            raise ValueError(f"No directory: {area_path}")
        self.files = [
            os.path.join(area_path, f)
            for f in os.listdir(area_path) if f.endswith('.ply')
        ]

        # —— 新增字段 ——  
        # 如果外部传了 num_classes，则用它，否则默认 5
        self.num_classes = num_classes or 5
        # 如果外部传了 classes，则用它，否则生成 [0,1,...]
        self.classes = classes if classes is not None else list(range(self.num_classes))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]

        # 1) 读坐标 & 颜色
        pcd  = o3d.io.read_point_cloud(ply_path)
        pts  = np.asarray(pcd.points, dtype=np.float32)   # [N,3]
        cols = np.asarray(pcd.colors, dtype=np.float32)   # [N,3]

        # —— 新增：下采样到至多 max_pts 点，避免内存爆炸 ——  
        max_pts = 150000  # 根据显存情况调整，比如 1e5~2e5
        N = pts.shape[0]
        if N > max_pts:
            choice = np.random.choice(N, max_pts, replace=False)
            pts  = pts[choice]
            cols = cols[choice]
        # ———————————————————————————————

        # 2) 读取 class 标签（最后一列）
        with open(ply_path, 'r') as f:
            line = f.readline()
            while line and line.strip() != 'end_header':
                line = f.readline()
            lbl = np.loadtxt(f, dtype=np.int64, usecols=(-1,))
        # 同步下采样标签
        if N > max_pts:
            lbl = lbl[choice]

        # 3) 返回
        return {
            'pos': torch.from_numpy(pts),
            'x'  : torch.from_numpy(cols),
            'y'  : torch.from_numpy(lbl),
        }
