import os
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
import open3d as o3d  # 用于在缺失法向量时估计算法
from openpoints.dataset.build import DATASETS

@DATASETS.register_module()
class OpenTrench3D(Dataset):
    def __init__(self, *, root, split, area='water',
                 num_classes=None, classes=None, **kwargs):
        # 构造数据路径
        area_path = os.path.join(root, area, split)
        if not os.path.isdir(area_path):
            raise ValueError(f"No directory: {area_path}")
        self.files = [
            os.path.join(area_path, f)
            for f in os.listdir(area_path) if f.endswith('.ply')
        ]
        self.num_classes    = num_classes or 5
        self.classes        = classes if classes is not None else list(range(self.num_classes))
        self.ignore_label   = kwargs.get('ignore_label', None)
        self.trench_class_id= kwargs.get('trench_class_id', None)
        print(f"[DEBUG Dataset] ignore_label={self.ignore_label}, trench_class_id={self.trench_class_id}")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]
        

        # 1) 读取所有顶点属性字段，打印字段名以便调试
        ply = PlyData.read(ply_path)
        v   = next(e.data for e in ply.elements if e.name == 'vertex')
        

        # 2) 提取坐标
        pts = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)  # (N,3)

        # 3) 提取或估算法向量 & 曲率
        if set(['nx','ny','nz','curvature']).issubset(v.dtype.names):
            normals   = np.stack([v['nx'], v['ny'], v['nz']], axis=-1).astype(np.float32)  # (N,3)
            curvature = np.expand_dims(v['curvature'].astype(np.float32), 1)               # (N,1)
        else:
            print("⚠️ Missing normals/curvature, estimating normals …")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            normals   = np.asarray(pcd.normals, dtype=np.float32)
            curvature = np.zeros((normals.shape[0],1), dtype=np.float32)

        # 4) 下采样到最多 150k 点，同时同步 normals & curvature
        max_pts = 150000
        N = pts.shape[0]
        if N > max_pts:
            choice    = np.random.choice(N, max_pts, replace=False)
            pts       = pts[choice]
            normals   = normals[choice]
            curvature = curvature[choice]

        # 5) 读取 class 标签（最后一列），并同步下采样
        with open(ply_path, 'r') as f:
            line = f.readline()
            while line and line.strip() != 'end_header':
                line = f.readline()
            lbl = np.loadtxt(f, dtype=np.int64, usecols=(-1,))
        if N > max_pts:
            lbl = lbl[choice]

        # 6) 过滤 ignore_label
        if self.ignore_label is not None:
            mask      = lbl != self.ignore_label
            pts       = pts[mask]
            normals   = normals[mask]
            curvature = curvature[mask]
            lbl       = lbl[mask]

        # 7) 二分类重映射：Other→0, Trench→1
        if self.trench_class_id is not None:
            lbl = np.where(lbl == self.trench_class_id, 1, 0).astype(np.int64)

        # 8) 返回字典
        feat = np.concatenate([normals, curvature], axis=1).astype(np.float32)  # (N,4)
        return {
            'pos'      : torch.from_numpy(pts),        # (N,3)
            'x'        : torch.from_numpy(feat),       # (N,4)
            'normals'  : torch.from_numpy(normals),    # (N,3)
            'curvature': torch.from_numpy(curvature),  # (N,1)
            'y'        : torch.from_numpy(lbl),        # (N,)
        }
