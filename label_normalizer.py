import yaml
from pathlib import Path
from typing import Dict, List

import torch


class LabelNormalizer:
    """
    统一的标签归一化/反归一化工具，基于 z-score。

    用法:
        normalizer = LabelNormalizer.from_yaml("configs/label_stats.yaml")
        norm = normalizer.normalize(raw_tensor)        # raw_tensor shape: (B, num_params)
        denorm = normalizer.denormalize(norm)          # 恢复到原始物理量
    """

    def __init__(self, stats: Dict[str, Dict[str, float]], param_order: List[str]):
        self.stats = stats
        self.param_order = param_order

        self.means = torch.tensor([stats[p]["mean"] for p in param_order], dtype=torch.float32)
        self.stds = torch.tensor([stats[p]["std"] for p in param_order], dtype=torch.float32)

    @classmethod
    def from_yaml(cls, path: str, param_order: List[str]):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"未找到标签统计文件: {path}")
        with open(path_obj, "r", encoding="utf-8") as f:
            stats = yaml.safe_load(f)
        return cls(stats, param_order)

    def normalize(self, raw_values: torch.Tensor) -> torch.Tensor:
        if raw_values.shape[-1] != len(self.param_order):
            raise ValueError(
                f"输入维度 {raw_values.shape[-1]} 与 param_order 长度 {len(self.param_order)} 不匹配"
            )
        return (raw_values - self.means.to(raw_values.device)) / self.stds.to(raw_values.device)

    def denormalize(self, norm_values: torch.Tensor) -> torch.Tensor:
        if norm_values.shape[-1] != len(self.param_order):
            raise ValueError(
                f"输入维度 {norm_values.shape[-1]} 与 param_order 长度 {len(self.param_order)} 不匹配"
            )
        return norm_values * self.stds.to(norm_values.device) + self.means.to(norm_values.device)

    def to_dict(self):
        return {
            "param_order": self.param_order,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
        }

