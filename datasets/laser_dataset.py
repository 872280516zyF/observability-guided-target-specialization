import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

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


class LaserParamDataset(Dataset):
    """
    支持“洗前图 + 目标图案 → 参数 / 洗后图”任务的数据集：

    - annotation_path: CSV / Excel，至少包含参数列；可选 pattern_id、after_path
    - before_dir: 洗前布料图像目录
    - after_dir: (可选) 洗后布料图像目录
    - pattern_dir: 目标图案目录
    - pattern_manifest: CSV，包含 pattern_id, filename 列，用于映射
    - label_stats_path: YAML，提供 mean/std 做统一归一化
    """

    VALID_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    def __init__(
            self,
            annotation_path: str,
            before_dir: str,
            pattern_dir: str,
            label_stats_path: str,
            after_dir: Optional[str] = None,
            pattern_manifest: Optional[str] = None,
            id_column: str = "编号",
            param_cols: Optional[list] = None,
            has_after: bool = True,
            transforms_cfg: Optional[dict] = None,
    ):
        self.annotation_path = annotation_path
        self.df = self._load_table(annotation_path)
        self.before_dir = Path(before_dir)
        self.after_dir = Path(after_dir) if after_dir else None
        self.pattern_dir = Path(pattern_dir)
        self.id_column = id_column if id_column in self.df.columns else None
        self.has_after = has_after and (self.after_dir is not None)

        if param_cols is None:
            self.param_cols = [c for c in ["频率", "脉宽", "速度", "DPI"] if c in self.df.columns]
        else:
            self.param_cols = param_cols

        if len(self.param_cols) == 0:
            raise ValueError("参数列未找到，请检查 annotation_path 是否包含 频率/脉宽/速度/DPI 等列。")

        # 目标图案映射
        self.pattern_manifest = None
        if pattern_manifest:
            self.pattern_manifest = self._load_pattern_manifest(pattern_manifest)

        # 标签归一化
        self.normalizer = LabelNormalizer.from_yaml(label_stats_path, self.param_cols)

        # 图像变换
        self.train_transform, self.eval_transform = self._build_transforms(transforms_cfg)

    @staticmethod
    def _load_table(path: str) -> pd.DataFrame:
        suffix = Path(path).suffix.lower()
        if suffix in [".csv", ".txt"]:
            # 尝试多种编码格式读取CSV文件
            encodings = ["utf-8", "gbk", "gb2312", "latin1"]
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    print(f"成功使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise UnicodeDecodeError(f"无法使用任何编码读取文件: {path}")
            return df
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        raise ValueError(f"不支持的标注文件格式: {suffix}")

    @staticmethod
    def _load_pattern_manifest(path: str) -> Dict[str, str]:
        # 尝试多种编码格式读取CSV文件
        encodings = ["utf-8", "gbk", "gb2312", "latin1"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                print(f"成功使用编码读取pattern_manifest: {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise UnicodeDecodeError(f"无法使用任何编码读取文件: {path}")

        if "pattern_id" not in df.columns or "filename" not in df.columns:
            raise ValueError("pattern_manifest 需包含 columns: pattern_id, filename")
        return dict(zip(df["pattern_id"].astype(str), df["filename"]))

    def _build_transforms(self, cfg: Optional[dict]):
        resize = cfg.get("resize", [224, 224]) if cfg else [224, 224]

        # 支持矩形尺寸 (宽×高: 374×164)
        if isinstance(resize, list) and len(resize) == 2:
            target_size = tuple(resize)  # (width, height)
        else:
            # 向后兼容正方形尺寸
            target_size = (resize, resize)

        train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        eval_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return train_tf, eval_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        before_img = self._load_image(self.before_dir, row[self.id_column])
        pattern_img = self._load_pattern(row)

        transform = self.train_transform if self._is_train(row) else self.eval_transform
        before_img = transform(before_img)
        pattern_img = transform(pattern_img)

        raw_params = torch.tensor([row[c] for c in self.param_cols], dtype=torch.float32)
        norm_params = self.normalizer.normalize(raw_params)

        sample = {
            "before_img": before_img,
            "pattern_img": pattern_img,
            "targets": norm_params,
            "targets_raw": raw_params,
        }

        if self.has_after:
            after_img = self._load_image(self.after_dir, row[self.id_column])
            sample["after_img"] = self.eval_transform(after_img)

        return sample

    def _load_pattern(self, row):
        # 优先使用 manifest
        if self.pattern_manifest is not None and "pattern_id" in row:
            pattern_id = str(row["pattern_id"])
            filename = self.pattern_manifest.get(pattern_id)
            if filename is None:
                raise FileNotFoundError(f"pattern_id {pattern_id} 未在 pattern_manifest 中找到映射。")
            pattern_path = self.pattern_dir / filename
        else:
            # 默认与编号同名
            identifier = row[self.id_column]
            pattern_path = self._locate_by_id(self.pattern_dir, identifier)

        img = cv2.imread(str(pattern_path))
        if img is None:
            raise FileNotFoundError(f"pattern image not found: {pattern_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_image(self, folder: Path, identifier):
        if folder is None:
            raise ValueError("未提供图像目录。")
        path = self._locate_by_id(folder, identifier)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _locate_by_id(self, folder: Path, identifier):
        raw = str(identifier).strip()
        candidates = [raw, raw.zfill(3), raw.zfill(4), raw.zfill(5)]
        for name in candidates:
            for ext in self.VALID_IMAGE_EXTS:
                path = folder / f"{name}{ext}"
                if path.exists():
                    return path
        raise FileNotFoundError(f"{identifier} 对应的图片未在目录 {folder} 中找到。")

    def _is_train(self, row) -> bool:
        # 简单依据 annotation 是否包含 train 标记，可扩展
        return row.get("split", "train") == "train"

