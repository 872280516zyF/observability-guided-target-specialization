from pathlib import Path
from typing import Dict, Optional

import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from label_normalizer import LabelNormalizer


def adaptive_collate_fn(batch):
    """
    自适应批处理函数，支持不同尺寸的图像
    
    当使用自适应尺寸时，不同图像可能有不同的尺寸。
    这个函数会检查批次内图像的尺寸，如果不一致则进行调整。
    
    Args:
        batch: 数据集返回的样本列表
        
    Returns:
        处理后的批次数据
    """
    # 检查批次内图像的尺寸
    if len(batch) == 0:
        return {}
    
    # 获取第一张图像的尺寸
    first_before_shape = batch[0]['before_img'].shape
    first_pattern_shape = batch[0]['pattern_img'].shape
    
    # 检查所有图像的尺寸是否一致
    all_same_size = True
    for sample in batch:
        if sample['before_img'].shape != first_before_shape or \
           sample['pattern_img'].shape != first_pattern_shape:
            all_same_size = False
            break
    
    if all_same_size:
        # 所有图像尺寸一致，直接堆叠
        result = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in batch])
            elif key == 'sample_id':
                result[key] = [item[key] for item in batch]
            elif key == 'before_id':
                result[key] = [item.get(key) for item in batch if key in item]
            elif key == 'pattern_id':
                result[key] = [item.get(key) for item in batch if key in item]
            elif isinstance(batch[0][key], (int, float)):
                result[key] = torch.tensor([item[key] for item in batch])
            else:
                result[key] = [item[key] for item in batch]
        return result
    else:
        # 图像尺寸不一致，需要调整
        # 找到最常见的尺寸
        size_counts = {}
        for sample in batch:
            size = (sample['before_img'].shape[1], sample['before_img'].shape[2])  # (H, W)
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # 选择最常见的尺寸
        target_size = max(size_counts.keys(), key=lambda x: size_counts[x])
        
        # 调整所有图像到目标尺寸
        result = {}
        for key in batch[0].keys():
            if key in ['before_img', 'pattern_img', 'after_img']:
                # 调整图像尺寸
                resized_images = []
                for sample in batch:
                    if key in sample:
                        img = sample[key]
                        if img.shape[1:] != (target_size[0], target_size[1]):
                            # 使用双线性插值调整尺寸
                            img = torch.nn.functional.interpolate(
                                img.unsqueeze(0), 
                                size=target_size, 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                        resized_images.append(img)
                    else:
                        resized_images.append(None)
                
                # 堆叠图像（过滤掉None值）
                valid_images = [img for img in resized_images if img is not None]
                if valid_images:
                    result[key] = torch.stack(valid_images)
            elif isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in batch])
            elif key == 'sample_id':
                result[key] = [item[key] for item in batch]
            elif key == 'before_id':
                result[key] = [item.get(key) for item in batch if key in item]
            elif key == 'pattern_id':
                result[key] = [item.get(key) for item in batch if key in item]
            elif isinstance(batch[0][key], (int, float)):
                result[key] = torch.tensor([item[key] for item in batch])
            else:
                result[key] = [item[key] for item in batch]
        
        return result


class LaserParamDatasetV2(Dataset):
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
        if id_column in self.df.columns:
            self.id_column = id_column
        elif "sample_id" in self.df.columns:
            self.id_column = "sample_id"
        elif "编号" in self.df.columns:
            self.id_column = "编号"
        else:
            raise ValueError(
                "样本编号列未找到；需要 编号 或 sample_id 列。"
            )
        self.has_after = has_after and (self.after_dir is not None)

        # 可选：单独的洗前图编号列，用于复用同一张洗前图
        # 如果标签中包含 before_id 列，则优先使用该列去检索洗前图；
        # 否则退回到使用 id_column（通常是“编号”）
        self.before_id_column = "before_id" if "before_id" in self.df.columns else None

        if param_cols is None:
            parameter_orders = [
                ["frequency", "pulse_width", "speed", "dpi"],
                ["频率", "脉宽", "速度", "DPI"],
                ["棰戠巼", "鑴夊", "閫熷害", "DPI"],
            ]
            self.param_cols = next(
                (
                    order
                    for order in parameter_orders
                    if all(column in self.df.columns for column in order)
                ),
                [],
            )
        else:
            self.param_cols = param_cols

        if len(self.param_cols) != 4:
            raise ValueError(
                "参数列未找到；需要完整的 "
                "frequency/pulse_width/speed/dpi 或 频率/脉宽/速度/DPI。"
            )

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
        raise ValueError(f"不支持的标签文件格式: {suffix}")

    @staticmethod
    def _load_pattern_manifest(path: str) -> Dict[str, str]:
        df = pd.read_csv(path)
        if "pattern_id" not in df.columns or "filename" not in df.columns:
            raise ValueError("pattern_manifest 需包含 columns: pattern_id, filename")
        return dict(zip(df["pattern_id"].astype(str), df["filename"]))

    def _build_transforms(self, cfg: Optional[dict]):
        resize = cfg.get("resize", [164, 374]) if cfg else [164, 374]
        augment = cfg.get("augment", True) if cfg else True
        
        # 支持自适应尺寸（不进行resize）
        adaptive_size = resize is None or resize == "adaptive" or resize == "none"
        
        if not adaptive_size:
            # 支持矩形尺寸 (宽×高: 374×164)
            if isinstance(resize, list) and len(resize) == 2:
                target_size = tuple(resize)  # (width, height)
            else:
                # 向后兼容正方形尺寸
                target_size = (resize, resize)
        
        train_tf_list = [transforms.ToPILImage()]
        if augment:
            train_tf_list.extend([
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),
                transforms.RandomHorizontalFlip(p=0.3),
            ])
        train_tf_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        eval_tf_list = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        
        # 如果不是自适应尺寸，则在ToPILImage后添加Resize
        if not adaptive_size:
            train_tf_list.insert(1, transforms.Resize(target_size))
            eval_tf_list.insert(1, transforms.Resize(target_size))
        
        train_tf = transforms.Compose(train_tf_list)
        eval_tf = transforms.Compose(eval_tf_list)
        
        return train_tf, eval_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 优先使用 before_id（如果存在），否则使用样本编号
        before_identifier = None
        if self.before_id_column is not None and self.before_id_column in row:
            before_identifier = row[self.before_id_column]
        else:
            before_identifier = row[self.id_column]

        before_img = self._load_image(self.before_dir, before_identifier)
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
            "sample_id": row[self.id_column],  # 添加样本ID
            "sample_idx": idx,  # 添加样本索引
        }

        # 记录洗前图ID（如果有 before_id，则单独暴露出来，便于分析/调试）
        if self.before_id_column is not None and self.before_id_column in row:
            sample["before_id"] = row[self.before_id_column]
        
        # 添加其他元数据（如果存在）
        if "pattern_id" in row:
            sample["pattern_id"] = row["pattern_id"]
        if hasattr(self, 'param_cols'):
            for col in self.param_cols:
                if col in row:
                    sample[f"param_{col}"] = row[col]

        if self.has_after:
            # 洗后图通常仍然是一行对应一张，继续使用样本编号
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

