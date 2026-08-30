"""
样板图增强模块
解决样板图种类少（3种）但参数组合多（400多种）的问题
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2


class PatternAugmenter:
    """样板图增强器，增加样板图多样性"""
    
    def __init__(self, augment_prob=0.8):
        self.augment_prob = augment_prob
        
    def __call__(self, pattern_img, is_train=True):
        """
        对样板图进行增强
        Args:
            pattern_img: PIL Image 或 torch.Tensor
            is_train: 是否训练模式
        Returns:
            增强后的图像
        """
        if not is_train or random.random() > self.augment_prob:
            return pattern_img
            
        if isinstance(pattern_img, torch.Tensor):
            # 转换为PIL进行增强
            pattern_np = pattern_img.permute(1, 2, 0).numpy()
            pattern_np = (pattern_np * 255).astype(np.uint8)
            pattern_pil = Image.fromarray(pattern_np)
            
            # 应用增强
            pattern_pil = self._apply_augmentations(pattern_pil)
            
            # 转换回tensor
            pattern_np = np.array(pattern_pil).astype(np.float32) / 255.0
            pattern_tensor = torch.from_numpy(pattern_np).permute(2, 0, 1)
            return pattern_tensor
        else:
            # 直接对PIL图像增强
            return self._apply_augmentations(pattern_img)
    
    def _apply_augmentations(self, img):
        """应用一系列增强操作"""
        
        # 随机选择2-4种增强操作
        augmentations = random.sample([
            self._random_rotate,
            self._random_scale,
            self._random_translate,
            self._color_jitter,
            self._random_blur,
            self._random_noise,
            self._random_contrast,
            self._random_brightness,
        ], random.randint(2, 4))
        
        # 按随机顺序应用增强
        random.shuffle(augmentations)
        
        for aug in augmentations:
            img = aug(img)
            
        return img
    
    def _random_rotate(self, img):
        """随机旋转"""
        angle = random.uniform(-30, 30)
        return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    
    def _random_scale(self, img):
        """随机缩放"""
        scale = random.uniform(0.7, 1.3)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # 如果缩放后尺寸不同，进行裁剪或填充
        if scale > 1:
            # 随机裁剪
            left = random.randint(0, new_w - w)
            top = random.randint(0, new_h - h)
            img = img.crop((left, top, left + w, top + h))
        elif scale < 1:
            # 填充
            result = Image.new('RGB', (w, h), (255, 255, 255))
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            result.paste(img, (left, top))
            img = result
            
        return img
    
    def _random_translate(self, img):
        """随机平移"""
        w, h = img.size
        dx = random.randint(-int(w * 0.2), int(w * 0.2))
        dy = random.randint(-int(h * 0.2), int(h * 0.2))
        
        result = Image.new('RGB', (w, h), (255, 255, 255))
        result.paste(img, (dx, dy))
        return result
    
    def _color_jitter(self, img):
        """颜色抖动"""
        # 随机调整亮度
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        # 随机调整对比度
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        
        # 随机调整饱和度
        saturation_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        
        return img
    
    def _random_blur(self, img):
        """随机模糊"""
        if random.random() < 0.3:
            radius = random.uniform(0.5, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius))
        return img
    
    def _random_noise(self, img):
        """随机噪声"""
        if random.random() < 0.2:
            img_np = np.array(img)
            noise = np.random.normal(0, 10, img_np.shape).astype(np.uint8)
            img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
        return img
    
    def _random_contrast(self, img):
        """随机对比度"""
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def _random_brightness(self, img):
        """随机亮度"""
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)


class PatternMixer:
    """样板图混合器，创建新的样板图"""
    
    def __init__(self, mix_prob=0.3):
        self.mix_prob = mix_prob
        
    def __call__(self, pattern1, pattern2=None):
        """混合两个样板图"""
        if pattern2 is None or random.random() > self.mix_prob:
            return pattern1
            
        # 转换为numpy进行混合
        if isinstance(pattern1, torch.Tensor):
            pattern1_np = pattern1.permute(1, 2, 0).numpy()
            pattern2_np = pattern2.permute(1, 2, 0).numpy()
        else:
            pattern1_np = np.array(pattern1)
            pattern2_np = np.array(pattern2)
            
        # 随机选择混合方式
        mix_method = random.choice([
            self._alpha_blend,
            self._overlay_blend,
            self._mask_blend,
        ])
        
        mixed_np = mix_method(pattern1_np, pattern2_np)
        
        if isinstance(pattern1, torch.Tensor):
            mixed_tensor = torch.from_numpy(mixed_np).permute(2, 0, 1)
            return mixed_tensor
        else:
            return Image.fromarray(mixed_np.astype(np.uint8))
    
    def _alpha_blend(self, img1, img2):
        """Alpha混合"""
        alpha = random.uniform(0.3, 0.7)
        return (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)
    
    def _overlay_blend(self, img1, img2):
        """叠加混合"""
        # 随机选择叠加模式
        mode = random.choice(['multiply', 'screen', 'overlay'])
        
        if mode == 'multiply':
            result = (img1.astype(np.float32) * img2.astype(np.float32) / 255.0).astype(np.uint8)
        elif mode == 'screen':
            result = (255 - (255 - img1.astype(np.float32)) * (255 - img2.astype(np.float32)) / 255.0).astype(np.uint8)
        else:  # overlay
            mask = img1 > 128
            result = np.where(mask, 
                            255 - 2 * (255 - img1) * (255 - img2) / 255,
                            2 * img1 * img2 / 255).astype(np.uint8)
            
        return result
    
    def _mask_blend(self, img1, img2):
        """掩码混合"""
        # 创建随机掩码
        h, w = img1.shape[:2]
        mask = np.random.rand(h, w, 1) > 0.5
        mask = mask.astype(np.float32)
        
        # 应用高斯模糊使边缘平滑
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask[..., np.newaxis]
        
        result = (img1.astype(np.float32) * mask + img2.astype(np.float32) * (1 - mask)).astype(np.uint8)
        return result


# 使用示例
def test_augmenter():
    """测试增强器"""
    augmenter = PatternAugmenter()
    mixer = PatternMixer()
    
    # 模拟一个样板图
    pattern = Image.new('RGB', (256, 256), (128, 128, 128))
    
    print("原始样板图尺寸:", pattern.size)
    
    # 应用增强
    augmented = augmenter(pattern, is_train=True)
    print("增强后尺寸:", augmented.size)
    
    # 保存示例
    pattern.save("original_pattern.png")
    augmented.save("augmented_pattern.png")
    
    print("增强测试完成！")


if __name__ == "__main__":
    test_augmenter()