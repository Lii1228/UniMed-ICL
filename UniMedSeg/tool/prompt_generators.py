import torch
import numpy as np
import cv2
import random
from typing import Tuple, List, Dict, Any
from scipy import ndimage as ndi
from scipy.spatial import distance 
from skimage.morphology import ball
from scipy.ndimage import distance_transform_edt

# ==============================================================================
# 基础辅助函数
# ==============================================================================

def get_slice(t: torch.Tensor, ax: int, idx: int) -> torch.Tensor:
    if ax == 0: return t[idx, :, :]
    elif ax == 1: return t[:, idx, :]
    else: return t[:, :, idx]

def set_slice(t: torch.Tensor, ax: int, idx: int, val: torch.Tensor) -> None:
    if ax == 0: t[idx, :, :] = val
    elif ax == 1: t[:, idx, :] = val
    else: t[:, :, idx] = val

# ==============================================================================
# 核心：全局切片采样器 (Global Slice Sampler)
# ==============================================================================

def _sample_global_indices(mask_3d: torch.Tensor, num_samples: int = 1, strategy: str = 'random') -> List[Tuple[int, int]]:
    """
    在三个轴向的所有切片池中进行混合竞争。
    Return: List of (axis, slice_index)
    """
    # 计算三个轴向的投影和 (即每一层的像素数)
    w_ax0 = mask_3d.sum(dim=(1, 2)) 
    w_ax1 = mask_3d.sum(dim=(0, 2))
    w_ax2 = mask_3d.sum(dim=(0, 1))
    
    # 拼接成一个全局权重向量
    all_weights = torch.cat([w_ax0, w_ax1, w_ax2]).float()
    
    # 过滤掉全黑切片
    eligible_mask = (all_weights > 0)
    if eligible_mask.sum() == 0:
        return []
    
    # 策略选择
    if strategy == 'max':
        # 选择像素最多的那个切片 (num_samples 强制为 1)
        global_indices = [torch.argmax(all_weights).item()]
    else:
        # random: 基于权重的随机采样
        num_valid = eligible_mask.sum().item()
        actual_samples = min(num_samples, num_valid)
        if actual_samples < 1: actual_samples = 1
        
        # 使用 multinomial 进行加权无放回采样
        global_indices = torch.multinomial(all_weights, actual_samples, replacement=False).tolist()

    # 解码全局索引为 (axis, local_index)
    results = []
    d0, d1, d2 = mask_3d.shape
    
    for g_idx in global_indices:
        if g_idx < d0:
            results.append((0, g_idx))
        elif g_idx < d0 + d1:
            results.append((1, g_idx - d0))
        else:
            results.append((2, g_idx - d0 - d1))
            
    return results

# ==============================================================================
# 1. Scribble 生成器 (已优化防止卡死)
# ==============================================================================

class _BaseScribbleGenerator:
    def __init__(self, thickness: int = 3, truncate_ratio_range: Tuple[float, float] = (0.3, 0.7),
                 deformation_strength_factor: float = 0.1, deformation_sigma: float = 15.0, **kwargs):
        self.thickness = thickness
        self.truncate_ratio_range = truncate_ratio_range
        self.deformation_strength_factor = deformation_strength_factor
        self.deformation_sigma = deformation_sigma

    def generate_raw_scribble(self, component_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _truncate_scribble(self, scribble_mask: np.ndarray, component_mask: np.ndarray) -> np.ndarray:
        if scribble_mask.max() == 0 or component_mask.max() == 0: return scribble_mask
        coords = np.argwhere(component_mask > 0)
        y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
        axis = random.choice([0, 1])
        truncated_scribble = np.zeros_like(scribble_mask)
        min_ratio, max_ratio = self.truncate_ratio_range
        length_ratio = random.uniform(min_ratio, max_ratio)
        start_ratio = random.uniform(0.0, 1.0 - length_ratio)

        if axis == 0:
            axis_len = y_max - y_min
            if axis_len > 1:
                start_idx = y_min + int(axis_len * start_ratio)
                end_idx = min(start_idx + int(axis_len * length_ratio), y_max + 1)
                truncated_scribble[start_idx:end_idx, :] = scribble_mask[start_idx:end_idx, :]
        else:
            axis_len = x_max - x_min
            if axis_len > 1:
                start_idx = x_min + int(axis_len * start_ratio)
                end_idx = min(start_idx + int(axis_len * length_ratio), x_max + 1)
                truncated_scribble[:, start_idx:end_idx] = scribble_mask[:, start_idx:end_idx]
        return truncated_scribble if np.any(truncated_scribble) else scribble_mask

    def _apply_deformation(self, mask: np.ndarray, component_mask: np.ndarray) -> np.ndarray:
        if not np.any(mask): return mask
        object_coords = np.argwhere(component_mask > 0)
        if len(object_coords) == 0: return mask
        h, w = mask.shape
        diagonal = np.linalg.norm(object_coords.max(axis=0) - object_coords.min(axis=0))
        strength = max(5.0, diagonal * self.deformation_strength_factor)
        rand_x = ndi.gaussian_filter((np.random.rand(h, w) * 2 - 1), self.deformation_sigma, mode='reflect')
        rand_y = ndi.gaussian_filter((np.random.rand(h, w) * 2 - 1), self.deformation_sigma, mode='reflect')
        edt = ndi.distance_transform_edt(component_mask > 0)
        if edt.max() > 0: edt /= edt.max()
        dx, dy = strength * rand_x * edt, strength * rand_y * edt
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        map_y, map_x = np.clip(y_indices + dy, 0, h - 1).astype(np.float32), np.clip(x_indices + dx, 0, w - 1).astype(np.float32)
        deformed_mask = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        return (deformed_mask > 0.5).astype(mask.dtype)

    def _process_scribble(self, raw_scribble: np.ndarray, component_mask: np.ndarray) -> np.ndarray:
        processed_mask = self._truncate_scribble(raw_scribble, component_mask)
        processed_mask = self._apply_deformation(processed_mask, component_mask)
        if self.thickness > 1 and np.any(processed_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.thickness, self.thickness))
            processed_mask = cv2.dilate(processed_mask, kernel)
        final_mask = processed_mask * component_mask
        return final_mask.astype(np.uint8) if np.any(final_mask) else (raw_scribble * component_mask).astype(np.uint8)

    def __call__(self, component_mask: np.ndarray) -> np.ndarray:
        raw_scribble = self.generate_raw_scribble(component_mask)
        return self._process_scribble(raw_scribble, component_mask)

class LineScribbleGenerator(_BaseScribbleGenerator):
    def __init__(self, smart_line_min_points: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.smart_line_min_points = smart_line_min_points

    def generate_raw_scribble(self, component_mask: np.ndarray) -> np.ndarray:
        points = np.argwhere(component_mask > 0)
        if len(points) < 2: return np.zeros_like(component_mask, dtype=np.uint8)
        
        # [优化] 如果点太少，随机选俩；如果够多，执行智能长线逻辑
        if len(points) < self.smart_line_min_points:
            idx1, idx2 = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx1], points[idx2]
        else:
            # 随机选起点
            p1_idx = random.randint(0, len(points) - 1)
            p1 = points[p1_idx]
            
            # [关键优化] 大数组保护：如果点数过多(>2000)，进行随机下采样来计算距离
            # 这防止了 cdist 在处理 10000+ 点时导致 CPU 卡死
            if len(points) > 2000:
                sample_indices = np.random.choice(len(points), 2000, replace=False)
                calc_points = points[sample_indices]
            else:
                calc_points = points
            
            # 计算距离
            dists = distance.cdist([p1], calc_points)[0]
            percentile = np.percentile(dists, 90)
            far_points_indices = np.where(dists >= percentile)[0]
            
            if len(far_points_indices) > 0:
                p2_local_idx = random.choice(far_points_indices)
                p2 = calc_points[p2_local_idx]
            else:
                p2_idx = random.randint(0, len(points) - 1)
                p2 = points[p2_idx]
                
        line_mask = np.zeros_like(component_mask, dtype=np.uint8)
        cv2.line(line_mask, (p1[1], p1[0]), (p2[1], p2[0]), 1, 1)
        return line_mask

class CenterlineScribbleGenerator(_BaseScribbleGenerator):
    def generate_raw_scribble(self, component_mask: np.ndarray) -> np.ndarray:
        if not np.any(component_mask): return np.zeros_like(component_mask, dtype=np.uint8)
        opencv_mask = (component_mask * 255).astype(np.uint8)
        try:
            # 优先尝试 OpenCV 的快速算法
            skeleton_255 = cv2.ximgproc.thinning(opencv_mask)
        except (cv2.error, AttributeError):
            # [关键优化] 备选算法增加最大迭代次数限制，防止死循环
            skeleton_255 = np.zeros(opencv_mask.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp_img = opencv_mask.copy()
            
            max_iter = 100 # 安全刹车
            count = 0
            
            while cv2.countNonZero(temp_img) != 0 and count < max_iter:
                eroded = cv2.erode(temp_img, element)
                temp = cv2.dilate(eroded, element)
                skeleton_255 = cv2.bitwise_or(skeleton_255, cv2.subtract(temp_img, temp))
                temp_img = eroded.copy()
                count += 1
                
        return (skeleton_255 > 0).astype(np.uint8)

class ContourScribbleGenerator(_BaseScribbleGenerator):
    def __init__(self, blur_sigma_range: Tuple[float, float] = (5.0, 15.0), **kwargs):
        super().__init__(**kwargs)
        self.blur_sigma_range = blur_sigma_range

    def generate_raw_scribble(self, component_mask: np.ndarray) -> np.ndarray:
        if not np.any(component_mask): return np.zeros_like(component_mask, dtype=np.uint8)
        rev_mask = 1 - component_mask
        sigma = random.uniform(*self.blur_sigma_range)
        blurred_rev_mask = ndi.gaussian_filter(rev_mask.astype(float), sigma=sigma)
        corrected_blur_mask = np.maximum(blurred_rev_mask, rev_mask)
        min_val = corrected_blur_mask.min(); max_val = corrected_blur_mask[component_mask > 0].max()
        if max_val <= min_val:
            eroded = cv2.erode(component_mask, np.ones((3, 3), np.uint8))
            return component_mask - eroded
        threshold = random.uniform(min_val, max_val)
        internal_mask = (corrected_blur_mask <= threshold).astype(np.uint8)
        if not np.any(internal_mask): return np.zeros_like(component_mask, dtype=np.uint8)
        dilated = cv2.dilate(internal_mask, np.ones((3, 3), np.uint8))
        return dilated - internal_mask

class InteractiveScribbleGenerator:
    def __init__(self, line_params: Dict[str, Any] = {}, centerline_params: Dict[str, Any] = {}, contour_params: Dict[str, Any] = {}):
        self.generators = {
            'line': LineScribbleGenerator(**line_params),
            'centerline': CenterlineScribbleGenerator(**centerline_params),
            'contour': ContourScribbleGenerator(**contour_params)
        }

    def _select_slices(self, mask_3d: torch.Tensor, max_prompts: int) -> List[Tuple[int, int]]:
        return _sample_global_indices(mask_3d, max_prompts, strategy='random')

    def generate_3d_scribbles(self, mask_3d: torch.Tensor, scribble_type: str = 'random', max_prompts: int = 1) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        output_mask = torch.zeros_like(mask_3d, dtype=mask_3d.dtype)
        device = mask_3d.device
        selected_slices = self._select_slices(mask_3d, max_prompts)
        for axis, slice_idx in selected_slices:
            slice_tensor = get_slice(mask_3d, axis, slice_idx)
            slice_np = slice_tensor.cpu().numpy()
            unique_labels = np.unique(slice_np); unique_labels = unique_labels[unique_labels > 0]
            if len(unique_labels) == 0: continue
            combined_prompts_2d = torch.zeros_like(slice_tensor, dtype=mask_3d.dtype)
            for organ_label in unique_labels:
                labeled_slice, num_labels = ndi.label(slice_np == organ_label)
                if num_labels == 0: continue
                for component_id in range(1, num_labels + 1):
                    component_mask_np = (labeled_slice == component_id).astype(np.uint8)
                    ctype = random.choice(list(self.generators.keys())) if scribble_type == 'random' else scribble_type
                    prompt_np = self.generators[ctype](component_mask_np)
                    combined_prompts_2d[torch.from_numpy(prompt_np > 0).to(device)] = organ_label.item()
            set_slice(output_mask, axis, slice_idx, combined_prompts_2d)
        return output_mask, selected_slices

# ==============================================================================
# 2. Lasso 生成器 (保留自定义 Rough Lasso 逻辑)
# ==============================================================================
'''
#规则lasso
def _apply_deformation_lasso(coarse_mask: np.ndarray, deformation_strength: float) -> np.ndarray:
    h, w = coarse_mask.shape
    rand_x = ndi.gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=20, mode='reflect')
    rand_y = ndi.gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=20, mode='reflect')
    edt = ndi.distance_transform_edt(coarse_mask)
    if edt.max() > 0: edt = edt / edt.max()
    dx, dy = deformation_strength * rand_x * edt, deformation_strength * rand_y * edt
    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    map_y = np.clip(y_indices + dy, 0, h - 1).astype(np.float32)
    map_x = np.clip(x_indices + dx, 0, w - 1).astype(np.float32)
    deformed_mask = cv2.remap(coarse_mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    return (deformed_mask > 0.5).astype(coarse_mask.dtype)

def _generate_single_lasso_np(component_mask_np: np.ndarray, deformation_strength: float = 10.0, **kwargs) -> np.ndarray:
    if component_mask_np.max() == 0: return np.zeros_like(component_mask_np)
    uint8_mask = component_mask_np.astype(np.uint8)
    dist_transform = cv2.distanceTransform(uint8_mask, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    kernel_size = 3 if max_dist < 2 else int(max_dist * 0.75)
    kernel_size = max(3, kernel_size); kernel_size += 1 if kernel_size % 2 == 0 else 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    coarse_mask = cv2.morphologyEx(uint8_mask, cv2.MORPH_CLOSE, kernel)
    coarse_mask = cv2.dilate(coarse_mask, kernel, iterations=1)
    return _apply_deformation_lasso(coarse_mask, deformation_strength)
'''
def _apply_deformation(coarse_mask: np.ndarray, deformation_strength: float) -> np.ndarray:
    """
    修改版：在保留 EDT 的前提下，制造粗糙的 Lasso 效果。
    """
    h, w = coarse_mask.shape
    
    # [修改点 1] 产生高频噪声
    sigma_val = 7.0  #################### 关键修改1：sigma=50 是平滑波浪，sigma=4 是粗糙抖动
    
    rand_x = np.random.rand(h, w) * 2 - 1
    rand_y = np.random.rand(h, w) * 2 - 1
    
    rand_x = ndi.gaussian_filter(rand_x, sigma=sigma_val, mode='reflect')
    rand_y = ndi.gaussian_filter(rand_y, sigma=sigma_val, mode='reflect')
    
    # [修改点 2] 保留并计算 EDT
    edt = ndi.distance_transform_edt(coarse_mask)
    if edt.max() > 0: 
        edt = edt / edt.max()
    
    # [关键调整] 给 EDT 加上一个底数 (Bias)
    weighted_map = edt + 0.4 
    
    norm_factor = 10.0 
    
    dx = deformation_strength * rand_x * weighted_map * norm_factor
    dy = deformation_strength * rand_y * weighted_map * norm_factor
    
    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    map_y = np.clip(y_indices + dy, 0, h - 1)
    map_x = np.clip(x_indices + dx, 0, w - 1)
    
    # 使用线性插值
    deformed_mask = cv2.remap(
        coarse_mask.astype(np.float32), 
        map_x.astype(np.float32), 
        map_y.astype(np.float32), 
        cv2.INTER_LINEAR
    )
    
    # 阈值化保持二值
    return (deformed_mask > 0.5).astype(coarse_mask.dtype)

def _generate_single_lasso_np(
    component_mask_np: np.ndarray,
    deformation_strength: float = 9.0, ########################关键修改2： 建议默认强度稍微调大
    **kwargs
    ) -> np.ndarray:
    """
    保留了原有的形态学预处理逻辑，仅调整了扩张系数以配合粗糙变形。
    """
    if component_mask_np.max() == 0:
        return np.zeros_like(component_mask_np)

    uint8_mask = component_mask_np.astype(np.uint8)
    
    dist_transform = cv2.distanceTransform(uint8_mask, cv2.DIST_L2, 5)
    
    max_dist = dist_transform.max()
    
    # 稍微增大一点初始扩张，给向内凹陷的扰动留出空间
    kernel_size = int(max_dist * 0.8) + 3
    kernel_size = max(5, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    coarse_mask = cv2.morphologyEx(uint8_mask, cv2.MORPH_CLOSE, kernel)
    coarse_mask = cv2.dilate(coarse_mask, kernel, iterations=1)
    
    return _apply_deformation(coarse_mask, deformation_strength)

def mask_to_lasso_2d(mask_2d: torch.Tensor, **kwargs) -> torch.Tensor:
    device, dtype = mask_2d.device, mask_2d.dtype
    mask_np = mask_2d.cpu().numpy().astype(np.uint8).copy()
    try:
        lasso_np = _generate_single_lasso_np(mask_np, **kwargs)
        return torch.from_numpy(lasso_np).to(device, dtype=dtype)
    except Exception:
        return torch.zeros_like(mask_2d)

def interactive_lasso_3d(mask_3d: torch.Tensor, prompt_mode: str = 'positive', max_prompts: int = 3, **kwargs) -> Tuple[torch.Tensor, List]:
    output_mask = torch.zeros_like(mask_3d, dtype=torch.float32)
    device = mask_3d.device
    mask_to_process = (mask_3d > 0) if prompt_mode == 'positive' else (mask_3d == 0)
    if not torch.any(mask_to_process): return output_mask, []
    
    num_prompts_to_gen = random.randint(1, max_prompts)
    selected_indices = _sample_global_indices(mask_to_process.float(), num_prompts_to_gen, strategy='random')
    
    generated_prompts = []
    
    for ax, idx in selected_indices:
        slice_np = get_slice(mask_3d, ax, idx).cpu().numpy()
        unique_intensities = np.unique(slice_np); unique_intensities = unique_intensities[unique_intensities > 0]
        combined_prompts_2d = torch.zeros_like(get_slice(mask_3d, 0, 0), device=device, dtype=torch.float32)
        
        for intensity in unique_intensities:
            labeled_mask, num_comps = ndi.label(slice_np == intensity, structure=np.ones((3, 3)))
            for i in range(1, num_comps + 1):
                comp_mask = torch.from_numpy(labeled_mask == i).to(device)
                if torch.any(comp_mask):
                    lasso = mask_to_lasso_2d(comp_mask, **kwargs)
                    combined_prompts_2d[lasso > 0] = intensity.item()
        
        if torch.any(combined_prompts_2d > 0):
            for k_offset in [-1, 0, 1]:
                k_abs = idx + k_offset
                if 0 <= k_abs < mask_3d.shape[ax]:
                    set_slice(output_mask, ax, k_abs, torch.maximum(get_slice(output_mask, ax, k_abs), combined_prompts_2d))
            generated_prompts.append(((ax, idx), combined_prompts_2d))
            
    return output_mask, generated_prompts

# ==============================================================================
# 3. Box, Point 和 Randomized Dense Slice 生成器
# ==============================================================================

def randomized_slice_rects_3d(mask_3d: torch.Tensor, max_slice: int = 5, var: float = 0.5, connectivity: int = 8) -> torch.Tensor:
    if mask_3d.ndim != 3:
        if mask_3d.ndim == 4 and mask_3d.shape[0] == 1: mask_3d = mask_3d[0]
        else: return torch.zeros_like(mask_3d)
        
    mask_bbox = torch.zeros_like(mask_3d)
    
    # 全局采样
    num_box = int(torch.randint(1, max_slice + 1, (1,)).item())
    selected_indices = _sample_global_indices(mask_3d, num_samples=num_box, strategy='random')
    
    for axis, k in selected_indices:
        slice_2d = get_slice(mask_3d, axis, k).cpu().numpy().astype(np.uint8)
        labeled, num = ndi.label(slice_2d, structure=np.ones((3,3), dtype=np.uint8))
        if num == 0: continue
        slices = ndi.find_objects(labeled)
        rect2d = torch.zeros_like(get_slice(mask_3d, axis, k))
        H, W = slice_2d.shape
        std = float(var) ** 0.5 if var > 0 else 0.0
        
        for slc in slices:
            if slc is None: continue
            ymin, ymax = slc[0].start, slc[0].stop
            xmin, xmax = slc[1].start, slc[1].stop
            if std > 0:
                off = torch.normal(mean=0.0, std=std, size=(4,)).round().int().tolist()
            else: off = [0,0,0,0]
            y0 = max(0, int(ymin) + off[0]); y1 = min(H, int(ymax) + off[1])
            x0 = max(0, int(xmin) + off[2]); x1 = min(W, int(xmax) + off[3])
            if y0 < y1 and x0 < x1:
                rect2d[y0:y1, x0:x1] = 1.0
        
        # 3层厚度
        for kk in (k - 1, k, k + 1):
            if 0 <= kk < mask_3d.shape[axis]:
                set_slice(mask_bbox, axis, kk, torch.maximum(get_slice(mask_bbox, axis, kk), rect2d))
                
    return mask_bbox

def randomized_slice_points_3d(mask_3d: torch.Tensor, radius: int = 6, max_slice: int = 3, connectivity: int = 8) -> torch.Tensor:
    if mask_3d.ndim != 3:
         if mask_3d.ndim == 4 and mask_3d.shape[0] == 1: mask_3d = mask_3d[0]
         else: return torch.zeros_like(mask_3d)
    device = mask_3d.device
    mask_out = torch.zeros_like(mask_3d, dtype=torch.float32)
    
    # 全局采样
    num_points = int(torch.randint(low=1, high=max_slice+1, size=(1,)).item())
    selected_indices = _sample_global_indices(mask_3d, num_samples=num_points, strategy='random')
    
    strel = ball(radius).astype(np.float32)
    sphere = torch.from_numpy(distance_transform_edt(strel)).to(device)
    sphere /= (sphere.max() + 1e-8)
    sz = sphere.shape
    
    for axis, k in selected_indices:
        slice_2d = get_slice(mask_3d, axis, k).cpu().numpy().astype(np.uint8)
        labeled, num = ndi.label(slice_2d, structure=np.ones((3,3), dtype=np.uint8))
        
        for l in range(1, num+1):
            coords = np.argwhere(labeled == l)
            if len(coords) == 0: continue
            pt_2d = coords[np.random.randint(len(coords))] 
            
            if axis == 0:   center = (k, pt_2d[0], pt_2d[1])
            elif axis == 1: center = (pt_2d[0], k, pt_2d[1])
            else:           center = (pt_2d[0], pt_2d[1], k)
            
            label_val = 1.0
            starts = [c - s//2 for c, s in zip(center, sz)]
            ends = [s + sz_i for s, sz_i in zip(starts, sz)]
            ranges = []
            sphere_ranges = []
            valid = True
            for i in range(3):
                s, e, d, sz_i = starts[i], ends[i], mask_3d.shape[i], sz[i]
                if s >= d or e <= 0: 
                    valid = False; break
                r_start = max(s, 0)
                r_end = min(e, d)
                ranges.append(slice(r_start, r_end))
                sp_start = max(0, -s)
                sp_end = min(sz_i, sz_i - (e - d))
                sphere_ranges.append(slice(sp_start, sp_end))
            if valid:
                mask_out[tuple(ranges)] = torch.maximum(mask_out[tuple(ranges)], sphere[tuple(sphere_ranges)] * label_val)
                
    return mask_out

def randomized_dense_slice_3d(mask_3d: torch.Tensor, max_slice: int = 1) -> torch.Tensor:
    """
    [New] Randomized Dense Slice 生成器 (原 MaxSlice)
    逻辑参考 Box:
    1. 决定生成几个切片 (1 到 max_slice)。
    2. 全局混合竞争 (strategy='random')：像素越多的切片被选中概率越大。
    3. 保持 GT 形状 (不是矩形)。
    4. 执行 3 层 (k-1, k, k+1) 赋值。
    """
    if mask_3d.ndim != 3:
        if mask_3d.ndim == 4 and mask_3d.shape[0] == 1: mask_3d = mask_3d[0]
        else: return torch.zeros_like(mask_3d)
        
    mask_out = torch.zeros_like(mask_3d)
    
    # 1. 决定生成数量
    num_prompts = int(torch.randint(1, max_slice + 1, (1,)).item())
    
    # 2. 全局加权采样
    selected_indices = _sample_global_indices(mask_3d, num_samples=num_prompts, strategy='random')
    
    # 3. 赋值操作
    for axis, k in selected_indices:
        # 获取该切片的原始 Mask
        # 使用 clone 确保在赋值时不会出现内存视图问题
        slice_mask = get_slice(mask_3d, axis, k).clone()
        
        # 3层赋值
        for kk in (k - 1, k, k + 1):
            if 0 <= kk < mask_3d.shape[axis]:
                # 使用 maximum 叠加，防止不同 slice 的扩展区域互相覆盖导致信息丢失
                set_slice(mask_out, axis, kk, torch.maximum(get_slice(mask_out, axis, kk), slice_mask))
            
    return mask_out



# ============================================================================================================================================================#
#                                                                             2D版本
# ============================================================================================================================================================#
from skimage.morphology import disk  # 注意这里对应 3D 的 ball

# ==============================================================================
# 4. 2D 专用 Prompt 生成器 (严谨复刻版)
# ==============================================================================

def _extract_2d_mask(mask_input: torch.Tensor) -> Tuple[np.ndarray, torch.device, torch.dtype, tuple]:
    """
    通用辅助函数：从 5D/4D/3D Tensor 中提取出唯一的 2D HxW Numpy 面。
    """
    device = mask_input.device
    dtype = mask_input.dtype
    original_shape = mask_input.shape
    
    # 尝试挤压维度
    mask_sq = mask_input.squeeze() 
    
    # 鲁棒性处理：确保获取 HxW
    if mask_sq.ndim == 2:
        mask_np = mask_sq.cpu().numpy()
    else:
        # 如果 squeeze 后不是 2D (例如 batch size 不为 1 或者是 1x1 图像)，尝试智能推断
        mask_np_raw = mask_input.cpu().numpy()
        # 找到所有空间维度 (>1)
        spatial_dims = [s for s in mask_np_raw.shape if s > 1]
        if len(spatial_dims) >= 2:
            # 假设最后两个有效维度是 H, W
            mask_np = mask_np_raw.reshape(-1, spatial_dims[-2], spatial_dims[-1])[0]
        else:
            # 极端保底
            mask_np = np.zeros((128, 128), dtype=np.uint8)
            
    return mask_np.astype(np.uint8), device, dtype, original_shape

def randomized_rects_2d(mask_input: torch.Tensor, max_rects: int = 1, var: float = 0.5) -> torch.Tensor:
    """
    [严谨复刻 Box] 对应 randomized_slice_rects_3d
    逻辑：
    1. 连通域分析 (ndi.label)
    2. 获取每个连通域的切片范围 (find_objects)
    3. 对坐标应用高斯扰动 (var)
    4. 填充矩形
    """
    mask_np, device, dtype, original_shape = _extract_2d_mask(mask_input)
    mask_bbox = np.zeros_like(mask_np, dtype=np.float32)
    
    # 1. 连通域标记
    labeled, num = ndi.label(mask_np, structure=np.ones((3, 3), dtype=np.uint8))
    
    if num > 0:
        # 2. 获取对象切片
        slices = ndi.find_objects(labeled)
        
        # 严格复刻 3D 逻辑：对每个连通域生成 Box
        # 注意：原 3D 代码是遍历所有 slices。如果需要限制数量，应在这里 shuffle 并截断
        # 这里为了保持一致性，处理所有连通域，但受外部概率控制
        
        H, W = mask_np.shape
        std = float(var) ** 0.5 if var > 0 else 0.0
        
        for slc in slices:
            if slc is None: continue
            
            # 提取原始边界
            ymin, ymax = slc[0].start, slc[0].stop
            xmin, xmax = slc[1].start, slc[1].stop
            
            # 3. 计算扰动 (严格复刻 torch.normal 逻辑)
            if std > 0:
                off = np.random.normal(loc=0.0, scale=std, size=4).round().astype(int)
            else:
                off = [0, 0, 0, 0]
            
            # 应用扰动并进行边界裁剪
            y0 = max(0, int(ymin) + off[0])
            y1 = min(H, int(ymax) + off[1])
            x0 = max(0, int(xmin) + off[2])
            x1 = min(W, int(xmax) + off[3])
            
            # 4. 填充
            if y0 < y1 and x0 < x1:
                mask_bbox[y0:y1, x0:x1] = 1.0

    prompt_tensor = torch.from_numpy(mask_bbox).to(device=device, dtype=dtype)
    return prompt_tensor.view(original_shape)

def randomized_points_2d(mask_input: torch.Tensor, radius: int = 6, max_points: int = 1) -> torch.Tensor:
    """
    [严谨复刻 Point] 对应 randomized_slice_points_3d
    逻辑：
    1. 生成 soft sphere (这里退化为 soft disk)
    2. 连通域分析
    3. 每个连通域随机取一点
    4. 叠加 soft disk
    """
    mask_np, device, dtype, original_shape = _extract_2d_mask(mask_input)
    mask_out = np.zeros_like(mask_np, dtype=np.float32)
    
    # 1. 准备 Soft Disk (严格复刻 ball -> distance_transform -> normalize)
    # 3D 用 ball(radius), 2D 用 disk(radius)
    strel = disk(radius).astype(np.float32)
    soft_disk = distance_transform_edt(strel)
    soft_disk /= (soft_disk.max() + 1e-8)
    sz = soft_disk.shape # (h_k, w_k)
    
    # 2. 连通域标记
    labeled, num = ndi.label(mask_np, structure=np.ones((3, 3), dtype=np.uint8))
    
    # 决定采样多少个连通域 (复刻 num_points 逻辑)
    # 原 3D 逻辑是先选切片，再在切片里的每个连通域都打点。
    # 这里我们对应为：在当前 2D 面上，对每个连通域打点。
    
    for l in range(1, num + 1):
        # 3. 随机取点
        coords = np.argwhere(labeled == l)
        if len(coords) == 0: continue
        
        # 随机选择中心点 center (y, x)
        pt_idx = np.random.randint(len(coords))
        center = coords[pt_idx] # [y, x]
        
        # 4. 放置 Soft Disk (处理边界)
        # 严格复刻 3D 代码中的切片计算逻辑
        starts = [c - s // 2 for c, s in zip(center, sz)]
        ends = [s + sz_i for s, sz_i in zip(starts, sz)]
        
        ranges = []        # 原图切片
        disk_ranges = []   # Disk 切片
        valid = True
        
        for i in range(2): # 遍历 y, x 轴
            s, e, d, sz_i = starts[i], ends[i], mask_np.shape[i], sz[i]
            if s >= d or e <= 0: 
                valid = False; break
            
            r_start = max(s, 0)
            r_end = min(e, d)
            ranges.append(slice(r_start, r_end))
            
            d_start = max(0, -s)
            d_end = min(sz_i, sz_i - (e - d))
            disk_ranges.append(slice(d_start, d_end))
            
        if valid:
            # 使用 maximum 叠加，模拟 soft sphere 效果
            mask_out[tuple(ranges)] = np.maximum(mask_out[tuple(ranges)], soft_disk[tuple(disk_ranges)])
            
    prompt_tensor = torch.from_numpy(mask_out).to(device=device, dtype=dtype)
    return prompt_tensor.view(original_shape)

def randomized_dense_2d(mask_input: torch.Tensor) -> torch.Tensor:
    """
    [严谨复刻 Dense] 对应 randomized_dense_slice_3d
    逻辑：
    直接返回 Mask 副本，因为 2D 模式下没有 "k-1, k, k+1" 的层间扩展。
    """
    return mask_input.clone()

# ==============================================================================
# Wrappers for Lasso and Scribble (适配接口)
# ==============================================================================

def interactive_lasso_2d(mask_input: torch.Tensor, deformation_strength: float = 9.0) -> torch.Tensor:
    """
    Wrapper for Lasso
    """
    mask_np, device, dtype, original_shape = _extract_2d_mask(mask_input)
    try:
        # 复用核心 Lasso 生成逻辑
        lasso_np = _generate_single_lasso_np(mask_np, deformation_strength=deformation_strength)
    except Exception:
        lasso_np = np.zeros_like(mask_np)
    
    prompt_tensor = torch.from_numpy(lasso_np).to(device=device, dtype=dtype)
    return prompt_tensor.view(original_shape)

def generate_scribbles_2d(mask_input: torch.Tensor, generator) -> torch.Tensor:
    """
    Wrapper for Scribble
    """
    mask_np, device, dtype, original_shape = _extract_2d_mask(mask_input)
    try:
        # 复用核心 Scribble 生成逻辑 (Line, Centerline, Contour)
        scribble_np = generator(mask_np)
    except Exception:
        scribble_np = np.zeros_like(mask_np)
        
    prompt_tensor = torch.from_numpy(scribble_np).to(device=device, dtype=dtype)
    return prompt_tensor.view(original_shape)
