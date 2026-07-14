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
# Basic Helper Functions
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
# Core: Global Slice Sampler
# ==============================================================================

def _sample_global_indices(mask_3d: torch.Tensor, num_samples: int = 1, strategy: str = 'random') -> List[Tuple[int, int]]:
    """
    Perform mixed competition among all slice pools across the three axes.
    Return: List of (axis, slice_index)
    """
    # Calculate the projection sum along the three axes (i.e., number of pixels per slice)
    w_ax0 = mask_3d.sum(dim=(1, 2)) 
    w_ax1 = mask_3d.sum(dim=(0, 2))
    w_ax2 = mask_3d.sum(dim=(0, 1))
    
    # Concatenate into a global weight vector
    all_weights = torch.cat([w_ax0, w_ax1, w_ax2]).float()
    
    # Filter out entirely empty (black) slices
    eligible_mask = (all_weights > 0)
    if eligible_mask.sum() == 0:
        return []
    
    # Strategy selection
    if strategy == 'max':
        # Select the slice with the maximum number of pixels (num_samples forced to 1)
        global_indices = [torch.argmax(all_weights).item()]
    else:
        # random: Weighted random sampling based on valid pixel counts
        num_valid = eligible_mask.sum().item()
        actual_samples = min(num_samples, num_valid)
        if actual_samples < 1: actual_samples = 1
        
        # Use multinomial for weighted sampling without replacement
        global_indices = torch.multinomial(all_weights, actual_samples, replacement=False).tolist()

    # Decode global indices into (axis, local_index)
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
# 1. Scribble Generators (Optimized to prevent execution deadlock)
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
        
        # [Optimization] If points are too few, select two randomly; if sufficient, execute smart long-line logic
        if len(points) < self.smart_line_min_points:
            idx1, idx2 = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx1], points[idx2]
        else:
            # Randomly select starting point
            p1_idx = random.randint(0, len(points) - 1)
            p1 = points[p1_idx]
            
            # [Crucial Optimization] Large array protection: randomly downsample if points exceed 2000
            # This prevents CPU deadlock caused by cdist when processing 10000+ points
            if len(points) > 2000:
                sample_indices = np.random.choice(len(points), 2000, replace=False)
                calc_points = points[sample_indices]
            else:
                calc_points = points
            
            # Calculate distances
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
            # Prioritize OpenCV's fast thinning algorithm
            skeleton_255 = cv2.ximgproc.thinning(opencv_mask)
        except (cv2.error, AttributeError):
            # [Crucial Optimization] Add max iteration limit to the fallback algorithm to prevent infinite loops
            skeleton_255 = np.zeros(opencv_mask.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp_img = opencv_mask.copy()
            
            max_iter = 100 # Safety brake
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
# 2. Lasso Generator (Retain custom Rough Lasso logic)
# ==============================================================================
'''
# Regular lasso
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
    Modified version: Create a rough Lasso effect while preserving the Exact Distance Transform (EDT).
    """
    h, w = coarse_mask.shape
    
    # [Modification 1] Generate high-frequency noise
    sigma_val = 7.0  # Crucial modification 1: sigma=50 yields smooth waves, sigma=4 yields rough jitter
    
    rand_x = np.random.rand(h, w) * 2 - 1
    rand_y = np.random.rand(h, w) * 2 - 1
    
    rand_x = ndi.gaussian_filter(rand_x, sigma=sigma_val, mode='reflect')
    rand_y = ndi.gaussian_filter(rand_y, sigma=sigma_val, mode='reflect')
    
    # [Modification 2] Retain and compute EDT
    edt = ndi.distance_transform_edt(coarse_mask)
    if edt.max() > 0: 
        edt = edt / edt.max()
    
    # [Crucial Adjustment] Add a baseline (Bias) to EDT
    weighted_map = edt + 0.4 
    
    norm_factor = 10.0 
    
    dx = deformation_strength * rand_x * weighted_map * norm_factor
    dy = deformation_strength * rand_y * weighted_map * norm_factor
    
    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    map_y = np.clip(y_indices + dy, 0, h - 1)
    map_x = np.clip(x_indices + dx, 0, w - 1)
    
    # Use linear interpolation
    deformed_mask = cv2.remap(
        coarse_mask.astype(np.float32), 
        map_x.astype(np.float32), 
        map_y.astype(np.float32), 
        cv2.INTER_LINEAR
    )
    
    # Thresholding to maintain binary mask
    return (deformed_mask > 0.5).astype(coarse_mask.dtype)

def _generate_single_lasso_np(
    component_mask_np: np.ndarray,
    deformation_strength: float = 9.0, # Crucial modification 2: Suggested to slightly increase default strength
    **kwargs
    ) -> np.ndarray:
    """
    Retains original morphological preprocessing logic, only adjusting the dilation coefficient 
    to match the rough deformation.
    """
    if component_mask_np.max() == 0:
        return np.zeros_like(component_mask_np)

    uint8_mask = component_mask_np.astype(np.uint8)
    
    dist_transform = cv2.distanceTransform(uint8_mask, cv2.DIST_L2, 5)
    
    max_dist = dist_transform.max()
    
    # Slightly increase initial dilation to leave room for inward concave perturbations
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
# 3. Box, Point, and Randomized Dense Slice Generators
# ==============================================================================

def randomized_slice_rects_3d(mask_3d: torch.Tensor, max_slice: int = 5, var: float = 0.5, connectivity: int = 8) -> torch.Tensor:
    if mask_3d.ndim != 3:
        if mask_3d.ndim == 4 and mask_3d.shape[0] == 1: mask_3d = mask_3d[0]
        else: return torch.zeros_like(mask_3d)
        
    mask_bbox = torch.zeros_like(mask_3d)
    
    # Global sampling
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
        
        # 3-layer thickness
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
    
    # Global sampling
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
    [New] Randomized Dense Slice Generator (formerly MaxSlice)
    Logic reference Box:
    1. Determine the number of slices to generate (1 to max_slice).
    2. Global mixed competition (strategy='random'): Slices with more pixels have a higher probability of being selected.
    3. Maintain GT shape (not rectangular).
    4. Execute 3-layer (k-1, k, k+1) assignments.
    """
    if mask_3d.ndim != 3:
        if mask_3d.ndim == 4 and mask_3d.shape[0] == 1: mask_3d = mask_3d[0]
        else: return torch.zeros_like(mask_3d)
        
    mask_out = torch.zeros_like(mask_3d)
    
    # 1. Determine the generation count
    num_prompts = int(torch.randint(1, max_slice + 1, (1,)).item())
    
    # 2. Global weighted sampling
    selected_indices = _sample_global_indices(mask_3d, num_samples=num_prompts, strategy='random')
    
    # 3. Assignment operation
    for axis, k in selected_indices:
        # Obtain the original Mask of the slice
        # Use clone to ensure no memory view issues during assignment
        slice_mask = get_slice(mask_3d, axis, k).clone()
        
        # 3-layer assignment
        for kk in (k - 1, k, k + 1):
            if 0 <= kk < mask_3d.shape[axis]:
                # Use maximum overlay to prevent expanded regions of different slices from overwriting each other, preventing information loss
                set_slice(mask_out, axis, kk, torch.maximum(get_slice(mask_out, axis, kk), slice_mask))
            
    return mask_out


# ============================================================================================================================================================#
#                                                                         2D Version
# ============================================================================================================================================================#
from skimage.morphology import disk  # Note: This corresponds to the 3D 'ball'

# ==============================================================================
# 4. 2D Specific Prompt Generators (Strictly Replicated Version)
# ==============================================================================

def _extract_2d_mask(mask_input: torch.Tensor) -> Tuple[np.ndarray, torch.device, torch.dtype, tuple]:
    """
    General helper function: Extract the unique 2D HxW Numpy plane from a 5D/4D/3D Tensor.
    """
    device = mask_input.device
    dtype = mask_input.dtype
    original_shape = mask_input.shape
    
    # Attempt to squeeze dimensions
    mask_sq = mask_input.squeeze() 
    
    # Robustness handling: Ensure HxW is obtained
    if mask_sq.ndim == 2:
        mask_np = mask_sq.cpu().numpy()
    else:
        # If not 2D after squeezing (e.g., batch size != 1 or 1x1 image), attempt smart inference
        mask_np_raw = mask_input.cpu().numpy()
        # Find all spatial dimensions (>1)
        spatial_dims = [s for s in mask_np_raw.shape if s > 1]
        if len(spatial_dims) >= 2:
            # Assume the last two valid dimensions are H, W
            mask_np = mask_np_raw.reshape(-1, spatial_dims[-2], spatial_dims[-1])[0]
        else:
            # Extreme fallback
            mask_np = np.zeros((128, 128), dtype=np.uint8)
            
    return mask_np.astype(np.uint8), device, dtype, original_shape

def randomized_rects_2d(mask_input: torch.Tensor, max_rects: int = 1, var: float = 0.5) -> torch.Tensor:
    """
    [Strict Box Replication] Corresponds to randomized_slice_rects_3d
    Logic:
    1. Connected component labeling (ndi.label)
    2. Obtain object slices (find_objects)
    3. Compute perturbation (var)
    4. Fill rectangles
    """
    mask_np, device, dtype, original_shape = _extract_2d_mask(mask_input)
    mask_bbox = np.zeros_like(mask_np, dtype=np.float32)
    
    # 1. Connected component labeling
    labeled, num = ndi.label(mask_np, structure=np.ones((3, 3), dtype=np.uint8))
    
    if num > 0:
        # 2. Obtain object slices
        slices = ndi.find_objects(labeled)
        
        # Strictly replicate 3D logic: Generate Box for each connected component
        # Note: Original 3D code iterates through all slices. If quantity limits are needed, shuffle and truncate here
        # Processed all connected components here to maintain consistency, controlled by external probabilities
        
        H, W = mask_np.shape
        std = float(var) ** 0.5 if var > 0 else 0.0
        
        for slc in slices:
            if slc is None: continue
            
            # Extract original boundaries
            ymin, ymax = slc[0].start, slc[0].stop
            xmin, xmax = slc[1].start, slc[1].stop
            
            # 3. Compute perturbation (Strictly replicate torch.normal logic)
            if std > 0:
                off = np.random.normal(loc=0.0, scale=std, size=4).round().astype(int)
            else:
                off = [0, 0, 0, 0]
            
            # Apply perturbation and perform boundary clipping
            y0 = max(0, int(ymin) + off[0])
            y1 = min(H, int(ymax) + off[1])
            x0 = max(0, int(xmin) + off[2])
            x1 = min(W, int(xmax) + off[3])
            
            # 4. Fill
            if y0 < y1 and x0 < x1:
                mask_bbox[y0:y1, x0:x1] = 1.0

    prompt_tensor = torch.from_numpy(mask_bbox).to(device=device, dtype=dtype)
    return prompt_tensor.view(original_shape)

def randomized_points_2d(mask_input: torch.Tensor, radius: int = 6, max_points: int = 1) -> torch.Tensor:
    """
    [Strict Point Replication] Corresponds to randomized_slice_points_3d
    Logic:
    1. Generate soft sphere (degrades to soft disk here)
    2. Connected component analysis
    3. Randomly select one point per connected component
    4. Overlay soft disk
    """
    mask_np, device, dtype, original_shape = _extract_2d_mask(mask_input)
    mask_out = np.zeros_like(mask_np, dtype=np.float32)
    
    # 1. Prepare Soft Disk (Strictly replicate ball -> distance_transform -> normalize)
    # Use ball(radius) for 3D, disk(radius) for 2D
    strel = disk(radius).astype(np.float32)
    soft_disk = distance_transform_edt(strel)
    soft_disk /= (soft_disk.max() + 1e-8)
    sz = soft_disk.shape # (h_k, w_k)
    
    # 2. Connected component labeling
    labeled, num = ndi.label(mask_np, structure=np.ones((3, 3), dtype=np.uint8))
    
    # Determine how many connected components to sample (replicate num_points logic)
    # Original 3D logic: Select slice first, then place points in each connected component within the slice.
    # Here we map this to: Place points on each connected component on the current 2D plane.
    
    for l in range(1, num + 1):
        # 3. Randomly select points
        coords = np.argwhere(labeled == l)
        if len(coords) == 0: continue
        
        # Randomly select center point (y, x)
        pt_idx = np.random.randint(len(coords))
        center = coords[pt_idx] # [y, x]
        
        # 4. Place Soft Disk (Handle boundaries)
        # Strictly replicate slice calculation logic from 3D code
        starts = [c - s // 2 for c, s in zip(center, sz)]
        ends = [s + sz_i for s, sz_i in zip(starts, sz)]
        
        ranges = []        # Original image slice
        disk_ranges = []   # Disk slice
        valid = True
        
        for i in range(2): # Iterate over y, x axes
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
            # Use maximum overlay to simulate soft sphere effect
            mask_out[tuple(ranges)] = np.maximum(mask_out[tuple(ranges)], soft_disk[tuple(disk_ranges)])
            
    prompt_tensor = torch.from_numpy(mask_out).to(device=device, dtype=dtype)
    return prompt_tensor.view(original_shape)

def randomized_dense_2d(mask_input: torch.Tensor) -> torch.Tensor:
    """
    [Strict Dense Replication] Corresponds to randomized_dense_slice_3d
    Logic:
    Directly return the Mask copy, as 2D mode lacks the "k-1, k, k+1" inter-layer expansion.
    """
    return mask_input.clone()

# ==============================================================================
# Wrappers for Lasso and Scribble (Adapter Interfaces)
# ==============================================================================

def interactive_lasso_2d(mask_input: torch.Tensor, deformation_strength: float = 9.0) -> torch.Tensor:
    """
    Wrapper for Lasso
    """
    mask_np, device, dtype, original_shape = _extract_2d_mask(mask_input)
    try:
        # Reuse core Lasso generation logic
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
        # Reuse core Scribble generation logic (Line, Centerline, Contour)
        scribble_np = generator(mask_np)
    except Exception:
        scribble_np = np.zeros_like(mask_np)
        
    prompt_tensor = torch.from_numpy(scribble_np).to(device=device, dtype=dtype)
    return prompt_tensor.view(original_shape)
