import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, List
import torchvision.transforms.functional as TF
from pathlib import Path


class ImagePreprocessor:
    """Preprocessing utilities for RGB and IR images"""
    
    @staticmethod
    def normalize_rgb(image: np.ndarray, range_out: Tuple[float, float] = (-1, 1)) -> np.ndarray:
        """Normalize RGB image to specified range"""
        image = image.astype(np.float32) / 255.0
        if range_out == (-1, 1):
            image = image * 2.0 - 1.0
        return image
    
    @staticmethod
    def normalize_ir(image: np.ndarray, 
                     range_out: Tuple[float, float] = (-1, 1),
                     mean: float = 0.5, 
                     std: float = 0.25) -> np.ndarray:
        """Normalize IR image with thermal-aware normalization"""
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        if range_out == (-1, 1):
            image = torch.clamp(torch.from_numpy(image), -2, 2) / 2.0
        
        return image.numpy() if isinstance(image, torch.Tensor) else image
    
    @staticmethod
    def compute_edge_map(image: np.ndarray, 
                        method: str = 'canny',
                        threshold1: int = 50,
                        threshold2: int = 150) -> np.ndarray:
        """
        Compute edge map from image
        
        Args:
            image: Input image
            method: 'canny' or 'sobel'
            threshold1: Lower threshold for Canny
            threshold2: Upper threshold for Canny
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        if method == 'canny':
            edges = cv2.Canny(image, threshold1, threshold2)
        elif method == 'sobel':
            sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = ((edges - edges.min()) / (edges.max() - edges.min() + 1e-8) * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown edge method: {method}")
        
        return edges.astype(np.float32) / 255.0


class ImagePostprocessor:
    """Postprocessing utilities for generated IR images"""
    
    @staticmethod
    def denormalize_ir(image: torch.Tensor,
                       range_in: Tuple[float, float] = (-1, 1),
                       mean: float = 0.5,
                       std: float = 0.25) -> torch.Tensor:
        """Denormalize IR image from normalized range"""
        if range_in == (-1, 1):
            image = image * 2.0 + 1.0
        
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        return image
    
    @staticmethod
    def thermal_color_map(ir_image: torch.Tensor, 
                         colormap: str = 'turbo') -> np.ndarray:
        """
        Apply thermal colormap to IR image for visualization
        
        Args:
            ir_image: IR image in [0, 1] range (C, H, W) or (H, W)
            colormap: 'turbo', 'jet', 'hot', 'plasma'
        
        Returns:
            RGB image (H, W, 3) in [0, 255]
        """
        if isinstance(ir_image, torch.Tensor):
            ir_image = ir_image.cpu().numpy()
        
        # Handle different input shapes
        if ir_image.ndim == 3:
            ir_image = ir_image[0] if ir_image.shape[0] == 1 else ir_image.mean(axis=0)
        
        # Normalize to [0, 255]
        ir_image = ((ir_image - ir_image.min()) / (ir_image.max() - ir_image.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        if colormap == 'turbo':
            colormap_cv = cv2.COLORMAP_TURBO
        elif colormap == 'jet':
            colormap_cv = cv2.COLORMAP_JET
        elif colormap == 'hot':
            colormap_cv = cv2.COLORMAP_HOT
        elif colormap == 'plasma':
            colormap_cv = cv2.COLORMAP_PLASMA
        else:
            colormap_cv = cv2.COLORMAP_TURBO
        
        colored = cv2.applyColorMap(ir_image, colormap_cv)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def denoise_ir(ir_image: torch.Tensor, 
                   strength: float = 0.5) -> torch.Tensor:
        """
        Apply mild denoising to IR image while preserving edges
        Uses bilateral filter characteristics
        """
        if isinstance(ir_image, torch.Tensor):
            ir_np = ir_image.cpu().numpy()
        else:
            ir_np = ir_image
        
        # Normalize to [0, 255] for filtering
        if ir_np.ndim == 3 and ir_np.shape[0] == 1:
            ir_np = ir_np[0]
        
        ir_normalized = ((ir_np - ir_np.min()) / (ir_np.max() - ir_np.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply bilateral filter (preserves edges)
        denoised = cv2.bilateralFilter(ir_normalized, 9, 75, 75)
        
        # Blend with original based on strength
        blended = (strength * denoised + (1 - strength) * ir_normalized).astype(np.float32) / 255.0
        
        if isinstance(ir_image, torch.Tensor):
            blended = torch.from_numpy(blended).to(ir_image.dtype).to(ir_image.device)
            if ir_image.ndim == 3:
                blended = blended.unsqueeze(0)
        
        return blended


def create_comparison_image(rgb: torch.Tensor,
                           ir_pred: torch.Tensor,
                           ir_gt: Optional[torch.Tensor] = None,
                           size: int = 512) -> Image.Image:
    """
    Create side-by-side comparison image
    
    Args:
        rgb: RGB image (3, H, W) in [-1, 1]
        ir_pred: Predicted IR (1, H, W) in [-1, 1]
        ir_gt: Ground truth IR (1, H, W) in [-1, 1], optional
        size: Output size
    
    Returns:
        PIL Image with comparison
    """
    # Convert to numpy and denormalize
    rgb_np = ((rgb.cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
    ir_pred_np = ImagePostprocessor.thermal_color_map(
        (ir_pred + 1) / 2, colormap='turbo'
    )
    
    # Resize to consistent size
    rgb_np = cv2.resize(rgb_np, (size, size))
    ir_pred_np = cv2.resize(ir_pred_np, (size, size))
    
    if ir_gt is not None:
        ir_gt_np = ImagePostprocessor.thermal_color_map(
            (ir_gt + 1) / 2, colormap='turbo'
        )
        ir_gt_np = cv2.resize(ir_gt_np, (size, size))
        # Create 3-row comparison
        comparison = np.vstack([rgb_np, ir_pred_np, ir_gt_np])
    else:
        # Create 2-row comparison
        comparison = np.vstack([rgb_np, ir_pred_np])
    
    return Image.fromarray(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))


def save_checkpoint(model, optimizer, scheduler, epoch, save_path: str):
    """Save training checkpoint"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else {},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else {},
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0)


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_learning_rate(optimizer, lr):
    """Set learning rate for optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class WarmupScheduler:
    """Learning rate warmup scheduler"""
    
    def __init__(self, optimizer, warmup_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
            set_learning_rate(self.optimizer, lr)


if __name__ == '__main__':
    # Test preprocessing
    dummy_rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    normalized = ImagePreprocessor.normalize_rgb(dummy_rgb)
    print(f"RGB normalized: min={normalized.min():.2f}, max={normalized.max():.2f}")
    
    # Test edge detection
    edges = ImagePreprocessor.compute_edge_map(dummy_rgb)
    print(f"Edge map: min={edges.min():.2f}, max={edges.max():.2f}")
