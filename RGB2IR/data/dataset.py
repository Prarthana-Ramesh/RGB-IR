import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import cv2
from PIL import Image
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RGBIRPairedDataset(Dataset):
    """
    Dataset for paired RGB and IR images with optional depth and structure maps
    
    Directory structure:
    dataset_root/
    ├── split/  (train, val, test)
    │   ├── rgb/
    │   │   ├── image_001.png
    │   │   └── ...
    │   ├── ir/
    │   │   ├── image_001.png
    │   │   └── ...
    │   ├── depth/  (optional)
    │   │   ├── image_001.npy or .png
    │   │   └── ...
    │   └── masks/  (optional, for material segmentation)
    │       ├── image_001.png
    │       └── ...
    """
    
    def __init__(self,
                 dataset_root: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (512, 512),
                 use_depth: bool = True,
                 use_augmentation: bool = True,
                 rgb_normalize: bool = True,
                 ir_normalize: bool = True,
                 normalize_params: Optional[Dict] = None):
        """
        Args:
            dataset_root: Root directory of dataset
            split: 'train', 'val', or 'test'
            image_size: Target image size (H, W)
            use_depth: Whether to load depth maps for ControlNet
            use_augmentation: Whether to apply augmentation
            rgb_normalize: Normalize RGB images to [-1, 1]
            ir_normalize: Normalize IR images to [-1, 1]
            normalize_params: Dict with 'mean' and 'std' for IR normalization
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.use_depth = use_depth
        self.use_augmentation = use_augmentation
        self.rgb_normalize = rgb_normalize
        self.ir_normalize = ir_normalize
        
        # Default IR normalization parameters (8-14 micron thermal range)
        if normalize_params is None:
            self.normalize_params = {
                'mean': 0.5,  # Center around middle value
                'std': 0.25   # Scale to roughly [-1, 1]
            }
        else:
            self.normalize_params = normalize_params
        
        # Construct paths
        self.rgb_dir = self.dataset_root / split / 'rgb'
        self.ir_dir = self.dataset_root / split / 'ir'
        self.depth_dir = self.dataset_root / split / 'depth'
        self.mask_dir = self.dataset_root / split / 'masks'
        
        # Get list of RGB images (use as reference)
        self.rgb_images = sorted(list(self.rgb_dir.glob('*.png'))) + \
                          sorted(list(self.rgb_dir.glob('*.jpg'))) + \
                          sorted(list(self.rgb_dir.glob('*.jpeg')))
        
        if len(self.rgb_images) == 0:
            raise ValueError(f"No RGB images found in {self.rgb_dir}")
        
        # Setup augmentation
        self.transform = self._get_augmentation() if use_augmentation else None
    
    def _get_augmentation(self):
        """Define albumentations pipeline"""
        # Note: Images are already resized in _load_image, so no need for Resize here
        # Note: GaussNoise removed because it doesn't work well with mixed channel images
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ToFloat(max_value=255.0),
        ], additional_targets={
            'ir': 'image',
            'depth': 'image',
            'mask': 'image'
        })
    
    def _load_image(self, path: Path, is_ir: bool = False) -> np.ndarray:
        """Load image from file"""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Convert BGR to RGB for RGB images
        if not is_ir:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # IR images are typically single channel, expand if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
        
        # Resize to target size to ensure consistent shapes before augmentation
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        
        return img
    
    def _load_depth(self, img_name: str) -> Optional[np.ndarray]:
        """Load depth map if available"""
        if not self.use_depth or not self.depth_dir.exists():
            return None
        
        stem = Path(img_name).stem
        depth_npy = self.depth_dir / f"{stem}.npy"
        depth_png = self.depth_dir / f"{stem}.png"
        
        if depth_npy.exists():
            depth = np.load(depth_npy).astype(np.float32)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
            # Resize to target size for consistency
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
        elif depth_png.exists():
            depth = self._load_image(depth_png, is_ir=True)
        else:
            return None
        
        return depth
    
    def _load_mask(self, img_name: str) -> Optional[np.ndarray]:
        """Load material segmentation mask if available"""
        if not self.mask_dir.exists():
            return None
        
        stem = Path(img_name).stem
        mask_path = self.mask_dir / f"{stem}.png"
        
        if mask_path.exists():
            return self._load_image(mask_path, is_ir=True)
        return None
    
    def _compute_canny_edges(self, rgb_image: np.ndarray) -> np.ndarray:
        """Compute Canny edges for ControlNet"""
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_image.astype(np.uint8)
        
        edges = cv2.Canny(gray, 100, 200)
        edges = edges.astype(np.float32) / 255.0
        return np.expand_dims(edges, axis=-1)
    
    def _compute_depth_from_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB if depth not available
        Uses Laplacian as proxy for structure
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (rgb_image * 255).astype(np.uint8)
        
        # Compute Laplacian as edge/structure indicator
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        depth = cv2.GaussianBlur(np.abs(laplacian), (5, 5), 0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return np.expand_dims(depth.astype(np.float32), axis=-1)
    
    def __len__(self) -> int:
        return len(self.rgb_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path = self.rgb_images[idx]
        img_name = rgb_path.name
        stem = rgb_path.stem
        
        # Load RGB image
        rgb_img = self._load_image(rgb_path, is_ir=False)
        
        # Load IR image (with same name)
        ir_path = self.ir_dir / rgb_path.name
        ir_img = self._load_image(ir_path, is_ir=True)
        
        # Load optional depth
        depth = self._load_depth(img_name)
        if depth is None:
            # Estimate from RGB if not available
            depth = self._compute_depth_from_rgb(rgb_img)
        
        # Compute Canny edges from RGB
        canny_edges = self._compute_canny_edges(rgb_img)
        
        # Load optional mask
        mask = self._load_mask(img_name)
        
        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(
                image=rgb_img,
                ir=ir_img,
                depth=depth,
                mask=mask if mask is not None else np.zeros_like(ir_img)
            )
            rgb_img = augmented['image']
            ir_img = augmented['ir']
            depth = augmented['depth']
            mask = augmented['mask'] if mask is not None else None
        else:
            # Just resize
            rgb_img = cv2.resize(rgb_img, (self.image_size[1], self.image_size[0]))
            ir_img = cv2.resize(ir_img, (self.image_size[1], self.image_size[0]))
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))
        
        # Normalize RGB to [-1, 1]
        rgb_tensor = torch.from_numpy(rgb_img).float()
        if self.rgb_normalize:
            rgb_tensor = rgb_tensor / 127.5 - 1.0
        else:
            rgb_tensor = rgb_tensor / 255.0
        
        # Normalize IR to [-1, 1]
        ir_tensor = torch.from_numpy(ir_img).float()
        if self.ir_normalize:
            ir_tensor = (ir_tensor / 255.0 - self.normalize_params['mean']) / self.normalize_params['std']
        else:
            ir_tensor = ir_tensor / 255.0
        
        # Ensure correct shape (C, H, W)
        if rgb_tensor.dim() == 2:
            rgb_tensor = rgb_tensor.unsqueeze(0)
        elif rgb_tensor.shape[0] != 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)
        
        if ir_tensor.dim() == 2:
            ir_tensor = ir_tensor.unsqueeze(0)
        elif ir_tensor.shape[0] != 1:
            ir_tensor = ir_tensor.permute(2, 0, 1)[:1]  # Take first channel
        
        # Normalize depth to [0, 1]
        depth_tensor = torch.from_numpy(depth).float()
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        elif depth_tensor.shape[0] != 1:
            depth_tensor = depth_tensor.permute(2, 0, 1)[:1]
        
        depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)
        depth_tensor = depth_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Compute canny edges (already float and normalized)
        canny_tensor = torch.from_numpy(canny_edges).float()
        if canny_tensor.dim() == 2:
            canny_tensor = canny_tensor.unsqueeze(0)
        elif canny_tensor.shape[0] != 1:
            canny_tensor = canny_tensor.permute(2, 0, 1)[:1]
        
        canny_tensor = canny_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        
        sample = {
            'rgb': rgb_tensor,
            'ir': ir_tensor,
            'depth': depth_tensor,
            'canny_edges': canny_tensor,
            'filename': img_name,
            'stem': stem
        }
        
        # Add mask if available
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.shape[0] != 1:
                mask_tensor = mask_tensor.permute(2, 0, 1)[:1]
            sample['mask'] = mask_tensor
        
        return sample


def create_dataloaders(dataset_root: str,
                       batch_size: int = 4,
                       num_workers: int = 4,
                       image_size: Tuple[int, int] = (512, 512),
                       use_augmentation: bool = True,
                       splits: List[str] = ['train', 'val']) -> Dict[str, DataLoader]:
    """
    Create dataloaders for different splits
    
    Args:
        dataset_root: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Target image size
        use_augmentation: Whether to apply augmentation
        splits: List of splits to load ('train', 'val', 'test')
    
    Returns:
        Dictionary of dataloaders for each split
    """
    dataloaders = {}
    
    for split in splits:
        dataset = RGBIRPairedDataset(
            dataset_root=dataset_root,
            split=split,
            image_size=image_size,
            use_augmentation=use_augmentation and (split == 'train'),
            use_depth=True,
            rgb_normalize=True,
            ir_normalize=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders


if __name__ == '__main__':
    # Example usage
    dataset = RGBIRPairedDataset(
        dataset_root='./data/RGB2IR_dataset',
        split='train',
        image_size=(512, 512),
        use_augmentation=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"IR shape: {sample['ir'].shape}")
    print(f"Depth shape: {sample['depth'].shape}")
    print(f"Canny edges shape: {sample['canny_edges'].shape}")
