#!/usr/bin/env python3
"""
Data preparation utility for RGB-IR dataset
Helps organize, validate, and prepare your aligned image pairs
"""

import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import json
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Utility class for dataset preparation"""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.splits = ['train', 'val', 'test']
    
    def validate_pairs(self) -> dict:
        """Validate that RGB and IR image pairs exist and match"""
        logger.info("Validating image pairs...")
        results = {}
        
        for split in self.splits:
            split_dir = self.dataset_root / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                results[split] = {'valid': 0, 'missing_ir': 0, 'missing_rgb': 0, 'size_mismatch': 0}
                continue
            
            rgb_dir = split_dir / 'rgb'
            ir_dir = split_dir / 'ir'
            
            valid = 0
            missing_ir = 0
            missing_rgb = 0
            size_mismatch = 0
            
            if not rgb_dir.exists():
                logger.error(f"RGB directory not found: {rgb_dir}")
                continue
            
            rgb_images = list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg'))
            
            for rgb_path in tqdm(rgb_images, desc=f"Validating {split}"):
                # Check IR exists
                ir_path = ir_dir / rgb_path.name
                if not ir_path.exists():
                    missing_ir += 1
                    continue
                
                # Check sizes match
                try:
                    rgb_img = Image.open(rgb_path)
                    ir_img = Image.open(ir_path)
                    
                    if rgb_img.size != ir_img.size:
                        size_mismatch += 1
                        logger.warning(f"Size mismatch: {rgb_path.name} "
                                     f"{rgb_img.size} vs {ir_img.size}")
                    else:
                        valid += 1
                except Exception as e:
                    logger.error(f"Error loading {rgb_path.name}: {e}")
            
            results[split] = {
                'valid': valid,
                'missing_ir': missing_ir,
                'missing_rgb': missing_rgb,
                'size_mismatch': size_mismatch
            }
        
        return results
    
    def compute_depth_maps(self, split: str = 'train'):
        """Compute and save depth maps from RGB images using edge detection"""
        logger.info(f"Computing depth maps for {split}...")
        
        split_dir = self.dataset_root / split
        rgb_dir = split_dir / 'rgb'
        depth_dir = split_dir / 'depth'
        depth_dir.mkdir(parents=True, exist_ok=True)
        
        rgb_images = list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg'))
        
        for rgb_path in tqdm(rgb_images, desc=f"Computing depths"):
            # Load RGB
            rgb_img = cv2.imread(str(rgb_path))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            
            # Compute depth as Laplacian edges
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            depth = cv2.GaussianBlur(np.abs(laplacian), (5, 5), 0)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            # Save depth
            depth_path = depth_dir / f"{rgb_path.stem}.npy"
            np.save(depth_path, depth.astype(np.float32))
        
        logger.info(f"Saved {len(list(depth_dir.glob('*.npy')))} depth maps")
    
    def compute_statistics(self, split: str = 'train') -> dict:
        """Compute dataset statistics for normalization"""
        logger.info(f"Computing statistics for {split}...")
        
        split_dir = self.dataset_root / split
        rgb_dir = split_dir / 'rgb'
        ir_dir = split_dir / 'ir'
        
        rgb_means = []
        rgb_stds = []
        ir_means = []
        ir_stds = []
        ir_mins = []
        ir_maxs = []
        
        rgb_images = list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg'))
        
        for rgb_path in tqdm(rgb_images, desc=f"Computing stats"):
            # RGB stats
            rgb_img = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
            rgb_means.append(rgb_img.mean())
            rgb_stds.append(rgb_img.std())
            
            # IR stats
            ir_path = ir_dir / rgb_path.name
            ir_img = np.array(Image.open(ir_path)).astype(np.float32) / 255.0
            ir_means.append(ir_img.mean())
            ir_stds.append(ir_img.std())
            ir_mins.append(ir_img.min())
            ir_maxs.append(ir_img.max())
        
        stats = {
            'rgb': {
                'mean': np.mean(rgb_means),
                'std': np.mean(rgb_stds),
                'global_mean': np.concatenate([[np.array(Image.open(p)).astype(np.float32) / 255.0].mean() 
                                              for p in rgb_images]).mean()
            },
            'ir': {
                'mean': np.mean(ir_means),
                'std': np.mean(ir_stds),
                'min': np.mean(ir_mins),
                'max': np.mean(ir_maxs)
            }
        }
        
        return stats
    
    def create_metadata(self, split: str = 'train'):
        """Create metadata JSON file"""
        logger.info(f"Creating metadata for {split}...")
        
        split_dir = self.dataset_root / split
        rgb_dir = split_dir / 'rgb'
        
        rgb_images = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
        
        metadata = {
            'split': split,
            'num_samples': len(rgb_images),
            'samples': []
        }
        
        for rgb_path in rgb_images:
            metadata['samples'].append({
                'filename': rgb_path.name,
                'stem': rgb_path.stem,
                'size': Image.open(rgb_path).size
            })
        
        # Save metadata
        metadata_path = split_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def resize_images(self, split: str, size: Tuple[int, int]):
        """Resize all images to target size"""
        logger.info(f"Resizing images in {split} to {size}...")
        
        split_dir = self.dataset_root / split
        rgb_dir = split_dir / 'rgb'
        ir_dir = split_dir / 'ir'
        
        for rgb_path in tqdm(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg'))):
            # Resize RGB
            rgb_img = Image.open(rgb_path)
            rgb_resized = rgb_img.resize(size, Image.Resampling.LANCZOS)
            rgb_resized.save(rgb_path)
            
            # Resize IR
            ir_path = ir_dir / rgb_path.name
            ir_img = Image.open(ir_path)
            ir_resized = ir_img.resize(size, Image.Resampling.LANCZOS)
            ir_resized.save(ir_path)
        
        logger.info(f"Resized {len(list(rgb_dir.glob('*.png')))} image pairs")


def main():
    parser = argparse.ArgumentParser(description='Prepare RGB-IR dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--validate', action='store_true', help='Validate image pairs')
    parser.add_argument('--compute_depths', action='store_true', help='Compute depth maps')
    parser.add_argument('--compute_stats', action='store_true', help='Compute dataset statistics')
    parser.add_argument('--create_metadata', action='store_true', help='Create metadata files')
    parser.add_argument('--resize', type=int, nargs=2, default=None, help='Resize to height width')
    parser.add_argument('--split', type=str, default='train', help='Split to process')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.dataset_root)
    
    if args.validate:
        results = preparer.validate_pairs()
        logger.info("\nValidation Results:")
        for split, metrics in results.items():
            logger.info(f"\n{split.upper()}:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
    
    if args.compute_depths:
        preparer.compute_depth_maps(args.split)
    
    if args.compute_stats:
        stats = preparer.compute_statistics(args.split)
        logger.info(f"\nStatistics for {args.split}:")
        logger.info(f"RGB: mean={stats['rgb']['mean']:.4f}, std={stats['rgb']['std']:.4f}")
        logger.info(f"IR: mean={stats['ir']['mean']:.4f}, std={stats['ir']['std']:.4f}, "
                   f"min={stats['ir']['min']:.4f}, max={stats['ir']['max']:.4f}")
    
    if args.create_metadata:
        preparer.create_metadata(args.split)
    
    if args.resize:
        preparer.resize_images(args.split, tuple(args.resize))


if __name__ == '__main__':
    main()
