#!/usr/bin/env python3
"""
Rename RGB-IR dataset to sequential format
Renames paired RGB and IR images to scene_001.png, scene_002.png, etc.
Also handles splitting all images from a single folder into train/val/test splits.
"""

import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import shutil
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetRenamer:
    """Utility class for renaming RGB-IR dataset to sequential format"""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.splits = ['train', 'val', 'test']
    
    def split_and_rename(self, rgb_folder: str, ir_folder: str,
                        train_ratio: float = 0.7, val_ratio: float = 0.15,
                        test_ratio: float = 0.15, prefix: str = 'scene',
                        start_index: int = 1, padding: int = 3,
                        extension: str = '.png', seed: int = 42,
                        dry_run: bool = False):
        """
        Split all images from a single folder into train/val/test and rename sequentially
        
        Args:
            rgb_folder: Path to folder containing all RGB images
            ir_folder: Path to folder containing all IR images
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.15)
            test_ratio: Proportion for test set (default: 0.15)
            prefix: Prefix for renamed files (default: 'scene')
            start_index: Starting index (default: 1)
            padding: Number of digits for zero-padding (default: 3)
            extension: File extension to use (default: '.png')
            seed: Random seed for reproducible splits (default: 42)
            dry_run: If True, only show what would be done without actually doing it
        """
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Splitting and renaming dataset...")
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            logger.error(f"Split ratios must sum to 1.0, got {total}")
            return
        
        rgb_folder = Path(rgb_folder)
        ir_folder = Path(ir_folder)
        
        if not rgb_folder.exists():
            logger.error(f"RGB folder not found: {rgb_folder}")
            return
        if not ir_folder.exists():
            logger.error(f"IR folder not found: {ir_folder}")
            return
        
        # Get all RGB images and sort
        rgb_images = sorted(list(rgb_folder.glob('*.png')) + 
                           list(rgb_folder.glob('*.jpg')) + 
                           list(rgb_folder.glob('*.jpeg')))
        
        if len(rgb_images) == 0:
            logger.error(f"No RGB images found in {rgb_folder}")
            return
        
        logger.info(f"Found {len(rgb_images)} RGB images")
        
        # Verify all RGB images have corresponding IR images
        valid_pairs = []
        for rgb_path in rgb_images:
            ir_path = ir_folder / rgb_path.name
            if ir_path.exists():
                valid_pairs.append((rgb_path, ir_path))
            else:
                logger.warning(f"Missing IR pair for {rgb_path.name}")
        
        logger.info(f"Found {len(valid_pairs)} valid RGB-IR pairs")
        
        if len(valid_pairs) == 0:
            logger.error("No valid RGB-IR pairs found!")
            return
        
        # Shuffle for random split
        random.seed(seed)
        random.shuffle(valid_pairs)
        
        # Calculate split sizes
        n_total = len(valid_pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Remainder goes to test
        
        # Split the data
        train_pairs = valid_pairs[:n_train]
        val_pairs = valid_pairs[n_train:n_train + n_val]
        test_pairs = valid_pairs[n_train + n_val:]
        
        logger.info(f"\nSplit distribution:")
        logger.info(f"  Train: {len(train_pairs)} images ({len(train_pairs)/n_total*100:.1f}%)")
        logger.info(f"  Val:   {len(val_pairs)} images ({len(val_pairs)/n_total*100:.1f}%)")
        logger.info(f"  Test:  {len(test_pairs)} images ({len(test_pairs)/n_total*100:.1f}%)")
        
        if dry_run:
            logger.info("\n[DRY RUN] Preview of first 5 images per split:")
            logger.info("\nTrain:")
            for i, (rgb, ir) in enumerate(train_pairs[:5], 1):
                new_name = f"{prefix}_{i:0{padding}d}{extension}"
                logger.info(f"  {rgb.name} → {new_name}")
            
            logger.info("\nVal:")
            for i, (rgb, ir) in enumerate(val_pairs[:5], 1):
                new_name = f"{prefix}_{i:0{padding}d}{extension}"
                logger.info(f"  {rgb.name} → {new_name}")
            
            logger.info("\nTest:")
            for i, (rgb, ir) in enumerate(test_pairs[:5], 1):
                new_name = f"{prefix}_{i:0{padding}d}{extension}"
                logger.info(f"  {rgb.name} → {new_name}")
            
            logger.info("\n[DRY RUN] No files were created or moved. Remove --dry_run to apply changes.")
            return
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for subdir in ['rgb', 'ir', 'depth', 'masks']:
                (self.dataset_root / split / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info("\nCreated directory structure")
        
        # Process each split
        splits_data = [
            ('train', train_pairs),
            ('val', val_pairs),
            ('test', test_pairs)
        ]
        
        for split_name, pairs in splits_data:
            logger.info(f"\nProcessing {split_name} split...")
            self._copy_and_rename_pairs(
                pairs, split_name, prefix, start_index, padding, extension
            )
        
        logger.info(f"\n{'='*60}")
        logger.info("Dataset splitting and renaming complete!")
        logger.info(f"Output directory: {self.dataset_root}")
        logger.info(f"{'='*60}\n")
    
    def _copy_and_rename_pairs(self, pairs: List, split_name: str,
                               prefix: str, start_index: int,
                               padding: int, extension: str):
        """Copy and rename RGB-IR pairs to split directory"""
        split_dir = self.dataset_root / split_name
        rgb_out = split_dir / 'rgb'
        ir_out = split_dir / 'ir'
        
        success_count = 0
        errors = []
        
        for idx, (rgb_path, ir_path) in enumerate(tqdm(pairs, desc=f"Copying {split_name}"), start=start_index):
            new_name = f"{prefix}_{idx:0{padding}d}{extension}"
            
            try:
                # Copy RGB
                shutil.copy2(rgb_path, rgb_out / new_name)
                
                # Copy IR
                shutil.copy2(ir_path, ir_out / new_name)
                
                success_count += 1
                
            except Exception as e:
                error_msg = f"Error processing {rgb_path.name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.info(f"  Successfully copied: {success_count} pairs")
        if errors:
            logger.info(f"  Errors: {len(errors)}")
            for error in errors[:5]:
                logger.info(f"    - {error}")
    
    def rename_to_sequential(self, split: str, prefix: str = 'scene', 
                            start_index: int = 1, padding: int = 3,
                            extension: str = '.png', dry_run: bool = False):
        """
        Rename RGB-IR image pairs to sequential format (e.g., scene_001.png, scene_002.png)
        
        Args:
            split: 'train', 'val', or 'test'
            prefix: Prefix for renamed files (default: 'scene')
            start_index: Starting index (default: 1)
            padding: Number of digits for zero-padding (default: 3)
            extension: File extension to use (default: '.png')
            dry_run: If True, only show what would be renamed without actually renaming
        """
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Renaming images in {split} to sequential format...")
        
        split_dir = self.dataset_root / split
        rgb_dir = split_dir / 'rgb'
        ir_dir = split_dir / 'ir'
        depth_dir = split_dir / 'depth'
        mask_dir = split_dir / 'masks'
        
        if not rgb_dir.exists():
            logger.error(f"RGB directory not found: {rgb_dir}")
            return
        
        # Get all RGB images and sort them for consistent ordering
        rgb_images = sorted(list(rgb_dir.glob('*.png')) + 
                           list(rgb_dir.glob('*.jpg')) + 
                           list(rgb_dir.glob('*.jpeg')))
        
        if len(rgb_images) == 0:
            logger.warning(f"No RGB images found in {rgb_dir}")
            return
        
        logger.info(f"Found {len(rgb_images)} RGB images")
        
        # Create mapping of old names to new names
        rename_mapping = []
        for idx, rgb_path in enumerate(rgb_images, start=start_index):
            new_name = f"{prefix}_{idx:0{padding}d}{extension}"
            rename_mapping.append({
                'old_name': rgb_path.name,
                'new_name': new_name,
                'index': idx
            })
        
        # Show preview
        logger.info("\nRename preview (first 10):")
        for item in rename_mapping[:10]:
            logger.info(f"  {item['old_name']} → {item['new_name']}")
        if len(rename_mapping) > 10:
            logger.info(f"  ... and {len(rename_mapping) - 10} more")
        
        if dry_run:
            logger.info("\n[DRY RUN] No files were renamed. Remove --dry_run to apply changes.")
            return
        
        # Confirm before proceeding
        logger.info(f"\nAbout to rename {len(rename_mapping)} image pairs")
        
        # Perform renaming
        renamed_count = 0
        errors = []
        
        for mapping in tqdm(rename_mapping, desc="Renaming files"):
            old_name = mapping['old_name']
            new_name = mapping['new_name']
            
            try:
                # Get old paths
                old_rgb_path = rgb_dir / old_name
                old_ir_path = ir_dir / old_name
                
                # Get new paths
                new_rgb_path = rgb_dir / new_name
                new_ir_path = ir_dir / new_name
                
                # Check if IR exists
                if not old_ir_path.exists():
                    logger.warning(f"IR image not found for {old_name}, skipping")
                    errors.append(f"Missing IR: {old_name}")
                    continue
                
                # Rename RGB
                old_rgb_path.rename(new_rgb_path)
                
                # Rename IR
                old_ir_path.rename(new_ir_path)
                
                # Rename depth if exists
                if depth_dir.exists():
                    old_stem = Path(old_name).stem
                    old_depth_npy = depth_dir / f"{old_stem}.npy"
                    old_depth_png = depth_dir / f"{old_stem}.png"
                    new_stem = Path(new_name).stem
                    
                    if old_depth_npy.exists():
                        new_depth_path = depth_dir / f"{new_stem}.npy"
                        old_depth_npy.rename(new_depth_path)
                    elif old_depth_png.exists():
                        new_depth_path = depth_dir / f"{new_stem}.png"
                        old_depth_png.rename(new_depth_path)
                
                # Rename mask if exists
                if mask_dir.exists():
                    old_mask_path = mask_dir / old_name
                    if old_mask_path.exists():
                        new_mask_path = mask_dir / new_name
                        old_mask_path.rename(new_mask_path)
                
                renamed_count += 1
                
            except Exception as e:
                error_msg = f"Error renaming {old_name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Renaming complete!")
        logger.info(f"Successfully renamed: {renamed_count} image pairs")
        logger.info(f"Errors: {len(errors)}")
        if errors:
            logger.info("\nErrors encountered:")
            for error in errors[:10]:
                logger.info(f"  - {error}")
            if len(errors) > 10:
                logger.info(f"  ... and {len(errors) - 10} more errors")
        logger.info(f"{'='*60}\n")
    
    def rename_all_splits(self, prefix: str = 'scene', 
                         start_index: int = 1, padding: int = 3,
                         extension: str = '.png', dry_run: bool = False):
        """
        Rename all splits (train, val, test) to sequential format
        
        Args:
            prefix: Prefix for renamed files (default: 'scene')
            start_index: Starting index (default: 1)
            padding: Number of digits for zero-padding (default: 3)
            extension: File extension to use (default: '.png')
            dry_run: If True, only show what would be renamed without actually renaming
        """
        for split in self.splits:
            split_dir = self.dataset_root / split
            if split_dir.exists():
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {split} split")
                logger.info(f"{'='*60}")
                self.rename_to_sequential(
                    split=split,
                    prefix=prefix,
                    start_index=start_index,
                    padding=padding,
                    extension=extension,
                    dry_run=dry_run
                )
            else:
                logger.warning(f"Split directory not found: {split_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Split and rename RGB-IR dataset to sequential format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  MODE 1: Split all images from single folder into train/val/test
  --------------------------------------------------------------
  # Preview split (dry run)
  python rename_dataset.py --mode split --dataset_root data/RGB2IR_dataset \\
    --rgb_folder path/to/all/rgb --ir_folder path/to/all/ir --dry_run
  
  # Actually split and rename (70/15/15 split)
  python rename_dataset.py --mode split --dataset_root data/RGB2IR_dataset \\
    --rgb_folder path/to/all/rgb --ir_folder path/to/all/ir
  
  # Custom split ratios (80/10/10)
  python rename_dataset.py --mode split --dataset_root data/RGB2IR_dataset \\
    --rgb_folder path/to/all/rgb --ir_folder path/to/all/ir \\
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
  
  MODE 2: Rename existing splits
  ------------------------------
  # Preview changes (dry run)
  python rename_dataset.py --mode rename --dataset_root data/RGB2IR_dataset \\
    --split train --dry_run
  
  # Rename train split
  python rename_dataset.py --mode rename --dataset_root data/RGB2IR_dataset \\
    --split train
  
  # Rename all splits
  python rename_dataset.py --mode rename --dataset_root data/RGB2IR_dataset --all
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['split', 'rename'],
                       help='split: Split all images from single folder | rename: Rename existing splits')
    parser.add_argument('--dataset_root', type=str, required=True, 
                       help='Path to output dataset root directory')
    
    # Mode: split
    parser.add_argument('--rgb_folder', type=str,
                       help='[split mode] Path to folder with all RGB images')
    parser.add_argument('--ir_folder', type=str,
                       help='[split mode] Path to folder with all IR images')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='[split mode] Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='[split mode] Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='[split mode] Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='[split mode] Random seed for reproducible splits (default: 42)')
    
    # Mode: rename
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'val', 'test'],
                       help='[rename mode] Split to process (default: train)')
    parser.add_argument('--all', action='store_true', 
                       help='[rename mode] Process all splits (train, val, test)')
    
    # Common arguments
    parser.add_argument('--prefix', type=str, default='scene', 
                       help='Prefix for renamed files (default: scene)')
    parser.add_argument('--start_index', type=int, default=1, 
                       help='Starting index for renaming (default: 1)')
    parser.add_argument('--padding', type=int, default=3, 
                       help='Number of digits for zero-padding (default: 3)')
    parser.add_argument('--extension', type=str, default='.png',
                       help='File extension to use (default: .png)')
    parser.add_argument('--dry_run', action='store_true', 
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Create renamer
    renamer = DatasetRenamer(args.dataset_root)
    
    # Execute based on mode
    if args.mode == 'split':
        # Validate required arguments
        if not args.rgb_folder or not args.ir_folder:
            parser.error("--rgb_folder and --ir_folder are required for split mode")
        
        renamer.split_and_rename(
            rgb_folder=args.rgb_folder,
            ir_folder=args.ir_folder,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            prefix=args.prefix,
            start_index=args.start_index,
            padding=args.padding,
            extension=args.extension,
            seed=args.seed,
            dry_run=args.dry_run
        )
    
    elif args.mode == 'rename':
        # Rename existing splits
        if args.all:
            renamer.rename_all_splits(
                prefix=args.prefix,
                start_index=args.start_index,
                padding=args.padding,
                extension=args.extension,
                dry_run=args.dry_run
            )
        else:
            renamer.rename_to_sequential(
                split=args.split,
                prefix=args.prefix,
                start_index=args.start_index,
                padding=args.padding,
                extension=args.extension,
                dry_run=args.dry_run
            )


if __name__ == '__main__':
    main()
