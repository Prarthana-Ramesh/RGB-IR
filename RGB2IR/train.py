import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import yaml

from models.rgb2ir_model import RGB2IRLoHaModel
from data.dataset import create_dataloaders
from losses.physics_losses import CombinedPhysicsLoss
from utils.preprocessing import AverageMeter, get_learning_rate, WarmupScheduler, save_checkpoint, load_checkpoint


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RGB2IRTrainer:
    """Main trainer class for RGB to IR translation"""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        """
        Args:
            config: Configuration dictionary
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        
        # Create directories
        self.output_dir = Path(config['training']['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Initialize model
        logger.info("Initializing RGB2IR model...")
        self.model = RGB2IRLoHaModel(
            config_path=config['model']['config_path'],
            device=device,
            enable_controlnet=config['model'].get('enable_controlnet', True),
            pretrained_model_name_or_path=config['model']['pretrained']
        )
        
        # Set model to training mode (for LoHA adapters)
        self.model.pipeline.unet.train()
        self.model.pipeline.text_encoder.train()
        self.model.material_recognition.train()
        self.model.emissivity_calculation.train()
        
        # Initialize loss function
        logger.info("Initializing loss function...")
        self.loss_fn = CombinedPhysicsLoss(
            lambda_l1=config['training']['loss_weights']['l1'],
            lambda_hadar=config['training']['loss_weights']['hadar'],
            lambda_emissivity=config['training']['loss_weights']['emissivity'],
            lambda_transmitivity=config['training']['loss_weights']['transmitivity'],
            lambda_perceptual=config['training']['loss_weights']['perceptual'],
            lambda_structure=config['training']['loss_weights']['structure']
        )
        
        # Get trainable parameters (only LoHA adapters)
        trainable_params = self.model.get_trainable_parameters()
        self.trainable_param_list = [p for _, p in trainable_params]
        
        total_params = sum(p.numel() for p in self.trainable_param_list)
        logger.info(f"Trainable parameters: {total_params:,}")
        
        # Debug: Check if UNet LoRA parameters are actually trainable
        unet_lora_count = sum(1 for name, _ in trainable_params if 'unet' in name and 'lora' in name.lower())
        text_lora_count = sum(1 for name, _ in trainable_params if 'text_encoder' in name and 'lora' in name.lower())
        logger.info(f"LoRA parameters - UNet: {unet_lora_count}, TextEncoder: {text_lora_count}")
        
        if unet_lora_count == 0:
            logger.warning("WARNING: No UNet LoRA parameters found! Checking UNet parameter status...")
            lora_params_total = 0
            lora_params_trainable = 0
            for name, param in self.model.pipeline.unet.named_parameters():
                if 'lora' in name.lower():
                    lora_params_total += 1
                    if param.requires_grad:
                        lora_params_trainable += 1
            logger.info(f"UNet LoRA params: {lora_params_trainable}/{lora_params_total} have requires_grad=True")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.trainable_param_list,
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        # Initialize scheduler
        if config['training']['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs']
            )
        else:
            self.scheduler = None
        
        # Warmup scheduler
        self.warmup_scheduler = WarmupScheduler(
            self.optimizer,
            warmup_steps=config['training']['warmup_steps'],
            base_lr=float(config['training']['learning_rate'])
        )
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch"""
        self.model.pipeline.unet.train()
        self.model.pipeline.text_encoder.train()
        self.model.material_recognition.train()
        self.model.emissivity_calculation.train()
        
        metrics = {
            'loss': AverageMeter(),
            'noise': AverageMeter(),
            'l1': AverageMeter(),
            'hadar': AverageMeter(),
            'emissivity': AverageMeter(),
            'transmitivity': AverageMeter(),
            'perceptual': AverageMeter(),
            'structure': AverageMeter()
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(self.device, dtype=torch.float16)
            ir_gt = batch['ir'].to(self.device, dtype=torch.float16)
            depth = batch['depth'].to(self.device, dtype=torch.float16)
            canny_edges = batch['canny_edges'].to(self.device, dtype=torch.float16)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Training prompt
            prompt = "high quality infrared thermal image, temperature gradient visible"
            
            # Use training-optimized forward pass
            output = self.model.forward_train(
                rgb_image=rgb,
                ir_target=ir_gt,
                prompt=prompt,
                depth_map=depth,
                canny_edges=canny_edges
            )
            
            # Get predicted and target noise
            noise_pred = output['noise_pred']
            noise_target = output['noise_target']
            ir_pred = output['ir_pred']  # For visualization/logging only
            
            # Compute noise prediction loss (primary training objective)
            noise_loss = F.mse_loss(noise_pred, noise_target)
            
            # Compute physics-informed losses on predicted IR (for regularization)
            # Scale down since these are secondary objectives
            physics_loss, loss_dict = self.loss_fn(ir_pred, ir_gt)
            physics_loss = physics_loss * 0.1  # 10% weight on physics losses
            
            # Total loss
            loss = noise_loss + physics_loss
            
            # Add noise loss to metrics
            loss_dict['noise'] = noise_loss.item()
            loss_dict['total'] = loss.item()  # Update total to include noise loss
            
            # Backward pass
            loss.backward()
            
            # Check gradients on first batch to verify training
            if batch_idx == 0 and epoch == 0:
                grad_norms = []
                for name, param in self.model.get_trainable_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                if grad_norms:
                    avg_grad = sum(grad_norms) / len(grad_norms)
                    logger.info(f"First batch gradient check: avg norm = {avg_grad:.6f}, params with grads = {len(grad_norms)}")
                else:
                    logger.warning("WARNING: No gradients detected on trainable parameters!")
            
            torch.nn.utils.clip_grad_norm_(self.trainable_param_list, max_norm=1.0)
            self.optimizer.step()
            
            # Warmup step
            self.warmup_scheduler.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                if key in metrics:
                    metrics[key].update(value)
            
            # Log to tensorboard
            if self.global_step % self.config['training']['log_interval'] == 0:
                for key, meter in metrics.items():
                    self.writer.add_scalar(f'train/{key}', meter.avg, self.global_step)
                self.writer.add_scalar('train/lr', get_learning_rate(self.optimizer), self.global_step)
            
            # Update progress bar with better formatting
            pbar.set_postfix({
                'loss': f'{metrics["loss"].avg:.4f}',
                'noise': f'{metrics["noise"].avg:.4f}',
                'l1': f'{metrics["l1"].avg:.2f}',
                'hadar': f'{metrics["hadar"].avg:.2f}',
                'lr': f'{get_learning_rate(self.optimizer):.2e}'
            })
            
            self.global_step += 1
        
        # Step scheduler after epoch
        if self.scheduler is not None:
            self.scheduler.step()
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int):
        """Validate model"""
        self.model.pipeline.unet.eval()
        self.model.pipeline.text_encoder.eval()
        
        metrics = {
            'loss': AverageMeter(),
            'noise': AverageMeter(),
            'l1': AverageMeter(),
            'hadar': AverageMeter(),
            'emissivity': AverageMeter(),
            'transmitivity': AverageMeter(),
            'perceptual': AverageMeter(),
            'structure': AverageMeter()
        }
        
        # Create sample output directory
        sample_dir = Path(self.config['training']['output_dir']) / 'samples' / f'epoch_{epoch+1}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving validation samples to: {sample_dir}")
        
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}", dynamic_ncols=True)
        
        num_samples = 0
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(self.device, dtype=torch.float16)
            ir_gt = batch['ir'].to(self.device, dtype=torch.float16)
            depth = batch['depth'].to(self.device, dtype=torch.float16)
            canny_edges = batch['canny_edges'].to(self.device, dtype=torch.float16)
            
            # Training prompt
            prompt = "high quality infrared thermal image, temperature gradient visible"
            
            # Use training forward pass for validation too (faster, more consistent)
            output = self.model.forward_train(
                rgb_image=rgb,
                ir_target=ir_gt,
                prompt=prompt,
                depth_map=depth,
                canny_edges=canny_edges
            )
            
            noise_pred = output['noise_pred']
            noise_target = output['noise_target']
            ir_pred = output['ir_pred']
            
            # Compute losses
            noise_loss = F.mse_loss(noise_pred, noise_target)
            physics_loss, loss_dict = self.loss_fn(ir_pred, ir_gt)
            physics_loss = physics_loss * 0.1
            
            loss = noise_loss + physics_loss
            
            # Add noise loss to metrics
            loss_dict['noise'] = noise_loss.item()
            loss_dict['total'] = loss.item()
            
            # Update metrics
            for key, value in loss_dict.items():
                if key in metrics:
                    metrics[key].update(value)
            
            # Save first 5 batches as samples
            if batch_idx < 5:
                import cv2
                try:
                    for i in range(min(2, rgb.size(0))):  # Save 2 images per batch
                        # Denormalize from [-1, 1] to [0, 1]
                        rgb_img = ((rgb[i].cpu().float() + 1.0) / 2.0).permute(1, 2, 0).numpy()
                        ir_gt_img = ((ir_gt[i].cpu().float() + 1.0) / 2.0).squeeze().numpy()
                        ir_pred_img = ((ir_pred[i].cpu().float() + 1.0) / 2.0).squeeze().numpy()
                        
                        # Convert to uint8
                        rgb_img = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
                        ir_gt_img = (np.clip(ir_gt_img, 0, 1) * 255).astype(np.uint8)
                        ir_pred_img = (np.clip(ir_pred_img, 0, 1) * 255).astype(np.uint8)
                        
                        # Convert grayscale IR to RGB for concatenation
                        ir_gt_rgb = cv2.cvtColor(ir_gt_img, cv2.COLOR_GRAY2RGB)
                        ir_pred_rgb = cv2.cvtColor(ir_pred_img, cv2.COLOR_GRAY2RGB)
                        
                        # Create side-by-side comparison: RGB | GT IR | Pred IR
                        comparison = np.concatenate([rgb_img, ir_gt_rgb, ir_pred_rgb], axis=1)
                        
                        # Save
                        sample_idx = batch_idx * 2 + i
                        sample_path = sample_dir / f'sample_{sample_idx:03d}.png'
                        cv2.imwrite(str(sample_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                        
                        if i == 0 and batch_idx == 0:
                            logger.info(f"Saved first sample to: {sample_path}")
                except Exception as e:
                    logger.error(f"Error saving sample {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
            
            pbar.set_postfix({'loss': metrics['loss'].avg})
        
        # Log to tensorboard
        for key, meter in metrics.items():
            self.writer.add_scalar(f'val/{key}', meter.avg, epoch)
        
        # Log sample count
        num_samples = len(list(sample_dir.glob('*.png')))
        logger.info(f"Saved {num_samples} validation samples to {sample_dir}")
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Config: {self.config}")
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        dataloaders = create_dataloaders(
            dataset_root=self.config['data']['dataset_root'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            image_size=tuple(self.config['data']['image_size']),
            use_augmentation=True,
            splits=['train', 'val']
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_metrics = self.validate(val_loader, epoch)
                
                val_loss = val_metrics['loss'].avg
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                    
                    # Save best model
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        str(self.checkpoint_dir / f'best.pt')
                    )
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    str(self.checkpoint_dir / f'epoch_{epoch:03d}.pt')
                )
            
            logger.info(f"Train Loss: {train_metrics['loss'].avg:.4f}")
            logger.info(f"Learning Rate: {get_learning_rate(self.optimizer):.6f}")
        
        logger.info("Training completed!")
        self.writer.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train RGB to IR translation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = RGB2IRTrainer(config, device=args.device)
    
    # Resume if checkpoint provided
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.start_epoch = load_checkpoint(
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            args.resume
        )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
