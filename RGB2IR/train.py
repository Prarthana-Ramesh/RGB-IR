import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
import os
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
            enable_controlnet=config['model'].get('enable_controlnet', True)
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
        
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.trainable_param_list):,}")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.trainable_param_list,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
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
            base_lr=config['training']['learning_rate']
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
            
            # Generate IR image
            prompt = "high quality infrared thermal image, temperature gradient visible"
            negative_prompt = "blurry, low quality, artifacts, noise"
            
            output = self.model(
                rgb_image=rgb,
                prompt=prompt,
                negative_prompt=negative_prompt,
                depth_map=depth,
                canny_edges=canny_edges,
                strength=0.8,
                guidance_scale=7.5,
                num_inference_steps=20,  # Fewer steps during training for speed
                return_features=False
            )
            
            ir_pred = output['ir_image']
            
            # Compute loss
            loss, loss_dict = self.loss_fn(ir_pred, ir_gt)
            
            # Backward pass
            loss.backward()
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
            
            # Update progress bar
            pbar.set_postfix({
                'loss': metrics['loss'].avg,
                'l1': metrics['l1'].avg,
                'hadar': metrics['hadar'].avg,
                'lr': get_learning_rate(self.optimizer)
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
            'l1': AverageMeter(),
            'hadar': AverageMeter(),
            'emissivity': AverageMeter(),
            'transmitivity': AverageMeter(),
            'perceptual': AverageMeter(),
            'structure': AverageMeter()
        }
        
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}", dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(self.device, dtype=torch.float16)
            ir_gt = batch['ir'].to(self.device, dtype=torch.float16)
            depth = batch['depth'].to(self.device, dtype=torch.float16)
            canny_edges = batch['canny_edges'].to(self.device, dtype=torch.float16)
            
            # Generate IR image
            prompt = "high quality infrared thermal image, temperature gradient visible"
            negative_prompt = "blurry, low quality, artifacts, noise"
            
            output = self.model(
                rgb_image=rgb,
                prompt=prompt,
                negative_prompt=negative_prompt,
                depth_map=depth,
                canny_edges=canny_edges,
                strength=0.8,
                guidance_scale=7.5,
                num_inference_steps=30,
                return_features=False
            )
            
            ir_pred = output['ir_image']
            
            # Compute loss
            loss, loss_dict = self.loss_fn(ir_pred, ir_gt)
            
            # Update metrics
            for key, value in loss_dict.items():
                if key in metrics:
                    metrics[key].update(value)
            
            pbar.set_postfix({'loss': metrics['loss'].avg})
        
        # Log to tensorboard
        for key, meter in metrics.items():
            self.writer.add_scalar(f'val/{key}', meter.avg, epoch)
        
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
