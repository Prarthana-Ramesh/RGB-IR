import torch
import torch.nn.functional as F
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, List

from models.rgb2ir_model import RGB2IRLoHaModel
from data.dataset import create_dataloaders
from utils.preprocessing import ImagePreprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Peak Signal-to-Noise Ratio"""
        # Denormalize from [-1, 1] to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        psnr_val = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
        return psnr_val.item()
    
    @staticmethod
    def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Structural Similarity Index"""
        from torchvision.models.feature_extraction import create_feature_extractor
        
        # Denormalize
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to grayscale if needed
        if pred.shape[1] == 3:
            pred = 0.299 * pred[:, 0:1, :, :] + 0.587 * pred[:, 1:2, :, :] + 0.114 * pred[:, 2:3, :, :]
        if target.shape[1] == 3:
            target = 0.299 * target[:, 0:1, :, :] + 0.587 * target[:, 1:2, :, :] + 0.114 * target[:, 2:3, :, :]
        
        # Compute means
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances
        sigma1_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Absolute Error"""
        return F.l1_loss(pred, target).item()
    
    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Squared Error"""
        return F.mse_loss(pred, target).item()
    
    @staticmethod
    def thermal_consistency(ir_image: torch.Tensor) -> float:
        """
        Measure thermal consistency (smoothness of temperature transitions)
        Lower is better (smoother transitions)
        """
        # Compute Laplacian (local variance)
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                 dtype=torch.float32, device=ir_image.device).view(1, 1, 3, 3)
        laplacian_map = F.conv2d(ir_image, laplacian, padding=1)
        consistency = torch.mean(laplacian_map ** 2)
        return consistency.item()
    
    @staticmethod
    def gradient_matching(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Measure gradient matching (edge preservation)
        Compute gradients using Sobel and compare
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
        
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)
        
        grad_matching = F.mse_loss(grad_pred_x, grad_target_x) + \
                       F.mse_loss(grad_pred_y, grad_target_y)
        
        return grad_matching.item()


class RGB2IREvaluator:
    """Evaluation interface"""
    
    def __init__(self,
                 config_path: str,
                 checkpoint_path: str,
                 device: str = 'cuda'):
        """
        Args:
            config_path: Path to LoHA configuration
            checkpoint_path: Path to LoHA weights
            device: Device to use
        """
        self.device = device
        logger.info(f"Loading model on {device}...")
        
        self.model = RGB2IRLoHaModel(
            config_path=config_path,
            device=device,
            enable_controlnet=True
        )
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.model.load_lora_weights(checkpoint_path)
        
        # Set to eval mode
        self.model.pipeline.unet.eval()
        self.model.pipeline.text_encoder.eval()
        
        # Memory efficient inference
        self.model.enable_model_cpu_offload()
        self.model.enable_attention_slicing()
        
        self.metric_calc = MetricCalculator()
    
    @torch.no_grad()
    def evaluate(self, 
                 dataset_root: str,
                 split: str = 'val',
                 batch_size: int = 1) -> Dict[str, float]:
        """
        Evaluate on dataset
        
        Args:
            dataset_root: Path to dataset root
            split: 'val' or 'test'
            batch_size: Batch size
        
        Returns:
            Dictionary with metrics
        """
        logger.info(f"Loading {split} dataset...")
        dataloaders = create_dataloaders(
            dataset_root=dataset_root,
            batch_size=batch_size,
            num_workers=4,
            splits=[split]
        )
        
        dataloader = dataloaders[split]
        
        # Metrics accumulators
        metrics = {
            'psnr': [],
            'ssim': [],
            'mae': [],
            'mse': [],
            'gradient_matching': [],
            'thermal_consistency': []
        }
        
        logger.info(f"Evaluating on {len(dataloader)} batches...")
        
        for batch in tqdm(dataloader):
            rgb = batch['rgb'].to(self.device, dtype=torch.float16)
            ir_gt = batch['ir'].to(self.device, dtype=torch.float16)
            depth = batch['depth'].to(self.device, dtype=torch.float16)
            canny_edges = batch['canny_edges'].to(self.device, dtype=torch.float16)
            
            # Generate IR
            output = self.model(
                rgb_image=rgb,
                prompt="high quality infrared thermal image",
                negative_prompt="blurry, low quality",
                depth_map=depth,
                canny_edges=canny_edges,
                strength=0.8,
                guidance_scale=7.5,
                num_inference_steps=50
            )
            
            ir_pred = output['ir_image']
            
            # Compute metrics
            psnr = self.metric_calc.psnr(ir_pred, ir_gt)
            ssim = self.metric_calc.ssim(ir_pred, ir_gt)
            mae = self.metric_calc.mae(ir_pred, ir_gt)
            mse = self.metric_calc.mse(ir_pred, ir_gt)
            grad_match = self.metric_calc.gradient_matching(ir_pred, ir_gt)
            therm_consist = self.metric_calc.thermal_consistency(ir_pred)
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['mae'].append(mae)
            metrics['mse'].append(mse)
            metrics['gradient_matching'].append(grad_match)
            metrics['thermal_consistency'].append(therm_consist)
        
        # Compute averages
        result = {}
        for key in metrics.keys():
            result[f'{key}_mean'] = np.mean(metrics[key])
            result[f'{key}_std'] = np.std(metrics[key])
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate RGB to IR model')
    parser.add_argument('--config', type=str, required=True, help='Path to LoHA config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to LoHA weights')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = RGB2IREvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Evaluate
    logger.info("Starting evaluation...")
    metrics = evaluator.evaluate(
        dataset_root=args.dataset,
        split=args.split,
        batch_size=args.batch_size
    )
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    for metric, value in sorted(metrics.items()):
        logger.info(f"{metric}: {value:.4f}")
    logger.info("="*50)


if __name__ == '__main__':
    main()
