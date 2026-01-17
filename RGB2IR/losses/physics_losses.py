import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod


class ThermalPhysicsLoss(ABC):
    """Base class for physics-informed thermal losses"""
    
    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass


class HADARLoss(ThermalPhysicsLoss):
    """
    Heat-Aided Deep learning Attentive Reasoning (HADAR) Loss
    Incorporates thermal dynamics and physical constraints
    """
    
    def __init__(self, lambda_hadar: float = 0.5, weight_thermal: float = 1.0):
        """
        Args:
            lambda_hadar: Weight for HADAR loss component
            weight_thermal: Weight for thermal consistency
        """
        self.lambda_hadar = lambda_hadar
        self.weight_thermal = weight_thermal
        self.mse_loss = nn.MSELoss()
        
    def compute_thermal_gradient(self, ir_image: torch.Tensor) -> torch.Tensor:
        """Compute spatial temperature gradients (Sobel filter)"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=ir_image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=ir_image.device).view(1, 1, 3, 3)
        
        # Compute gradients
        grad_x = F.conv2d(ir_image, sobel_x, padding=1)
        grad_y = F.conv2d(ir_image, sobel_y, padding=1)
        
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    
    def compute_thermal_consistency(self, ir_image: torch.Tensor) -> torch.Tensor:
        """
        Enforce thermal consistency: smooth temperature transitions
        Real IR images don't have abrupt temperature changes
        """
        # Compute Laplacian (local temperature variance)
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                 dtype=torch.float32, device=ir_image.device).view(1, 1, 3, 3)
        laplacian_map = F.conv2d(ir_image, laplacian, padding=1)
        
        # Penalize high-frequency noise
        consistency_loss = torch.mean(laplacian_map ** 2)
        return consistency_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted IR image (B, C, H, W)
            target: Target IR image (B, C, H, W)
        """
        # Reconstruction loss
        recon_loss = self.mse_loss(pred, target)
        
        # Thermal gradient matching
        grad_pred = self.compute_thermal_gradient(pred)
        grad_target = self.compute_thermal_gradient(target)
        gradient_loss = self.mse_loss(grad_pred, grad_target)
        
        # Thermal consistency (smoothness)
        consistency_loss = self.compute_thermal_consistency(pred)
        
        # Combined HADAR loss
        total_loss = recon_loss + self.lambda_hadar * gradient_loss + \
                     self.weight_thermal * consistency_loss
        
        return total_loss


class EmissivityLoss(ThermalPhysicsLoss):
    """
    Material emissivity loss
    Different materials have different thermal emissivity values
    Helps the to_k projection learn material properties
    """
    
    def __init__(self, lambda_emissivity: float = 0.1):
        self.lambda_emissivity = lambda_emissivity
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                material_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted IR image
            target: Target IR image
            material_map: Optional material segmentation map
        """
        # Emissivity relates to local texture and material
        # High-frequency components in RGB correspond to material boundaries
        
        # Compute local contrast (proxy for emissivity variation)
        pred_std = torch.std(pred, dim=(2, 3), keepdim=True)
        target_std = torch.std(target, dim=(2, 3), keepdim=True)
        
        emissivity_loss = F.mse_loss(pred_std, target_std)
        
        return self.lambda_emissivity * emissivity_loss


class TransmitivityLoss(ThermalPhysicsLoss):
    """
    Atmospheric transmitivity loss
    Models how IR radiation is affected by atmospheric conditions
    Preserves spatial coherence and long-range dependencies
    """
    
    def __init__(self, lambda_transmitivity: float = 0.05, window_size: int = 16):
        self.lambda_transmitivity = lambda_transmitivity
        self.window_size = window_size
        
    def compute_transmitivity_map(self, ir_image: torch.Tensor) -> torch.Tensor:
        """
        Compute local transmitivity as average intensity in local windows
        Higher transmitivity = brighter regions = better atmospheric transmission
        """
        unfold = F.unfold(ir_image, kernel_size=self.window_size, stride=self.window_size)
        transmitivity = unfold.mean(dim=1).view(ir_image.size(0), 1, -1)
        return transmitivity
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Enforce consistency in atmospheric transmitivity
        """
        trans_pred = self.compute_transmitivity_map(pred)
        trans_target = self.compute_transmitivity_map(target)
        
        transmitivity_loss = F.mse_loss(trans_pred, trans_target)
        
        return self.lambda_transmitivity * transmitivity_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss based on VQ features (from taming VAE)
    Encourages perceptually similar outputs
    """
    
    def __init__(self, lambda_perceptual: float = 0.1):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        
    def forward(self, pred_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_features: Features from pred image (from VAE encoder)
            target_features: Features from target image (from VAE encoder)
        """
        perceptual_loss = F.mse_loss(pred_features, target_features)
        return self.lambda_perceptual * perceptual_loss


class StructurePreservationLoss(nn.Module):
    """
    Preserves structural consistency using attention maps
    Leverages self-attention layers for semantic structure
    """
    
    def __init__(self, lambda_structure: float = 0.2):
        super().__init__()
        self.lambda_structure = lambda_structure
        
    def forward(self, pred_attention: Dict[str, torch.Tensor], 
                target_attention: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compare attention maps from self-attention layers
        Args:
            pred_attention: Dict of attention maps from pred model
            target_attention: Dict of attention maps from target model
        """
        loss = 0.0
        for key in pred_attention.keys():
            if key in target_attention:
                # Normalize attention maps
                pred_attn = F.softmax(pred_attention[key].flatten(2), dim=-1)
                target_attn = F.softmax(target_attention[key].flatten(2), dim=-1)
                
                # KL divergence for attention maps
                loss += F.kl_div(torch.log(pred_attn + 1e-8), target_attn, reduction='batchmean')
        
        return self.lambda_structure * (loss / max(len(pred_attention), 1))


class CombinedPhysicsLoss(nn.Module):
    """
    Combined loss function incorporating all physics-informed components
    """
    
    def __init__(self, 
                 lambda_l1: float = 1.0,
                 lambda_hadar: float = 0.5,
                 lambda_emissivity: float = 0.1,
                 lambda_transmitivity: float = 0.05,
                 lambda_perceptual: float = 0.1,
                 lambda_structure: float = 0.2):
        super().__init__()
        
        self.l1_loss = nn.L1Loss()
        self.hadar_loss = HADARLoss(lambda_hadar=lambda_hadar)
        self.emissivity_loss = EmissivityLoss(lambda_emissivity=lambda_emissivity)
        self.transmitivity_loss = TransmitivityLoss(lambda_transmitivity=lambda_transmitivity)
        self.perceptual_loss = PerceptualLoss(lambda_perceptual=lambda_perceptual)
        self.structure_loss = StructurePreservationLoss(lambda_structure=lambda_structure)
        
        self.lambda_l1 = lambda_l1
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                pred_features: Optional[torch.Tensor] = None,
                target_features: Optional[torch.Tensor] = None,
                pred_attention: Optional[Dict[str, torch.Tensor]] = None,
                target_attention: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: Predicted IR image
            target: Target IR image
            pred_features: VQ features from pred (optional)
            target_features: VQ features from target (optional)
            pred_attention: Attention maps from pred (optional)
            target_attention: Attention maps from target (optional)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # L1 Reconstruction
        l1_loss = self.l1_loss(pred, target)
        loss_dict['l1'] = l1_loss.item()
        
        # HADAR Loss (thermal dynamics)
        hadar_loss = self.hadar_loss(pred, target)
        loss_dict['hadar'] = hadar_loss.item()
        
        # Emissivity Loss
        emissivity_loss = self.emissivity_loss(pred, target)
        loss_dict['emissivity'] = emissivity_loss.item()
        
        # Transmitivity Loss
        transmitivity_loss = self.transmitivity_loss(pred, target)
        loss_dict['transmitivity'] = transmitivity_loss.item()
        
        # Perceptual Loss (if features provided)
        perceptual_loss = torch.tensor(0.0, device=pred.device)
        if pred_features is not None and target_features is not None:
            perceptual_loss = self.perceptual_loss(pred_features, target_features)
            loss_dict['perceptual'] = perceptual_loss.item()
        
        # Structure Loss (if attention maps provided)
        structure_loss = torch.tensor(0.0, device=pred.device)
        if pred_attention is not None and target_attention is not None:
            structure_loss = self.structure_loss(pred_attention, target_attention)
            loss_dict['structure'] = structure_loss.item()
        
        # Combined loss
        total_loss = (self.lambda_l1 * l1_loss + 
                      hadar_loss + 
                      emissivity_loss + 
                      transmitivity_loss +
                      perceptual_loss +
                      structure_loss)
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
