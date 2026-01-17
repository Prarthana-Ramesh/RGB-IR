import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from diffusers import StableDiffusionXLImg2ImgPipeline, ControlNetModel
from peft import LoHaConfig, get_peft_model
import yaml


class MaterialRecognitionModule(nn.Module):
    """
    Material recognition module using to_k projections
    Learns to map RGB textures to material properties
    """
    
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Material property predictor
        self.material_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16)  # 16 material types
        )
        
        # Reflectance predictor
        self.reflectance_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 0-1 reflectance
            nn.Sigmoid()
        )
    
    def forward(self, key_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            key_features: Key projections from to_k (B, N, D)
        
        Returns:
            material_logits: (B, 16) material class logits
            reflectance: (B, 1) reflectance values
        """
        # Average over spatial dimension
        avg_features = key_features.mean(dim=1) if key_features.dim() == 3 else key_features
        
        material_logits = self.material_classifier(avg_features)
        reflectance = self.reflectance_predictor(avg_features)
        
        return material_logits, reflectance


class EmissivityCalculationModule(nn.Module):
    """
    Emissivity and thermal property calculation module using to_v projections
    Learns to map image features to thermal emissivity values
    """
    
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Emissivity predictor (0-1 for most materials)
        self.emissivity_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Temperature offset predictor (relative to ambient)
        self.temperature_offset_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, value_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            value_features: Value projections from to_v (B, N, D)
        
        Returns:
            emissivity: (B, 1) emissivity values [0, 1]
            temperature_offset: (B, 1) relative temperature [-1, 1]
        """
        # Average over spatial dimension
        avg_features = value_features.mean(dim=1) if value_features.dim() == 3 else value_features
        
        emissivity = self.emissivity_predictor(avg_features)
        temperature_offset = self.temperature_offset_predictor(avg_features)
        
        return emissivity, temperature_offset


class RGB2IRLoHaModel(nn.Module):
    """
    RGB to IR translation model using SDXL with LoHA adaptation
    Incorporates ControlNet for structure preservation
    """
    
    def __init__(self, 
                 config_path: str,
                 device: str = 'cuda',
                 enable_controlnet: bool = True):
        """
        Args:
            config_path: Path to LoHA configuration YAML
            device: Device to use ('cuda' or 'cpu')
            enable_controlnet: Whether to use ControlNet
        """
        super().__init__()
        
        self.device = device
        self.enable_controlnet = enable_controlnet
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load base SDXL model
        print("Loading SDXL base model...")
        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.config['model_loading']['pretrained_model'],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipeline = self.pipeline.to(device)
        
        # Load ControlNet models
        if self.enable_controlnet:
            print("Loading ControlNet models...")
            self.controlnet_depth = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(device)
            
            self.controlnet_canny = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(device)
        
        # Apply LoHA adapters to text encoder
        print("Applying LoHA to text encoder...")
        self._apply_loha_to_text_encoder()
        
        # Apply LoHA adapters to UNet
        print("Applying LoHA to UNet...")
        self._apply_loha_to_unet()
        
        # Initialize specialized modules
        self.material_recognition = MaterialRecognitionModule(embedding_dim=768)
        self.emissivity_calculation = EmissivityCalculationModule(embedding_dim=768)
        
        # Move to device
        self.material_recognition.to(device)
        self.emissivity_calculation.to(device)
    
    def _apply_loha_to_text_encoder(self):
        """Apply LoHA configuration to text encoder"""
        config = self.config['text_encoder']
        
        loha_config = LoHaConfig(
            r=config['r'],
            alpha=config['alpha'],
            rank_dropout=config['rank_dropout'],
            module_dropout=config['module_dropout'],
            init_weights=config['init_weights'],
            target_modules=config['target_modules'],
            peft_type="LOHA"
        )
        
        self.pipeline.text_encoder = get_peft_model(
            self.pipeline.text_encoder, 
            loha_config
        )
    
    def _apply_loha_to_unet(self):
        """Apply LoHA configuration to UNet"""
        config = self.config['unet']
        
        loha_config = LoHaConfig(
            r=config['r'],
            alpha=config['alpha'],
            rank_dropout=config['rank_dropout'],
            module_dropout=config['module_dropout'],
            init_weights=config['init_weights'],
            target_modules=config['target_modules'],
            use_effective_conv2d=config['use_effective_conv2d'],
            peft_type="LOHA"
        )
        
        self.pipeline.unet = get_peft_model(
            self.pipeline.unet,
            loha_config
        )
    
    def enable_model_cpu_offload(self):
        """Enable CPU offloading for memory efficiency"""
        self.pipeline.enable_model_cpu_offload()
    
    def enable_attention_slicing(self):
        """Enable attention slicing for memory efficiency"""
        self.pipeline.enable_attention_slicing()
    
    def disable_attention_slicing(self):
        """Disable attention slicing for speed"""
        self.pipeline.disable_attention_slicing()
    
    def forward(self,
                rgb_image: torch.Tensor,
                prompt: str = "a thermal infrared image",
                negative_prompt: str = "blurry, low quality",
                depth_map: Optional[torch.Tensor] = None,
                canny_edges: Optional[torch.Tensor] = None,
                strength: float = 0.8,
                guidance_scale: float = 7.5,
                num_inference_steps: int = 50,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate IR image from RGB input
        
        Args:
            rgb_image: Input RGB image (B, 3, H, W) in [-1, 1]
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            depth_map: Depth map for ControlNet (B, 1, H, W) in [-1, 1]
            canny_edges: Canny edge map for ControlNet (B, 1, H, W) in [-1, 1]
            strength: Denoising strength
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary with:
                'ir_image': Generated IR image (B, 1, H, W)
                'features': Intermediate features (if return_features=True)
        """
        # Denormalize RGB to [0, 1] for VAE
        rgb_input = (rgb_image + 1.0) / 2.0
        
        # Prepare ControlNet inputs if provided
        control_image = None
        controlnet_conditioning_scale = None
        
        if self.enable_controlnet and (depth_map is not None or canny_edges is not None):
            # Combine control signals
            if depth_map is not None and canny_edges is not None:
                # Denormalize to [0, 1]
                depth_normalized = (depth_map + 1.0) / 2.0
                canny_normalized = (canny_edges + 1.0) / 2.0
                
                # Stack as multi-channel condition (3 channels for compatibility)
                control_image = torch.cat([
                    depth_normalized,
                    canny_normalized,
                    canny_normalized  # Repeat canny for 3 channels
                ], dim=1)
                
                controlnet_conditioning_scale = [
                    self.config['inference']['controlnet']['depth']['scale'],
                    self.config['inference']['controlnet']['canny']['scale'],
                    self.config['inference']['controlnet']['canny']['scale']
                ]
        
        # Generate IR image
        with torch.no_grad():
            ir_output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=rgb_input,
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="latent" if return_features else "pil"
            )
            
            if return_features:
                ir_latent = ir_output.images
                # Decode latent to image
                ir_image = self.pipeline.vae.decode(ir_latent).sample
            else:
                ir_image = ir_output.images
        
        # Normalize IR output to [-1, 1]
        ir_image = ir_image * 2.0 - 1.0
        
        result = {'ir_image': ir_image}
        
        if return_features:
            result['features'] = {
                'latent': ir_latent
            }
        
        return result
    
    def get_trainable_parameters(self):
        """Get trainable parameters (LoHA adapters only)"""
        trainable_params = []
        
        # Get LoHA parameters from text encoder
        if hasattr(self.pipeline.text_encoder, 'peft_config'):
            for name, param in self.pipeline.text_encoder.named_parameters():
                if 'lora' in name.lower():
                    trainable_params.append((f"text_encoder.{name}", param))
        
        # Get LoHA parameters from UNet
        if hasattr(self.pipeline.unet, 'peft_config'):
            for name, param in self.pipeline.unet.named_parameters():
                if 'lora' in name.lower():
                    trainable_params.append((f"unet.{name}", param))
        
        # Get specialized module parameters
        for name, param in self.material_recognition.named_parameters():
            trainable_params.append((f"material_recognition.{name}", param))
        
        for name, param in self.emissivity_calculation.named_parameters():
            trainable_params.append((f"emissivity_calculation.{name}", param))
        
        return trainable_params
    
    def save_lora_weights(self, save_path: str):
        """Save LoHA adapter weights"""
        if hasattr(self.pipeline.text_encoder, 'save_pretrained'):
            self.pipeline.text_encoder.save_pretrained(f"{save_path}/text_encoder_lora")
        
        if hasattr(self.pipeline.unet, 'save_pretrained'):
            self.pipeline.unet.save_pretrained(f"{save_path}/unet_lora")
        
        # Save specialized modules
        torch.save(self.material_recognition.state_dict(), 
                   f"{save_path}/material_recognition.pt")
        torch.save(self.emissivity_calculation.state_dict(), 
                   f"{save_path}/emissivity_calculation.pt")
    
    def load_lora_weights(self, load_path: str):
        """Load LoHA adapter weights"""
        if hasattr(self.pipeline.text_encoder, 'load_adapter'):
            self.pipeline.text_encoder.load_adapter(
                f"{load_path}/text_encoder_lora", 
                adapter_name="default"
            )
        
        if hasattr(self.pipeline.unet, 'load_adapter'):
            self.pipeline.unet.load_adapter(
                f"{load_path}/unet_lora", 
                adapter_name="default"
            )
        
        # Load specialized modules
        self.material_recognition.load_state_dict(
            torch.load(f"{load_path}/material_recognition.pt")
        )
        self.emissivity_calculation.load_state_dict(
            torch.load(f"{load_path}/emissivity_calculation.pt")
        )
