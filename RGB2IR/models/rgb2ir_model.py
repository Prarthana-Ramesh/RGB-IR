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
                 enable_controlnet: bool = True,
                 pretrained_model_name_or_path: str = None):
        """
        Args:
            config_path: Path to LoHA configuration YAML
            device: Device to use ('cuda' or 'cpu')
            enable_controlnet: Whether to use ControlNet
            pretrained_model_name_or_path: Override pretrained model path (from train config)
        """
        super().__init__()
        
        self.device = device
        self.enable_controlnet = enable_controlnet
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Use override if provided, otherwise use config
        model_name = pretrained_model_name_or_path or self.config['model_loading']['pretrained_model']
        
        # Load base model (SD 1.5 or SDXL)
        print(f"Loading base model: {model_name}...")
        if "xl" in model_name.lower():
            from diffusers import StableDiffusionXLImg2ImgPipeline as PipelineClass
        else:
            from diffusers import StableDiffusionImg2ImgPipeline as PipelineClass
        
        self.pipeline = PipelineClass.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,  # Disable safety checker (not needed for training)
            requires_safety_checker=False
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
    
    def forward_train(self,
                      rgb_image: torch.Tensor,
                      ir_target: torch.Tensor,
                      prompt: str = "a thermal infrared image",
                      depth_map: Optional[torch.Tensor] = None,
                      canny_edges: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Training forward pass - predicts noise at random timestep
        
        Args:
            rgb_image: Input RGB image (B, 3, H, W) in [-1, 1]
            ir_target: Target IR image (B, 1, H, W) in [-1, 1]
            prompt: Text prompt for generation
            depth_map: Depth map for ControlNet (B, 1, H, W) in [-1, 1]
            canny_edges: Canny edges for ControlNet (B, 1, H, W) in [-1, 1]
        
        Returns:
            Dictionary with:
                'ir_pred': Predicted IR image (B, 1, H, W) - for visualization/logging only
                'noise_pred': Predicted noise
                'noise_target': Target noise (ground truth)
        """
        batch_size = rgb_image.shape[0]
        
        # Encode IR target to latent space (this is what we want to learn to generate)
        # Expand IR to 3 channels for VAE
        ir_input = (ir_target + 1.0) / 2.0  # Denormalize to [0, 1]
        if ir_input.shape[1] == 1:
            ir_input = ir_input.repeat(1, 3, 1, 1)
        
        # VAE is frozen, encode target latents (no gradients needed here)
        with torch.no_grad():
            target_latents = self.pipeline.vae.encode(ir_input).latent_dist.sample()
            target_latents = target_latents * self.pipeline.vae.config.scaling_factor
        
        # Sample random timestep for each image
        timesteps = torch.randint(
            0, 
            self.pipeline.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # Add noise to target latents
        noise = torch.randn_like(target_latents)
        noisy_latents = self.pipeline.scheduler.add_noise(target_latents, noise, timesteps)
        
        # Encode text prompt (frozen text encoder)
        with torch.no_grad():
            text_inputs = self.pipeline.tokenizer(
                [prompt] * batch_size,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            text_embeddings = self.pipeline.text_encoder(text_input_ids)[0]
        
        # Prepare ControlNet conditioning if enabled
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        
        if self.enable_controlnet and depth_map is not None:
            depth_input = (depth_map + 1.0) / 2.0  # Denormalize to [0, 1]
            depth_input = depth_input.repeat(1, 3, 1, 1)  # Repeat to 3 channels
            
            with torch.no_grad():  # ControlNet is frozen
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_input,
                    return_dict=False
                )
                
                depth_scale = self.config['inference']['controlnet']['depth']['scale']
                down_block_additional_residuals = [sample * depth_scale for sample in down_block_res_samples]
                mid_block_additional_residual = mid_block_res_sample * depth_scale
        
        # Predict noise using UNet - THIS is where LoRA adapters get trained!
        # UNet has trainable LoRA parameters, so gradients flow here
        noise_pred = self.pipeline.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=False
        )[0]
        
        # For visualization/logging: predict clean latent and decode
        # This part is optional and only used for monitoring (keep in no_grad)
        with torch.no_grad():
            alpha_prod_t = self.pipeline.scheduler.alphas_cumprod[timesteps].to(self.device, dtype=torch.float16)
            alpha_prod_t = alpha_prod_t.flatten()
            while len(alpha_prod_t.shape) < len(noisy_latents.shape):
                alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            
            pred_latents = (noisy_latents - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
            ir_pred = self.pipeline.vae.decode(pred_latents / self.pipeline.vae.config.scaling_factor).sample
            ir_pred = torch.clamp(ir_pred, -1.0, 1.0)
            
            # Convert to single channel
            if ir_pred.shape[1] == 3:
                ir_pred = ir_pred.mean(dim=1, keepdim=True)
        
        return {
            'ir_pred': ir_pred,  # For visualization only (no gradients)
            'noise_pred': noise_pred,  # For loss computation (HAS gradients from UNet)
            'noise_target': noise  # Ground truth noise
        }
    
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
        batch_size = rgb_image.shape[0]
        
        # Denormalize RGB to [0, 1] for VAE
        rgb_input = (rgb_image + 1.0) / 2.0
        
        # Encode RGB image to latent space
        with torch.no_grad():
            latents = self.pipeline.vae.encode(rgb_input).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor
        
        # Encode text prompts directly using text encoder
        with torch.no_grad():
            text_inputs = self.pipeline.tokenizer(
                [prompt] * batch_size,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            text_embeddings = self.pipeline.text_encoder(text_input_ids)[0]
        
        # Add noise to latents for diffusion training
        timestep = torch.randint(0, self.pipeline.scheduler.config.num_train_timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timestep)
        
        # Prepare ControlNet conditioning if available
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        
        if self.enable_controlnet and depth_map is not None:
            # Use depth for ControlNet conditioning
            depth_input = (depth_map + 1.0) / 2.0  # Denormalize to [0, 1]
            # Repeat to 3 channels for ControlNet
            depth_input = depth_input.repeat(1, 3, 1, 1)
            
            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    noisy_latents,  # ControlNet works in latent space
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_input,  # Conditioning in pixel space
                    return_dict=False
                )
                
                # Scale the ControlNet outputs
                depth_scale = self.config['inference']['controlnet']['depth']['scale']
                down_block_additional_residuals = [sample * depth_scale for sample in down_block_res_samples]
                mid_block_additional_residual = mid_block_res_sample * depth_scale
        
        # Predict noise
        noise_pred = self.pipeline.unet(
            noisy_latents,
            timestep,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=False
        )[0]
        
        # For training, we predict the clean latent from the noise prediction
        # Using the simple prediction: x0 = (xt - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
        alpha_prod_t = self.pipeline.scheduler.alphas_cumprod[timestep].to(self.device, dtype=torch.float16)
        alpha_prod_t = alpha_prod_t.flatten()
        while len(alpha_prod_t.shape) < len(noisy_latents.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        
        # Predict original latent (keep in float16)
        pred_latents = (noisy_latents - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
        
        # Decode to image space (VAE is frozen, but we need gradients to flow back through the prediction)
        ir_image = self.pipeline.vae.decode(pred_latents / self.pipeline.vae.config.scaling_factor).sample
        
        # Normalize IR output to [-1, 1]
        ir_image = torch.clamp(ir_image, -1.0, 1.0)
        
        # Convert RGB output to single channel IR by averaging
        if ir_image.shape[1] == 3:
            ir_image = ir_image.mean(dim=1, keepdim=True)
        
        result = {'ir_image': ir_image}
        
        if return_features:
            result['features'] = {
                'latent': pred_latents,
                'noise_pred': noise_pred,
                'noise_target': noise
            }
        
        return result
    
    def get_trainable_parameters(self):
        """Get trainable parameters (LoHA adapters only)"""
        trainable_params = []
        
        # Get LoHA parameters from text encoder
        if hasattr(self.pipeline.text_encoder, 'peft_config'):
            for name, param in self.pipeline.text_encoder.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    trainable_params.append((f"text_encoder.{name}", param))
        
        # Get LoHA parameters from UNet
        if hasattr(self.pipeline.unet, 'peft_config'):
            for name, param in self.pipeline.unet.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    trainable_params.append((f"unet.{name}", param))
        
        # Get specialized module parameters
        for name, param in self.material_recognition.named_parameters():
            if param.requires_grad:
                trainable_params.append((f"material_recognition.{name}", param))
        
        for name, param in self.emissivity_calculation.named_parameters():
            if param.requires_grad:
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
