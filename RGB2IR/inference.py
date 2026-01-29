import torch
import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from typing import Optional, List

from models.rgb2ir_model import RGB2IRLoHaModel
from utils.preprocessing import ImagePreprocessor, ImagePostprocessor, create_comparison_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RGB2IRInference:
    """Inference interface for RGB to IR translation"""
    
    def __init__(self, 
                 config_path: str,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda'):
        """
        Args:
            config_path: Path to LoHA configuration
            checkpoint_path: Optional path to saved LoHA weights
            device: Device to use
        """
        self.device = device
        logger.info(f"Loading model on {device}...")
        
        self.model = RGB2IRLoHaModel(
            config_path=config_path,
            device=device,
            enable_controlnet=False  # Disabled to match training configuration
        )
        
        # Load checkpoint if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                # Loading from training checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                # Loading from raw state dict
                self.model.load_state_dict(checkpoint, strict=False)
        
        # Set to eval mode
        self.model.pipeline.unet.eval()
        self.model.pipeline.text_encoder.eval()
        self.model.material_recognition.eval()
        self.model.emissivity_calculation.eval()
        
        # Enable memory-efficient inference
        self.model.enable_model_cpu_offload()
        self.model.enable_attention_slicing()
        
        logger.info("Model ready for inference")
    
    def infer(self,
              rgb_image: torch.Tensor,
              use_controlnet: bool = True,
              controlnet_type: str = 'depth',
              guidance_scale: float = 7.5,
              num_inference_steps: int = 50,
              strength: float = 0.8,
              denoise: bool = True,
              apply_colormap: bool = False) -> dict:
        """
        Translate RGB image to IR
        
        Args:
            rgb_image: RGB tensor (3, H, W) in [-1, 1]
            use_controlnet: Whether to use ControlNet guidance
            controlnet_type: 'depth' or 'canny' or 'both'
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of inference steps
            strength: Denoising strength
            denoise: Whether to apply post-processing denoising
            apply_colormap: Whether to apply thermal colormap
        
        Returns:
            Dictionary with:
                'ir_image': Generated IR image (1, H, W) in [-1, 1]
                'ir_colored': Colored IR visualization (H, W, 3) if apply_colormap=True
        """
        with torch.no_grad():
            # Prepare control inputs if needed
            depth_map = None
            canny_edges = None
            
            if use_controlnet:
                rgb_np = ((rgb_image.cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
                
                if controlnet_type in ['depth', 'both']:
                    # Estimate depth from RGB (using Laplacian edges)
                    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
                    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
                    depth_map = cv2.GaussianBlur(np.abs(laplacian), (5, 5), 0)
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                    depth_map = torch.from_numpy(np.expand_dims(depth_map, 0)).float().to(self.device)
                    depth_map = depth_map * 2.0 - 1.0
                
                if controlnet_type in ['canny', 'both']:
                    # Compute canny edges
                    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    edges = edges.astype(np.float32) / 255.0
                    canny_edges = torch.from_numpy(np.expand_dims(edges, 0)).float().to(self.device)
                    canny_edges = canny_edges * 2.0 - 1.0
            
            # Prepare RGB for model (convert to float16 for the pipeline)
            rgb_model = rgb_image.unsqueeze(0).to(self.device, dtype=torch.float16) if rgb_image.dim() == 3 else rgb_image.to(self.device, dtype=torch.float16)
            
            # Convert control inputs to float16 as well
            if depth_map is not None:
                depth_map = depth_map.unsqueeze(0).to(dtype=torch.float16) if depth_map.dim() == 2 else depth_map.to(dtype=torch.float16)
            if canny_edges is not None:
                canny_edges = canny_edges.unsqueeze(0).to(dtype=torch.float16) if canny_edges.dim() == 2 else canny_edges.to(dtype=torch.float16)
            
            # Generate IR
            output = self.model(
                rgb_image=rgb_model,
                prompt="high quality infrared thermal image, temperature distribution, emissivity",
                negative_prompt="blurry, low quality, artifacts, noise",
                depth_map=depth_map,
                canny_edges=canny_edges,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                return_features=False
            )
            
            ir_image = output['ir_image'].squeeze(0)
            
            # Post-processing
            if denoise:
                ir_image = ImagePostprocessor.denoise_ir(ir_image, strength=0.5)
            
            result = {'ir_image': ir_image}
            
            if apply_colormap:
                ir_colored = ImagePostprocessor.thermal_color_map(ir_image, colormap='turbo')
                result['ir_colored'] = ir_colored
            
            return result
    
    def infer_batch(self,
                    rgb_images: torch.Tensor,
                    **kwargs) -> List[dict]:
        """Infer on batch of images"""
        results = []
        for i in range(rgb_images.shape[0]):
            result = self.infer(rgb_images[i], **kwargs)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description='RGB to IR inference')
    parser.add_argument('--config', type=str, required=True, help='Path to LoHA config')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to LoHA weights')
    parser.add_argument('--rgb_image', type=str, required=True, help='Input RGB image path')
    parser.add_argument('--output', type=str, required=True, help='Output IR image path')
    parser.add_argument('--use_controlnet', action='store_true', default=True)
    parser.add_argument('--controlnet_type', type=str, default='both', choices=['depth', 'canny', 'both'])
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--denoise', action='store_true', default=True)
    parser.add_argument('--colormap', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Load input image
    logger.info(f"Loading RGB image from {args.rgb_image}")
    rgb_pil = Image.open(args.rgb_image).convert('RGB')
    rgb_np = np.array(rgb_pil).astype(np.float32)
    rgb_tensor = ImagePreprocessor.normalize_rgb(rgb_np)
    rgb_tensor = torch.from_numpy(rgb_tensor).permute(2, 0, 1).float()
    
    # Initialize inference
    inferencer = RGB2IRInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Run inference
    logger.info("Running inference...")
    result = inferencer.infer(
        rgb_image=rgb_tensor,
        use_controlnet=args.use_controlnet,
        controlnet_type=args.controlnet_type,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        denoise=args.denoise,
        apply_colormap=args.colormap
    )
    
    # Save IR image
    ir_image = result['ir_image']
    ir_np = ((ir_image.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)[0]
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving IR image to {args.output}")
    Image.fromarray(ir_np, mode='L').save(args.output)
    
    # Save colormap version if requested
    if args.colormap and 'ir_colored' in result:
        colormap_path = output_path.parent / f"{output_path.stem}_colormap.png"
        logger.info(f"Saving colormap to {colormap_path}")
        Image.fromarray(result['ir_colored']).save(colormap_path)
    
    logger.info("Inference completed!")


if __name__ == '__main__':
    main()
