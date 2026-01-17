import os
import cv2
import numpy as np
from PIL import Image

def create_collages(rgb_folder, ir_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    rgb_images = [f for f in os.listdir(rgb_folder)]
    
    for rgb_image in rgb_images:
        rgb_path = os.path.join(rgb_folder, rgb_image)
        image_id = rgb_image.replace("DJI_", "").replace(".jpg", "")
        print(image_id)
        ir_image = f"DJI_{str(int(image_id) + 1).zfill(4)}_R.JPG"
        ir_path = os.path.join(ir_folder, ir_image)
        
        if os.path.exists(ir_path):
            rgb_img = cv2.imread(rgb_path)
            ir_img = cv2.imread(ir_path)
            
            if rgb_img is None or ir_img is None:
                print(f"Skipping {rgb_image} as one of the images is not readable.")
                continue
            
            # Resize images to the same height
            target_height = max(rgb_img.shape[0], ir_img.shape[0])
            rgb_img = cv2.resize(rgb_img, (int(rgb_img.shape[1] * target_height / rgb_img.shape[0]), target_height))
            ir_img = cv2.resize(ir_img, (int(ir_img.shape[1] * target_height / ir_img.shape[0]), target_height))
            
            collage = np.hstack((rgb_img, ir_img))
            output_path = os.path.join(output_folder, f"collage_{image_id}.jpg")
            cv2.imwrite(output_path, collage)
            print(f"Collage saved: {output_path}")
        else:
            print(f"No matching IR image found for {rgb_image}")

# Example usage:
rgb_folder = "RGB"
ir_folder = "Thermal"
output_folder = "merged"
create_collages(rgb_folder, ir_folder, output_folder)
