import numpy as np
from PIL import Image
from typing import Optional, Callable, List
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.RenderingAlgorithims import Algorithm
from src.Raytracing import Raytracer, JitterRayGenerator, RayMarchingIntersection, SimpleMaterialInteraction, RecursiveLambertShading
from src.Sampling import SamplingManager, SampleSettings, PixelFilter
from src.Postprocessing import PostProcessingPipeline
from src.Scene import Scene
from src.Luminance import Color

# Import your scenes
from test_scenes import *

def render_process(scene: Scene, algorithim: Algorithm) -> np.ndarray:
    """
    Renders the scene and returns a NumPy float32 array (H, W, 3).
    Optimized to avoid slow Python loops during list-to-array conversion.
    """
    W, H = scene.camera.width, scene.camera.height
    
    # 1. Render (Returns flat list of Color objects)
    pixel_colors: List[Color] = algorithim.render(scene)
    
    # 2. Vectorized Conversion (Much faster than for-loops)
    # We extract r,g,b into a numpy array in one go
    # Assumes Color class has .r, .g, .b attributes
    pixels_np = np.array([[c.r, c.g, c.b] for c in pixel_colors], dtype=np.float32)
    
    # Reshape flattened list to Image dimensions (H, W, 3)
    raw_buffer = pixels_np.reshape((H, W, 3))
    
    return raw_buffer

def save_image(img_data: np.ndarray, out_path="render_out.png"):
    """
    Saves the float image data to PNG.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Quantize: Float(0..1) -> Int(0..255)
    final_pixels = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)
    
    im = Image.fromarray(final_pixels, mode="RGB")
    im.save(out_path)
    print(f" > Saved to {out_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "simple_scene")
    os.makedirs(OUT_DIR, exist_ok=True)

    img_w, img_h = 64, 36 # 256, 144 for higher resolution images
    enable_post_processing = True

    all_scenes = [
        get_minimal_scene(img_w, img_h),
        get_gradient_scene(img_w, img_h),
        get_emissive_scene(img_w, img_h),
        get_lit_studio_scene(img_w, img_h),
        get_rgb_room_with_objects_scene(img_w, img_h),
        get_cyberpunk_scene(img_w, img_h),
        get_material_deck_scene(img_w, img_h),
        get_refraction_lab_scene(img_w, img_h)
    ]

    sample_settings = SampleSettings(img_w, img_h, 1, PixelFilter.BOX, 2)
    sampling_manager = SamplingManager(sample_settings, "halton")

    generator = JitterRayGenerator(sampling_manager._sampler)
    intersection = RayMarchingIntersection()
    interactor = SimpleMaterialInteraction(sampling_manager._sampler)
    shading = RecursiveLambertShading(ambient_color=Color.from_hex("#24272B"), ambient_intensity=0.67)

    raytracer = Raytracer(
        sampling_manager=sampling_manager,
        ray_generator=generator,
        intersection_strategy=intersection,
        interaction_strategy=interactor,
        shading_strategy=shading
    )

    for scene in all_scenes:
        sanitized_name = scene.name.replace(" ", "_").lower()
        out_path = os.path.join(OUT_DIR, f"{sanitized_name}.png")
        print(f"Rendering '{scene.name}' -> {out_path} ({scene.camera.width}x{scene.camera.height})")
        
        try:
            # 1. Render to Float Array
            raw_img_data = render_process(scene, raytracer)

            # 2. Post-Process (The Pipeline)
            processed_img = raw_img_data
            if enable_post_processing:
                # We chain the effects directly on the numpy array
                
                # A. Bloom (Make bright lights glow)
                processed_img = PostProcessingPipeline.apply_bloom(
                    processed_img, 
                    threshold=1, 
                    intensity=0.3, 
                    radius=4,
                    fast=True
                )
                
                # B. Cromatic Aberration (Shifts Red and Blue channels)
                processed_img = PostProcessingPipeline.apply_chromatic_aberration(
                    processed_img,
                    strength=0.5
                )

                # C. Vignette (Darken corners slightly)
                processed_img = PostProcessingPipeline.apply_vignette(
                    processed_img, 
                    strength=0
                )

                # D. Tone Mapping (Compress HDR values to 0-1)
                # Without this, bright spots just clip to white
                processed_img = PostProcessingPipeline.aces_tone_map(processed_img)

                # E. Gamma Correction (Linear -> sRGB)
                # Without this, the image looks too dark
                processed_img = PostProcessingPipeline.gamma_correct(processed_img, gamma=2.2)

            # 3. Save Image
            save_image(processed_img, out_path=out_path)
            print("-" * 50)
            
        except Exception as e:
            print(f"Failed to render '{scene.name}': {e}")
            import traceback
            traceback.print_exc()