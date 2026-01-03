import numpy as np
from PIL import Image
import sys, os
import argparse
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.RenderingAlgorithims import Algorithm
from src.Raytracing import Raytracer, JitterRayGenerator, RayMarchingIntersection, InverseSDFIntersection, TerminalInteraction, StandardInteraction, RecursiveLambertShading, XRayThicknessShading, TracingStats
from src.Sampling import SamplingManager, SampleSettings, PixelFilter

PostProcessingPipeline = None
from src.Scene import Scene
from test_scenes import *

def render_process(scene: Scene, algorithim: Algorithm):
    """
    Execute the rendering algorithim on the scene and return the rendered image as a numpy array.
    """
    # Render using the algorithim
    pixel_colors = algorithim.render(scene)
    
    # Convert List[Color] to numpy array (width x height x 3)
    width = scene.camera.width
    height = scene.camera.height
    img_array = np.zeros((height, width, 3), dtype=np.float32)
    
    for i, color in enumerate(pixel_colors):
        y = i // width
        x = i % width
        if hasattr(color, 'to_np_array'):
            img_array[y, x] = color.to_np_array()[:3]
        else:
            img_array[y, x] = [color[0], color[1], color[2]]
    
    return img_array

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-post", dest="disable_post", action="store_true", help="Disable post-processing to reduce memory and runtime")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "simple_scene")
    os.makedirs(OUT_DIR, exist_ok=True)

    img_w, img_h = 64, 36 # 256, 144 for higher resolution images

    all_scenes = [
        get_minimal_scene(img_w, img_h),
        get_gradient_scene(img_w, img_h),
        get_emissive_scene(),
        get_lit_studio_scene(),
        get_rgb_room_with_objects_scene(img_w, img_h),
        get_cyberpunk_scene(img_w, img_h),
        get_material_deck_scene(),
        get_refraction_lab_scene()
    ]

    sample_settings = SampleSettings(samples_per_pixel=1, filter_type=PixelFilter.BOX, filter_width=2)
    sampling_manager = SamplingManager(sample_settings)

    generator = JitterRayGenerator()
    intersection = RayMarchingIntersection(max_distance=100)
    test_intersection = InverseSDFIntersection(max_distance=100)
    interactor = StandardInteraction()
    test_interactor = TerminalInteraction()
    shading = RecursiveLambertShading(ambient_color=Color.from_hex("#24272B"), ambient_intensity=0.3, shadow_samples=8)
    test_shading = XRayThicknessShading()

    raytracer = Raytracer(
        max_depth=1,
        sampling_manager=sampling_manager,
        ray_generator=generator,
        intersection_strategy=intersection,
        interaction_strategy=interactor,
        shading_strategy=shading,
        custom_background=Color(1.0, 1.0, 1.0),
        enable_scene_background=True
    )

    enable_postprocessing = not args.disable_post

    for scene in all_scenes:
        # Reset tracing stats per-scene to avoid accumulation and keep reported memory accurate
        raytracer.stats = TracingStats()

        sanitized_name = scene.name.replace(" ", "_").lower()
        out_path = os.path.join(OUT_DIR, f"{sanitized_name}_python")
        print(f"Rendering '{scene.name}' -> {OUT_DIR} ({scene.camera.width}x{scene.camera.height})")
        
        try:
            # 1. Render to Float Array
            raw_img_data = render_process(scene, raytracer)

            save_image(raw_img_data, out_path=out_path + "_raw.png")

            # 2. Post-Process (The Pipeline)
            processed_img = raw_img_data
            # We chain the effects directly on the numpy array

            if enable_postprocessing:
                # Import lazily so heavy deps (scipy) are only loaded when needed
                from src.PostProcessing import PostProcessingPipeline

                # A. Bloom (Make bright lights glow)
                processed_img = PostProcessingPipeline.apply_bloom(
                    processed_img, 
                    threshold=0.8, 
                    intensity=0.05,
                    softness=0.75,
                    radius=1.5,
                    fast=False
                )
                
                # B. Cromatic Aberration (Shifts Red and Blue channels)
                processed_img = PostProcessingPipeline.apply_chromatic_aberration(
                    processed_img,
                    strength=0
                )

                # C. Vignette (Darken corners slightly)
                processed_img = PostProcessingPipeline.apply_vignette(
                    processed_img, 
                    strength=0.2
                )

                # D. Tone Mapping (Compress HDR values to 0-1)
                # Without this, bright spots just clip to white
                processed_img = PostProcessingPipeline.aces_tone_map(processed_img)

                # E. Gamma Correction (Linear -> sRGB)
                # Without this, the image looks too dark
                processed_img = PostProcessingPipeline.gamma_correct(processed_img, gamma=2.0)

            # 3. Save Image
            save_image(processed_img, out_path=out_path + ".png")

            # Free large buffers promptly to avoid accumulation between scenes
            try:
                del raw_img_data
                del processed_img
            except NameError:
                pass
            gc.collect()

            print("-" * 50)
            
        except Exception as e:
            print(f"Failed to render '{scene.name}': {e}")
            import traceback
            traceback.print_exc()