import numpy as np
from PIL import Image
import sys, os
import argparse
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.Scene import Scene
from py_src.src.Rendering.Core import Algorithm
from src.Raytracing import *
from src.Generation import *
from py_src.src.Rendering.Intersections import *
from src.Shading import *
from py_src.src.Rendering.Interactions import *
from src.Sampling import SamplingManager, SampleSettings, PixelFilter, reconstruct_pixel
from py_src.src.Utilities.Memory.Profiler import MemoryProfiler
from test_scenes import *
PostProcessingPipeline = None

def render_process(scene: Scene, algorithm: Algorithm):
    """
    Execute the rendering algorithm on the scene and return the rendered image as a numpy array.
    """
    # Render using the algorithm
    pixel_colors = algorithm.render(scene, tile_size=16)
    
    # Convert List[Color] to numpy array (width x height x 3)
    cam = scene.camera
    width = cam.width if cam is not None else 64
    height = cam.height if cam is not None else 32
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
    parser = argparse.ArgumentParser(description="Raytracer CLI")
    parser.add_argument("--no-post", dest="disable_post", action="store_true", help="Disable post-processing to reduce memory and runtime")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "simple_scene")
    os.makedirs(OUT_DIR, exist_ok=True)

    img_w, img_h = 16 * 8, 9 * 8

    all_scenes = [
        get_minimal_scene(img_w, img_h),
        get_gradient_scene(img_w, img_h),
        get_emissive_scene(img_w, img_h),
        get_lit_studio_scene(img_w, img_h),
        get_rgb_room_with_objects_scene(img_w, img_h),
        get_cyberpunk_scene(img_w, img_h),
        get_material_deck_scene(img_w, img_h),
        get_refraction_lab_scene(img_w, img_h),
        get_scifi_corridor_scene(img_w, img_h),
        get_sunset_monolith_scene(img_w, img_h),
        get_pastel_blocks_scene(img_w, img_h),
        get_glass_prism_scene(img_w, img_h),
        get_glass_sculpture_scene(img_w, img_h),
        get_100_spheres_grid_scene(img_w, img_h),
        get_low_ior_scene(img_w, img_h),
    ]

    sample_settings = SampleSettings(samples_per_pixel=1, filter_type=PixelFilter.BOX, filter_width=2)
    sampling_manager = SamplingManager(sample_settings, "halton")

    for scene in all_scenes:
        generator = RayGenerator()
        intersection = BVHIntersection(max_distance=500, max_steps=64)
        interactor = TerminalInteraction()
        shading = FlatShading(
            ambience_settings=AmbienceSettings(False, getattr(scene, "ambient_color", Color(0.03, 0.03, 0.03, 1.0)), getattr(scene, "ambient_intensity", 0.1)),
            shadow_settings=ShadowSettings(False, 8, 2e-3),
            background_settings=BackgroundSettings(True, Color(0.0, 0.0, 0.0, 0.0), getattr(scene, "background_color", None))
        )

        raytracer = Raytracer(
            max_recursions=1,
            sampling_manager=sampling_manager,
            ray_generator=generator,
            intersection_strategy=intersection,
            interaction_strategy=interactor,
            shading_strategy=shading,
        )
        # Reset tracing stats per-scene to avoid accumulation and keep reported memory accurate
        raytracer.stats = TracingStats()

        sanitized_name = scene.name.replace(" ", "_").lower()
        out_path = os.path.join(OUT_DIR, f"{sanitized_name}_python")
        width = scene.camera.width if scene.camera is not None else img_w
        height = scene.camera.height if scene.camera is not None else img_h
        print(f"Rendering '{scene.name}' -> {OUT_DIR} ({width}x{height})")
        
        try:
            # 1. Render to Float Array (profile memory during render)
            mem_report_path = out_path + "_mem.txt"
            stats_report_path = out_path + "_stats.txt"
            with MemoryProfiler(enable_tracemalloc=True, top=6) as mp:
                raw_img_data = render_process(scene, raytracer)
            
            try:
                with open(stats_report_path, "w", encoding="utf-8") as f:
                    f.write(raytracer.stats.format_report())
                    print(" + Wrote rendering statistics")
            except Exception as e:
                print(f" / Failed to write rendering statistics: {e}")
                import traceback
                traceback.print_exc()
            try:
                with open(mem_report_path, "w", encoding="utf-8") as f:
                    f.write(mp.format_report())
                    print(" + Wrote memory report")
            except Exception as e:
                print(f" / Failed to write memory report: {e}")
                import traceback
                traceback.print_exc()

            save_image(raw_img_data, out_path=out_path + "_raw.png")

            # 2. Post-Process (The Pipeline)
            processed_img = raw_img_data

            if not args.disable_post:
                with MemoryProfiler(enable_tracemalloc=True, top=6) as mp:
                    print(" > Running post-processing")
                    # Import lazily so heavy deps (scipy) are only loaded when needed
                    from src.PostProcessing import *

                    # We chain the effects directly on the numpy array
                    # A. Bloom (Make bright lights glow)
                    processed_img = PostProcessingPipeline.apply_bloom(
                        processed_img, 
                        threshold=0.8, 
                        intensity=0.05,
                        softness=0.75,
                        radius=1,
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
                
                try:
                    with open(mem_report_path, "a", encoding="utf-8") as f:
                        f.write("\n\nPostprocessing:\n")
                        f.write(mp.format_report())
                        print(" + Appended to memory report")
                except Exception as e:
                    print(f" / Failed to append to memory report: {e}")
                    import traceback
                    traceback.print_exc()

                save_image(processed_img, out_path=out_path + ".png")
            else:
                print(" > Skipping post-processing (--no-post active)")

            # Free large buffers promptly to avoid accumulation between scenes
            try:
                del raw_img_data
                del processed_img
            except NameError:
                pass
            gc.collect()

            print("|-" * 32 + "|")
            
        except Exception as e:
            print(f"Failed to render '{scene.name}': {e}")
            import traceback
            traceback.print_exc()