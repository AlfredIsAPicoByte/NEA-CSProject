import os
import argparse
import gc

from src.Rendering.Core import Algorithm
from src.Rendering.RayTracing.Core import *
from src.Rendering.RayTracing.Intersections import *
from src.Rendering.RayTracing.Shading import *
from src.Image.Film import Film
from src.Data.Scene import Scene
from src.Data.Sampling.Core import SamplingManager, SampleSettings, PixelFilter
from src.Utilities.Memory.Profiler import MemoryProfiler
from tests.test_scenes import *
PostProcessingPipeline = None

def render_process(scene: Scene, algorithm: Algorithm):
    """
    Execute the rendering algorithm on the scene and return the rendered image as a numpy array.
    """
    # Render using the algorithm
    algorithm.generate_film(scene)
    print(" + Rendering complete")
    print(f"{algorithm.settings.film}")

    # Convert List[Color] to numpy array (width x height x 3)
    cam = scene.camera
    width = cam.width if cam is not None else 64
    height = cam.height if cam is not None else 32

    if len(algorithm.settings.film.accum_color) * len(algorithm.settings.film.accum_color[1]) != width * height:
        print(f"Warning: Rendered pixel count ({len(algorithm.settings.film.accum_color) * len(algorithm.settings.film.accum_color[1])}) does not match Camera dimensions ({width}x{height}={width*height}).")

def apply_post_processing(raw_img):
    """
    Apply post-processing pipeline to the raw image.
    """
    from src.Image.PostProcessing.Pipeline import ImagePipeline
    from src.Image.PostProcessing.Passes import (
        AutoExposure,
        Bloom,
        ChromaticAberration,
        Vignette,
        ACESFilmicToneMapping,
        GammaCorrection
    )
    
    pipeline = ImagePipeline()
    pipeline.add_pass(AutoExposure())
    pipeline.add_pass(Bloom(1, 5, 0.67, 0.75))
    pipeline.add_pass(ChromaticAberration())
    pipeline.add_pass(Vignette(0.15, 0.6))
    pipeline.add_pass(ACESFilmicToneMapping())
    pipeline.add_pass(GammaCorrection(2.2))
    
    return pipeline.execute(raw_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RayTracer CLI")
    parser.add_argument("--no-post", dest="disable_post", action="store_true", help="Disable post-processing to reduce memory and runtime")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    IMG_OUT_DIR = os.path.join(PROJECT_ROOT, "images", "benchmarking", "scenes")
    REP_OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "benchmarking", "scenes")
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    os.makedirs(REP_OUT_DIR, exist_ok=True)

    img_width, img_height = 640, 480 # 480p

    all_scenes = [
        get_minimal_scene(img_width, img_height),
        get_gradient_scene(img_width, img_height),
        get_emissive_scene(img_width, img_height),
        get_lit_studio_scene(img_width, img_height),
        get_rgb_room_with_objects_scene(img_width, img_height),
        get_cyberpunk_scene(img_width, img_height),
        get_material_deck_scene(img_width, img_height),
        get_refraction_lab_scene(img_width, img_height),
        get_scifi_corridor_scene(img_width, img_height),
        get_sunset_monolith_scene(img_width, img_height),
        get_pastel_blocks_scene(img_width, img_height),
        get_glass_prism_scene(img_width, img_height),
        get_glass_sculpture_scene(img_width, img_height),
        get_100_spheres_grid_scene(img_width, img_height),
        get_low_ior_scene(img_width, img_height),
    ]

    sample_settings = SampleSettings(width=img_width, height=img_height, samples_per_pixel=8, filter_type=PixelFilter.GAUSSIAN, filter_width=4)
    sampling_manager = SamplingManager(sample_settings, "halton")

    for scene in all_scenes:
        intersection = BVHIntersection(max_distance=scene.camera.far * 10, max_steps=2048)
        shading = LambertShading(
            ambience_settings=AmbienceSettings(True, getattr(scene, "ambient_color", Color(0.03, 0.03, 0.03)), getattr(scene, "ambient_intensity", 0.07)),
            shadow_settings=ShadowSettings(True, 16, 1e-3),
            background_settings=BackgroundSettings(True, Color(0.0, 0.0, 0.0, 0.0), getattr(scene, "background_color", None))
        )

        raytracer = RayTracer(RayTracingSettings(
            image_width=img_width,
            image_height=img_height,
            sampling_manager=sampling_manager,
            max_recursions=6, #7
            intersection_method=intersection,
            shading_method=shading,
            use_tiling=True,
            tile_size=64,
            debug_mode=True,
            verbose_logging=True
        ))
        # Reset tracing stats per-scene to avoid accumulation and keep reported memory accurate
        raytracer.stats = TracingStats()

        sanitized_name = scene.name.replace(" ", "_").lower()
        raw_image_out_path = os.path.join(IMG_OUT_DIR, "raw")
        processed_image_out_path = os.path.join(IMG_OUT_DIR, "processed")
        mem_report_out_path = os.path.join(REP_OUT_DIR, "memory")
        stats_report_out_path = os.path.join(REP_OUT_DIR, "statistics")
            
        os.makedirs(raw_image_out_path, exist_ok=True)
        os.makedirs(processed_image_out_path, exist_ok=True)
        os.makedirs(mem_report_out_path, exist_ok=True)
        os.makedirs(stats_report_out_path, exist_ok=True)

        raw_image_out_path = os.path.join(raw_image_out_path, f"{sanitized_name}_python.png")
        processed_image_out_path = os.path.join(processed_image_out_path, f"{sanitized_name}_python.png")
        mem_report_out_path = os.path.join(mem_report_out_path, f"{sanitized_name}_python.txt")
        stats_report_out_path = os.path.join(stats_report_out_path, f"{sanitized_name}_python.txt")

        width = scene.camera.width if scene.camera is not None else img_width
        height = scene.camera.height if scene.camera is not None else img_height
        print(f"Rendering '{scene.name}' -> {raw_image_out_path} ({width}x{height})")
        
        try:
            with MemoryProfiler(enable_tracemalloc=True, top=6) as mp:
                render_process(scene, raytracer)
            
            try:
                with open(stats_report_out_path, "w", encoding="utf-8") as f:
                    f.write(raytracer.stats.format_report())
                    print(" + Wrote rendering statistics")
            except Exception as e:
                print(f" / Failed to write rendering statistics:\n{e}\n")
                import traceback
                traceback.print_exc()
            try:
                with open(mem_report_out_path, "w", encoding="utf-8") as f:
                    f.write(mp.format_report())
                    print(" + Wrote memory report")
            except Exception as e:
                print(f" / Failed to write memory report:\n{e}\n")
                import traceback
                traceback.print_exc()

            raw_img_data = raytracer.settings.film.get_image()
            Film.save(raw_img_data, raw_image_out_path)

            # 2. Post-Process (The Pipeline)
            processed_img = raw_img_data

            if not args.disable_post:
                with MemoryProfiler(enable_tracemalloc=True, top=6) as mp:
                    processed_img = apply_post_processing(raw_img_data)

                try:
                    with open(mem_report_out_path, "a", encoding="utf-8") as f:
                        f.write("\n\nPostprocessing:\n")
                        f.write(mp.format_report())
                        print(" + Appended to memory report")
                except Exception as e:
                    print(f" / Failed to append to memory report:\n{e}\n")
                    import traceback
                    traceback.print_exc()

                Film.save(processed_img, processed_image_out_path)
            else:
                print(" > Skipping post-processing (--no-post active)")

            # Free large buffers promptly to avoid accumulation between scenes
            try:
                del raw_img_data
                del processed_img
            except NameError:
                pass
            gc.collect()

            print("|-" * 32 + "|\n")
            
        except Exception as e:
            print(f"Failed to render '{scene.name}': {e}")
            import traceback
            traceback.print_exc()