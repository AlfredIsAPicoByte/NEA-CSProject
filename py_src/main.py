import sys, os
import argparse
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.Rendering.Core import Algorithm
from src.Rendering.Raytracing import *
from src.Rendering.Intersections import *
from src.Rendering.Shading import *
from src.Rendering.Interactions import *
from src.Image.Film import Film
from src.Utilities.Scene import Scene
from src.Utilities.Sampling import SamplingManager, SampleSettings, PixelFilter
from src.Utilities.Memory.Profiler import MemoryProfiler
from tests.test_scenes import *
PostProcessingPipeline = None

def render_process(scene: Scene, algorithm: Algorithm):
    """
    Execute the rendering algorithm on the scene and return the rendered image as a numpy array.
    """
    # Render using the algorithm
    film = algorithm.render(scene, tile_size=16)
    
    # Convert List[Color] to numpy array (width x height x 3)
    cam = scene.camera
    width = cam.width if cam is not None else 64
    height = cam.height if cam is not None else 32
    
    if len(film.accum_color) * len(film.accum_color[1]) != width * height:
        print(f"Warning: Rendered pixel count ({len(film.accum_color) * len(film.accum_color[1])}) does not match Camera dimensions ({width}x{height}={width*height}).")

    return film

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raytracer CLI")
    parser.add_argument("--no-post", dest="disable_post", action="store_true", help="Disable post-processing to reduce memory and runtime")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUT_DIR = os.path.join(PROJECT_ROOT, "image", "benchmarking", "scenes")
    os.makedirs(OUT_DIR, exist_ok=True)

    img_width, img_height = 16 * 8, 9 * 8

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

    sample_settings = SampleSettings(samples_per_pixel=1, filter_type=PixelFilter.BOX, filter_width=2)
    sampling_manager = SamplingManager(sample_settings, "halton")

    for scene in all_scenes:
        intersection = BVHIntersection(max_distance=500, max_steps=128)
        interactor = TerminalInteraction()
        shading = LambertShading(
            ambience_settings=AmbienceSettings(False, getattr(scene, "ambient_color", Color(0.03, 0.03, 0.03)), getattr(scene, "ambient_intensity", 0.1)),
            shadow_settings=ShadowSettings(False, 8, 2e-3),
            background_settings=BackgroundSettings(True, Color(0.0, 0.0, 0.0, 0.0), getattr(scene, "background_color", None))
        )

        raytracer = Raytracer(
            max_recursions=1,
            sampling_manager=sampling_manager,
            intersection_strategy=intersection,
            interaction_strategy=interactor,
            shading_strategy=shading,
        )
        # Reset tracing stats per-scene to avoid accumulation and keep reported memory accurate
        raytracer.stats = TracingStats()

        sanitized_name = scene.name.replace(" ", "_").lower()
        out_path = os.path.join(OUT_DIR, f"{sanitized_name}_python")
        width = scene.camera.width if scene.camera is not None else img_width
        height = scene.camera.height if scene.camera is not None else img_height
        print(f"Rendering '{scene.name}' -> {OUT_DIR} ({width}x{height})")
        
        try:
            mem_report_path = out_path + "_mem.txt"
            stats_report_path = out_path + "_stats.txt"
            with MemoryProfiler(enable_tracemalloc=True, top=6) as mp:
                film_data = render_process(scene, raytracer)
            
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

            raw_img_data = film_data.get_image()
            raw_img_data = np.rot90(raw_img_data, k=-1)
            Film.save(raw_img_data, out_path + "_raw.png")

            # 2. Post-Process (The Pipeline)
            processed_img = raw_img_data

            if not args.disable_post:
                from src.Image.PostProcessing.Pipeline import ImagePipeline
                from src.Image.PostProcessing.Passes import *

                with MemoryProfiler(enable_tracemalloc=True, top=6) as mp:
                    pipeline =  ImagePipeline()

                    pipeline.add_pass(Exposure(1.0))
                    
                    pipeline.add_pass(Bloom(0.8, 1, 0.5, 0.75))

                    # pipeline.add_pass(ChromaticAberration(1))

                    pipeline.add_pass(Vignette(0.15, 0.6))

                    pipeline.add_pass(ACESFilmicToneMapping())

                    pipeline.add_pass(GammaCorrection(2.2))

                    processed_img = pipeline.execute(processed_img)

                try:
                    with open(mem_report_path, "a", encoding="utf-8") as f:
                        f.write("\n\nPostprocessing:\n")
                        f.write(mp.format_report())
                        print(" + Appended to memory report")
                except Exception as e:
                    print(f" / Failed to append to memory report: {e}")
                    import traceback
                    traceback.print_exc()

                Film.save(processed_img, out_path + ".png")
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