import numpy as np
import math
from enum import Enum
from typing import List, Tuple, Optional

from src.Data.Transform import Transform
from src.Data.Ray import Ray, TracingRay
from src.Data.Ratio import Ratio
from .Sampling.Core import Sampler

class CameraType (Enum):
    """
    Enum for different camera projection types.

    PERSPECTIVE: Perspective projection.
    ORTHOGRAPHIC: Orthographic projection.
    """
    PERSPECTIVE = 1
    ORTHOGRAPHIC = 2

class CameraMode (Enum):
    """
    Enum for different camera movement modes.

    FIRST_PERSON: First-person movement.
    PLANE: Plane movement (like an airplane).
    ORBIT: Orbit around a target.
    """
    FIRST_PERSON = 1
    PLANE = 2
    ORBIT = 3

class Camera:
    """
    A virtual camera class used to store properties and process calculations.
    """
    def __init__(
        self,
        transform: Transform = Transform.Identity(),
        resolution_width: int = 800,
        resolution_height: int = 600,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 1000.0,
        camera_type: CameraType = CameraType.PERSPECTIVE,
        othographic_distance: float = 10,
        aperture_radius: float = 0.0, # For Depth of Field
        focal_distance: float = 10.0  # Distance to focus plane
    ):
        self.transform = transform
        self.transform.update_orientations()
        
        self.width = resolution_width
        self.height = resolution_height
        self.aspect_ratio = Ratio(self.width, self.height)
        
        self.fov = fov
        self.near = near
        self.far = far
        self.othographic_distance = othographic_distance
        self.type = camera_type
        self.mode = CameraMode.FIRST_PERSON
        
        self.name = "Camera"
        
        # Ray Tracing Specifics
        self.aperture_radius = aperture_radius
        self.focal_distance = focal_distance
    
    def __post_init__(self):
        if self.fov <= 0:
            raise ValueError("FOV must be greater than 0")
        
        if self.near <= 0 or self.far <= self.near:
            raise ValueError("Invalid near and far plane values")
    
        if self.resolution_width is None or self.resolution_height is None:
            raise ValueError("Width and Height must be provided (as 'resolution_width, resolution_height')")

    def get_view_matrix(self) -> np.ndarray:
        """
        Returns the view matrix for the camera.
        """
        # The view matrix is typically the inverse of the camera's global transform
        return np.linalg.inv(np.array(self.transform.get_global_matrix()))

    def get_projection_matrix(self) -> np.ndarray:
        """
        Returns the projection matrix for the camera.
        """
        import math

        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        near, far = self.near, self.far
        
        if self.type == CameraType.PERSPECTIVE:
            inv_tan = 1.0 / tan_half_fov
            aspect = self.aspect_ratio.value
            return np.array([
                [inv_tan / aspect, 0.0, 0.0, 0.0],
                [0.0, inv_tan, 0.0, 0.0],
                [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
                [0.0, 0.0, -1.0, 0.0]
            ])
        
        elif self.type == CameraType.ORTHOGRAPHIC:
            # Orthographic projection matrix calculation
            # 'fov' here acts as half-height for the orthographic plane
            half_height = tan_half_fov * self.othographic_distance
            half_width = half_height * float(self.aspect_ratio.value)

            left = -half_width
            right = half_width
            bottom = -half_height
            top = half_height

            near, far = self.near, self.far
            return np.array([
                [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=float)
        
        return np.identity(4)
    
    def resize(self, width: float, height: float):
        """
        Resize the camera resolution and update aspect ratio.

        :param width: New width in pixels.
        :param height: New height in pixels.
        """
        self.resolution_width = width
        self.resolution_height = height
        self.aspect_ratio = Ratio(width, height)

    def resize_aspect(self, aspect: Ratio, scale: float = 1.0):
        """
        Resize the camera resolution based on a new aspect ratio and scale.
        :param aspect: New aspect ratio.
        :param scale: Scale factor to apply to the resolution.
        """

        self.aspect_ratio = aspect
        self.resolution_width = int(aspect.width * scale)
        self.resolution_height = int(aspect.height * scale)

    def generate_ray(self, u: float, v: float) -> Ray:
        """
        Converts a pixel (x, y) into a World Space Ray.

        :param u: Normalized horizontal coordinate [0,1].
        :param v: Normalized vertical coordinate [0,1].
        :return: Ray in world space.
        """
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)

        # --- ORTHOGRAPHIC ---
        if self.type == CameraType.ORTHOGRAPHIC:
            # 1. Direction is always forward
            direction = self.transform.forward
            
            # 2. Origin shifts along the camera plane
            height_size = tan_half_fov * getattr(self, "distance", 1)
            width_size = height_size * self.aspect_ratio.value

            # Map 0..1 to -Width/2 .. +Width/2
            offset_x = (u - 0.5) * width_size
            offset_y = (0.5 - v) * height_size 

            origin = (
                self.transform.position + 
                (self.transform.right * offset_x) + 
                (self.transform.up * offset_y)
            )
            return Ray(origin + direction * self.near, direction)

        # --- PERSPECTIVE ---
        # Geometric approach (Snippet 1 style) is usually faster for CPU raytracing 
        # than matrix inversion if you aren't caching the inverse matrix.
        ndc_x = (2.0 * u) - 1.0
        ndc_y = 1.0 - (2.0 * v)
        
        # Camera Space Direction
        # Assuming standard camera: Right=+X, Up=+Y, Forward=+Z (or -Z depending on convention)
        # We use the Basis vectors directly to go straight to World Space
        
        x_scale = ndc_x * self.aspect_ratio.value * tan_half_fov
        y_scale = ndc_y * tan_half_fov
        
        # Construct direction in world space by summing basis vectors
        # This assumes 'forward' is the direction the camera looks.
        direction = (
            (self.transform.forward) +         # 1 unit forward
            (self.transform.right * x_scale) + # Horizontal spread
            (self.transform.up * y_scale)      # Vertical spread
        )
        
        # Normalize
        direction = direction / np.linalg.norm(direction)
        
        return Ray(self.transform.position + direction * self.near, direction)
    
    def generate_lens_ray(self, u: float, v: float, sampler: Sampler) -> Ray:
        """
        Generate a ray from the camera considering Depth of Field (DOF).
        Uses the aperture radius and focal distance to simulate lens effects.

        :param u: Normalized horizontal coordinate [0,1].
        :param v: Normalized vertical coordinate [0,1].
        :param sampler: Sampler object to provide random samples for lens jitter.
        :return: Ray in world space with DOF applied.
        """
        # 1. Calculate the standard "Perfect Pinhole" Direction
        #    (This part is the same as before)
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        ndc_x = (2.0 * u) - 1.0
        ndc_y = 1.0 - (2.0 * v)
        
        x_scale = ndc_x * self.aspect_ratio.value * tan_half_fov
        y_scale = ndc_y * tan_half_fov
        
        # This is the direction a perfect ray would travel in Camera Space
        # (Assuming Camera looks down -Z)
        pinhole_dir = np.array([x_scale, y_scale, -1.0])
        pinhole_dir /= np.linalg.norm(pinhole_dir)

        # 2. Handle field of depth
        # If aperture is 0, we are a perfect Pinhole camera.
        if self.aperture_radius <= 0:
            ray_origin_cam = np.array([0.0, 0.0, 0.0])
            ray_direction_cam = pinhole_dir
            
        else:
            # A. Determine the Focal Point
            # We know the ray MUST hit the focal plane at this specific point
            # regardless of where it starts on the lens.
            # Scaling pinhole_dir by (focal_dist / z) extends it to the focal plane.
            ft = abs(self.focal_distance / pinhole_dir[2])
            focal_point = pinhole_dir * ft

            # B. Sample the Lens (Crucial Step)
            # We ask the sampler for 2 random numbers (0..1)
            rand_u, rand_v = sampler.next_2d()
            
            # C. Map those random numbers to a disk (Concentric mapping is best)
            # Simplified version: Polar coordinates
            r = np.sqrt(rand_u) * self.aperture_radius
            theta = rand_v * 2.0 * math.pi
            
            lens_u = r * math.cos(theta)
            lens_v = r * math.sin(theta)
            
            # This is our new "jittered" starting point on the lens
            ray_origin_cam = np.array([lens_u, lens_v, 0.0])
            
            # D. Calculate new direction
            # From the random point on the lens -> To the fixed focal point
            ray_direction_cam = focal_point - ray_origin_cam
            ray_direction_cam /= np.linalg.norm(ray_direction_cam)

        # ---------------------------------------------------------
        # 3. Transform to World Space
        # ---------------------------------------------------------
        
        # Rotate the local direction into world space
        world_direction = (
            (self.transform.right * ray_direction_cam[0]) +
            (self.transform.up * ray_direction_cam[1]) +
            (self.transform.forward * (-ray_direction_cam[2])) # -Z becomes forward
        )
        
        # Translate the origin (Camera Pos + Lens Offset rotated to world)
        world_origin = (
            self.transform.position + 
            (self.transform.right * ray_origin_cam[0]) + 
            (self.transform.up * ray_origin_cam[1])
        )

        return Ray(world_origin, world_direction)

    def generate_screen_rays(
            self,
            sampler: Sampler,
            region: Optional[Tuple[int, int, int, int]] = None, # (x, y, w, h)
        ) -> List[TracingRay]:
        """
        Generate camera rays

        :param sampler: Sampler object to provide pixel samples.
        :param region: Optional region (x, y, width, height) to generate rays for.
        :return: List of TracingRay objects for the specified region.
        """

        # 1. Resolve Region
        if region is None:
            x_start, y_start, region_w, region_h = 0, 0, self.height, self.height
        else:
            x_start, y_start, req_w, req_h = region
            # Clamp to image bounds
            x_start = max(0, min(x_start, self.width))
            y_start = max(0, min(y_start, self.height))
            region_w = max(0, min(req_w, self.width - x_start))
            region_h = max(0, min(req_h, self.height - y_start))

        rays: List[TracingRay] = []

        # 2. Iterate over pixels
        for y in range(y_start, y_start + region_h):
            for x in range(x_start, x_start + region_w):

                # 3. Get Samples
                pixel_samples = sampler.get_samples_per_pixel(x, y)
                for i, sample in enumerate(pixel_samples):
                    # Normalize sample coordinates to 0..1 for camera functions
                    # to avoid noisy renders with one sample only perform when there are more samples to work with
                    screen_x = (x + sample.u) / float(self.width)
                    screen_y = (y + sample.v) / float(self.height)
                    
                    pixle_x, pixle_y = sampler.sample_pixel(x, y, i % region_w + i // region_w) if i > 0 else [0.0, 0.0]
                    
                    # 4. Calculate Ray Geometry
                    _r = self.generate_ray(screen_x, screen_y)

                    # 5. Build Ray
                    ray = TracingRay(
                        origin=_r.origin,
                        orientation=_r.orientation,
                        pixel_x=x,
                        pixel_y=y,
                        sample_u=sample.u,
                        sample_v=sample.v,
                        name=f"ray#{i}_({x},{y})",
                    )
                    rays.append(ray)
        return rays

    def screen_to_world(self, screen_x: float, screen_y: float, depth: float) -> np.ndarray:
        """
        Convert normalized screen coordinates (0..1) and depth to world space position.

        :param screen_x: Normalized horizontal coordinate [0,1].
        :param screen_y: Normalized vertical coordinate [0,1].
        :param depth: Depth value from near to far plane.
        :return: World space position as a numpy array.
        """
        # 1. Convert screen coords to NDC
        ndc_x = (2.0 * screen_x) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y)
        ndc_z = (2.0 * depth) - 1.0  # Assuming depth is normalized [0,1]

        # 2. Create clip space position
        clip_space_pos = np.array([ndc_x, ndc_y, ndc_z, 1.0])

        # 3. Inverse Projection to View Space
        inv_proj = np.linalg.inv(self.get_projection_matrix())
        view_space_pos = inv_proj @ clip_space_pos
        view_space_pos /= view_space_pos[3]  # Perspective divide

        # 4. Inverse View to World Space
        inv_view = np.linalg.inv(self.get_view_matrix())
        world_space_pos = inv_view @ np.array([view_space_pos[0], view_space_pos[1], view_space_pos[2], 1.0])
        
        return world_space_pos[:3]
    
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a world space position to normalized screen coordinates (0..1) and depth.

        :param world_pos: World space position as a numpy array.
        :return: Tuple of (screen_x, screen_y, depth) in normalized coordinates.
        """
        # 1. World to View Space
        view_space_pos = self.get_view_matrix() @ np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])

        # 2. View to Clip Space
        clip_space_pos = self.get_projection_matrix() @ view_space_pos

        # 3. Perspective Divide to NDC
        if clip_space_pos[3] == 0:
            return (0.0, 0.0, 0.0)  # Avoid division by zero

        ndc_pos = clip_space_pos / clip_space_pos[3]

        # 4. NDC to Screen Space
        screen_x = (ndc_pos[0] + 1.0) / 2.0
        screen_y = (1.0 - ndc_pos[1]) / 2.0
        depth = (ndc_pos[2] + 1.0) / 2.0  # Normalize depth to [0,1]

        return (screen_x, screen_y, depth)

    # ==========================================================
    # Methods for OpenGL / Rasterization (Legacy Support)
    # ==========================================================

    def to_perspective(self) -> dict:
        """
        Export camera parameters in GLM/OpenGL-friendly format.
        C++ can read this and call glm::perspective(fov, aspect, near, far).
        """
        return {
            "fov_radians": np.radians(self.fov),
            "fov_degrees": self.fov,
            "aspect": float(self.aspect_ratio.value),
            "near": float(self.near),
            "far": float(self.far),
        }

    def to_matrices(self) -> dict:
        """
        Export matrices in OpenGL-friendly format (column-major, transposed if needed).
        """
        return {
            "view": self.get_view_matrix().T.tolist(),  # Transpose for column-major
            "projection": self.get_projection_matrix().T.tolist(),
            "view_projection": (self.get_projection_matrix() @ self.get_view_matrix()).T.tolist(),
        }

    def get_camera_matrix (self) -> np.ndarray:
        # Projection * View (standard order)
        return self.get_projection_matrix() @ self.get_view_matrix()

    def get_frustum_planes(self) -> dict:
        """
        Compute the 6 frustum planes (left, right, top, bottom, near, far).
        Useful for C++ culling and intersection tests.
        Returns: {"left": (normal, d), "right": ..., etc.}
        """
        # Extract planes from combined matrix for frustum culling
        M = self.get_camera_matrix()
        planes = {}
        
        # Normalize and extract each plane
        planes["left"] = (M[3] + M[0], M[3, 3] + M[0, 3])
        planes["right"] = (M[3] - M[0], M[3, 3] - M[0, 3])
        planes["top"] = (M[3] - M[1], M[3, 3] - M[1, 3])
        planes["bottom"] = (M[3] + M[1], M[3, 3] + M[1, 3])
        planes["near"] = (M[3] + M[2], M[3, 3] + M[2, 3])
        planes["far"] = (M[3] - M[2], M[3, 3] - M[2, 3])
        
        return planes
    
    def export_uniforms(self) -> str:
        """
        Generate GLSL uniform declarations for copy-paste into vertex/fragment shaders.
        """
        matrices = self.to_matrices()
        return f"""
// Camera uniforms (auto-generated from VCamera)
uniform mat4 uView;           // View matrix
uniform mat4 uProjection;     // Projection matrix
uniform mat4 uViewProjection; // Combined VP matrix
uniform vec3 uCameraPos;      // Camera position (world-space)
uniform float uNear;          // Near plane
uniform float uFar;           // Far plane
uniform float uFOV;           // Field of view (radians)
"""

    def __repr__(self):
        return (f"VCamera(name={self.name}, type={self.type.name}, mode={self.mode.name}, "
                f"fov={self.fov}, aspect={self.aspect_ratio}, near={self.near}, far={self.far}, "
                f"res={self.resolution_width}x{self.resolution_height})")

"""
Camera module: Providing the Camera class with properties, projection types and movement modes.
"""