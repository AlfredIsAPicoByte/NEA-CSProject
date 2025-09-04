from src.Basic import *
from src.Camera import *
from src.Projections import *
from src.Shapes import *

if __name__ == "__main__":
    # Example usage
    camera = Camera(Vector(0, 0, 10), Vector(0, 0, -1), Vector(0, 1, 0), 60, 1.33)
    ortho_proj = OrthographicProjection(camera)
    
    circle = Circle(Vector(0, 0, 0), 5)
    projected_circle = ortho_proj.DrawShape(circle)
    
    ray = Ray(Vector(0, 0, 10), Vector(0, 0, -1))
    intersection = ortho_proj.ProjectIntersections(ray, circle)
    
    print(f"Original Circle: {circle}")
    print(f"Projected Circle: {projected_circle}")
    print(f"Ray intersects Circle: {intersection}")