from src.Basic import *
from src.Camera import *
from src.Projections import *
from py_src.src.Gemometry import *

if __name__ == "__main__":
    circle = Circle(Vector(0, 0, 0), 5)
    
    ray = Ray(Vector(0, 0, 0), Vector(-1, 0, 0))
    
    intersections = circle.GetIntersection(ray)
    
    print(f"Original Circle: {circle}")
    print(f"Original Ray: {ray}")
    print(f"Ray intersects Circle: {intersections}")