from src.Basic import Transform, Matrix, Ratio


class Camera:
    def __init__ (self, transform: Transform, fov: float, near: float, far: float, aspect: Ratio):
        self.transform = transform
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect = aspect
    
    def get_view_matrix (self):
        pass

    def get_projection_matrix (self):
        pass