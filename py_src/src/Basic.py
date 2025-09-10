from math import cos, sin, acos

class Vector:
    """
    Represents a mathematical vector in N-dimensional space.
    Attributes:
        data (list[float]): The components of the vector.
        component_size (int): The number of operations performed on the vector.
    Methods:
        __add__(other): Adds this vector to another vector or a scalar.
        __sub__(other): Subtracts another vector or a scalar from this vector.
        __mul__(scalar): Multiplies this vector by a scalar.
        __truediv__(scalar): Divides this vector by a scalar.
        Dot(other): Returns the dot product of this vector and another vector.
        Cross(other): Returns the cross product of this vector and another vector (3D only).
        Magnitude(): Returns the magnitude (length) of the vector.
        Normalize(): Returns the normalized (unit length) version of the vector.
        AngleBetween(other): Returns the angle (in radians) between this vector and another vector.
        Rotate(axis, angle): Returns the rotated version of this vector around a given axis by a specified angle (3D only).
        __getitem__(index): Gets the component at the specified index or a slice of components.
        __setitem__(index, value): Sets the component at the specified index or a slice of components.
        __len__(): Returns the number of components in the vector.
        __repr__(): Returns a string representation of the vector.
    """
    def __init__(self, id: int = 0, *args: list[float]):
        """
        A vector can be of any dimension, but most commonly used are 2D and 3D.
        Example usage:
        v2d = Vector(1, 2)          # 2D vector
        v3d = Vector(1, 2, 3)       # 3D vector
        v4d = Vector(1, 2, 3, 4)    # 4D vector
        vNd = Vector(1, 2, 3, ..., N) # N-dimensional vector
        2D and 3D vectors have special methods for cross product and rotation.
        """
        self.data: list[float] = args
        self.component_size: int = 0

        self.id = id

    def __add__(self, other):
        if not (isinstance(other, Vector) or isinstance(other, (int, float))):
            raise TypeError("Other must be a vector or in the form (x, y)")
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must be of the same dimension")

        self.component_size += 1

        if isinstance(other, Vector):
            result = []
            for i in range(len(self.data)):
                result.append(self.data[i] + other.data[i])
            return Vector(*result)
        if isinstance(other, (int, float)):
            result = []
            for i in self.data:
                result.append(i + other)
            return Vector(*result)

    def __sub__(self, other):
        if not (isinstance(other, Vector) or isinstance(other, (int, float))):
            raise TypeError("Other must be a vector or in the form (x, y)")
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must be of the same dimension")
        
        self.component_size += 1

        if isinstance(other, Vector):
            result = []
            for i in range(len(self.data)):
                result.append(self.data[i] - other.data[i])
            return Vector(*result)
        if isinstance(other, (int, float)):
            result = []
            for i in self.data:
                result.append(i - other)
            return Vector(*result)

    def __mul__(self, scalar: int|float):
        self.component_size += 1
        
        result = []

        for i in self.data:
            result.append(i * scalar)

        return Vector(*result)
    
    def __truediv__(self, scalar: int|float):
        self.component_size += 1
        if scalar == 0:
            raise ValueError("Division by zero is not allowed")
        
        result = []

        for i in self.data:
            result.append(i / scalar)

        return Vector(*result)
    
    def Dot(self, other):
        """
        Returns the dot product of this vector and another vector.
        """
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must be of the same dimension")
        
        result = 0
        for i in range(len(self.data)):
            result += float(self.data[i]) * float(other.data[i])
        return result
    
    def Cross(self, other):
        """
        Returns the cross product of this vector and another vector.
        3D vectors only.
        """
        if len(self.data) != 3 or len(other.data) != 3:
            raise ValueError("Cross product is only defined for 3D vectors")
        
        return Vector(
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0]
        )

    def Magnitude(self):
        """
        Returns the magnitude (length) of the vector.
        """
        return sum([x ** 2 for x in self.data]) ** 0.5
    
    def Normalize(self):
        """
        Returns the normalized vector.
        """
        magnitude = self.Magnitude()

        if magnitude == 0:
            raise ValueError("Cannot normalize a zero vector")
        
        normalized_vector = Vector(*self.data)
        return normalized_vector / magnitude
    
    def AngleBetween(self, other):
        """
        Returns the angle (in radians) between this vector and another vector.
        """
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must be of the same dimension")
        
        dot_product = self.Dot(other)
        magnitudes = self.Magnitude() * other.Magnitude()
        if magnitudes == 0:
            raise ValueError("Cannot calculate angle with a zero vector")
        cos_angle = dot_product / magnitudes
        return acos(max(-1, min(1, cos_angle)))

    def Rotate(self, axis, angle: float):
        """
        Returns the rotated vector. 3D vectors only.
        """
        if len(self.data) != 3 or len(axis.data) != 3:
            raise ValueError("Rotation is only defined for 3D vectors")
        
        # Normalize the axis
        n = axis.Normalize()
        
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        rotation_matrix = Matrix(
            [cos_angle + n[0] * n[0] * (1 - cos_angle), n[0] * n[1] * (1 - cos_angle) - n[2] * sin_angle, n[0] * n[2] * (1 - cos_angle) + n[1] * sin_angle],
            [n[1] * n[0] * (1 - cos_angle) + n[2] * sin_angle, cos_angle + n[1] * n[1] * (1 - cos_angle), n[1] * n[2] * (1 - cos_angle) - n[0] * sin_angle],
            [n[2] * n[0] * (1 - cos_angle) - n[1] * sin_angle, n[2] * n[1] * (1 - cos_angle) + n[0] * sin_angle, cos_angle + n[2] * n[2] * (1 - cos_angle)]
        )
        
        return rotation_matrix * self

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, slice):
            return Vector(*self.data[index])
        else:
            raise TypeError("Index must be an integer or a slice")
        
    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.data[index] = value
        elif isinstance(index, slice):
            if len(value) != len(self.data[index]):
                raise ValueError("Value length must match the slice length")
            self.data[index] = value
        else:
            raise TypeError("Index must be an integer or a slice")
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        result = ", ".join(str(x) for x in self.data)
        return f"Vector({result})"

class Matrix:
    """
    Represents a mathematical matrix.
    Attributes:
        data (list[list[float]]): The elements of the matrix.
        rows (int): The number of rows in the matrix.
        cols (int): The number of columns in the matrix.
    Methods:
        __add__(other): Adds this matrix to another matrix.
        __sub__(other): Subtracts another matrix from this matrix.
        __mul__(other): Multiplies this matrix by another matrix, a vector, or a scalar.
        __rmul__(scalar): Multiplies this matrix by a scalar (right-hand side).
        __truediv__(scalar): Divides this matrix by a scalar.
        __repr__(): Returns a string representation of the matrix.)
    """
    def __init__(self, name: str = "Matrix", id: int = 0, *args: list[list[float]]):
        """
        A matrix can be of any dimension, but most commonly used are 2x2, 3x3, and 4x4.
        Example usage:
        m2x2 = Matrix([1, 2], [3, 4])
        m3x3 = Matrix([1, 2, 3], [4, 5, 6], [7, 8, 9])
        m4x4 = Matrix([1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16])
        mNxM = Matrix([ ... ], [ ... ], ..., [ ... ]) # N rows, M columns
        2x2, 3x3, and 4x4 matrices have special methods for transformations.
        2D transformations use 3x3 matrices, and 3D transformations use 4x4 matrices.
        2D and 3D matrices can be used for rotation, translation, scaling, and shearing.
        2D and 3D matrices can also be used for projection, view, camera, object, texture, lighting and normal transformations.
        """
        self.data: list[list[float]] = args
        self.rows: int = len(self.data)
        self.cols: int = len(self.data[0]) if self.rows > 0 else 0

    def __add__(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have the same dimensions for addition")
            return Matrix(
                [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            )
        raise NotImplementedError("Matrix addition with this type is not implemented")
    
    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have the same dimensions for subtraction")
            return Matrix(
                [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            )
        raise NotImplementedError("Matrix subtraction with this type is not implemented")
    
    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Matrix multiplication requires compatible dimensions")
            return Matrix(
                [
                    [sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) for j in range(other.cols)]
                    for i in range(self.rows)
                ]
            )
        elif isinstance(other, (int, float)):
            return Matrix(self.data) * other
        elif isinstance(other, Vector):
            if self.cols != len(other):
                raise ValueError("Matrix and vector dimensions do not match")
            return Vector(
                sum(self.data[i][j] * other[j] for j in range(self.cols))
                for i in range(self.rows)
            )
        else:
            raise NotImplementedError("Matrix multiplication with this type is not implemented")

    def __rmul__(self, scalar: int|float):
        if isinstance(scalar, (int, float)):
            return Matrix(
                [self.data[i][j] * scalar for j in range(self.cols)]
                for i in range(self.rows)
            )
        raise NotImplementedError("Right multiplication with this type is not implemented")
    
    def __truediv__(self, scalar: int|float):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ValueError("Division by zero is not allowed")
            return Matrix(
                [self.data[i][j] / scalar for j in range(self.cols)]
                for i in range(self.rows)
            )
        raise NotImplementedError("Matrix division with this type is not implemented")

    def Transpose(self):
        """
        Returns the transpose of the matrix.
        """
        if self.rows == 0:
            return Matrix()
        
        if self.cols == 0:
            return Matrix([[] for _ in range(self.rows)])
        
        return Matrix(
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        )
    
    def Determinant(self):
        """
        Returns the determinant of the matrix.
        Only defined for square matrices.
        """
        if self.rows != self.cols:
            raise ValueError("Determinant is only defined for square matrices")
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        elif self.rows == 3:
            return (
                self.data[0][0] * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1]) -
                self.data[0][1] * (self.data[1][0] * self.data[2][2] - self.data[1][2] * self.data[2][0]) +
                self.data[0][2] * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0])
            )
        else:
            raise NotImplementedError("Determinant calculation for matrices larger than 3x3 is not implemented")

    def Inverse(self):
        """
        Returns the inverse of the matrix.
        Only defined for square matrices.
        """
        if self.rows != self.cols:
            raise ValueError("Inverse is only defined for square matrices")
        if self.rows == 2:
            det = self.Determinant()
            if det == 0:
                raise ValueError("Matrix is singular and cannot be inverted")
            return Matrix(
                [self.data[1][1] / det, -self.data[0][1] / det],
                [-self.data[1][0] / det, self.data[0][0] / det]
            )
        elif self.rows == 3:
            # Implementing a simple method for 3x3 matrices
            det = self.Determinant()
            if det == 0:
                raise ValueError("Matrix is singular and cannot be inverted")
            inv_det = 1 / det
            return Matrix(
                [
                    (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1]) * inv_det,
                    (self.data[0][2] * self.data[2][1] - self.data[0][1] * self.data[2][2]) * inv_det,
                    (self.data[0][1] * self.data[1][2] - self.data[0][2] * self.data[1][1]) * inv_det
                ],
                [
                    (self.data[1][2] * self.data[2][0] - self.data[1][0] * self.data[2][2]) * inv_det,
                    (self.data[0][0] * self.data[2][2] - self.data[0][2] * self.data[2][0]) * inv_det,
                    (self.data[0][2] * self.data[1][0] - self.data[0][0] * self.data[1][2]) * inv_det
                ],
                [
                    (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0]) * inv_det,
                    (self.data[0][1] * self.data[2][0] - self.data[0][0] * self.data[2][1]) * inv_det,
                    (self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]) * inv_det
                ])
        else:
            raise NotImplementedError("Inverse calculation for matrices larger than 3x3 is not implemented")

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, slice):
            return Matrix(*self.data[index])
        else:
            raise TypeError("Index must be an integer or a slice")
    
    def __setitem__(self, index, value):
        if isinstance(index, int):
            if len(value) != self.cols:
                raise ValueError("Value length must match the number of columns")
            self.data[index] = value
        elif isinstance(index, slice):
            if len(value) != len(self.data[index]):
                raise ValueError("Value length must match the slice length")
            self.data[index] = value
        else:
            raise TypeError("Index must be an integer or a slice")

    def __repr__(self):
        return f"Matrix({self.elements})"

class Ray:
    """
    Represents a ray in N-dimensional space, defined by an origin point and a direction vector.
    Attributes:
        origin (Vector): The starting point of the ray.
        direction (Vector): The direction of the ray (should be normalized).
    Methods:
        PointAtParameter(t): Returns the point along the ray at parameter t.
        CheckPointOnRay(point): Checks if a given point lies on the ray.
        CheckPointInFront(point): Checks if a given point is in front of the ray's origin along its direction.
        CheckPointBehind(point): Checks if a given point is behind the ray's origin opposite its direction.
        CheckIntersection(other): Checks if this ray intersects with another ray (not implemented).
        GetIntersection(other): Gets the intersection point with another ray (not implemented).
        __repr__(): Returns a string representation of the ray.
    """
    def __init__(self, origin: Vector, direction: Vector, name: str = "Ray", id: int = 0):
        """
        A ray defined by an origin point and a direction vector.
        """
        self.origin = origin

        if direction.Magnitude() == 0:
            raise ValueError("Direction vector cannot be zero-length")
        self.direction = direction.Normalize()
        
        self.name = name

    def PointAtParameter(self, t: float):
        """
        Returns the point along the ray at parameter t.
        """
        return self.origin + self.direction * t

    def CheckPointOnRay(self, point: Vector):
        """
        Checks if a given point lies on the ray.
        """
        if len(point) != len(self.origin):
            raise ValueError("Point and ray origin must be of the same dimension")
        
        to_point = point - self.origin
        if to_point.Magnitude() == 0:
            return True
    
        to_point_normalized = to_point.Normalize()
        return to_point_normalized == self.direction
    
    def CheckPointInFront(self, point: Vector):
        """
        Checks if a given point is in front of the ray's origin along its direction.
        """
        if len(point) != len(self.origin):
            raise ValueError("Point and ray origin must be of the same dimension")
        
        to_point = point - self.origin
        return self.direction.Dot(to_point) > 0
    
    def CheckPointBehind(self, point: Vector):
        """
        Checks if a given point is behind the ray's origin opposite its direction.
        """
        if len(point) != len(self.origin):
            raise ValueError("Point and ray origin must be of the same dimension")
        
        to_point = point - self.origin
        return self.direction.Dot(to_point) < 0

    def CheckIntersection(self, other):
        pass

    def GetIntersection(self, other):
        pass

    def __repr__(self):
        return f"Ray(origin={self.origin}, direction={self.direction})"

class Transform2D:
    """
    Represents a 2D transformation with position, rotation, scale, and hierarchical parent support.
    Attributes:
        position (Vector): The global/world position of the object.
        rotation (float): The global/world rotation of the object (in radians).
        scale (Vector): The global/world scale of the object.
        local_position (Vector): The local position relative to the parent.
        local_rotation (float): The local rotation relative to the parent (in radians).
        local_scale (Vector): The local scale relative to the parent.
        parent (Transform2D, optional): The parent transform, if any.
        up (Vector): The up direction vector of the object.
        right (Vector): The right direction vector of the object.
    Methods:
        update_directions():
            Updates the up and right vectors based on the current rotation.

        get_local_matrix():
            Returns the local transformation matrix (scale, rotate, translate) for this object.

        get_global_matrix():
            Returns the global/world transformation matrix, including parent transforms.
        get_global_position():
            Returns the global/world position as a Vector.
        get_global_rotation():
            Returns the global/world rotation (in radians).
        get_global_scale():
            Returns the global/world scale as a Vector.

        Translate(vector: Vector, isWorld: bool = False):
            Translates the object by the given vector in local or world space.
        Translate(origin: Vector, map: Matrix, isWorld: bool = False):
            Translates the object by a transformed vector in local or world space.
            
        Rotate(angle: float, isWorld: bool = False):
            Rotates the object by the given angle in local or world space.
        Rotate(origin: Vector, map: Matrix, isWorld: bool = False):
            Rotates the object by the angle of a transformed vector in local or world space.

        Enlarge(vector: Vector, isWorld: bool = False):
            Enlarges (scales) the object by the given vector in local or world space.
        Enlarge(origin: Vector, map: Matrix, isWorld: bool = False):
            Enlarges (scales) the object by a transformed vector in local or world space.

        Reflect(axis: Ray, isWorld: bool = False):
            Reflects the object across the given axis in local or world space.
        Reflect(axis: Ray, map: Matrix, isWorld: bool = False):
            Reflects the object across a transformed axis in local or world space.

        Sheering is not implemented yet.
    
    2D vectors only.
    """
    def __init__(self, position: Vector, rotation: float, scale: Vector, parent=None, name: str = "Transform2D", id: int = 0):
        """
        Initializes a 2D transform with position, rotation, and scale.
        - position: The position of the object in world (Vector2)
        - rotation: The rotation of the object in world (in radians)
        - scale: The scale of the object in world (Vector2)
        - parent: The parent transform (if any)
        2D vectors only.
        """
        self.position = position
        self.rotation = rotation  # in radians
        self.scale = scale
        self.local_position = Vector(0, 0)
        self.local_rotation = 0.0  # in radians
        self.local_scale = Vector(1, 1)
        self.parent = parent

        self.name = name
        self.id = id

        # Up and right directions (default for 2D: up +Y, right +X)
        self.update_directions()

    def update_directions(self):
        """
        Updates the up and right vectors based on the current rotation.
        """
        r = self.rotation
        cos_r = cos(r)
        sin_r = sin(r)
        # Right is rotated +X, Up is rotated +Y
        self.right = Vector(cos_r, sin_r)
        self.up = Vector(-sin_r, cos_r)

    def get_local_matrix(self):
        """
        Returns the local transformation matrix.
        """
        sx, sy = self.local_scale[0], self.local_scale[1]
        tx, ty = self.local_position[0], self.local_position[1]
        r = self.local_rotation
        # Rotation matrix
        rotation = Matrix(
            [cos(r), -sin(r), 0],
            [sin(r), cos(r), 0],
            [0, 0, 1]
        )
        # Scale matrix
        scale = Matrix(
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        )
        # Translation matrix
        translate = Matrix(
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        )
        # Combine: T * R * S (order: scale, rotate, translate)
        return translate * rotation * scale

    def get_global_matrix(self):
        """
        Returns the global transformation matrix.
        """
        local = self.get_local_matrix()
        if self.parent is not None:
            parent_global = self.parent.get_global_matrix()
            return parent_global * local
        else:
            return local
    
    def get_global_position(self):
        """
        Returns the global position of the transform.
        """
        global_matrix = self.get_global_matrix()
        return Vector(global_matrix[0][2], global_matrix[1][2])

    def get_global_rotation(self):
        """
        Returns the global rotation of the transform.
        """
        global_matrix = self.get_global_matrix()
        return acos(global_matrix[0][0])  # Assuming no scaling for simplicity

    def get_global_scale(self):
        """
        Returns the global scale of the transform.
        """
        global_matrix = self.get_global_matrix()
        return Vector(global_matrix[0][0], global_matrix[1][1])

    def Translate(self, vector: Vector, isWorld: bool = False):
        """
        Translates the transform in the specified space (world or local).
        """
        if isWorld:
            # Translate in world space
            self.position += vector
        else:
            # Translate in local space
            self.local_position += vector
        self.update_directions()
    
    def Translate(self, origin: Vector, map: Matrix, isWorld: bool = False):
        """
        Translates the transform in the specified space (world or local).
        """
        transformed_vector = map * origin

        if isWorld:
            # Translate in world space
            self.position += transformed_vector
        else:
            # Translate in local space
            self.local_position += transformed_vector
        self.update_directions()
    
    def Rotate(self, angle: float, isWorld: bool = False):
        """
        Rotates the transform in the specified space (world or local).
        """
        if isWorld:
            # Rotate in world space
            self.rotation += angle
        else:
            # Rotate in local space
            self.local_rotation += angle
        self.update_directions()
    
    def Rotate(self, origin: Vector, map: Matrix, isWorld: bool = False):
        """
        Rotates the transform in the specified space (world or local).
        """
        transformed_vector = map * origin
        angle = acos(transformed_vector.Normalize().Dot(Vector(1, 0)))  # Angle with respect to x-axis

        if isWorld:
            # Rotate in world space
            self.rotation += angle
        else:
            # Rotate in local space
            self.local_rotation += angle
        self.update_directions()
    
    def Enlarge(self, vector: Vector, isWorld: bool = False):
        """
        Enlarge the transform in the specified space (world or local).
        """
        if isWorld:
            # Enlarge in world space
            self.scale += vector
        else:
            # Enlarge in local space
            self.local_scale += vector
        self.update_directions()
    
    def Enlarge(self, origin: Vector, map: Matrix, isWorld: bool = False):
        """
        Enlarge the transform in the specified space (world or local).
        """
        transformed_vector = map * origin
        
        if isWorld:
            # Enlarge in world space
            self.scale += transformed_vector
        else:
            # Enlarge in local space
            self.local_scale += transformed_vector
        self.update_directions()
    
    def Reflect(self, axis: Ray, isWorld: bool = False):
        """
        Reflects the transform across a specified axis.
        """
        if axis.direction.Magnitude() == 0:
            raise ValueError("Cannot reflect across a zero-length vector")
        
        # Normalize the axis direction vector
        n = axis.direction.Normalize()
        # Create the reflection matrix based on the normal vector
        reflection_matrix = Matrix(
            [1 - 2 * n[0] * n[0], -2 * n[0] * n[1], 0],
            [-2 * n[1] * n[0], 1 - 2 * n[1] * n[1], 0],
            [0, 0, 1]
        )
        
        if isWorld:
            self.position = reflection_matrix * self.position
            self.rotation = acos(reflection_matrix[0][0])  # Assuming no scaling for simplicity
        else:
            self.local_position = reflection_matrix * self.local_position
            self.local_rotation = acos(reflection_matrix[0][0])  # Assuming no scaling for simplicity
        self.update_directions()
    
    def Reflect(self, axis: Ray, map: Matrix, isWorld: bool = False):
        """
        Reflects the transform across a transformed axis.
        """
        transformed_vector = map * axis.direction
        if transformed_vector.Magnitude() == 0:
            raise ValueError("Cannot reflect across a zero-length vector")
        
        # Normalize the axis direction vector
        n = transformed_vector.Normalize()
        # Create the reflection matrix based on the normal vector
        reflection_matrix = Matrix(
            [1 - 2 * n[0] * n[0], -2 * n[0] * n[1], 0],
            [-2 * n[1] * n[0], 1 - 2 * n[1] * n[1], 0],
            [0, 0, 1]
        )
        
        if isWorld:
            self.position = reflection_matrix * self.position
            self.rotation = acos(reflection_matrix[0][0])  # Assuming no scaling for simplicity
        else:
            self.local_position = reflection_matrix * self.local_position
            self.local_rotation = acos(reflection_matrix[0][0])  # Assuming no scaling for simplicity
        self.update_directions()
        
    def __repr__(self):
        return f"Transform2D(position={self.position}, rotation={self.rotation}, scale={self.scale})"

class Transform:
    """
    Represents a 3D transformation with position, rotation, scale, and hierarchical parent support.
    Attributes:
        position (Vector): The global/world position of the object.
        rotation (Vector): The global/world rotation of the object (Euler angles in radians).
        scale (Vector): The global/world scale of the object.
        local_position (Vector): The local position relative to the parent.
        local_rotation (Vector): The local rotation relative to the parent (Euler angles in radians).
        local_scale (Vector): The local scale relative to the parent.
        parent (Transform, optional): The parent transform, if any.
        forward (Vector): The forward direction vector of the object.
        right (Vector): The right direction vector of the object.
        up (Vector): The up direction vector of the object.
    Methods:
        update_directions():
            Updates the forward, right, and up vectors based on the current rotation.

        get_local_matrix():
            Returns the local transformation matrix (scale, rotate, translate) for this object.

        get_global_matrix():
            Returns the global/world transformation matrix, including parent transforms.
        get_global_position():
            Returns the global/world position as a Vector.
        get_global_rotation():
            Returns the global/world rotation (Euler angles in radians).
        get_global_scale():
            Returns the global/world scale as a Vector.

        Translate(vector: Vector, isWorld: bool = False):
            Translates the object by the given vector in local or world space.
        Translate(origin: Vector, map: Matrix, isWorld: bool = False):
            Translates the object by a transformed vector in local or world space.
        
        Rotate(direction: Vector, angle: float, isWorld: bool = False):
            Rotates the object around a given axis by a specified angle in local or world space.
        Rotate(direction: Vector, map: Matrix, isWorld: bool = False):
            Rotates the object around a transformed axis by a specified angle in local or world space.

        Enlarge(vector: Vector, isWorld: bool = False):
            Enlarges (scales) the object by the given vector in local or world space.
        Enlarge(origin: Vector, map: Matrix, isWorld: bool = False):
            Enlarges (scales) the object by a transformed vector in local or world space.

        Reflect(axis: Ray, isWorld: bool = False):
            Reflects the object across the given axis in local or world space.
        Reflect(axis: Ray, map: Matrix, isWorld: bool = False):
            Reflects the object across a transformed axis in local or world space.

        Sheering is not implemented yet.
    
    3D vectors only.
    """
    def __init__(self, position: Vector, rotation: Vector, scale: Vector, parent=None, name: str = "Transform", id: int = 0):
        """
        Initializes a transform with position, rotation, and scale.
        - position: The position of the object in world
        - rotation: The rotation of the object in world (Euler angles in radians)
        - scale: The scale of the object in world
        - parent: The parent transform (if any)
        3D vectors only.
        """
        self.position = position
        self.rotation = rotation # in radians
        self.scale = scale
        self.local_position = Vector(0, 0, 0)
        self.local_rotation = Vector(0, 0, 0) # in radians
        self.local_scale = Vector(1, 1, 1)
        self.parent = parent

        self.name = name
        self.id = id

        # Forward, right, and up directions (default for 3D: +Z, +X, +Y)
        self.update_directions()

    def update_directions(self):
        """
        Updates the forward, right, and up vectors based on the current rotation.
        Assumes rotation is in Euler angles (rx, ry, rz).
        """
        rx, ry, rz = self.rotation[0], self.rotation[1], self.rotation[2]
        # Rotation matrices
        cx, sx = cos(rx), sin(rx)
        cy, sy = cos(ry), sin(ry)
        cz, sz = cos(rz), sin(rz)

        # Combined rotation matrix (Z * Y * X)
        m00 = cy * cz
        m01 = cz * sx * sy - cx * sz
        m02 = cx * cz * sy + sx * sz
        m10 = cy * sz
        m11 = cx * cz + sx * sy * sz
        m12 = -cz * sx + cx * sy * sz
        m20 = -sy
        m21 = cy * sx
        m22 = cx * cy

        # Forward (+Z), Right (+X), Up (+Y)
        self.forward = Vector(m02, m12, m22)
        self.right = Vector(m00, m10, m20)
        self.up = Vector(m01, m11, m21)

    def get_local_matrix(self):
        """
        Returns the local transformation matrix.
        """
        sx, sy, sz = self.local_scale[0], self.local_scale[1], self.local_scale[2]
        tx, ty, tz = self.local_position[0], self.local_position[1], self.local_position[2]
        rx, ry, rz = self.local_rotation[0], self.local_rotation[1], self.local_rotation[2]
        # Rotation around X axis
        rot_x = Matrix(
            [1, 0, 0, 0],
            [0, cos(rx), -sin(rx), 0],
            [0, sin(rx), cos(rx), 0],
            [0, 0, 0, 1]
        )
        # Rotation around Y axis
        rot_y = Matrix(
            [cos(ry), 0, sin(ry), 0],
            [0, 1, 0, 0],
            [-sin(ry), 0, cos(ry), 0],
            [0, 0, 0, 1]
        )
        # Rotation around Z axis
        rot_z = Matrix(
            [cos(rz), -sin(rz), 0, 0],
            [sin(rz), cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        )

        # Scale matrix
        scale = Matrix(
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        )

        # Translation matrix
        translate = Matrix(
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        )

        # Combine: T * Rz * Ry * Rx * S (order: scale, rotate x, rotate y, rotate z, translate)
        # You may adjust the order as needed for your convention
        return translate * rot_z * rot_y * rot_x * scale

    def get_global_matrix(self):
        """
        Returns the global transformation matrix.
        """
        local = self.get_local_matrix()
        if self.parent is not None:
            parent_global = self.parent.get_global_matrix()
            return parent_global * local
        else:
            return local

    def get_global_position(self):
        """
        Returns the global position of the transform.
        """
        global_matrix = self.get_global_matrix()
        return Vector(global_matrix[0][3], global_matrix[1][3], global_matrix[2][3])
    
    def get_global_rotation(self):
        """
        Returns the global rotation of the transform.
        """
        global_matrix = self.get_global_matrix()
        return Vector(global_matrix[0][0], global_matrix[1][1], global_matrix[2][2])
    
    def get_global_scale(self):
        """
        Returns the global scale of the transform.
        """
        global_matrix = self.get_global_matrix()
        return Vector(global_matrix[0][0], global_matrix[1][1], global_matrix[2][2])

    def Translate(self, vector: Vector, isWorld: bool = False):
        """
        Translates the transform in the specified space (world or local).
        """
        if isWorld:
            # Translate in world space
            self.position += vector
        else:
            # Translate in local space
            self.local_position += vector
        self.update_directions()
    
    def Translate(self, origin: Vector, map: Matrix, isWorld: bool = False):
        """
        Translates the transform in the specified space (world or local).
        """
        transformed_vector = map * origin

        if isWorld:
            # Translate in world space
            self.position += transformed_vector
        else:
            # Translate in local space
            self.local_position += transformed_vector
        self.update_directions()

    def Rotate(self, direction: Vector, angle: float, isWorld: bool = False):
        """
        Rotates the transform around a specified axis.
        """
        if direction.Magnitude() == 0:
            raise ValueError("Cannot rotate around a zero-length vector")
        
        # Normalize the direction vector
        n = direction.Normalize()
        
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        rotation_matrix = Matrix(
            [cos_angle + n[0] * n[0] * (1 - cos_angle), n[0] * n[1] * (1 - cos_angle) - n[2] * sin_angle, n[0] * n[2] * (1 - cos_angle) + n[1] * sin_angle],
             [n[1] * n[0] * (1 - cos_angle) + n[2] * sin_angle, cos_angle + n[1] * n[1] * (1 - cos_angle), n[1] * n[2] * (1 - cos_angle) - n[0] * sin_angle],
             [n[2] * n[0] * (1 - cos_angle) - n[1] * sin_angle, n[2] * n[1] * (1 - cos_angle) + n[0] * sin_angle, cos_angle + n[2] * n[2] * (1 - cos_angle)]
        )
        
        if isWorld:
            # Rotate in world space
            self.rotation = rotation_matrix * self.rotation
        else:
            # Rotate in local space
            self.local_rotation = rotation_matrix * self.local_rotation
        self.update_directions()
    
    def Rotate(self, origin: Vector, map: Matrix, isWorld: bool = False):
        """
        Rotate the transform in the specified space (world or local).
        """
        transformed_vector = map * origin

        if isWorld:
            # Rotalte in world space
            self.rotation += transformed_vector
        else:
            # Rotatle in local space
            self.local_rotation += transformed_vector
        self.update_directions()

    
    def Enlarge(self, vector: Vector, isWorld: bool = False):
        """
        Enlarge the transform in the specified space (world or local).
        """
        if isWorld:
            # Enlarge in world space
            self.scale += vector
        else:
            # Enlarge in local space
            self.local_scale += vector
    
    def Enlarge(self, origin: Vector, map: Matrix, isWorld: bool = False):
        """
        Enlarge the transform in the specified space (world or local).
        """
        transformed_vector = map * origin
        
        if isWorld:
            # Enlarge in world space
            self.scale += transformed_vector
        else:
            # Enlarge in local space
            self.local_scale += transformed_vector
    
    def Reflect(self, axis: Ray, isWorld: bool = False):
        """
        Reflects the transform across a specified axis.
        """
        if axis.direction.Magnitude() == 0:
            raise ValueError("Cannot reflect across a zero-length vector")
        
        # Normalize the axis direction vector
        n = axis.direction.Normalize()
        # Create the reflection matrix based on the normal vector
        reflection_matrix = Matrix(
            [1 - 2 * n[0] * n[0], -2 * n[0] * n[1], -2 * n[0] * n[2]],
            [-2 * n[1] * n[0], 1 - 2 * n[1] * n[1], -2 * n[1] * n[2]],
            [-2 * n[2] * n[0], -2 * n[2] * n[1], 1 - 2 * n[2] * n[2]]
        )
        
        if isWorld:
            self.position = reflection_matrix * self.position
            self.rotation = reflection_matrix * self.rotation
        else:
            self.local_position = reflection_matrix * self.local_position
            self.local_rotation = reflection_matrix * self.local_rotation
        self.update_directions()
    
    def Reflect(self, origin: Vector, map: Matrix, isWorld: bool = False):
        """
        Reflects the transform across a specified plane.
        """
        transformed_vector = map * origin
        if transformed_vector.Magnitude() == 0:
            raise ValueError("Cannot reflect across a zero-length vector")
        
        # Normalize the transformed vector
        n = transformed_vector.Normalize()

        # Calculate the reflection matrix
        reflection_matrix = Matrix(
            [1 - 2 * n[0] * n[0], -2 * n[0] * n[1], -2 * n[0] * n[2]],
            [-2 * n[1] * n[0], 1 - 2 * n[1] * n[1], -2 * n[1] * n[2]],
            [-2 * n[2] * n[0], -2 * n[2] * n[1], 1 - 2 * n[2] * n[2]]
        )
        
        if isWorld:
            self.position = reflection_matrix * self.position
            self.rotation = reflection_matrix * self.rotation
        else:
            self.local_position = reflection_matrix * self.local_position
            self.local_rotation = reflection_matrix * self.local_rotation
        self.update_directions()
    
    def __repr__(self):
        return f"Transform(position={self.position}, rotation={self.rotation}, scale={self.scale})"

class Ratio:
    """
    Represents a ratio (fraction) with a denominator and numerator.
    Supports basic arithmetic operations and comparisons.
    """
    def __init__(self, denominator: float, numerator: float, id: int = 0):
        """
        Creates a ratio (fraction) with a denominator and numerator.
        """
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        self.denominator = denominator
        self.numerator = numerator

        self.id = id
    
    def __add__(self, other):
        if not isinstance(other, Ratio):
            raise TypeError("Can only add Ratio to another Ratio")
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Ratio(new_denominator, new_numerator)
    
    def __sub__(self, other):
        if not isinstance(other, Ratio):
            raise TypeError("Can only subtract Ratio from another Ratio")
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Ratio(new_denominator, new_numerator)

    def __mul__(self, other):
        if not isinstance(other, Ratio):
            raise TypeError("Can only multiply Ratio by another Ratio")
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Ratio(new_denominator, new_numerator)
    
    def __truediv__(self, other):
        if not isinstance(other, Ratio):
            raise TypeError("Can only divide Ratio by another Ratio")
        if other.numerator == 0:
            raise ValueError("Cannot divide by a Ratio with a numerator of zero")
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return Ratio(new_denominator, new_numerator)
    
    def __repr__(self):
        return f"Ratio({self.numerator}/{self.denominator})"
    
    def __float__(self):
        return self.numerator / self.denominator
    
    def __neg__(self):
        return Ratio(-self.denominator, -self.numerator)
    
    def __eq__(self, other):
        if not isinstance(other, Ratio):
            return False
        return self.numerator * other.denominator == self.denominator * other.numerator
    
    def __lt__(self, other):
        if not isinstance(other, Ratio):
            return False
        return self.numerator * other.denominator < self.denominator * other.numerator
    def __le__(self, other):
        if not isinstance(other, Ratio):
            return False
        return self.numerator * other.denominator <= self.denominator * other.numerator
    def __gt__(self, other):
        if not isinstance(other, Ratio):
            return False
        return self.numerator * other.denominator > self.denominator * other.numerator
    def __ge__(self, other):
        if not isinstance(other, Ratio):
            return False
        return self.numerator * other.denominator >= self.denominator * other.numerator
    def __ne__(self, other):
        if not isinstance(other, Ratio):
            return True
        return self.numerator * other.denominator != self.denominator * other.numerator
    