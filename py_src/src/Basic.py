from math import cos, sin

class Vector:
    def __init__(self, *args: list[float]):
        self.data: list[float] = args
        self.component_size: int = 0

    def __add__(self, other):
        if not (isinstance(other, Vector) or isinstance(other, (int, float))):
            raise TypeError("Other must be a vector or in the form (x, y)")
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must be of the same dimension")

        self.component_size += 1

        if isinstance(other, Vector):
            return Vector(self.data[i] + other.data[i] for i in range(len(self.data)))
        if isinstance(other, (int, float)):
            return Vector(self.data[i] + other for i in range(len(self.data)))

    def __sub__(self, other):
        self.component_size += 1

        if isinstance(other, Vector):
            return Vector(self.data[i] - other.data[i] for i in range(len(self.data)))
        if isinstance(other, (int, float)):
            return Vector(self.data[i] - other for i in range(len(self.data)))
        elif isinstance(other, Vector):
            if len(self.data) != len(other.data):
                raise ValueError("Vectors must be of the same dimension")

    def __mul__(self, scalar: int|float):
        self.component_size += 1

        return Vector(self.data[i] * scalar for i in range(len(self.data)))
    
    def __truediv__(self, scalar: int|float):
        self.component_size += 1

        if scalar == 0:
            raise ValueError("Division by zero is not allowed")
        return Vector(self.data[i] / scalar for i in range(len(self.data)))
    
    def Dot(self, other):
        result = 0
        for i in range(len(self.data)):
            result += self.data[i] * other.data[i]
        return result
    
    def Cross(self, other):
        if len(self.data) != 3 or len(other.data) != 3:
            raise ValueError("Cross product is only defined for 3D vectors")
        return Vector(
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0]
        )

    def Magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
    
    def Normalize(self):
        magnitude = self.Magnitude()
        if magnitude == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector(x / magnitude for x in self.data)
    
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
    def __init__(self, *args: list[list[float]]):
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
        return Matrix(
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        )
    
    def Determinant(self):
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
    def __init__(self, origin: Vector, direction: Vector):
        self.origin = origin

        if direction.Magnitude() == 0:
            raise ValueError("Direction vector cannot be zero-length")
        self.direction = direction.Normalize()

    def PointAtParameter(self, t: float):
        return self.origin + self.direction * t

    def __repr__(self):
        return f"Ray(origin={self.origin}, direction={self.direction})"

class Transform:
    def __init__(self, position: Vector, rotation: Vector, scale: Vector, parent=None):
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.local_position = Vector(0, 0, 0)
        self.local_rotation = Vector(0, 0, 0)
        self.local_scale = Vector(1, 1, 1)
        self.parent = parent

    def get_local_matrix(self):
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
        local = self.get_local_matrix()
        if self.parent is not None:
            parent_global = self.parent.get_global_matrix()
            return parent_global * local
        else:
            return local

    def get_global_position(self):
        global_matrix = self.get_global_matrix()
        return Vector(global_matrix[0][3], global_matrix[1][3], global_matrix[2][3])
    
    def get_global_rotation(self):
        global_matrix = self.get_global_matrix()
        return Vector(global_matrix[0][0], global_matrix[1][1], global_matrix[2][2])
    
    def get_global_scale(self):
        global_matrix = self.get_global_matrix()
        return Vector(global_matrix[0][0], global_matrix[1][1], global_matrix[2][2])

    def Translate(self, vector: Vector, isWorld: bool = False):
        if isWorld:
            # Translate in world space
            self.position += vector
        else:
            # Translate in local space
            self.local_position += vector
    
    def Translate(self, origin: Vector, map: Matrix, isWorld: bool = False):
        transformed_vector = map * origin

        if isWorld:
            # Translate in world space
            self.position += transformed_vector
        else:
            # Translate in local space
            self.local_position += transformed_vector

    def Rotate(self, direction: Vector, angle: float, isWorld: bool = False):
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
    
    def Rotate(self, origin: Vector, map: Matrix, isWorld: bool = False):
        transformed_vector = map * origin

        if isWorld:
            # Rotalte in world space
            self.rotation += transformed_vector
        else:
            # Rotatle in local space
            self.local_rotation += transformed_vector

    
    def Enlarge(self, vector: Vector, isWorld: bool = False):
        if isWorld:
            # Enlarge in world space
            self.scale += vector
        else:
            # Enlarge in local space
            self.local_scale += vector
    
    def Enlarge(self, origin: Vector, map: Matrix, isWorld: bool = False):
        transformed_vector = map * origin
        
        if isWorld:
            # Enlarge in world space
            self.scale += transformed_vector
        else:
            # Enlarge in local space
            self.local_scale += transformed_vector
    
    def Reflect(self, axis: Ray, isWorld: bool = False):
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
    
    def Reflect(self, origin: Vector, map: Matrix, isWorld: bool = False):
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
    
    def Sheer(self, origin: Vector, map: Matrix, isWorld: bool = False):
        transformed_vector = map * origin
        if transformed_vector.Magnitude() == 0:
            raise ValueError("Cannot sheer across a zero-length vector")
        
        # Normalize the transformed vector
        n = transformed_vector.Normalize()

        # Create the sheer matrix
        sheer_matrix = Matrix(
            [1, n[0], n[1]],
            [n[0], 1, n[2]],
            [n[1], n[2], 1]
        )
        
        # Apply the sheer to the position and rotation
        if isWorld:
            # Sheer in world space
            self.position = sheer_matrix * self.position
            self.rotation = sheer_matrix * self.rotation
        else:
            # Sheer in local space
            self.local_position = sheer_matrix * self.local_position
            self.local_rotation = sheer_matrix * self.local_rotation

class Ratio:
    def __init__(self, denominator: float, numerator: float):
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        self.denominator = denominator
        self.numerator = numerator
    
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
    