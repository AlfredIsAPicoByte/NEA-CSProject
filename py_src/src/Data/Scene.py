import numpy as np
from typing import List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field

from .Transform import Transform
from .Camera import Camera
from src.Geometry.AABB import AABB

@dataclass
class SceneNode:
    """
    A node in the scene graph. 
    Combines a position in 3D space and possibley some data.
    Can act as a parent to other SceneNodes.
    The data object contains the contents of the node (e.g., mesh, light, SDF, etc). This prevents inheritance explosion hell lol.
    """
    name: str = "Object"
    context: Optional[Any] = None  # Generic data field; can be shapes, meshs or lights, etc.
    transform: Transform = field(default_factory=Transform.Identity)

    active: bool = True  # Whether this node is active in the scene

    # Hierarchy
    children: List['SceneNode'] = field(default_factory=list)
    parent: Optional['SceneNode'] = field(default=None, repr=False)
    
    # Caching / Optimization
    _world_matrix: Optional[np.ndarray] = None # 4x4 World Transformation Matrix
    _inverse_world_matrix: Optional[np.ndarray] = None # Inverse of the world matrix
    _cache_objects: Optional[List['SceneNode']] = None # A list of this and all descendants in a flat list
    _aabb_bounds: Optional[AABB] = None # Axis-Aligned Bounding Box for this SceneNode

    def __post_init__(self):
        # Ensure children know their parent
        for child in self.children:
            child.parent = self
        
    def add_child(self, child: 'SceneNode'):
        """Attaches a child node to this node."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def remove_child(self, child: 'SceneNode'):
        """Detaches a child node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def update_matrices(self, parent_matrix: Optional[np.ndarray] = None):
        """
        Recursive pass to update the world matrix of this node and all children.
        Call this ONCE before rendering starts.
        """
        # 1. Get Local Matrix
        # FIX: Use self.transform (Local), NOT self.world_transform (Computed Global)
        local_mat = self.transform.to_matrix() 

        # 2. Multiply by Parent (if exists)
        if parent_matrix is not None:
            self._world_matrix = parent_matrix @ local_mat
        else:
            self._world_matrix = local_mat
            
        # 3. Calculate Inverse (Needed for Ray Intersection: World -> Local)
        try:
            self._inverse_world_matrix = np.linalg.inv(self._world_matrix)
        except np.linalg.LinAlgError:
            self._inverse_world_matrix = np.eye(4)

        # 4. Propagate down the tree
        for child in self.children:
            child.update_matrices(self._world_matrix)

    def get_world_matrix(self):
        return self._world_matrix
    
    def get_world_inverse_matrix(self):
        return self._inverse_world_matrix

    @property
    def world_transform(self) -> Transform:
        """Returns a `Transform` representing the object's world transform (position/rotation/scale).
        Useful for APIs that expect a `Transform` object rather than raw matrices."""
        # Ensure matrices are up-to-date
        mat = self.get_world_matrix()
        return Transform.from_matrix(mat)

    def flatten_children(self, include_self: bool = True):
        """
        Returns a flat list of this object and all descendants.
        Useful for building the global list of objects for the BVH or Renderer.
        """
        result = []
        stack = [self] if include_self else list(reversed(self.children))
        
        while stack:
            current = stack.pop()
            result.append(current)
            
            # Add children in reversed order so they're popped in correct order
            for child in reversed(current.children):
                stack.append(child)

        self._cache_objects = result

    def get_scene_objects_flattened(self, include_self: bool = True):
        if self._cache_objects is None:
            self.flatten_children(include_self)
        
        return self._cache_objects
    
    def get_bounds(self) -> AABB:
        """
        Delegates the bounds calculation to the data object if it exists.
        """
        # Check if context exists and has the method we need
        if self.context is None:
            return AABB.empty()
        
        if hasattr(self.context, "bounding_box"):
            return self.context.bounding_box
        
        return AABB.unit_cube()
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"SceneNode(name={self.name})"

class Scene:
    """
    A container for Scene Node objects used in rendering algorithms.
    """
    def __init__(self, name: str = "Scene", camera: Optional[Camera] = None, **kwargs):
        self.name = name
        self.camera: Camera = camera or Camera()

        self.objects: List[SceneNode] = []
        
        self._version: int = 1
        self._cache_objects: Optional[List[SceneNode]] = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def version(self) -> int:
        """
        A counter for the amount of changes that have occured on the scene since initilization.
        Starts at `1` when initialized.
        
        :return: Description
        :rtype: int
        """
        return self._version

    def add_object(self, obj: SceneNode):
        """
        Adds a new object to the scene and updates the version counter.
        """
        obj.update_matrices()
        self.objects.append(obj)
        self.update_version()

    def add_object_by_context(self, context: Any, name: str = "Object", transform: Transform = Transform.Identity()) -> SceneNode:
        """
        Creates a new SceneNode with the given context and adds it to the scene.
        
        :param context: The context data to attach to the new SceneNode.
        :param name: The name of the new SceneNode.
        :return: The newly created SceneNode.
        """
        new_node = SceneNode(name=name, context=context, transform=transform)
        self.add_object(new_node)
        return new_node
    
    @staticmethod
    def get_object(objects: list[SceneNode], name: str) -> Optional[SceneNode]:
        for obj in objects:
            if obj.name == name:
                return obj
        return None

    @staticmethod
    def get_object_by_id(objects: list[SceneNode], id: int) -> Optional[SceneNode]:
        for obj in objects:
            if hash(obj) == id:
                return obj
        
        return None

    @staticmethod
    def _matches_context(context: Any, type_or_name: Union[type, str]) -> bool:
        """Helper to check if a context matches a type (class) or name (str)."""
        if isinstance(type_or_name, type):
            return isinstance(context, type_or_name)
        return type(context).__name__ == type_or_name

    @staticmethod
    def get_objects_by_type(objects: list[SceneNode], context_type: Union[type, str]) -> List[SceneNode]:
        """
        Returns a list of all scene objects that contain data of the specified context type.
        """
        result = []
        for obj in objects:
            if obj.context is not None:
                if Scene._matches_context(obj.context, context_type):
                    result.append(obj)
        return result

    @staticmethod
    def get_objects_by_types(objects: list[SceneNode], context_types: List[Union[type, str]]) -> List[SceneNode]:
        """
        Returns a list of all scene objects that contain data of ANY of the specified context types.
        """
        result = []
        
        # Optimization: Separate real classes from string names once
        real_types = tuple(t for t in context_types if isinstance(t, type))
        type_names = {t for t in context_types if isinstance(t, str)}

        for obj in objects:
            if obj.context is None:
                continue
                
            # Check 1: Is it an instance of the real classes? (Fast)
            if real_types and isinstance(obj.context, real_types):
                result.append(obj)
                continue
            
            # Check 2: Does the class name match one of the strings? (Slower fallback)
            if type_names and type(obj.context).__name__ in type_names:
                result.append(obj)
                
        return result

    @staticmethod
    def get_objects_not_of_type(objects: list[SceneNode], context_type: Union[type, str]) -> List[SceneNode]:
        """
        Returns a list of all scene objects that do NOT contain data of the specified context type.
        """
        result = []
        for obj in objects:
            # If context is None, it definitely doesn't match the type, so we include it
            if obj.context is None:
                result.append(obj)
                continue

            if not Scene._matches_context(obj.context, context_type):
                result.append(obj)
        return result

    @staticmethod
    def get_objects_not_of_types(objects: list[SceneNode], context_types: List[Union[type, str]]) -> List[SceneNode]:
        """
        Returns a list of all scene objects that do NOT contain data of ANY of the specified context types.
        """
        result = []
        
        real_types = tuple(t for t in context_types if isinstance(t, type))
        type_names = {t for t in context_types if isinstance(t, str)}

        for obj in objects:
            if obj.context is None:
                result.append(obj)
                continue

            # Check 1: If it matches a real type, EXCLUDE it
            if real_types and isinstance(obj.context, real_types):
                continue
            
            # Check 2: If it matches a string name, EXCLUDE it
            if type_names and type(obj.context).__name__ in type_names:
                continue

            # If we reached here, it didn't match anything
            result.append(obj)
            
        return result
    
    @staticmethod
    def get_objects_subclass_of(objects: list[SceneNode], parent_class: type) -> List[SceneNode]:
        """
        Returns all objects whose context is a subclass of the given parent_class.
        Useful for polymorphism (e.g., get all Light subclasses).
        """
        result = []
        for obj in objects:
            if obj.context is not None and isinstance(obj.context, parent_class):
                result.append(obj)
        return result

    @staticmethod
    def get_objects_with_attribute(objects: list[SceneNode], attribute_name: str) -> List[SceneNode]:
        """
        Returns all objects whose context has a specific attribute (variable/property).
        Example: get_objects_with_attribute("intensity")
        """
        result = []
        for obj in objects:
            if obj.context is not None and hasattr(obj.context, attribute_name):
                result.append(obj)
        return result

    @staticmethod
    def get_objects_by_attribute_value(objects: list[SceneNode], attribute_name: str, value: Any) -> List[SceneNode]:
        """
        Returns all objects whose context has a specific attribute equal to a specific value.
        Example: get_objects_by_attribute_value("is_visible", True)
        """
        result = []
        for obj in objects:
            if obj.context is not None:
                # Check if attribute exists
                if hasattr(obj.context, attribute_name):
                    # Check if value matches
                    attr_val = getattr(obj.context, attribute_name)
                    if attr_val == value:
                        result.append(obj)
        return result

    @staticmethod
    def get_objects_by_condition(objects: list[SceneNode], condition_func: Callable[[SceneNode], bool]) -> List[SceneNode]:
        """
        Returns all objects that satisfy a custom lambda condition.
        
        Example: 
            # Get all lights brighter than 500
            scene.get_objects_by_condition(
                lambda node: isinstance(node.context, Light) and node.context.intensity > 500
            )
        """
        result = []
        for obj in objects:
            if condition_func(obj):
                result.append(obj)
        return result
    
    def flatten_scene_objects(self):
        """
        Retruns a list of all objects in a flat list array and updated the cache for faster access
        """
        flat_list = []
        
        # We store the ID (memory address) of visited objects
        visited_ids = set() 
        
        # Initialize stack with the top-level objects
        stack = list(self.objects)

        while stack:
            current_obj = stack.pop()

            # 1. Cycle Detection: Have we seen this specific object instance before?
            if id(current_obj) in visited_ids:
                continue
                
            # 2. Mark as visited
            visited_ids.add(id(current_obj))
            
            # 3. Add to our result list
            flat_list.append(current_obj)

            # 4. Add children to the stack to be processed next
            # (Check if the object actually has children first)
            if hasattr(current_obj, 'children') and current_obj.children:
                stack.extend(current_obj.children)
        
        # remove duplicate objects (with the same id)
        self._cache_objects = list({id(obj): obj for obj in flat_list}.values())
    
    def get_scene_objects_flattened(self) -> List[SceneNode]:
        """
        Returns a flat list of all objects in the scene, including children.
        Caches the result for faster access on subsequent calls.
        
        :return: Description
        :rtype: List[SceneNode]
        """
        if self._cache_objects is None:
            self.flatten_scene_objects()

        return self._cache_objects or self.objects
    
    def remove_object(self, object_node: SceneNode) -> bool:
        """
        Removes a specific object node from the scene. 
        Returns True if successful, False if the object was not found.
        """
        # 1. Check root level
        if object_node in self.objects:
            self.objects.remove(object_node)
            return True

        # 2. Check children (recursive search)
        # Note: This is expensive for deep trees. 
        # Better to store parent references in SceneNode if frequent removal is needed.
        for parent in self.get_scene_objects_flattened():
            if object_node in parent.children:
                parent.children.remove(object_node)
                return True
        
        return False

    def remove_object_by_name(self, name: str) -> bool:
        """Removes the first object found with the given name."""
        obj = self.get_object(self.get_scene_objects_flattened(), name)
        if obj:
            return self.remove_object(obj)
        return False

    def reparent(self, object_node: SceneNode, new_parent: Optional[SceneNode]):
        """
        Moves an object from its current parent (or root) to a new parent.
        If new_parent is None, moves the object to the Scene root.
        """
        # 1. Remove from old location
        removed = self.remove_object(object_node)
        if not removed:
            print(f"Warning: Object '{object_node.name}' not found in scene; cannot reparent.")
            return

        # 2. Add to new location
        if new_parent is None:
            self.objects.append(object_node)
        else:
            new_parent.add_child(object_node)


    def print_hierarchy(self):
        """Prints a visual tree structure of the scene to the console."""
        print(f"Scene: {self.name}")
        for obj in self.objects:
            self._print_node_recursive(obj, prefix="", is_last=True)

    def _print_node_recursive(self, node: SceneNode, prefix: str, is_last: bool):
        # Visual connectors
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{node.name} ({type(node.context).__name__ if node.context else 'Group'})")
        
        # Prepare prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        count = len(node.children)
        for i, child in enumerate(node.children):
            is_last_child = (i == count - 1)
            self._print_node_recursive(child, child_prefix, is_last_child)

    def get_stats(self) -> dict:
        """Returns a dictionary summary of the scene content."""
        all_objs = self.get_scene_objects_flattened()
        
        # Count types
        type_counts = {}
        for obj in all_objs:
            t_name = type(obj.context).__name__ if obj.context else "EmptyNode"
            type_counts[t_name] = type_counts.get(t_name, 0) + 1

        return {
            "total_objects": len(all_objs),
            "root_objects": len(self.objects),
            "type_breakdown": type_counts
        }
    
    def set_camera(self, camera: Camera):
        """
        Change the current camera that is used.
        Doesn't update the version counter.
        """
        self.camera = camera

    def clear(self):
        """
        Removes all the objects and light sources from the scene, while updating the version counter.
        """
        self.objects.clear()
        self.update_version()
    
    def update_version(self):
        """
        Signal a change in the scene.
        """
        self._version += 1
    

def find_scene_extremes(
    nodes: List['SceneNode'], 
    target_point: np.ndarray,
    ignore_empty: bool = True
) -> Tuple[Optional[SceneNode], Optional[SceneNode]]:
    """
    Finds the (Closest Node, Furthest Node) relative to a target point.
    
    :param nodes: A flat list of SceneNodes (use scene.get_objects_flat())
    :param target_point: A numpy array [x, y, z]
    :param ignore_empty: If True, skips nodes that have no 'data' (containers/folders)
    :return: Tuple (closest_node, furthest_node)
    """
    closest_node = None
    furthest_node = None
    
    # Initialize distances to infinity and negative infinity
    min_dist_sq = float('inf')
    max_dist_sq = float('-inf')

    for node in nodes:
        # 1. Skip logic
        if ignore_empty and node.context is None:
            continue
            
        # 2. Get Position
        node_pos = node.get_world_matrix()[:3, 3]
        
        # 3. Calculate Squared Euclidean Distance
        # (Square root is expensive, so we compare squared values for speed)
        diff = node_pos - target_point
        dist_sq = np.dot(diff, diff) 
        
        # 4. Check Closest
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_node = node
            
        # 5. Check Furthest
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            furthest_node = node
            
    return closest_node, furthest_node
