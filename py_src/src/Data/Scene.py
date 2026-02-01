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
    
    def get_local_matrix(self):
        return self._inverse_world_matrix

    @property
    def world_transform(self) -> Transform:
        """
        Returns a `Transform` representing the object's world transform (position/rotation/scale).
        Useful for APIs that expect a `Transform` object rather than raw matrices.
        """
        # Ensure matrices are up-to-date
        return Transform.from_matrix(self.get_world_matrix())

    @property
    def local_transfrom(self) -> Transform:
        """
        Returns a `Transform` representing the object's world transform (position/rotation/scale).
        Useful for APIs that expect a `Transform` object rather than raw matrices.
        """
        # Ensure matrices are up-to-date
        return Transform.from_matrix(self.get_local_matrix())
    
    def get_local_bounds(self) -> Optional[np.ndarray]:
        """
        Docstring for get_local_bounds

        :return: The local
        :rtype: np.ndarray
        """

        
        return None
    
    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float):
        local_bounds = self.get_local_bounds()
        if local_bounds is None:
            return AABB.unit_cube()
        
        world_bounds = AABB.transform_local_bounds_to_world_bounds(transformation_matrix, local_bounds)

        min_p = np.min(world_bounds, axis=0) - padding
        max_p = np.max(world_bounds, axis=0) + padding
        return AABB(min_p, max_p)
    
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

        self.nodes: List[SceneNode] = []
        
        self._version: int = 1
        self._cache_nodes: Optional[List[SceneNode]] = None

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
    
    def set_camera(self, camera: Camera):
        """
        Change the current camera that is used.
        Doesn't update the version counter.
        """
        self.camera = camera

    def add_node(self, node: SceneNode):
        """
        Adds a new object to the scene and updates the version counter.
        """
        node.update_matrices()
        self.nodes.append(node)
        self.update_version()

    def add_object_by_context(self, context: Any, name: str = "Object", transform: Transform = Transform.Identity()) -> SceneNode:
        """
        Creates a new SceneNode with the given context and adds it to the scene.
        
        :param context: The context data to attach to the new SceneNode.
        :param name: The name of the new SceneNode.
        :return: The newly created SceneNode.
        """
        new_node = SceneNode(name=name, context=context, transform=transform)
        self.add_node(new_node)
        return new_node
    
    @staticmethod
    def get_node(nodes: list[SceneNode], name: str) -> Optional[SceneNode]:
        for node in nodes:
            if node.name == name:
                return node
        return None

    @staticmethod
    def get_node_by_id(nodes: list[SceneNode], id: int) -> Optional[SceneNode]:
        for node in nodes:
            if hash(node) == id:
                return node
        
        return None

    @staticmethod
    def _matches_context(context: Any, type_or_name: Union[type, str]) -> bool:
        """Helper to check if a context matches a type (class) or name (str)."""
        if isinstance(type_or_name, type):
            return isinstance(context, type_or_name)
        return type(context).__name__ == type_or_name

    @staticmethod
    def get_nodes_by_type(nodes: list[SceneNode], context_type: Union[type, str]) -> List[SceneNode]:
        """
        Returns a list of all scene objects that contain data of the specified context type.
        """
        result = []
        for node in nodes:
            if node.context is not None:
                if Scene._matches_context(node.context, context_type):
                    result.append(node)
        return result

    @staticmethod
    def get_nodes_by_types(nodes: list[SceneNode], context_types: List[Union[type, str]]) -> List[SceneNode]:
        """
        Returns a list of all scene objects that contain data of ANY of the specified context types.
        """
        result = []
        
        # Optimization: Separate real classes from string names once
        real_types = tuple(t for t in context_types if isinstance(t, type))
        type_names = {t for t in context_types if isinstance(t, str)}

        for node in nodes:
            if node.context is None:
                continue
                
            # Check 1: Is it an instance of the real classes? (Fast)
            if real_types and isinstance(node.context, real_types):
                result.append(node)
                continue
            
            # Check 2: Does the class name match one of the strings? (Slower fallback)
            if type_names and type(node.context).__name__ in type_names:
                result.append(node)
                
        return result

    @staticmethod
    def get_nodes_not_of_type(nodes: list[SceneNode], context_type: Union[type, str]) -> List[SceneNode]:
        """
        Returns a list of all scene objects that do NOT contain data of the specified context type.
        """
        result = []
        for node in nodes:
            # If context is None, it definitely doesn't match the type, so we include it
            if node.context is None:
                result.append(node)
                continue

            if not Scene._matches_context(node.context, context_type):
                result.append(node)
        return result

    @staticmethod
    def get_nodes_not_of_types(nodes: list[SceneNode], context_types: List[Union[type, str]]) -> List[SceneNode]:
        """
        Returns a list of all scene objects that do NOT contain data of ANY of the specified context types.
        """
        result = []
        
        real_types = tuple(t for t in context_types if isinstance(t, type))
        type_names = {t for t in context_types if isinstance(t, str)}

        for node in nodes:
            if node.context is None:
                result.append(node)
                continue

            # Check 1: If it matches a real type, EXCLUDE it
            if real_types and isinstance(node.context, real_types):
                continue
            
            # Check 2: If it matches a string name, EXCLUDE it
            if type_names and type(node.context).__name__ in type_names:
                continue

            # If we reached here, it didn't match anything
            result.append(node)
            
        return result
    
    @staticmethod
    def get_nodes_subclass_of(nodes: list[SceneNode], parent_class: type) -> List[SceneNode]:
        """
        Returns all objects whose context is a subclass of the given parent_class.
        Useful for polymorphism (e.g., get all Light subclasses).
        """
        result = []
        for node in nodes:
            if node.context is not None and isinstance(node.context, parent_class):
                result.append(node)
        return result

    @staticmethod
    def get_nodes_with_attribute(nodes: list[SceneNode], attribute_name: str) -> List[SceneNode]:
        """
        Returns all objects whose context has a specific attribute (variable/property).
        Example: get_nodes_with_attribute("intensity")
        """
        result = []
        for node in nodes:
            if node.context is not None and hasattr(node.context, attribute_name):
                result.append(node)
        return result

    @staticmethod
    def get_nodes_by_attribute_value(nodes: list[SceneNode], attribute_name: str, value: Any) -> List[SceneNode]:
        """
        Returns all objects whose context has a specific attribute equal to a specific value.
        Example: get_nodes_by_attribute_value("is_visible", True)
        """
        result = []
        for node in nodes:
            if node.context is not None:
                # Check if attribute exists
                if hasattr(node.context, attribute_name):
                    # Check if value matches
                    attr_val = getattr(node.context, attribute_name)
                    if attr_val == value:
                        result.append(node)
        return result

    @staticmethod
    def get_nodes_by_condition(nodes: list[SceneNode], condition_func: Callable[[SceneNode], bool]) -> List[SceneNode]:
        """
        Returns all objects that satisfy a custom lambda condition.
        
        Example: 
            # Get all lights brighter than 500
            scene.get_nodes_by_condition(
                lambda node: isinstance(node.context, Light) and node.context.intensity > 500
            )
        """
        result = []
        for node in nodes:
            if condition_func(node):
                result.append(node)
        return result
    
    @staticmethod
    def flatten_scene_nodes(nodes: list[SceneNode]) -> list[SceneNode]:
        """
        Retruns a list of all objects in a flat list array and updated the cache for faster access
        """
        flat_list = []
        
        # We store the ID (memory address) of visited objects
        visited_ids = set() 
        
        # Initialize stack with the top-level objects
        stack = list(nodes)

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
        return list({id(node): node for node in flat_list}.values())
    
    def cache_scene_nodes_flat(self) -> List[SceneNode]:
        """
        Returns a flat list of all objects in the scene, including children.
        Caches the result for faster access on subsequent calls.
        
        :return: Description
        :rtype: List[SceneNode]
        """
        if self._cache_nodes is None:
            Scene.flatten_scene_nodes(self.nodes)

        return self._cache_nodes or self.nodes
    
    def remove_node(self, object_node: SceneNode) -> bool:
        """
        Removes a specific object node from the scene. 
        Returns True if successful, False if the object was not found.
        """
        # 1. Check root level
        if object_node in self.nodes:
            self.nodes.remove(object_node)
            return True

        # 2. Check children (recursive search)
        # Note: This is expensive for deep trees. 
        # Better to store parent references in SceneNode if frequent removal is needed.
        for parent in self.cache_scene_nodes_flat():
            if object_node in parent.children:
                parent.children.remove(object_node)
                return True
        
        return False

    def remove_object_by_name(self, name: str) -> bool:
        """Removes the first object found with the given name."""
        node = self.get_node(self.cache_scene_nodes_flat(), name)
        if node:
            return self.remove_node(node)
        return False

    def reparent(self, object_node: SceneNode, new_parent: Optional[SceneNode]):
        """
        Moves an object from its current parent (or root) to a new parent.
        If new_parent is None, moves the object to the Scene root.
        """
        # 1. Remove from old location
        removed = self.remove_node(object_node)
        if not removed:
            print(f"Warning: Object '{object_node.name}' not found in scene; cannot reparent.")
            return

        # 2. Add to new location
        if new_parent is None:
            self.nodes.append(object_node)
        else:
            new_parent.add_child(object_node)

    def print_hierarchy(self):
        """Prints a visual tree structure of the scene to the console."""
        print(f"Scene: {self.name}")
        for node in self.nodes:
            self._print_node_recursive(node, prefix="", is_last=True)

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

    def get_scene_summary(self) -> dict:
        """Returns a dictionary summary of the scene content."""
        all_objs = self.cache_scene_nodes_flat()
        
        # Count types
        type_counts = {}
        for node in all_objs:
            t_name = type(node.context).__name__ if node.context else "EmptyNode"
            type_counts[t_name] = type_counts.get(t_name, 0) + 1

        return {
            "total_nodes": len(all_objs),
            "root_nodes": len(self.nodes),
            "type_breakdown": type_counts
        }

    def clear_scene(self):
        """
        Removes all the objects and light sources from the scene, while updating the version counter.
        """
        self.nodes.clear()
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
    
    :param nodes: A flat list of SceneNodes (use scene.get_nodes_flat())
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
