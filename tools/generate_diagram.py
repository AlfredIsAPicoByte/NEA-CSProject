#!/usr/bin/env python3
"""
Script to regenerate class diagrams from Python source files.
Supports multiple graph types: full, relationships, data-structures, inheritance-only.

Usage:
    python3 generate_diagram.py --type full --output classes.graphml
    python3 generate_diagram.py --type relationships --output relationships.graphml
    python3 generate_diagram.py --type data-structures --output data.graphml
    python3 generate_diagram.py --type inheritance --output inheritance.graphml
    python3 generate_diagram.py --mermaid --output diagram.mmd
"""

import ast
from distutils import extension
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import argparse

MARKER_START = ""
MARKER_END = ""

class GraphType(Enum):
    """Types of graphs that can be generated."""
    FULL = "full"  # All classes with all attributes and methods
    RELATIONSHIPS = "relationships"  # Relationships and dependencies only
    DATA_STRUCTURES = "data-structures"  # Classes with attributes focus
    INHERITANCE = "inheritance"  # Inheritance hierarchy only
    CLASS_DEFINITIONS = "class-definitions"  # Class definitions without relationships

@dataclass
class Attribute:
    """Represents a class attribute with type information."""
    name: str
    type_hint: Optional[str] = None
    is_private: bool = False
    source: str = "unknown"  # 'annotation', 'assignment', 'parameter', 'docstring'
    
    def __str__(self) -> str:
        prefix = "-" if self.is_private else "+"
        if self.type_hint:
            return f"{prefix}{self.name}: {self.type_hint}"
        return f"{prefix}{self.name}"

@dataclass
class PyClass:
    """Represents a parsed Python class."""
    name: str
    module: str
    file_path: str
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    attributes: Dict[str, Attribute] = field(default_factory=dict)  # Changed from properties
    method_signatures: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    is_abstract: bool = False
    is_enum: bool = False
    docstring: str = ""

    @property
    def node_id(self) -> str:
        """Get unique node ID for GraphML."""
        return self.name
    
    @property
    def public_attributes(self) -> Dict[str, Attribute]:
        """Get only public attributes."""
        return {k: v for k, v in self.attributes.items() if not v.is_private}
    
    @property
    def private_attributes(self) -> Dict[str, Attribute]:
        """Get only private attributes."""
        return {k: v for k, v in self.attributes.items() if v.is_private}

@dataclass
class Relationship:
    """Represents a relationship between classes."""
    source: str  # Class name
    target: str  # Class name
    rel_type: str  # 'inherits', 'contains', 'uses', 'depends'
    multiplicity: str = ""  # e.g., "1", "*", "0..1"

class PythonClassParser:
    """Parse Python files to extract class information."""

    def __init__(self, source_dir: str, include_private: bool = False):
        self.source_dir = Path(source_dir)
        self.include_private = include_private
        self.classes: Dict[str, PyClass] = {}
        self.relationships: List[Relationship] = []

    def parse_directory(self):
        """Recursively parse all Python files in directory."""
        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" not in py_file.parts:
                self.parse_file(py_file)

    def parse_file(self, file_path: Path):
        """Parse a single Python file for classes."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            print(f"⚠️  Skipping {file_path}: parse error")
            return

        # Get relative module path
        try:
            rel_path = file_path.relative_to(self.source_dir)
            module_path = str(rel_path.with_suffix('')).replace('/', '.')
        except ValueError:
            module_path = file_path.stem

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Skip private classes if not included
                if node.name.startswith('_') and not self.include_private:
                    continue

                py_class = PyClass(
                    name=node.name,
                    module=module_path,
                    file_path=str(file_path),
                    bases=[self._get_base_name(base) for base in node.bases],
                    docstring=ast.get_docstring(node) or "",
                )

                # Check if abstract
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and any(
                        isinstance(dec, ast.Name) and dec.id == "abstractmethod"
                        for dec in item.decorator_list
                    ):
                        py_class.is_abstract = True
                        break

                # Check if enum
                for base in py_class.bases:
                    if "Enum" in base:
                        py_class.is_enum = True
                        break

                # Extract methods
                py_class.methods = [
                    item.name for item in node.body
                    if isinstance(item, ast.FunctionDef) 
                    and (self.include_private or not item.name.startswith('_'))
                ]
                
                # Extract method signatures
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        py_class.method_signatures[item.name] = self._extract_method_signature(item)
                
                # Extract attributes - IMPROVED
                py_class.attributes = self._extract_attributes(node)

                self.classes[node.name] = py_class

    def _extract_attributes(self, class_node: ast.ClassDef) -> Dict[str, Attribute]:
        """Extract all attributes from class with comprehensive detection."""
        attributes = {}
        
        # 1. From annotated assignments (highest priority)
        for item in class_node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attr_name = item.target.id
                type_hint = self._extract_type_hint(item.annotation)
                is_private = attr_name.startswith('_')
                
                attributes[attr_name] = Attribute(
                    name=attr_name,
                    type_hint=type_hint,
                    is_private=is_private,
                    source="annotation"
                )
        
        # 2. From class variable assignments
        for item in class_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attr_name = target.id
                        if attr_name not in attributes:  # Don't override annotations
                            is_private = attr_name.startswith('_')
                            # Try to infer type from value
                            type_hint = self._infer_type_from_value(item.value)
                            
                            attributes[attr_name] = Attribute(
                                name=attr_name,
                                type_hint=type_hint,
                                is_private=is_private,
                                source="assignment"
                            )
        
        # 3. From __init__ parameters (instance attributes)
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for arg in item.args.args[1:]:  # Skip 'self'
                    attr_name = arg.arg
                    if attr_name not in attributes:
                        type_hint = self._extract_type_hint(arg.annotation) if arg.annotation else None
                        is_private = attr_name.startswith('_')
                        
                        attributes[attr_name] = Attribute(
                            name=attr_name,
                            type_hint=type_hint,
                            is_private=is_private,
                            source="parameter"
                        )
                
                # Also check for self.x = ... assignments in __init__
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                                attr_name = target.attr
                                if attr_name not in attributes:
                                    type_hint = None
                                    is_private = attr_name.startswith('_')
                                    
                                    attributes[attr_name] = Attribute(
                                        name=attr_name,
                                        type_hint=type_hint,
                                        is_private=is_private,
                                        source="assignment"
                                    )
        
        return attributes

    def _infer_type_from_value(self, value_node) -> Optional[str]:
        """Attempt to infer type from an AST value node."""
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, bool):
                return "bool"
            elif isinstance(value_node.value, int):
                return "int"
            elif isinstance(value_node.value, float):
                return "float"
            elif isinstance(value_node.value, str):
                return "str"
            elif value_node.value is None:
                return "None"
        elif isinstance(value_node, ast.List):
            return "list"
        elif isinstance(value_node, ast.Dict):
            return "dict"
        elif isinstance(value_node, ast.Set):
            return "set"
        elif isinstance(value_node, ast.Tuple):
            return "tuple"
        elif isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Name):
                return value_node.func.id
        elif isinstance(value_node, ast.Name):
            return value_node.id
        
        return None

    def _extract_method_signature(self, func_node: ast.FunctionDef) -> str:
        """Extract method signature with parameters and return type."""
        method_name = func_node.name
        if method_name == "__init__":
            method_name = "new"
        
        args = []
        for arg in func_node.args.args[1:]:  # Skip 'self'
            arg_str = arg.arg
            if arg.annotation:
                arg_type = self._extract_type_hint(arg.annotation)
                arg_str += f": {arg_type}"
            args.append(arg_str)
        
        return_type = ""
        if func_node.returns:
            return_type = self._extract_type_hint(func_node.returns)
            return_type = f" -> {return_type}"
        
        params = ", ".join(args) if args else ""
        signature = f"{method_name}({params}){return_type}"
        return signature

    def _get_base_name(self, base_node) -> str:
        """Extract base class name from AST node."""
        if isinstance(base_node, ast.Name):
            return base_node.id
        elif isinstance(base_node, ast.Attribute):
            return base_node.attr
        elif isinstance(base_node, ast.Subscript):
            if isinstance(base_node.value, ast.Name):
                return base_node.value.id
        return "Unknown"

    def extract_relationships(self):
        """Extract inheritance and dependency relationships."""
        for class_obj in self.classes.values():
            # Extract inheritance relationships
            for base in class_obj.bases:
                if base in self.classes:
                    self.relationships.append(
                        Relationship(class_obj.name, base, "inherits")
                    )
                elif base not in ("object", "ABC", "Generic"):
                    self.relationships.append(Relationship(class_obj.name, base, "depends"))
            
            # Extract from attribute type hints
            for attr in class_obj.attributes.values():
                if attr.type_hint and attr.type_hint in self.classes:
                    rel_type = "contains" if not attr.is_private else "uses"
                    self.relationships.append(
                        Relationship(class_obj.name, attr.type_hint, rel_type)
                    )
            
            # Extract from method return types
            class_file = Path(class_obj.file_path)
            try:
                with open(class_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_obj.name:
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.returns:
                                target_class = self._extract_type_hint(item.returns)
                                if target_class and target_class in self.classes:
                                    self.relationships.append(
                                        Relationship(class_obj.name, target_class, "uses")
                                    )
            except (SyntaxError, UnicodeDecodeError):
                pass

    def _extract_type_hint(self, annotation_node) -> Optional[str]:
        """Extract class name from type annotation AST node."""
        if isinstance(annotation_node, ast.Name):
            return annotation_node.id
        elif isinstance(annotation_node, ast.Attribute):
            return annotation_node.attr
        elif isinstance(annotation_node, ast.Subscript):
            if isinstance(annotation_node.slice, ast.Name):
                return annotation_node.slice.id
            elif hasattr(annotation_node.slice, 'value') and isinstance(annotation_node.slice.value, ast.Name):
                return annotation_node.slice.value.id
        return None

class XMLGenerator:
    """Generate XML output from parsed classes."""

    def __init__(self, classes: Dict[str, PyClass], relationships: List[Relationship]):
        self.classes = classes
        self.relationships = relationships

    def generate_xml(self) -> str:
        """Generate XML representation of classes and relationships."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<classes>']
        
        for class_name in sorted(self.classes):
            class_obj = self.classes[class_name]
            lines.append(f'  <class name="{class_obj.name}" module="{class_obj.module}">')
            if class_obj.description:
                lines.append(f'    <description>{class_obj.description}</description>')
            if class_obj.bases:
                lines.append(f'    <bases>{";".join(class_obj.bases)}</bases>')
            if class_obj.attributes:
                lines.append('    <attributes>')
                for attr in class_obj.attributes.values():
                    lines.append(f'      <attribute name="{attr.name}" type="{attr.type_hint}" source="{attr.source}" private="{attr.is_private}"/>')
                lines.append('    </attributes>')
            if class_obj.methods:
                lines.append('    <methods>')
                for method in class_obj.methods:
                    signature = class_obj.method_signatures.get(method, method)
                    lines.append(f'      <method name="{method}" signature="{signature}"/>')
                lines.append('    </methods>')
            lines.append('  </class>')
    
        lines.append('  <relationships>')
        for rel in self.relationships:
            lines.append(f'    <relationship source="{rel.source}" target="{rel.target}" type="{rel.rel_type}" multiplicity="{rel.multiplicity}"/>')
        lines.append('  </relationships>')
        lines.append('</classes>')
        return "\n".join(lines)
    
class GraphMLGenerator:
    """Generate GraphML output from parsed classes."""

    MODULE_COLORS = {
        'Data': '#ADD8E6',
        'Geometry': '#90EE90',
        'Material': '#FFFFE0',
        'Lighting': '#FFE4B5',
        'Rendering': '#FFB6C1',
        'Image': '#DDA0DD',
        'Utilities': '#D3D3D3',
    }

    def __init__(self, classes: Dict[str, PyClass], relationships: List[Relationship]):
        self.classes = classes
        self.relationships = relationships

    def get_module_name(self, module_path: str) -> str:
        """Extract primary module from full module path."""
        for known_module in self.MODULE_COLORS.keys():
            if known_module.lower() in module_path.lower():
                return known_module
        parts = module_path.split('.')
        return parts[0].capitalize() if parts else "Other"

    def generate_graphml(self, graph_type: GraphType = GraphType.FULL) -> str:
        """Generate GraphML with optional filtering by graph type."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
            '  <key id="d0" for="node" attr.name="description" attr.type="string"/>',
            '  <key id="d1" for="node" attr.name="module" attr.type="string"/>',
            '  <key id="d2" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="d3" for="node" attr.name="attributes" attr.type="string"/>',
            '  <key id="d4" for="edge" attr.name="relationship" attr.type="string"/>',
            '',
            f'  <graph id="NEA-CSProject-{graph_type.value}" edgedefault="directed">',
            f'    <attr name="name">NEA-CSProject {graph_type.value.capitalize()} (Auto-Generated)</attr>',
            f'    <attr name="description">{self._get_graph_description(graph_type)}</attr>',
            ''
        ]

        # Filter classes based on graph type
        classes_to_include = self._filter_classes(graph_type)
        relationships_to_include = self._filter_relationships(graph_type, classes_to_include)

        # Add nodes
        for class_name in sorted(classes_to_include):
            class_obj = self.classes[class_name]
            module = self.get_module_name(class_obj.module)
            color = self.MODULE_COLORS.get(module, '#CCCCCC')
            class_type = "enum" if class_obj.is_enum else "abstract" if class_obj.is_abstract else "class"

            attributes_str = "; ".join(str(attr) for attr in class_obj.attributes.values()) if graph_type in (GraphType.FULL, GraphType.DATA_STRUCTURES, GraphType.CLASS_DEFINITIONS) else ""

            lines.append(f'    <node id="{class_name}" labels="{class_name}">')
            lines.append(f'      <data key="d0">{class_obj.description or class_type.capitalize()}</data>')
            lines.append(f'      <data key="d1">{module}</data>')
            lines.append(f'      <data key="d2">{class_type}</data>')
            if attributes_str:
                lines.append(f'      <data key="d3">{attributes_str}</data>')
            lines.append('    </node>')

        lines.append('')

        # Add edges
        for i, rel in enumerate(relationships_to_include):
            edge_id = f"e_{rel.source}_{rel.target}_{i}"
            lines.append(f'    <edge id="{edge_id}" source="{rel.source}" target="{rel.target}">')
            lines.append(f'      <data key="d4">{rel.rel_type}</data>')
            lines.append('    </edge>')

        lines.extend(['', '  </graph>', '</graphml>'])
        return '\n'.join(lines)

    def _get_graph_description(self, graph_type: GraphType) -> str:
        """Get description for graph type."""
        descriptions = {
            GraphType.FULL: "Complete class hierarchy with all attributes and methods",
            GraphType.RELATIONSHIPS: "Class relationships and dependencies only",
            GraphType.DATA_STRUCTURES: "Focus on data attributes and structures",
            GraphType.INHERITANCE: "Inheritance hierarchy only",
            GraphType.CLASS_DEFINITIONS: "Class definitions without relationships",
        }
        return descriptions.get(graph_type, "Class diagram")

    def _filter_classes(self, graph_type: GraphType) -> Set[str]:
        """Filter classes based on graph type."""
        return set(self.classes.keys())  # All graphs include all classes

    def _filter_relationships(self, graph_type: GraphType, classes: Set[str]) -> List[Relationship]:
        """Filter relationships based on graph type."""
        filtered = []
        
        for rel in self.relationships:
            if rel.source not in classes or rel.target not in classes:
                continue
            
            if graph_type == GraphType.INHERITANCE:
                if rel.rel_type in ("inherits",):
                    filtered.append(rel)
            elif graph_type == GraphType.RELATIONSHIPS:
                if rel.rel_type in ("inherits", "uses", "depends"):
                    filtered.append(rel)
            elif graph_type == GraphType.DATA_STRUCTURES:
                if rel.rel_type in ("contains",):
                    filtered.append(rel)
            elif graph_type == GraphType.CLASS_DEFINITIONS:
                continue  # No relationships for class definitions
            else:  # FULL
                filtered.append(rel)
        
        return filtered

class MermaidGenerator:
    """Generates Mermaid diagram syntax."""

    def __init__(self, classes: Dict[str, PyClass], relationships: List[Relationship]):
        self.classes = classes
        self.relationships = relationships

    def generate(self, graph_type: GraphType = GraphType.FULL) -> str:
        lines = ["classDiagram"]
        
        modules = {}
        for cls in self.classes.values():
            modules.setdefault(cls.module, []).append(cls)

        for module_name, class_list in modules.items():
            safe_module_name = module_name.replace("\\", ".").replace("/", ".")
            lines.append(f"    namespace {safe_module_name} {{")
            for cls in class_list:
                stereotype = ""
                if cls.is_enum:
                    stereotype = "<<enumeration>>"
                elif cls.is_abstract:
                    stereotype = "<<abstract>>"
                
                lines.append(f"        class {cls.name} {{")
                if stereotype:
                    lines.append(f"            {stereotype}")
                
                # Attributes based on graph type
                if graph_type in (GraphType.FULL, GraphType.DATA_STRUCTURES, GraphType.CLASS_DEFINITIONS):
                    for attr in cls.attributes.values():
                        lines.append(f"            {attr}")
                
                # Methods based on graph type
                if graph_type in (GraphType.FULL, GraphType.CLASS_DEFINITIONS):
                    for method in cls.methods:
                        signature = cls.method_signatures.get(method, method)
                        lines.append(f"            +{signature}")
                
                lines.append("        }")
            lines.append("    }")

        # Filter relationships
        relationships_to_show = self._filter_relationships(graph_type)
        
        for rel in relationships_to_show:
            if rel.rel_type == "inherits":
                lines.append(f"    {rel.source} <|-- {rel.target}")
            elif rel.rel_type == "contains":
                lines.append(f"    {rel.source}o-- {rel.target}")
            elif rel.rel_type in ("depends", "uses"):
                lines.append(f"    {rel.source} --> {rel.target}")

        return "\n".join(lines)

    def _filter_relationships(self, graph_type: GraphType) -> List[Relationship]:
        """Filter relationships for Mermaid based on graph type."""
        filtered = []
        
        for rel in self.relationships:
            if graph_type == GraphType.INHERITANCE and rel.rel_type in ("inherits",):
                filtered.append(rel)
            elif graph_type == GraphType.RELATIONSHIPS and rel.rel_type in ("inherits", "uses", "depends"):
                filtered.append(rel)
            elif graph_type == GraphType.DATA_STRUCTURES and rel.rel_type in ("contains",):
                filtered.append(rel)
            elif graph_type == GraphType.CLASS_DEFINITIONS:
                continue  # No relationships for class definitions
            elif graph_type == GraphType.FULL:
                filtered.append(rel)
        
        return filtered


def main():
    parser = argparse.ArgumentParser(description="Generate class diagrams from Python code")
    parser.add_argument("--source", default="py_src/src", help="Source directory")
    parser.add_argument("--output", default="diagram", help="Output file")
    parser.add_argument("--format", choices=["xml", "graphml", "mermaid"], default="graphml", help="Output format")
    parser.add_argument("--type", choices=[t.value for t in GraphType], default="full", 
                       help="Graph type to generate")
    parser.add_argument("--include-private", action="store_true", help="Include private classes")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")

    args = parser.parse_args()
    graph_type = GraphType(args.type)

    print(f"🔍 Parsing {args.source}...")
    python_parser = PythonClassParser(args.source, args.include_private)
    python_parser.parse_directory()
    python_parser.extract_relationships()

    print(f"📊 Found {len(python_parser.classes)} classes")
    print(f"📝 Found {sum(len(c.attributes) for c in python_parser.classes.values())} attributes")
    print(f"🔗 Found {len(python_parser.relationships)} relationships")

    if args.stats:
        print("\n📈 Attribute sources:")
        sources = {}
        for cls in python_parser.classes.values():
            for attr in cls.attributes.values():
                sources[attr.source] = sources.get(attr.source, 0) + 1
        for source, count in sorted(sources.items()):
            print(f"  - {source}: {count}")
        return

    print(f"📝 Generating {args.type} diagram...")
    
    if args.format.lower() == "xml":
        generator = XMLGenerator(python_parser.classes, python_parser.relationships)
        content = generator.generate(graph_type)
    elif args.format.lower() == "graphml":
        generator = GraphMLGenerator(python_parser.classes, python_parser.relationships)
        content = generator.generate_graphml(graph_type)
    elif args.format.lower() == "mermaid":
        generator = MermaidGenerator(python_parser.classes, python_parser.relationships)
        content = generator.generate(graph_type)
    else:
        print(f"❌ Unsupported format: {args.format}")
        return
    
    extension = "mmd" if args.format.lower() == "mermaid" else args.format.lower()

    with open(f"py_src/docs/{args.output}.{extension}", 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Done! Generated {args.output}.{extension} with {len(python_parser.classes)} classes and {len(python_parser.relationships)} relationships.")

if __name__ == '__main__':
    main()