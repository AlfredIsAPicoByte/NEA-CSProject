#!/usr/bin/env python3
"""
Script to regenerate classes_drawio.graphml from Python source files.
This keeps the class diagram synchronized with actual source code.

Usage:
    python3 generate_drawio_diagram.py
    python3 generate_drawio_diagram.py --output custom_diagram.graphml
    python3 generate_drawio_diagram.py --include-private

Script to generate a Mermaid class diagram from Python source files.
Can output to a file or update an existing Markdown documentation file.

Usage:
    python3 generate_mermaid.py
    python3 generate_mermaid.py --output docs/diagram.mmd
    python3 generate_mermaid.py --update-docs docs/ARCHITECTURE.md
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
import argparse

MARKER_START = ""
MARKER_END = ""


@dataclass
class PyClass:
    """Represents a parsed Python class."""
    name: str
    module: str
    file_path: str
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    description: str = ""
    is_abstract: bool = False
    is_enum: bool = False

    @property
    def node_id(self) -> str:
        """Get unique node ID for GraphML."""
        return self.name


@dataclass
class Relationship:
    """Represents a relationship between classes."""
    source: str  # Class name
    target: str  # Class name
    rel_type: str  # 'inherits', 'contains', 'uses'


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
                ]

                self.classes[node.name] = py_class

    def _get_base_name(self, base_node) -> str:
        """Extract base class name from AST node."""
        if isinstance(base_node, ast.Name):
            return base_node.id
        elif isinstance(base_node, ast.Attribute):
            return base_node.attr
        return "Unknown"

    def extract_relationships(self):
        """Extract inheritance and dependency relationships."""
        for class_obj in self.classes.values():
            for base in class_obj.bases:
                if base in self.classes:
                    self.relationships.append(
                        Relationship(class_obj.name, base, "inherits")
                    )

class MermaidGenerator:
    """Generates Mermaid diagram syntax."""
    
    def __init__(self, classes: Dict[str, PyClass], relationships: List[Relationship]):
        self.classes = classes
        self.relationships = relationships

    def generate(self) -> str:
        lines = ["classDiagram"]
        
        # Group by Module (Namespace)
        modules = {}
        for cls in self.classes.values():
            modules.setdefault(cls.module, []).append(cls)

        for module_name, class_list in modules.items():
            lines.append(f"    namespace {module_name} {{")
            for cls in class_list:
                stereotype = ""
                if cls.is_enum: stereotype = "<<enumeration>>"
                elif cls.is_abstract: stereotype = "<<abstract>>"
                
                # Class definition
                lines.append(f"        class {cls.name} {{")
                if stereotype:
                    lines.append(f"            {stereotype}")
                
                # Properties
                for prop in cls.bases:
                    lines.append(f"            +{prop}")
                
                # Methods
                for method in cls.methods:
                    method_clean = method.replace("__init__", "new")
                    lines.append(f"            +{method_clean}")
                
                lines.append("        }")
            lines.append("    }")

        # Relationships (Outside namespace to avoid syntax issues in some renderers)
        for rel in self.relationships:
            if rel.rel_type == "inheritance":
                # Parent <|-- Child
                lines.append(f"    {rel.source} <|-- {rel.target}")

        return "\n".join(lines)


def update_docs(file_path: str, mermaid_content: str):
    """Updates a markdown file between markers."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File {file_path} not found.")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if MARKER_START not in content or MARKER_END not in content:
        print(f"⚠️ Markers not found in {file_path}.")
        print(f"   Please add {MARKER_START} and {MARKER_END} to the file.")
        return

    # Replace content between markers
    pre = content.split(MARKER_START)[0]
    post = content.split(MARKER_END)[1]
    
    new_content = (
        f"{pre}{MARKER_START}\n"
        f"```mermaid\n{mermaid_content}\n```\n"
        f"{MARKER_END}{post}"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"✅ Updated documentation: {file_path}")

def mermaid_main():
    parser = argparse.ArgumentParser(description="Generate Mermaid Class Diagram from Python Code")
    parser.add_argument("--source", default="py_src/src", help="Source code directory")
    parser.add_argument("--output", help="Output .mmd file path")
    parser.add_argument("--update-docs", help="Markdown file to update (must contain markers)")
    parser.add_argument("--include-private", action="store_true", help="Include private classes")
    
    args = parser.parse_args()

    print(f"🔍 Scanning {args.source}...")
    scanner = PythonClassParser(args.source, args.include_private)
    scanner.parse_directory()
    
    print(f"📊 Found {len(scanner.classes)} classes.")
    generator = MermaidGenerator(scanner.classes, scanner.relationships)
    diagram = generator.generate()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(diagram)
        print(f"💾 Saved diagram to {args.output}")

    if args.update_docs:
        update_docs(args.update_docs, diagram)

    if not args.output and not args.update_docs:
        print("\n" + diagram)

class GraphMLGenerator:
    """Generate GraphML output from parsed classes."""

    # Module to color mapping
    MODULE_COLORS = {
        'Data': '#ADD8E6',      # Light blue
        'Geometry': '#90EE90',  # Light green
        'Material': '#FFFFE0',  # Light yellow
        'Lighting': '#FFE4B5',  # Moccasin (orange)
        'Rendering': '#FFB6C1', # Light pink (red)
        'Image': '#DDA0DD',     # Plum (purple)
        'Utilities': '#D3D3D3', # Light gray
    }

    def __init__(self, classes: Dict[str, PyClass], relationships: List[Relationship]):
        self.classes = classes
        self.relationships = relationships

    def get_module_name(self, module_path: str) -> str:
        """Extract primary module from full module path."""
        parts = module_path.split('.')
        if len(parts) > 1:
            return parts[0].capitalize()  # src -> Src (fallback)
        for known_module in self.MODULE_COLORS.keys():
            if known_module.lower() in module_path.lower():
                return known_module
        return "Other"

    def generate_graphml(self) -> str:
        """Generate complete GraphML XML."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
            '  <key id="d0" for="node" attr.name="description" attr.type="string"/>',
            '  <key id="d1" for="node" attr.name="module" attr.type="string"/>',
            '  <key id="d2" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="d3" for="edge" attr.name="relationship" attr.type="string"/>',
            '',
            '  <graph id="NEA-CSProject" edgedefault="directed">',
            '    <attr name="name">NEA-CSProject Class Hierarchy (Auto-Generated)</attr>',
            '    <attr name="description">Class hierarchy and inheritance relationships</attr>',
            ''
        ]

        # Add nodes
        for class_name, class_obj in sorted(self.classes.items()):
            module = self.get_module_name(class_obj.module)
            color = self.MODULE_COLORS.get(module, '#CCCCCC')
            class_type = "enum" if class_obj.is_enum else "abstract" if class_obj.is_abstract else "class"

            lines.append(f'    <node id="{class_name}" labels="{class_name}">')
            lines.append(f'      <data key="d0">{class_obj.description or class_type.capitalize()}</data>')
            lines.append(f'      <data key="d1">{module}</data>')
            lines.append(f'      <data key="d2">{class_type}</data>')
            lines.append('    </node>')

        lines.append('')

        # Add edges
        for i, rel in enumerate(self.relationships):
            if rel.target in self.classes:  # Only add if target exists
                edge_id = f"e_{rel.source}_{rel.target}"
                lines.append(f'    <edge id="{edge_id}" source="{rel.source}" target="{rel.target}">')
                lines.append(f'      <data key="d3">{rel.rel_type}</data>')
                lines.append('    </edge>')

        lines.extend([
            '',
            '  </graph>',
            '</graphml>'
        ])

        return '\n'.join(lines)

def graphml_main():
    parser = argparse.ArgumentParser(
        description='Generate draw.io class diagram from Python source code'
    )
    parser.add_argument(
        '--source',
        default='py_src/src',
        help='Source directory (default: py_src/src)'
    )
    parser.add_argument(
        '--output',
        default='classes_drawio.graphml',
        help='Output file (default: classes_drawio.graphml)'
    )
    parser.add_argument(
        '--include-private',
        action='store_true',
        help='Include private classes (starting with _)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics only'
    )

    args = parser.parse_args()

    print("🔍 Parsing Python files...")
    python_parser = PythonClassParser(args.source, args.include_private)
    python_parser.parse_directory()
    python_parser.extract_relationships()

    print(f"📊 Found {len(python_parser.classes)} classes")
    print(f"🔗 Found {len(python_parser.relationships)} relationships")

    if args.stats:
        # Print module breakdown
        module_stats = {}
        generator = GraphMLGenerator(python_parser.classes, python_parser.relationships)
        for class_obj in python_parser.classes.values():
            module = generator.get_module_name(class_obj.module)
            if module not in module_stats:
                module_stats[module] = 0
            module_stats[module] += 1

        print("\n📈 Classes by module:")
        for module, count in sorted(module_stats.items()):
            print(f"  - {module}: {count}")
        return

    print("📝 Generating GraphML...")
    generator = GraphMLGenerator(python_parser.classes, python_parser.relationships)
    graphml_content = generator.generate_graphml()

    print(f"💾 Writing to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(graphml_content)

    print(f"✅ Done! Generated {args.output}")
    print(f"\n📖 Next steps:")
    print(f"  1. Go to https://app.diagrams.net/")
    print(f"  2. File → Import from → GraphML")
    print(f"  3. Select {args.output}")
    print(f"  4. Your diagram is ready!")

if __name__ == '__main__':
    mermaid_main()
