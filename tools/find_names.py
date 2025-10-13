#!/usr/bin/env python3
# ...existing code...
"""
Generalized AST scanner.

Usage examples:
  python3 tools/find_names.py py_src --attr direction --class Ray --name origin
  python3 tools/find_names.py py_src --pattern "todo"        # regex on source lines
  python3 tools/find_names.py py_src --kwarg data --call render

Search options (can repeat):
  --attr / -a    attribute name (checks Attribute.attr)
  --class / -c   class definition name (checks ClassDef.name)
  --name / -n    identifier name (checks Name.id)
  --kwarg / -k   keyword-argument name (checks ast.keyword.arg)
  --call / -C    function/call target name (Name or Attribute)
  --pattern / -p regex searched on source lines (fallback grep style)
"""
import ast
import sys
import argparse
import re
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Find names/attributes/classes in Python AST")
    p.add_argument("root", help="root path to search")
    p.add_argument("-a", "--attr", action="append", help="attribute name to search (Attribute.attr)")
    p.add_argument("-c", "--class", dest="classes", action="append", help="class definition name to search")
    p.add_argument("-n", "--name", action="append", help="identifier name to search (Name.id)")
    p.add_argument("-k", "--kwarg", action="append", help="keyword-argument name to search (ast.keyword.arg)")
    p.add_argument("-C", "--call", action="append", help="call target name to search (Name or Attribute)")
    p.add_argument("-p", "--pattern", help="regex pattern to search in source lines")
    p.add_argument("--show-src", action="store_true", help="show source segment (if available)")
    return p.parse_args()

def get_segment(src, node):
    try:
        return ast.get_source_segment(src, node)
    except Exception:
        return None

def scan_file(path: Path, opts):
    try:
        src = path.read_text()
    except Exception:
        return
    try:
        tree = ast.parse(src, filename=str(path))
    except Exception:
        return

    pat = re.compile(opts.pattern, re.IGNORECASE) if opts.pattern else None

    class Visitor(ast.NodeVisitor):
        def report(self, node, kind, name):
            lineno = getattr(node, "lineno", "?")
            seg = get_segment(src, node) if opts.show_src else None
            line = src.splitlines()[lineno-1].strip() if lineno != "?" and lineno-1 < len(src.splitlines()) else ""
            out = f"{path}:{lineno}\ttype={kind}\tname={name}\tline={line}"
            if seg:
                out += f"\tsegment={seg!r}"
            print(out)

        def visit_ClassDef(self, node):
            if opts.classes and node.name in opts.classes:
                self.report(node, "ClassDef", node.name)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            # attribute access like obj.attr
            if opts.attr and getattr(node, "attr", None) in opts.attr:
                self.report(node, "Attribute", node.attr)
            self.generic_visit(node)

        def visit_Name(self, node):
            if opts.name and getattr(node, "id", None) in opts.name:
                self.report(node, "Name", node.id)
            self.generic_visit(node)

        def visit_Call(self, node):
            # call target can be Name or Attribute
            func = node.func
            if isinstance(func, ast.Name) and opts.call and func.id in opts.call:
                self.report(node, "Call(Name)", func.id)
            elif isinstance(func, ast.Attribute) and opts.call and getattr(func, "attr", None) in opts.call:
                self.report(node, "Call(Attribute)", func.attr)
            self.generic_visit(node)

        def visit_keyword(self, node):
            # keyword arguments in calls
            if opts.kwarg and getattr(node, "arg", None) in opts.kwarg:
                self.report(node, "kwarg", node.arg)
            self.generic_visit(node)

    Visitor().visit(tree)

    # optional regex pattern search on raw source lines (fallback / complementary)
    if pat:
        for i, l in enumerate(src.splitlines(), start=1):
            if pat.search(l):
                print(f"{path}:{i}\tpattern\t{pat.pattern}\tline={l.strip()}")

def main():
    opts = parse_args()
    root = Path(opts.root)
    for f in root.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        scan_file(f, opts)

if __name__ == "__main__":
    main()