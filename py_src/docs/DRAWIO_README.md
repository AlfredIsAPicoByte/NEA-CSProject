# NEA-CSProject - draw.io Import Files

## 📋 Quick Start

### For draw.io Import (RECOMMENDED)
**File:** `classes_drawio.graphml`
- **Format:** GraphML (Unicode standard for graphs)
- **Size:** Optimized for draw.io
- **Content:** 80 classes, 59 relationships
- **Import:** File → Import from → GraphML

### For API Documentation
**File:** `classes.xml`
- **Format:** Comprehensive XML with detailed documentation
- **Size:** Full API reference
- **Content:** Classes, methods, attributes, descriptions

### For Setup Help
**File:** `DRAWIO_IMPORT_GUIDE.md`
- **Content:** Step-by-step import instructions
- **Includes:** Customization tips, troubleshooting, advanced features

---

## 📦 File Locations

```
/
├── classes_drawio.graphml      ✓ Draw.io ready
├── classes.xml                 ✓ API documentation
└── DRAWIO_IMPORT_GUIDE.md      ✓ Setup guide
```

---

## 🚀 Quick Steps to View Diagram

1. Go to [draw.io](https://app.diagrams.net/)
2. Click: **File → Import from → GraphML**
3. Select: `classes_drawio.graphml`
4. Click **Open**
5. draw.io auto-generates the class diagram!

---

## 📊 What You'll See

### Nodes (Classes)
- **80 total classes** organized by module
- Color-coded by module (Data, Geometry, Material, Lighting, Rendering, Image)
- Descriptions visible on hover

### Relationships
- **59 edges** showing:
  - Inheritance (class hierarchies)
  - Composition (contains relationships)
  - Usage (class dependencies)

### Modules Covered
- 🔵 **Data** - Core data structures (Camera, Color, Transform, Ray, Scene, Sampler)
- 🟢 **Geometry** - 3D shapes, meshes, BVH, AABB, SDF shapes (26 classes)
- 🟡 **Material** - PBR material system
- 🟠 **Lighting** - Light sources and optics
- 🔴 **Rendering** - Ray tracing pipeline, intersection, shading strategies
- 🟣 **Image** - Film buffer, post-processing effects

---

## ✨ Key Features

### Immediate Benefits
✅ Visual class hierarchy  
✅ Understand inheritance chains  
✅ See module organization  
✅ Identify dependencies  
✅ Generate documentation  

### Customization Options
🎨 Change colors and styles  
📐 Auto-layout diagrams  
🔗 Add hyperlinks to source code  
📝 Add annotations and notes  
💾 Export as PNG/PDF/SVG  

---

## 💡 Tips

### Smart Navigation
- Use **Ctrl+F** to find classes
- Use **Outline Panel** (right side) for quick navigation
- Double-click node to see details

### Create Focused Views
1. Select related classes
2. Right-click → **Export Selection**
3. Create module-specific diagrams

### Add Context
1. Insert text boxes for module descriptions
2. Add swimlanes for organization
3. Create legend showing relationship types

---

## 📖 Advanced Usage

### Generate UML Diagrams
- Use GraphML as foundation
- Add UML notation (+, -, #, ~ for visibility)
- Export for documentation

### Create Architecture Diagrams
- Show data flow between modules
- Highlight critical paths
- Map rendering pipeline

### Generate from Code
To regenerate if classes change:
```bash
# Update from source code files
# (Script to be added for automatic generation)
```

---

## 🔗 Resources

- **draw.io**: https://app.diagrams.net/
- **GraphML Spec**: https://graphml.graphdrawing.org/
- **GitHub Repo**: https://github.com/AlfredIsAPicoByte/NEA-CSProject
- **Current Branch**: feature/python-raycasting-link-cpp

---

## 📝 File Statistics

| Aspect | Details |
|--------|---------|
| **Total Classes** | 80 |
| **Total Relationships** | 59 |
| **Modules** | 6 |
| **2D Shapes** | 7 (Circle, Rectangle, Square, Ellipse, Triangle, Polygon, RegularPolygon) |
| **3D Shapes** | 9 (Sphere, Cube, Cylinder, Cone, Pyramid, Torus, Capsule, Plane, EllipticSphere) |
| **CSG Operations** | 5 (Union, Intersection, Subtraction, SmoothUnion, Xor) |
| **Samplers** | 4 (Random, Stratified, Halton, Adaptive) |
| **Shading Strategies** | 6 (Normal, Distance, Flat, Physical, Lambert, RecursiveLambert) |
| **Intersection Methods** | 4 (RayMarching, InverseSDF, BVH, Analytical) |

---

## ⚙️ Technical Details

### GraphML Features Used
- Node IDs and labels
- Edge source/target relationships
- Edge type attributes (inherits, contains, uses)
- Data attributes for description and module

### Color Scheme (Module-Based)
- **Data**: Blue
- **Geometry**: Green  
- **Material**: Yellow
- **Lighting**: Orange
- **Rendering**: Red
- **Image**: Purple

---

## 🎯 Next Steps

1. ✅ Download/access `classes_drawio.graphml`
2. ✅ Import into draw.io
3. ✅ Explore the diagram
4. ✅ Customize for your needs
5. ✅ Export for presentations/documentation
6. ✅ Share with team

---

**Last Updated:** February 12, 2026  
**Status:** ✓ Ready for import
