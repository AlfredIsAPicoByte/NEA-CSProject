# Drawing Class Diagrams with draw.io

This guide explains how to import the NEA-CSProject class hierarchy into draw.io to visualize the software architecture.

## Files Available

### 1. `classes_drawio.graphml` (Recommended)
A GraphML format file optimized for draw.io import. This file contains:
- All classes organized by module (Data, Geometry, Material, Lighting, Rendering, Image)
- Inheritance relationships shown as edges
- Class descriptions and module information

### 2. `classes.xml` (Reference)
The original comprehensive XML documentation with detailed:
- Class attributes
- Method signatures
- Parameter types and return values
- Full API documentation

## How to Import into draw.io

### Option 1: Using GraphML File (Easiest)

1. Go to [draw.io](https://app.diagrams.net/)
2. Create a new diagram or open existing one
3. Click **File → Import from → GraphML**
4. Select `classes_drawio.graphml`
5. draw.io will automatically create a diagram with:
   - Nodes representing classes
   - Edges representing inheritance and relationships
   - Color coding by module
   - Auto-layout for better visualization

### Option 2: Using draw.io Desktop

If using draw.io desktop app:
1. Open draw.io
2. **File → Open**
3. Navigate to `classes_drawio.graphml`
4. Click **Import**

## Interpreting the Diagram

### Color Coding
- **Data Module** - Blue tones
- **Geometry Module** - Green tones
- **Material Module** - Yellow tones
- **Lighting Module** - Orange tones
- **Rendering Module** - Red tones
- **Image Module** - Purple tones

### Edge Types
- **Solid arrows** - Inheritance (class A inherits from class B)
- **Dashed arrows** - Composition (class A contains class B)
- **Dotted arrows** - Usage/Association (class A uses class B)

### Node Types
- **Square nodes** - Regular classes
- **Diamond nodes** - Abstract classes/interfaces
- **Hexagon nodes** - Enumerations

## Quick Navigation in Diagram

### Focusing on Specific Areas

1. **Ray Tracing Pipeline**
   - Start from `RayTracer` → `IntersectionStrategy` → `ShadingStrategy`

2. **Shape System**
   - Start from `SignedDistanceShape` → `SignedDistanceShape2D/3D` → Specific shapes

3. **Material System**
   - Start from `PBRMaterial` → `MaterialData` → `PBRMaterial` (circular uses)

4. **Scene Management**
   - Start from `Scene` → `SceneNode` → `Transform`

## Customizing the Diagram

### Add Styling
1. Right-click on element
2. **Format → Style** to change:
   - Colors
   - Line styles
   - Font sizes
   - Shapes

### Organize by Module
1. Use **Ctrl+A** to select all
2. Click **Arrange → Layout → Hierarchical** for automatic organization
3. Adjust layout direction: **Arrange → Layout Orientation**

### Add Annotations
1. Insert text boxes: **Insert → Text**
2. Add method details in boxes
3. Draw connecting lines with **Insert → Connector**

### Create Simplified Views
1. **File → New** to create filtered diagrams
2. Copy only relevant classes and relationships
3. Create separate diagrams for:
   - Data structures
   - Ray tracing pipeline
   - Shape hierarchy
   - Material system

## Export Options

After creating/customizing your diagram:

### Export as Image
1. **File → Export as → PNG/JPG/SVG**
2. Choose resolution and background

### Export as PDF
1. **File → Export as → PDF**
2. Great for documentation

### Share Diagram
1. **File → Share**
2. Get sharable link or embed code

## Advanced Features

### Add Hyperlinks
1. Right-click element
2. **Edit Link**
3. Link to GitHub source files: 
   ```
   https://github.com/AlfredIsAPicoByte/NEA-CSProject/blob/feature/python-raycasting-link-cpp/py_src/src/[path]/[file].py
   ```

### Create Swimlanes
1. **Insert → Swimlane**
2. Drag to create lanes for each module
3. Move classes into appropriate lanes

### Add Legend
1. **Insert → Text**
2. Create legend showing:
   - Node types (class, abstract, enum)
   - Edge relationships (inherits, contains, uses)
   - Module colors

## Tips & Tricks

1. **Use Search**: Ctrl+F to find specific classes
2. **Zoom**: Use Ctrl+Scroll wheel or view menu
3. **Mini Map**: View → Show Outline for navigation
4. **Undo/Redo**: Ctrl+Z / Ctrl+Y
5. **Auto-arrange**: After adding/removing elements
   - Ctrl+Shift+A to arrange
6. **Connection Mode**: Hold Shift while dragging to create connections
7. **Snap to Grid**: View → Snap helps align elements

## Generating Custom Diagrams

### UML Class Diagram
1. Create new diagram
2. Use `classes_drawio.graphml` as reference
3. Add UML notation:
   - Visibility symbols (+, -, #, ~)
   - Parameter lists in methods
   - Multiplicity on relationships

### Inheritance Tree
1. Focus on class hierarchies
2. Use tree layout
3. Highlight deepest inheritance chains

### Module Dependency Diagram
1. Create one node per module
2. Draw connections showing module dependencies
3. Label with key classes in each module

## Troubleshooting

### Import Fails
- Check file format is valid GraphML
- Try using **File → Import from → XML** instead
- Ensure file is not corrupted

### Diagram Won't Auto-Layout
- Select all nodes: Ctrl+A
- **Arrange → Layout → Hierarchical**
- Adjust layout options in bottom panel

### Text Too Small/Large
- Select all: Ctrl+A
- **Format → Text → Font Size**
- Or manually adjust with right panel

### Too Many Connections/Cluttered
- Right-click → **Show/Hide**
- Create separate diagrams for different relationship types
- Use swimlanes to organize by module

## Additional Resources

- [draw.io Documentation](https://www.diagrams.net/doc/)
- [GraphML Specification](https://graphml.graphdrawing.org/)
- [UML Class Diagram Guide](https://www.lucidchart.com/pages/uml-class-diagram)

## Contributing

To update the class hierarchy diagrams:
1. Modify `classes_drawio.graphml` with any new classes/relationships
2. Update node descriptions to match source code
3. Run validation: `xmllint classes_drawio.graphml`
4. Commit changes to repository
