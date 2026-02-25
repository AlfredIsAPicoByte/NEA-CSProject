# NEA-CSProject

## Raycasting and Integrations Project

Welcome to the Raycasting for potatoes project! This project explores the fascinating world of raycasting techniques, integrating graphics rendering using both Python and C++ and trying to optimize the technique for midranged and low end computer specs. Whether you're interested in creating simple 3D visualizations or learning about graphics programming, this project provides a solid foundation.

### Features

- **Raycasting Engine**: Implemented in **Python** for the added difficulty.
- **Graphics Rendering**: Visualize scenes with real-time rendering capabilities in **C++** and **OpenGL**.
- **Window Management**: Handle user input and display windows seamlessly across platforms using **GLFW**.
- **Cross-language Integration**: Combine Python's ease of use with C++'s performance for optimized results using **Pybind11**.

### Technologies Used

- **Python**: For scripting the algorithms and rapid prototyping.
- **C++**: For performance-critical components and the application functions.
- **Graphics Libraries**: OpenGL for handling 2D/3D objects.
- **Window Management**: GLFW for handling the user interfaces and user inputs.

## Quickstart (Python)

A short quickstart to activate the project virtual environment and run the renderer.

- Activate the virtual environment (bash/zsh):

```bash
source .venv/bin/activate
```

- Install Python dependencies:

```bash
pip install -r py_src/requirements.txt
```

- Run the renderer (writes images into `benchmark/simple_scene`):

```bash
python py_src/main.py
```

- Or run without activating the venv (explicit path):

```bash
/workspaces/NEA-CSProject/.venv/bin/python -m pip install -r py_src/requirements.txt
/workspaces/NEA-CSProject/.venv/bin/python py_src/main.py
```

Tip: to reduce memory and runtime during automated tests or CI, you can disable post-processing with:

```bash
python py_src/main.py --no-post
```
