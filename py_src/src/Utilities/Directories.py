from pathlib import Path

def find_project_root(marker_files=('pyproject.toml', 'setup.py', '.git')):
    """
    Find project root by looking for marker files.
    
    Args:
        marker_files: Tuple of filenames that indicate project root
        
    Returns:
        Path object pointing to project root
    """
    current = Path(__file__).resolve().parent
    
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    
    # Fallback: assume script is at root
    return current