import sys
import os
import platform
import psutil
import subprocess
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

from ..Directories import get_project_root


def get_cpu_info() -> Dict[str, Any]:
    """Collect detailed CPU information."""
    info = {
        'processor': platform.processor(),
        'architecture': platform.machine(),
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'max_frequency_mhz': None,
        'current_frequency_mhz': None,
    }
    
    try:
        freq = psutil.cpu_freq()
        if freq:
            info['max_frequency_mhz'] = f"{freq.max:.0f}"
            info['current_frequency_mhz'] = f"{freq.current:.0f}"
    except:
        pass
    
    return info

def get_memory_info() -> Dict[str, Any]:
    """Collect memory information."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_ram_gb': f"{memory.total / (1024**3):.2f}",
        'available_ram_gb': f"{memory.available / (1024**3):.2f}",
        'ram_usage_percent': f"{memory.percent:.1f}",
        'total_swap_gb': f"{swap.total / (1024**3):.2f}",
        'swap_usage_percent': f"{swap.percent:.1f}",
    }

def get_disk_info() -> Dict[str, Any]:
    """Collect disk information."""
    disk = psutil.disk_usage('/')
    
    return {
        'total_disk_gb': f"{disk.total / (1024**3):.2f}",
        'free_disk_gb': f"{disk.free / (1024**3):.2f}",
        'disk_usage_percent': f"{disk.percent:.1f}",
    }

def get_os_info() -> Dict[str, Any]:
    """Collect operating system information."""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'platform': platform.platform(),
    }

def get_python_info() -> Dict[str, Any]:
    """Collect Python environment information."""
    return {
        'version': platform.python_version(),
        'implementation': platform.python_implementation(),
        'compiler': platform.python_compiler(),
        'executable': sys.executable,
    }

def get_gpu_info() -> Dict[str, Any]:
    """Attempt to collect GPU information (requires nvidia-smi for NVIDIA)."""
    info = {'gpu_available': False}
    
    try:
        # Try NVIDIA GPUs
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(', ')
            if len(gpu_data) >= 2:
                info['gpu_available'] = True
                info['gpu_name'] = gpu_data[0]
                info['gpu_memory'] = gpu_data[1]
                if len(gpu_data) >= 3:
                    info['driver_version'] = gpu_data[2]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return info


def format_section(title: str, data: Dict[str, Any], indent: int = 2) -> str:
    """Format a section with key-value pairs."""
    lines = [f"{title}:"]
    indent_str = " " * indent
    
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
    
    for key, value in data.items():
        if value is not None:
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"{indent_str}{formatted_key:<{max_key_len + 2}}: {value}")
    
    return "\n".join(lines)


def generate_benchmark_report(custom_id: Optional[str] = None) -> str:
    """Generate a complete benchmark system report."""
    sections = []
    
    # Header
    sections.append("=" * 70)
    sections.append(f"SYSTEM BENCHMARK REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if custom_id:
        sections.append(f"ID: {custom_id}")
    sections.append("=" * 70)
    sections.append("")
    
    # Operating System
    sections.append(format_section("Operating System", get_os_info()))
    sections.append("")
    
    # Python Environment
    sections.append(format_section("Python Environment", get_python_info()))
    sections.append("")
    
    # CPU
    sections.append(format_section("CPU", get_cpu_info()))
    sections.append("")
    
    # Memory
    sections.append(format_section("Memory", get_memory_info()))
    sections.append("")
    
    # Disk
    sections.append(format_section("Disk", get_disk_info()))
    sections.append("")
    
    # GPU (if available)
    gpu_info = get_gpu_info()
    if gpu_info.get('gpu_available'):
        sections.append(format_section("GPU", gpu_info))
        sections.append("")
    
    sections.append("=" * 70)
    
    return "\n".join(sections)


def save_report(filepath: str, custom_id: Optional[str] = None):
    """
    Generate and save the benchmark report to a file.
    
    Args:
        filepath: Full path where the report will be saved
        custom_id: Optional identifier to include in the report
    """
    report = generate_benchmark_report(custom_id=custom_id)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
            print(" + Wrote system info")
            print(f" > Saved to {filepath}")
    except Exception as e:
        print(f" / Failed to write system info: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System Information Reporter for Benchmarking")
    parser.add_argument(
        "--name", "-n",
        dest="filename",
        default="system_info",
        help="Base filename for the report (without extension)"
    )
    parser.add_argument(
        "--id",
        dest="custom_id",
        default=None,
        help="Custom identifier to include in the report"
    )
    parser.add_argument(
        "--sections", "-s",
        nargs='+',
        choices=['os', 'python', 'cpu', 'memory', 'disk', 'gpu'],
        help="Specific sections to include (default: all)"
    )
    parser.add_argument(
        "--console-only", "-c",
        action="store_true",
        help="Only print to console, don't save to file"
    )
    
    args = parser.parse_args()

    # Setup output directory
    PROJECT_ROOT = os.path.join(get_project_root(), "py_src")
    REP_OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "benchmarking", "system")
    os.makedirs(REP_OUT_DIR, exist_ok=True)

    # Generate report based on selected sections
    if args.sections:
        # Custom sections
        section_map = {
            'os': ('Operating System', get_os_info),
            'python': ('Python Environment', get_python_info),
            'cpu': ('CPU', get_cpu_info),
            'memory': ('Memory', get_memory_info),
            'disk': ('Disk', get_disk_info),
            'gpu': ('GPU', get_gpu_info),
        }
        
        report_parts = []
        report_parts.append("=" * 70)
        report_parts.append(f"SYSTEM BENCHMARK REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.custom_id:
            report_parts.append(f"ID: {args.custom_id}")
        report_parts.append("=" * 70)
        report_parts.append("")
        
        for section_key in args.sections:
            title, func = section_map[section_key]
            data = func()
            # Skip GPU if not available
            if section_key == 'gpu' and not data.get('gpu_available'):
                continue
            report_parts.append(format_section(title, data))
            report_parts.append("")
        
        report_parts.append("=" * 70)
        report = "\n".join(report_parts)
    else:
        # Full report
        report = generate_benchmark_report(custom_id=args.custom_id)
    
    # Print to console
    print(report)
    print()
    
    # Save to file unless console-only
    if not args.console_only:
        output_path = os.path.join(REP_OUT_DIR, f"{args.filename}.txt")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
                print(" + Wrote system info")
                print(f" > Saved to {output_path}")
        except Exception as e:
            print(f" / Failed to write system info: {e}")
            import traceback
            traceback.print_exc()