import platform
import os
import psutil
import subprocess

print(f"System: {platform.system()}")
print(f"Release: {platform.release()}")
print(f"Version: {platform.version()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print(f"Python Version: {platform.python_version()}")

# CPU info
print(f"CPU Cores: {psutil.cpu_count()}")
print(f"CPU Usage: {psutil.cpu_percent()}%")

# Memory info
memory = psutil.virtual_memory()
print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
print(f"Available Memory: {memory.available / (1024**3):.2f} GB")

# Disk info
disk = psutil.disk_usage('/')
print(f"Total Disk: {disk.total / (1024**3):.2f} GB")
print(f"Free Disk: {disk.free / (1024**3):.2f} GB")

# Get system info
result = subprocess.run(['systeminfo'], capture_output=True, text=True)
print(result.stdout)