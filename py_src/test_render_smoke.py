import subprocess
import os
import time

RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), "main.py")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark", "simple_scene")
EXPECTED_FILE = os.path.join(OUT_DIR, "minimal_scene.png")


def test_renderer_smoke_runs_and_writes_image(tmp_path):
    # Ensure output dir exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Run the renderer (small images by default)
    res = subprocess.run(["python3", RENDER_SCRIPT], capture_output=True, text=True, timeout=120)
    print(res.stdout)
    print(res.stderr)

    # Check the command completed successfully
    assert res.returncode == 0, f"Renderer failed: {res.stderr}"

    # Check expected image was written
    assert os.path.exists(EXPECTED_FILE), f"Expected output not found: {EXPECTED_FILE}"
