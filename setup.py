#!/usr/bin/env python3
"""
One-command bootstrap for the ASD multimodal detection framework.

After cloning the repo, just run:
    python setup.py

What it does:
  1. Checks Python version (>= 3.10 required)
  2. Detects your CUDA version via nvidia-smi
  3. Creates a virtual environment (.venv/)
  4. Installs PyTorch with the correct CUDA build
  5. Installs all remaining dependencies from requirements.txt
  6. Downloads and preprocesses ABIDE I data  (~340 MB download, ~80 MB stored)
  7. Runs the smoke test to verify everything works

Flags:
    --skip-data     Skip ABIDE I download (use if you already have abide_processed/)
    --skip-smoke    Skip the smoke test
    --cpu-only      Install CPU-only PyTorch (slower training, no GPU needed)
    --no-venv       Don't create a venv; install into the current Python environment
"""

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
IS_WINDOWS = platform.system() == "Windows"

# Path to the venv Python / pip executables
if IS_WINDOWS:
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP    = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP    = VENV_DIR / "bin" / "pip"


def banner(msg: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def run(cmd: list[str], *, check: bool = True, capture: bool = False, **kw):
    if not capture:
        print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        **kw,
    )


def venv_run(cmd: list[str], **kw):
    """Run a command using the venv Python."""
    return run([str(VENV_PYTHON), *cmd], **kw)


def venv_pip(args: list[str], **kw):
    return run([str(VENV_PIP), *args], **kw)


# ---------------------------------------------------------------------------
# Step 1 — Python version check
# ---------------------------------------------------------------------------

def check_python() -> None:
    banner("Step 1 / 7 — Checking Python version")
    major, minor = sys.version_info[:2]
    print(f"  Python {major}.{minor} detected")
    if (major, minor) < (3, 10):
        print(f"\n  ERROR: Python 3.10 or later is required.")
        print(f"  Download from https://www.python.org/downloads/")
        sys.exit(1)
    print("  OK")


# ---------------------------------------------------------------------------
# Step 2 — CUDA detection
# ---------------------------------------------------------------------------

CUDA_TO_TORCH_INDEX = {
    (12, 6): "https://download.pytorch.org/whl/cu126",
    (12, 4): "https://download.pytorch.org/whl/cu124",
    (12, 1): "https://download.pytorch.org/whl/cu121",
    (11, 8): "https://download.pytorch.org/whl/cu118",
}


def detect_cuda(cpu_only: bool) -> str | None:
    """Return the PyTorch wheel index URL for the detected CUDA version, or None."""
    banner("Step 2 / 7 — Detecting CUDA")

    if cpu_only:
        print("  --cpu-only flag set; skipping GPU detection")
        return None

    try:
        result = run(["nvidia-smi"], capture=True, check=False)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("  nvidia-smi not found — will install CPU-only PyTorch")
        print("  (GPU training will not be available)")
        return None

    # Parse "CUDA Version: X.Y" from nvidia-smi output
    match = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", result.stdout)
    if not match:
        print("  Could not parse CUDA version from nvidia-smi — using CPU build")
        return None

    cuda_major = int(match.group(1))
    cuda_minor = int(match.group(2))
    print(f"  Detected CUDA {cuda_major}.{cuda_minor}")

    # Find the closest supported wheel (prefer exact, fall back to lower)
    for (maj, min_), url in sorted(CUDA_TO_TORCH_INDEX.items(), reverse=True):
        if (cuda_major, cuda_minor) >= (maj, min_):
            print(f"  Selected PyTorch wheel index: cu{maj}{min_}")
            return url

    print(f"  CUDA {cuda_major}.{cuda_minor} is older than 11.8 — using CPU build")
    return None


# ---------------------------------------------------------------------------
# Step 3 — Virtual environment
# ---------------------------------------------------------------------------

def create_venv(no_venv: bool) -> None:
    banner("Step 3 / 7 — Virtual environment")
    if no_venv:
        print("  --no-venv flag set; skipping venv creation")
        # Point VENV_PYTHON/VENV_PIP at the current interpreter
        global VENV_PYTHON, VENV_PIP
        VENV_PYTHON = Path(sys.executable)
        VENV_PIP    = VENV_PYTHON.parent / ("pip.exe" if IS_WINDOWS else "pip")
        return

    if VENV_DIR.exists():
        print(f"  .venv/ already exists — skipping creation")
    else:
        print(f"  Creating .venv/ ...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("  Done")

    if IS_WINDOWS:
        activate = VENV_DIR / "Scripts" / "activate.bat"
        print(f"\n  To activate later:  {activate}")
    else:
        activate = VENV_DIR / "bin" / "activate"
        print(f"\n  To activate later:  source {activate}")


# ---------------------------------------------------------------------------
# Step 4 — PyTorch
# ---------------------------------------------------------------------------

def install_torch(index_url: str | None) -> None:
    banner("Step 4 / 7 — Installing PyTorch")

    # Check if already installed
    result = venv_run(["-c", "import torch; print(torch.__version__)"],
                      capture=True, check=False)
    if result.returncode == 0:
        print(f"  PyTorch {result.stdout.strip()} already installed — skipping")
        return

    packages = ["torch>=2.1.0", "torchvision>=0.16.0"]
    if index_url:
        venv_pip(["install", *packages, "--index-url", index_url])
    else:
        venv_pip(["install", *packages])

    # Verify
    result = venv_run(["-c", "import torch; print('PyTorch', torch.__version__,"
                       "'| CUDA available:', torch.cuda.is_available())"],
                      capture=True, check=False)
    print(f"  {result.stdout.strip()}")


# ---------------------------------------------------------------------------
# Step 5 — Requirements
# ---------------------------------------------------------------------------

def install_requirements() -> None:
    banner("Step 5 / 7 — Installing dependencies (requirements.txt)")

    # Build a filtered requirements list that excludes torch/torchvision
    # (already installed in step 4 with the correct index URL)
    req_lines = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pkg_name = re.split(r"[>=<!@\s]", stripped)[0].lower()
        if pkg_name in ("torch", "torchvision"):
            continue
        req_lines.append(stripped)

    # Write a filtered temp requirements file
    tmp_req = ROOT / "_setup_requirements_tmp.txt"
    tmp_req.write_text("\n".join(req_lines))

    try:
        venv_pip(["install", "-r", str(tmp_req)])
    finally:
        tmp_req.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 6 — ABIDE I data
# ---------------------------------------------------------------------------

def prepare_data(skip_data: bool) -> None:
    banner("Step 6 / 7 — Downloading and preprocessing ABIDE I")

    processed_dir = ROOT / "abide_processed"

    if skip_data:
        print("  --skip-data flag set; skipping download")
        return

    metadata = processed_dir / "mri" / "metadata.csv"
    if metadata.exists():
        import csv
        with open(metadata) as f:
            n = sum(1 for _ in csv.reader(f)) - 1  # subtract header
        print(f"  abide_processed/ already contains {n} subjects — skipping download")
        return

    print("  Downloading ABIDE I ROI time series and computing FC vectors ...")
    print("  (This will download ~340 MB; processed output is ~80 MB)")
    print("  Expected time: 5-15 minutes depending on your internet speed\n")

    venv_run([
        str(ROOT / "prepare_abide.py"),
        "--dataset", "abide1",
        "--out_dir", str(processed_dir),
    ])


# ---------------------------------------------------------------------------
# Step 7 — Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(skip_smoke: bool) -> None:
    banner("Step 7 / 7 — Smoke test (synthetic data, ~2 min)")

    if skip_smoke:
        print("  --skip-smoke flag set; skipping")
        return

    print("  Running pipeline on 40 synthetic subjects ...")
    venv_run([str(ROOT / "run_experiment.py")])
    print("\n  Smoke test passed!")


# ---------------------------------------------------------------------------
# Final instructions
# ---------------------------------------------------------------------------

def print_next_steps(no_venv: bool) -> None:
    banner("Setup complete")

    if IS_WINDOWS and not no_venv:
        activate_cmd = f".venv\\Scripts\\activate"
    elif not no_venv:
        activate_cmd = "source .venv/bin/activate"
    else:
        activate_cmd = None

    print("\n  Your environment is ready.  To run the full experiment:\n")

    if activate_cmd:
        print(f"    {activate_cmd}")

    print("""
    python run_experiment.py \\
        --real_data \\
        --mri_dir ./abide_processed/mri \\
        --gen_dir ./abide_processed/gen \\
        --max_epochs 100 \\
        --n_folds 5 \\
        --skip_profile \\
        --n_hpo_trials 10

  After training, generate performance plots:

    python results/generate_plots.py

  Results are saved to results/
  See GETTING_STARTED.md for a full explanation of the project and dataset.
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-data",  action="store_true",
                        help="Skip ABIDE I download (abide_processed/ already exists)")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke test")
    parser.add_argument("--cpu-only",   action="store_true",
                        help="Install CPU-only PyTorch (no GPU)")
    parser.add_argument("--no-venv",    action="store_true",
                        help="Install into current Python instead of creating .venv/")
    args = parser.parse_args()

    print("\n  ASD Framework — Environment Bootstrap")
    print(f"  Project root: {ROOT}")
    print(f"  Python:       {sys.executable} ({sys.version.split()[0]})")
    print(f"  Platform:     {platform.system()} {platform.machine()}")

    check_python()
    index_url = detect_cuda(args.cpu_only)
    create_venv(args.no_venv)
    install_torch(index_url)
    install_requirements()
    prepare_data(args.skip_data)
    run_smoke_test(args.skip_smoke)
    print_next_steps(args.no_venv)


if __name__ == "__main__":
    main()
