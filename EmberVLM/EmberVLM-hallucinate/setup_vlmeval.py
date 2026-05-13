"""
Setup script for lmms-eval integration with EmberVLM.
Installs lmms-eval and downloads necessary benchmark data.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   EmberVLM + lmms-eval Setup                            ║
║   This will install lmms-eval for benchmarking          ║
╚══════════════════════════════════════════════════════════╝
""")

    # Check if we're on Linux and install system dependencies first
    if sys.platform.startswith('linux'):
        print("\n🔧 Step 0: Installing system dependencies (Linux)...")
        print("   Installing OpenGL libraries for OpenCV...")
        try:
            subprocess.run("apt-get update -qq", shell=True, check=False, capture_output=True)
            result = subprocess.run(
                "apt-get install -y --fix-missing libgl1-mesa-glx libglib2.0-0",
                shell=True,
                check=False,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("   ✅ System dependencies installed")
            else:
                print("   ⚠️  Could not auto-install system dependencies")
                print("   Trying with just libgl1...")
                result2 = subprocess.run(
                    "apt-get install -y --fix-missing libgl1",
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True
                )
                if result2.returncode == 0:
                    print("   ✅ Installed libgl1 successfully")
                else:
                    print("   ⚠️  System dependency installation had issues, but continuing...")
                    print("   If you see OpenCV errors, run: sudo apt-get install -y --fix-missing libgl1-mesa-glx")
        except Exception as e:
            print(f"   ⚠️  Warning: {e}")
            print("   Continuing anyway - if you see OpenCV errors, run:")
            print("   sudo apt-get install -y --fix-missing libgl1-mesa-glx")

    # Check if lmms-eval directory exists
    lmms_eval_path = Path("../lmms-eval")
    if not lmms_eval_path.exists():
        # Try absolute paths based on platform
        if sys.platform.startswith('win'):
            lmms_eval_path = Path("d:/BabyLM/lmms-eval")
        else:
            # Linux: try common locations
            for candidate in ["/root/lmms-eval", "../lmms-eval", str(Path.home() / "lmms-eval")]:
                if Path(candidate).exists():
                    lmms_eval_path = Path(candidate)
                    break
        
        if not lmms_eval_path.exists():
            print(f"⚠️  lmms-eval not found at {lmms_eval_path.absolute()}")
            print("Please ensure lmms-eval is cloned.")
            print("Run: git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git")
            sys.exit(1)

    # Step 1: Install lmms-eval in development mode
    print("\n🔧 Step 1/3: Installing lmms-eval...")
    os.chdir(lmms_eval_path)
    if not run_command("pip install -e .", "Installing lmms-eval"):
        print("❌ Failed to install lmms-eval")
        sys.exit(1)

    # Step 2: Install additional requirements
    print("\n🔧 Step 2/3: Installing additional dependencies...")
    additional_deps = [
        "opencv-python",  # For image processing
        "decord",        # For video processing (some benchmarks)
        "pandas",        # For data processing
    ]
    
    for dep in additional_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

    # Step 3: Verify installation
    print("\n🔧 Step 3/3: Verifying installation...")
    try:
        import lmms_eval
        print(f"✅ lmms-eval successfully installed!")
        print(f"   Version: {lmms_eval.__version__ if hasattr(lmms_eval, '__version__') else 'dev'}")
        
        # Verify EmberVLM model is registered
        from lmms_eval.models import AVAILABLE_SIMPLE_MODELS
        if 'embervlm' in AVAILABLE_SIMPLE_MODELS:
            print(f"✅ EmberVLM model adapter registered in lmms-eval")
        else:
            print(f"⚠️  Warning: EmberVLM not found in model registry")
            print(f"   You may need to re-clone lmms-eval from: https://github.com/euhidaman/lmms-eval.git")
            
    except ImportError as e:
        print(f"❌ lmms-eval import failed: {e}")
        print("\nTrying alternative verification...")
        
        # Try to verify installation by checking if package is installed
        import subprocess
        result = subprocess.run(
            ["pip", "show", "lmms-eval"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ lmms-eval package is installed (pip shows it)")
            print("   Note: You may need to restart your Python session or terminal")
            print("   Run: source /root/venv/bin/activate  # or restart your shell")
        else:
            print("❌ Installation verification failed")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Return to original directory
    os.chdir("../EmberVLM")

    print("""
╔══════════════════════════════════════════════════════════╗
║   ✅ Setup Complete!                                     ║
║                                                          ║
║   lmms-eval is now integrated with EmberVLM             ║
║                                                          ║
║   Benchmark data will be downloaded automatically       ║
║   when you run evaluation for the first time.           ║
║                                                          ║
║   To test benchmarking:                                 ║
║   python scripts/train_all.py --stage 2.5 \\             ║
║          --benchmark_subset quick                       ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
